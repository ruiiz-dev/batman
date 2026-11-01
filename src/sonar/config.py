"""Immutable sonar configuration with validation and derived quantities."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

from sonar.errors import ConfigError

Window = Literal["tukey", "hann"]

NEAR_ULTRASONIC_BAND: tuple[float, float] = (18_000.0, 22_000.0)
AUDIBLE_BAND: tuple[float, float] = (8_000.0, 16_000.0)

# Hard ceiling for the digital amplitude, regardless of what the user configures.
ABSOLUTE_MAX_AMPLITUDE = 1.0


def speed_of_sound(temperature_c: float) -> float:
    """Speed of sound in dry air (m/s) for a temperature in °C."""
    return 331.3 * math.sqrt(1.0 + temperature_c / 273.15)


@dataclass(frozen=True, slots=True)
class SonarConfig:
    # Sampling and waveform.
    fs: int = 48_000
    f_start: float = NEAR_ULTRASONIC_BAND[0]
    f_end: float = NEAR_ULTRASONIC_BAND[1]
    chirp_duration: float = 0.008
    window: Window = "tukey"
    tukey_alpha: float = 0.25

    # Ping train timing.
    n_pings: int = 16
    # Extra leading pings that are emitted but not averaged: laptop speaker amps
    # wake up from power saving during the first few hundred milliseconds.
    warmup_pings: int = 2
    ping_interval: float = 0.06
    pre_silence: float = 0.05
    latency_margin: float = 0.5
    # Largest buffer slip (samples) that per-ping re-alignment will search for.
    max_slip_samples: int = 128

    # Propagation. ``speed_of_sound_override`` wins over the temperature model.
    temperature_c: float = 20.0
    speed_of_sound_override: float | None = None
    # Speaker→microphone path length. Added to the measured round trip so that a
    # distance is referred to the device instead of the direct path (0 = spec formula).
    direct_path_m: float = 0.0

    # Safety.
    amplitude: float = 0.1
    max_amplitude: float = 0.5

    # Detection.
    min_range: float = 0.15
    max_range: float = 5.0
    n_echoes: int = 5
    min_snr_db: float = 12.0
    # A peak must also stand this far above its local (OS-CFAR) neighbourhood. Kept
    # separate from ``min_snr_db``: the global threshold bounds false alarms, the
    # local one only rejects extended clutter.
    cfar_snr_db: float = 8.0
    min_direct_snr_db: float = 15.0
    sidelobe_margin_db: float = 6.0
    clip_level: float = 0.99

    # Devices (index or name substring; None = system default).
    input_device: int | str | None = None
    output_device: int | str | None = None
    # Restrict name matching to one host API (e.g. "WASAPI"); devices are listed once per API.
    host_api: str | None = None
    # WASAPI exclusive mode bypasses Windows voice processing (AEC, noise gate, AGC).
    exclusive: bool = False

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ derived
    @property
    def speed_of_sound(self) -> float:
        if self.speed_of_sound_override is not None:
            return self.speed_of_sound_override
        return speed_of_sound(self.temperature_c)

    @property
    def bandwidth(self) -> float:
        return self.f_end - self.f_start

    @property
    def range_resolution(self) -> float:
        """Two echoes closer than this (m) merge into one matched-filter peak."""
        return self.speed_of_sound / (2.0 * self.bandwidth)

    @property
    def chirp_samples(self) -> int:
        return int(round(self.chirp_duration * self.fs))

    @property
    def frame_samples(self) -> int:
        return int(round(self.ping_interval * self.fs))

    @property
    def pre_silence_samples(self) -> int:
        return int(round(self.pre_silence * self.fs))

    @property
    def mainlobe_samples(self) -> int:
        """Approximate width of the matched-filter envelope main lobe."""
        return max(2, int(math.ceil(self.fs / self.bandwidth)))

    @property
    def guard_samples(self) -> int:
        """Samples kept before the direct path inside each averaged frame."""
        return self.chirp_samples

    def range_to_samples(self, distance_m: float) -> float:
        return (2.0 * distance_m - self.direct_path_m) / self.speed_of_sound * self.fs

    def samples_to_range(self, lag_samples: float) -> float:
        return (self.speed_of_sound * lag_samples / self.fs + self.direct_path_m) / 2.0

    def fingerprint(self) -> dict[str, Any]:
        """Parameters that must match for a background profile to be reusable."""
        return {
            "fs": self.fs,
            "f_start": self.f_start,
            "f_end": self.f_end,
            "chirp_duration": self.chirp_duration,
            "window": self.window,
            "tukey_alpha": self.tukey_alpha,
            "ping_interval": self.ping_interval,
        }

    def with_overrides(self, **changes: Any) -> SonarConfig:
        """Return a validated copy; ``None`` values are ignored."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})

    # --------------------------------------------------------------- validation
    def _validate(self) -> None:
        nyquist = self.fs / 2.0
        if self.fs <= 0:
            raise ConfigError(f"fs debe ser positivo (recibido {self.fs}).")
        if not 0 < self.f_start < self.f_end:
            raise ConfigError(
                f"Banda inválida: {self.f_start:.0f}–{self.f_end:.0f} Hz.",
                "Usa 0 < f_start < f_end, por ejemplo 18000–22000.",
            )
        if self.f_end >= 0.98 * nyquist:
            raise ConfigError(
                f"f_end={self.f_end:.0f} Hz está demasiado cerca de Nyquist ({nyquist:.0f} Hz).",
                "Baja f_end o sube fs (p. ej. 48000 o 96000 si el hardware lo soporta).",
            )
        if self.window not in ("tukey", "hann"):
            raise ConfigError(f"Ventana desconocida: {self.window!r}.", "Usa 'tukey' o 'hann'.")
        if not 0.0 <= self.tukey_alpha <= 1.0:
            raise ConfigError("tukey_alpha debe estar en [0, 1].")
        if self.chirp_duration <= 0 or self.chirp_samples < 16:
            raise ConfigError("chirp_duration es demasiado corto para fs.")
        if self.n_pings < 1:
            raise ConfigError("n_pings debe ser ≥ 1.")
        if self.max_slip_samples < 0:
            raise ConfigError("max_slip_samples no puede ser negativo.")
        if self.warmup_pings < 0:
            raise ConfigError("warmup_pings no puede ser negativo.")
        if self.pre_silence_samples < 2 * self.chirp_samples:
            raise ConfigError(
                "pre_silence debe durar al menos dos chirps (se usa para estimar el ruido)."
            )
        if self.latency_margin < 0:
            raise ConfigError("latency_margin no puede ser negativo.")
        if self.speed_of_sound <= 0:
            raise ConfigError("La velocidad del sonido debe ser positiva.")
        if self.direct_path_m < 0:
            raise ConfigError("direct_path_m no puede ser negativo.")
        if not 0 < self.max_amplitude <= ABSOLUTE_MAX_AMPLITUDE:
            raise ConfigError(f"max_amplitude debe estar en (0, {ABSOLUTE_MAX_AMPLITUDE}].")
        if not 0 < self.amplitude <= self.max_amplitude:
            raise ConfigError(
                f"amplitude={self.amplitude} excede el tope de seguridad max_amplitude="
                f"{self.max_amplitude}.",
                "Baja --amplitude o, si sabes lo que haces, sube --max-amplitude "
                "(el casi-ultrasonido fuerte molesta a mascotas y puede dañar tweeters).",
            )
        if not 0 <= self.min_range < self.max_range:
            raise ConfigError("Se requiere 0 ≤ min_range < max_range.")
        needed = self.guard_samples + self.range_to_samples(self.max_range) + self.chirp_samples
        if needed >= self.frame_samples:
            raise ConfigError(
                f"ping_interval={self.ping_interval * 1e3:.0f} ms es demasiado corto para "
                f"max_range={self.max_range} m.",
                f"Usa ping_interval ≥ {needed / self.fs * 1e3:.0f} ms o reduce max_range.",
            )
        if self.n_echoes < 1:
            raise ConfigError("n_echoes debe ser ≥ 1.")
        if not 0 < self.clip_level <= 1:
            raise ConfigError("clip_level debe estar en (0, 1].")
