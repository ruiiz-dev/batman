"""Audio port (``AudioDevice``) plus the sounddevice adapter and an in-memory fake."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from sonar.errors import (
    AudioIOError,
    ConfigError,
    DeviceNotFoundError,
    MicrophonePermissionError,
    UnsupportedSampleRateError,
    microphone_permission_hint,
)
from sonar.simulation import Scene, simulate_recording, simulate_silence

if TYPE_CHECKING:
    import types

FloatArray = NDArray[np.float64]

COMMON_SAMPLE_RATES: tuple[int, ...] = (44_100, 48_000, 88_200, 96_000, 192_000)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    supported_samplerates: tuple[int, ...]
    is_default_input: bool = False
    is_default_output: bool = False


@runtime_checkable
class AudioDevice(Protocol):
    """Port used by the domain. Implementations return mono float64 recordings."""

    def playrec(self, signal: FloatArray, fs: int) -> FloatArray:
        """Play ``signal`` and record simultaneously (same clock); same length as ``signal``."""
        ...

    def record(self, duration: float, fs: int) -> FloatArray:
        """Record ambient sound only."""
        ...

    def check_settings(self, fs: int) -> None:
        """Raise a ``SonarError`` if the devices cannot run at ``fs``."""
        ...

    def list_devices(self) -> list[DeviceInfo]: ...


def _guard_amplitude(signal: FloatArray, max_amplitude: float) -> None:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > max_amplitude + 1e-9:
        raise ConfigError(
            f"Se bloqueó la reproducción: pico {peak:.3f} > tope de seguridad {max_amplitude:.3f}.",
            "Baja --amplitude o sube --max-amplitude de forma consciente.",
        )


# --------------------------------------------------------------------------- real


class SoundDeviceAudio:
    """Adapter over ``sounddevice`` (PortAudio). Imported lazily so tests never need it."""

    def __init__(
        self,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        max_amplitude: float = 0.5,
        host_api: str | None = None,
        exclusive: bool = False,
    ) -> None:
        self._input = input_device
        self._output = output_device
        self._max_amplitude = max_amplitude
        self._host_api = host_api
        self._exclusive = exclusive
        self._sd_module: types.ModuleType | None = None

    # ----------------------------------------------------------------- helpers
    @property
    def _sd(self) -> types.ModuleType:
        if self._sd_module is None:
            try:
                import sounddevice
            except OSError as exc:  # PortAudio shared library missing.
                raise AudioIOError(
                    f"No se pudo cargar PortAudio: {exc}",
                    "En Linux instala `libportaudio2` (apt) o `portaudio` (dnf/pacman).",
                ) from exc
            self._sd_module = sounddevice
        return self._sd_module

    def _hostapi_name(self, index: int) -> str:
        info: Any = self._sd.query_devices(index)
        return str(self._sd.query_hostapis(info["hostapi"])["name"])

    def _resolve(self, device: int | str | None, kind: str) -> int:
        sd = self._sd
        label = "entrada" if kind == "input" else "salida"
        flag = "--input" if kind == "input" else "--output"
        if isinstance(device, str) and self._host_api:
            # Indices shift when devices come and go; name + host API is stable.
            channels = "max_input_channels" if kind == "input" else "max_output_channels"
            hostapis: Any = sd.query_hostapis()
            matches = [
                int(d["index"])
                for d in sd.query_devices()
                if device.lower() in str(d["name"]).lower()
                and self._host_api.lower() in str(hostapis[d["hostapi"]]["name"]).lower()
                and d[channels] > 0
            ]
            if len(matches) != 1:
                found = "ninguno" if not matches else f"varios: {matches}"
                raise DeviceNotFoundError(
                    f"Dispositivo de {label} {device!r} en host API {self._host_api!r}: {found}.",
                    f"Ejecuta `sonar devices` y usa un nombre más específico con {flag}.",
                )
            return matches[0]
        try:
            info: Any = sd.query_devices(device, kind=kind)
        except (ValueError, sd.PortAudioError) as exc:
            raise DeviceNotFoundError(
                f"Dispositivo de {label} {device!r} no encontrado: {exc}",
                f"Ejecuta `sonar devices` y usa el índice o parte del nombre con {flag} "
                "(combínalo con --host-api si el nombre se repite).",
            ) from exc
        return int(info["index"])

    def _extra_settings(self, index: int) -> Any:
        if not self._exclusive:
            return None
        if "wasapi" not in self._hostapi_name(index).lower():
            raise ConfigError(
                "El modo exclusivo sólo existe en dispositivos WASAPI (Windows).",
                "Usa --host-api WASAPI o quita --exclusive.",
            )
        return self._sd.WasapiSettings(exclusive=True)

    def _devices(self) -> tuple[int, int]:
        return self._resolve(self._input, "input"), self._resolve(self._output, "output")

    def _translate(self, exc: Exception, fs: int) -> Exception:
        msg = str(exc)
        lower = msg.lower()
        if "sample rate" in lower or "-9997" in msg:
            return UnsupportedSampleRateError(
                f"El dispositivo no soporta fs={fs} Hz ({msg}).",
                "Revisa `sonar devices` y usa --fs con una tasa soportada (48000 es la más común).",
            )
        if "invalid device" in lower or "-9996" in msg or "-9998" in msg:
            return DeviceNotFoundError(f"Dispositivo inválido ({msg}).", "Ejecuta `sonar devices`.")
        if "unanticipated host error" in lower or "-9999" in msg:
            return MicrophonePermissionError(
                f"El sistema rechazó abrir el audio ({msg}).", microphone_permission_hint()
            )
        if "unavailable" in lower or "busy" in lower or "-9985" in msg:
            return AudioIOError(
                f"El dispositivo está ocupado ({msg}).",
                "Cierra otras apps que usen el micrófono/parlante (Zoom, Discord, navegador).",
            )
        return AudioIOError(f"Error de audio: {msg}")

    # ------------------------------------------------------------------- port
    def check_settings(self, fs: int) -> None:
        sd = self._sd
        in_dev, out_dev = self._devices()
        try:
            sd.check_input_settings(device=in_dev, channels=1, samplerate=fs, dtype="float32",
                                    extra_settings=self._extra_settings(in_dev))
            sd.check_output_settings(device=out_dev, channels=self._output_channels(out_dev),
                                     samplerate=fs, dtype="float32",
                                     extra_settings=self._extra_settings(out_dev))
        except sd.PortAudioError as exc:
            raise self._translate(exc, fs) from exc

    def _output_channels(self, out_dev: int) -> int:
        info: Any = self._sd.query_devices(out_dev)
        return max(1, min(2, int(info["max_output_channels"])))

    def playrec(self, signal: FloatArray, fs: int) -> FloatArray:
        _guard_amplitude(signal, self._max_amplitude)
        self.check_settings(fs)
        sd = self._sd
        in_dev, out_dev = self._devices()
        channels = self._output_channels(out_dev)
        # Same chirp on every output channel (laptop speakers are usually a stereo pair).
        out = np.repeat(signal.astype(np.float32)[:, None], channels, axis=1)
        try:
            rec = sd.playrec(
                out,
                samplerate=fs,
                input_mapping=[1],
                device=(in_dev, out_dev),
                dtype="float32",
                blocking=True,
                extra_settings=(self._extra_settings(in_dev), self._extra_settings(out_dev)),
            )
        except sd.PortAudioError as exc:
            raise self._translate(exc, fs) from exc
        return np.asarray(rec[:, 0], dtype=np.float64)

    def record(self, duration: float, fs: int) -> FloatArray:
        self.check_settings(fs)
        sd = self._sd
        in_dev, _ = self._devices()
        try:
            rec = sd.rec(
                int(round(duration * fs)),
                samplerate=fs,
                channels=1,
                device=in_dev,
                dtype="float32",
                blocking=True,
                extra_settings=self._extra_settings(in_dev),
            )
        except sd.PortAudioError as exc:
            raise self._translate(exc, fs) from exc
        return np.asarray(rec[:, 0], dtype=np.float64)

    def list_devices(self) -> list[DeviceInfo]:
        sd = self._sd
        hostapis: Any = sd.query_hostapis()
        default_in, default_out = sd.default.device
        result = []
        for raw in sd.query_devices():
            dev: Any = raw
            idx = int(dev["index"])
            rates = []
            for fs in COMMON_SAMPLE_RATES:
                try:
                    if dev["max_input_channels"] > 0:
                        sd.check_input_settings(device=idx, channels=1, samplerate=fs)
                    if dev["max_output_channels"] > 0:
                        sd.check_output_settings(device=idx, channels=1, samplerate=fs)
                except sd.PortAudioError:
                    continue
                rates.append(fs)
            result.append(
                DeviceInfo(
                    index=idx,
                    name=str(dev["name"]),
                    hostapi=str(hostapis[dev["hostapi"]]["name"]),
                    max_input_channels=int(dev["max_input_channels"]),
                    max_output_channels=int(dev["max_output_channels"]),
                    default_samplerate=float(dev["default_samplerate"]),
                    supported_samplerates=tuple(rates),
                    is_default_input=idx == default_in,
                    is_default_output=idx == default_out,
                )
            )
        return result


# --------------------------------------------------------------------------- fake


@dataclass
class FakeAudioDevice:
    """Deterministic in-memory device driven by a synthetic ``Scene``."""

    scene: Scene = field(default_factory=Scene)
    seed: int = 0
    max_amplitude: float = 0.5
    supported_samplerates: tuple[int, ...] = (44_100, 48_000, 96_000)
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def check_settings(self, fs: int) -> None:
        if fs not in self.supported_samplerates:
            raise UnsupportedSampleRateError(
                f"El dispositivo no soporta fs={fs} Hz.",
                f"Tasas soportadas: {', '.join(map(str, self.supported_samplerates))}.",
            )

    def playrec(self, signal: FloatArray, fs: int) -> FloatArray:
        _guard_amplitude(signal, self.max_amplitude)
        self.check_settings(fs)
        self.calls.append("playrec")
        return simulate_recording(np.asarray(signal, dtype=np.float64), fs, self.scene, self._rng)

    def record(self, duration: float, fs: int) -> FloatArray:
        self.check_settings(fs)
        self.calls.append("record")
        return simulate_silence(int(round(duration * fs)), self.scene, self._rng)

    def list_devices(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(0, "Fake Microphone", "Simulated", 1, 0, 48_000.0,
                       self.supported_samplerates, is_default_input=True),
            DeviceInfo(1, "Fake Speaker", "Simulated", 0, 2, 48_000.0,
                       self.supported_samplerates, is_default_output=True),
        ]
