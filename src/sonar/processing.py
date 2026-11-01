"""Signal processing: band-pass, matched filter, envelope, coherent averaging and
background subtraction. Everything here is pure (arrays in, arrays out) except
``BackgroundProfile.save/load``, which only touch the path they are given.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, correlate, hilbert, sosfiltfilt

from sonar.config import SonarConfig
from sonar.errors import (
    AudioIOError,
    BandOutOfResponseError,
    ClippingError,
    IncompatibleBackgroundError,
    LowSNRError,
    MicrophonePermissionError,
    microphone_permission_hint,
)

FloatArray = NDArray[np.float64]

# Median of an exponential distribution is mean·ln2: converts a robust median of
# envelope power (Rayleigh noise → exponential power) into a mean-noise estimate.
_LN2 = math.log(2.0)
# In-band noise below this level counts as a "quiet room" when classifying failures.
QUIET_NOISE_DBFS = -55.0


# ----------------------------------------------------------------- primitives


def bandpass(x: FloatArray, fs: int, f_lo: float, f_hi: float, order: int = 4) -> FloatArray:
    """Zero-phase Butterworth band-pass (zero phase → no bias on arrival times)."""
    nyq = fs / 2.0
    lo = max(0.9 * f_lo, 20.0)
    hi = min(1.1 * f_hi, 0.99 * nyq)
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def matched_filter(x: FloatArray, template: FloatArray) -> FloatArray:
    """Cross-correlation indexed by lag: ``out[k]`` peaks when the template starts at ``x[k]``."""
    full = correlate(x, template, mode="full", method="fft")
    return full[template.size - 1 :]


def envelope(x: FloatArray) -> FloatArray:
    """Magnitude of the analytic signal (Hilbert envelope)."""
    return np.abs(hilbert(x))


def parabolic_peak(y: FloatArray, i: int) -> float:
    """Sub-sample location of the local maximum at ``i`` (3-point parabola)."""
    if i <= 0 or i >= y.size - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2.0 * b + c
    if den >= 0:
        return float(i)
    return i + 0.5 * (a - c) / den


def fractional_shift(x: FloatArray, shift: float) -> FloatArray:
    """Delay ``x`` by ``shift`` samples (may be fractional/negative), keeping its length."""
    n = x.size
    nfft = 1 << int(math.ceil(math.log2(n + 2 * abs(shift) + 2)))
    spectrum = np.fft.rfft(x, nfft)
    k = np.fft.rfftfreq(nfft)
    return np.fft.irfft(spectrum * np.exp(-2j * np.pi * k * shift), nfft)[:n]


def coherent_average(x: FloatArray, starts: list[int], length: int) -> FloatArray:
    """Average equal-length segments; uncorrelated noise drops by √N in amplitude."""
    return np.mean([x[s : s + length] for s in starts], axis=0)


def align_frames(frames: list[FloatArray], region: slice, max_lag: int) -> list[FloatArray]:
    """Shift each frame so that ``region`` lines up with the majority of frames.

    Real full-duplex streams occasionally drop or repeat a whole buffer, which
    moves every later ping by tens of samples; even one sample of misalignment
    destroys coherent averaging at several kHz. The lag of each frame is found by
    phase-sensitive cross-correlation (sub-sample, parabolic), first against frame
    0 and normalised to the median lag, then refined against the provisional mean.
    """
    centre = region.stop - region.start - 1

    def lags_against(ref: FloatArray) -> FloatArray:
        lags = []
        for f in frames:
            xc = correlate(f[region], ref[region], mode="full", method="fft")
            lo, hi = max(centre - max_lag, 0), min(centre + max_lag + 1, xc.size)
            j = lo + int(np.argmax(xc[lo:hi]))
            lags.append(parabolic_peak(xc, j) - centre)
        return np.asarray(lags)

    lags = lags_against(frames[0])
    lags -= np.median(lags)
    provisional = np.mean([fractional_shift(f, -lag) for f, lag in zip(frames, lags, strict=True)], axis=0)
    lags = lags_against(provisional)
    return [fractional_shift(f, -lag) for f, lag in zip(frames, lags, strict=True)]


def autocorrelation_envelope(template: FloatArray) -> FloatArray:
    """One-sided, peak-normalised envelope of the template autocorrelation.

    Gives the expected range-sidelobe level at each lag, used to reject sidelobes
    of strong arrivals as fake echoes.
    """
    full = correlate(template, template, mode="full", method="fft")
    env = envelope(full)[template.size - 1 :]
    return env / env[0]


def robust_noise_power(power: FloatArray) -> float:
    """Mean noise power estimated from the median (robust to sparse strong echoes)."""
    return float(np.median(power)) / _LN2 if power.size else 0.0


def check_recording(recording: FloatArray, cfg: SonarConfig) -> None:
    """Reject recordings that cannot be trusted: digital silence or ADC clipping."""
    if recording.size == 0 or not np.any(recording):
        raise MicrophonePermissionError(
            "La grabación es silencio digital absoluto: el sistema operativo probablemente "
            "está bloqueando el micrófono.",
            microphone_permission_hint(),
        )
    peak = float(np.max(np.abs(recording)))
    n_clipped = int(np.count_nonzero(np.abs(recording) >= cfg.clip_level))
    if n_clipped:
        raise ClippingError(
            f"Clipping: {n_clipped} muestras saturadas (pico {peak:.3f}).",
            "Baja --amplitude, el volumen del sistema o la ganancia del micrófono, o aleja "
            "el equipo de superficies muy cercanas.",
        )


# ------------------------------------------------------------------ profiles


def search_window(cfg: SonarConfig, t0: int, size: int) -> slice:
    """Profile indices covering ``[min_range, max_range]`` for a direct path at ``t0``."""
    lo = t0 + int(math.ceil(cfg.range_to_samples(cfg.min_range)))
    hi = t0 + int(math.floor(cfg.range_to_samples(cfg.max_range)))
    return slice(max(lo, 0), min(hi + 1, size))


@dataclass(frozen=True, slots=True)
class EchoProfile:
    """Range profile (A-scan) referenced to the direct path (index ``t0_index``)."""

    envelope: FloatArray
    frame: FloatArray
    t0_index: int
    noise_power: float
    direct_amplitude: float
    direct_snr_db: float
    n_averaged: int
    background_subtracted: bool
    cfg: SonarConfig

    @property
    def distances(self) -> FloatArray:
        lags = np.arange(self.envelope.size) - self.t0_index
        return self.cfg.samples_to_range(lags)

    @property
    def snr_db(self) -> FloatArray:
        floor = max(self.noise_power, np.finfo(float).tiny)
        return 10.0 * np.log10(np.maximum(self.envelope**2, floor * 1e-3) / floor)

    def in_range(self) -> slice:
        """Indices covering ``[min_range, max_range]``."""
        return search_window(self.cfg, self.t0_index, self.envelope.size)


@dataclass(frozen=True, slots=True)
class BackgroundProfile:
    """Aligned coherent matched-filter frame of the static scene (incl. direct path)."""

    frame: FloatArray
    fingerprint: dict

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez(fh, frame=self.frame, fingerprint=json.dumps(self.fingerprint))

    @classmethod
    def load(cls, path: Path, cfg: SonarConfig) -> BackgroundProfile:
        try:
            with np.load(path, allow_pickle=False) as data:
                frame = np.asarray(data["frame"], dtype=np.float64)
                fingerprint = json.loads(str(data["fingerprint"]))
        except (OSError, KeyError, ValueError) as exc:
            raise IncompatibleBackgroundError(
                f"No se pudo leer el perfil de fondo {path}: {exc}",
                "Vuelve a ejecutar `sonar calibrate`.",
            ) from exc
        expected = cfg.fingerprint()
        if fingerprint != expected or frame.size != cfg.frame_samples:
            diff = sorted(k for k in expected if fingerprint.get(k) != expected[k])
            raise IncompatibleBackgroundError(
                f"El perfil de fondo {path} se grabó con otra configuración ({', '.join(diff)}).",
                "Ejecuta `sonar calibrate` con los mismos parámetros, o usa --no-background.",
            )
        return cls(frame=frame, fingerprint=fingerprint)


# ------------------------------------------------------------------ pipeline


def _weak_signal_error(bp: FloatArray, cfg: SonarConfig, direct_snr_db: float) -> Exception:
    """Tell "hardware does not emit/capture this band" apart from "too much noise"."""
    pre = cfg.pre_silence_samples
    skip = cfg.chirp_samples
    noise_rms = float(np.sqrt(np.mean(bp[skip:pre] ** 2)))
    signal_rms = float(np.sqrt(np.mean(bp[pre:] ** 2)))
    noise_dbfs = 20 * math.log10(max(noise_rms, 1e-12))
    rise_db = 20 * math.log10(max(signal_rms, 1e-12) / max(noise_rms, 1e-12))
    band = f"{cfg.f_start / 1e3:.1f}–{cfg.f_end / 1e3:.1f} kHz"
    if noise_dbfs < QUIET_NOISE_DBFS and rise_db < 3.0:
        return BandOutOfResponseError(
            f"El ambiente está en silencio pero no se escucha el chirp en {band} "
            f"(SNR directo {direct_snr_db:.1f} dB): el parlante o el micrófono no responden "
            "en esa banda.",
            "Ejecuta `sonar diagnose` para medir la respuesta y usar la banda recomendada, "
            "p. ej. `sonar --band audible ping` (8–16 kHz). Verifica también que el volumen "
            "del sistema no esté en cero.",
        )
    return LowSNRError(
        f"SNR del camino directo demasiado bajo: {direct_snr_db:.1f} dB "
        f"(mínimo {cfg.min_direct_snr_db:.0f} dB, ruido en banda {noise_dbfs:.0f} dBFS).",
        "Busca un cuarto más silencioso, sube el volumen del sistema o --amplitude (con "
        "cuidado), aumenta --pings, o ejecuta `sonar diagnose` para elegir una mejor banda.",
    )


def process_recording(
    recording: FloatArray,
    cfg: SonarConfig,
    template: FloatArray,
    background: BackgroundProfile | None = None,
) -> EchoProfile:
    """Turn a raw ping-train recording into a range profile referenced to the direct path.

    The direct speaker→mic arrival is used as t0, which cancels the unknown latency
    of the audio stack: only *differences* of arrival times are ever used.
    """
    check_recording(recording, cfg)
    fs, pre, period = cfg.fs, cfg.pre_silence_samples, cfg.frame_samples
    guard, w = cfg.guard_samples, cfg.mainlobe_samples

    bp = bandpass(recording, fs, cfg.f_start, cfg.f_end)
    mf = matched_filter(bp, template)
    env = envelope(mf)

    # Noise reference: the leading silence (the latency can only delay the signal).
    noise_p = robust_noise_power(env[template.size : pre] ** 2)

    # Coarse direct path = earliest strong arrival after playback starts.
    search_end = min(env.size, pre + int(cfg.latency_margin * fs) + period)
    seg = env[pre:search_end]
    if seg.size == 0:
        raise AudioIOError("La grabación es más corta que el tren de pings.")
    peak = float(seg.max())
    direct_snr_db = 10 * math.log10(max(peak**2, 1e-300) / max(noise_p, 1e-300))
    if direct_snr_db < cfg.min_direct_snr_db:
        raise _weak_signal_error(bp, cfg, direct_snr_db)
    coarse = pre + int(np.argmax(seg >= 0.5 * peak))

    # Refine on the envelope folded over every ping: on real hardware the direct
    # path and a close reflection (lid, desk) can swap roles from ping to ping, so
    # a single ping is an unreliable reference.
    lo = max(coarse - guard, 0)
    measured = range(cfg.warmup_pings, cfg.warmup_pings + cfg.n_pings)
    windows = [env[lo + k * period : lo + k * period + 2 * guard] for k in measured]
    folded = np.mean([x for x in windows if x.size == 2 * guard], axis=0)
    rel = int(np.argmax(folded >= 0.5 * folded.max()))
    rel += int(np.argmax(folded[rel : rel + 2 * w + 1]))
    t_direct = lo + rel

    slip = cfg.max_slip_samples
    starts = [
        s
        for s in (t_direct - guard + k * period for k in measured)
        if s - slip >= 0 and s + period + slip <= mf.size
    ]
    if not starts:
        raise AudioIOError(
            "La latencia del audio superó el margen grabado: no hay pings completos.",
            "Aumenta latency_margin o usa un dispositivo/host API con menos latencia.",
        )
    # Frames carry ``slip`` extra samples on each side so they can be re-aligned on
    # their own content (duplex streams drop or repeat whole buffers) before averaging.
    padded = [mf[s - slip : s + period + slip] for s in starts]
    near = slice(slip + guard - 4 * w, slip + guard + 12 * w)
    avg = np.mean(align_frames(padded, near, slip), axis=0)

    # Sub-sample alignment so the direct path (earliest strong arrival) sits at ``guard``.
    avg_env = envelope(avg)
    zone = avg_env[guard : guard + 2 * slip + 1]
    i = guard + int(np.argmax(zone >= 0.5 * zone.max()))
    i += int(np.argmax(avg_env[i : i + 2 * w + 1]))
    aligned = fractional_shift(avg, slip + guard - parabolic_peak(avg_env, i))[slip : slip + period]

    residual = aligned
    if background is not None:
        # Least-squares gain on the direct path absorbs small volume changes.
        region = slice(guard - 2 * w, guard + 2 * w + 1)
        ref = background.frame
        # Envelope-peak alignment is biased by a fraction of a sample, which at the
        # carrier frequency is enough to leave strong static echoes behind. Refine
        # against the reference itself with a phase-sensitive cross-correlation.
        # The whole frame is used: the static scene adds structure that sharpens the
        # estimate; two passes remove most of the parabolic-interpolation bias.
        for _ in range(2):
            xc = correlate(aligned, ref, mode="full", method="fft")
            centre = ref.size - 1
            j = centre - w + int(np.argmax(xc[centre - w : centre + w + 1]))
            aligned = fractional_shift(aligned, centre - parabolic_peak(xc, j))
        scale = float(np.dot(aligned[region], ref[region]) / max(np.dot(ref[region], ref[region]), 1e-300))
        residual = aligned - scale * ref
    out_env = envelope(residual)

    # Noise of the averaged profile: robust estimate over the search window, floored
    # by the theoretical coherent-averaging gain on the leading-silence noise.
    search_noise = robust_noise_power(out_env[search_window(cfg, guard, out_env.size)] ** 2)
    return EchoProfile(
        envelope=out_env,
        frame=aligned,
        t0_index=guard,
        noise_power=max(search_noise, noise_p / len(starts)),
        direct_amplitude=float(out_env[guard]),
        direct_snr_db=direct_snr_db,
        n_averaged=len(starts),
        background_subtracted=background is not None,
        cfg=cfg,
    )
