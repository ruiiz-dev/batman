"""Synthetic acoustic scene used by the fake audio device and the tests.

Models: unknown system latency, a direct speaker→mic path, reflectors at known
ranges, an optional hardware passband, additive noise and ADC clipping.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Reflector:
    """A point reflector at ``distance_m`` (one way) with linear ``gain`` vs. the emitted signal."""

    distance_m: float
    gain: float


@dataclass(frozen=True, slots=True)
class Scene:
    reflectors: tuple[Reflector, ...] = ()
    direct_gain: float = 0.5
    latency_s: float = 0.03
    noise_std: float = 1e-4
    speed_of_sound: float = 343.0
    # Hardware passband (speaker × mic). ``None`` means flat response.
    response_band: tuple[float, float] | None = None
    clip_level: float = 1.0
    # Simulates an OS that silently blocks the microphone (records exact zeros).
    muted: bool = False
    # Duplex buffer glitches: at each (time_s, n) the capture repeats n samples
    # (n > 0) or drops -n samples, shifting everything recorded afterwards.
    slips: tuple[tuple[float, int], ...] = ()

    def with_reflectors(self, *reflectors: Reflector) -> Scene:
        return replace(self, reflectors=self.reflectors + reflectors)


def apply_band(x: FloatArray, fs: int, band: tuple[float, float]) -> FloatArray:
    lo, hi = band
    nyq = fs / 2.0
    sos = butter(10, [max(lo, 1.0) / nyq, min(hi, 0.999 * nyq) / nyq], btype="bandpass", output="sos")
    return sosfilt(sos, x)


def delay_and_sum(x: FloatArray, delays_s: list[float], gains: list[float], fs: int) -> FloatArray:
    """Sum of fractionally delayed copies of ``x`` (exact, FFT-based), same length as ``x``."""
    n = x.size
    pad = int(np.ceil(max(delays_s, default=0.0) * fs)) + 1
    nfft = 1 << int(np.ceil(np.log2(n + pad)))
    spectrum = np.fft.rfft(x, nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    h = np.zeros_like(spectrum)
    for tau, g in zip(delays_s, gains, strict=True):
        h += g * np.exp(-2j * np.pi * f * tau)
    return np.fft.irfft(spectrum * h, nfft)[:n]


def simulate_recording(
    played: FloatArray, fs: int, scene: Scene, rng: np.random.Generator
) -> FloatArray:
    """What the microphone would capture while ``played`` comes out of the speaker."""
    if scene.muted:
        return np.zeros(played.size)
    x = apply_band(played, fs, scene.response_band) if scene.response_band else played
    delays = [scene.latency_s]
    gains = [scene.direct_gain]
    for r in scene.reflectors:
        delays.append(scene.latency_s + 2.0 * r.distance_m / scene.speed_of_sound)
        gains.append(r.gain)
    y = delay_and_sum(x, delays, gains, fs)
    for t, n in sorted(scene.slips, reverse=True):
        i = int(round(t * fs))
        y = np.concatenate((y[:i], y[i - n : i], y[i:])) if n > 0 else np.concatenate((y[:i], y[i - n :]))
    y = np.resize(y, played.size) if y.size >= played.size else np.pad(y, (0, played.size - y.size))
    y = y + rng.normal(0.0, scene.noise_std, y.size)
    return np.clip(y, -scene.clip_level, scene.clip_level)


def simulate_silence(n: int, scene: Scene, rng: np.random.Generator) -> FloatArray:
    if scene.muted:
        return np.zeros(n)
    return rng.normal(0.0, scene.noise_std, n)
