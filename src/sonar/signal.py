"""Waveform generation. Pure functions: no I/O, no global state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import chirp as _scipy_chirp
from scipy.signal.windows import hann, tukey

from sonar.config import SonarConfig

FloatArray = NDArray[np.float64]


def taper(cfg: SonarConfig, n: int) -> FloatArray:
    """Amplitude window that tames matched-filter range sidelobes and speaker clicks."""
    if cfg.window == "hann":
        return hann(n, sym=True)
    return tukey(n, alpha=cfg.tukey_alpha, sym=True)


def linear_chirp(cfg: SonarConfig) -> FloatArray:
    """Unit-peak windowed linear chirp from ``f_start`` to ``f_end``.

    This is also the matched-filter template, so it is returned unscaled.
    """
    n = cfg.chirp_samples
    t = np.arange(n) / cfg.fs
    sweep = _scipy_chirp(t, f0=cfg.f_start, t1=t[-1], f1=cfg.f_end, method="linear")
    return sweep * taper(cfg, n)


@dataclass(frozen=True, slots=True)
class PingTrain:
    """Playback buffer with ``warmup_pings + n_pings`` chirps at a fixed period."""

    samples: FloatArray
    ping_starts: tuple[int, ...]
    period: int


def ping_train(cfg: SonarConfig, template: FloatArray | None = None) -> PingTrain:
    """Silence → warm-up + N pings every ``ping_interval`` → tail that absorbs the I/O latency.

    The leading silence guarantees a noise-only stretch at the start of the
    recording, since the audio latency can only delay the signal, never advance it.
    """
    template = linear_chirp(cfg) if template is None else template
    period = cfg.frame_samples
    pre = cfg.pre_silence_samples
    tail = period + int(round(cfg.latency_margin * cfg.fs))
    total = cfg.warmup_pings + cfg.n_pings
    out = np.zeros(pre + total * period + tail)
    starts = tuple(pre + k * period for k in range(total))
    for s in starts:
        out[s : s + template.size] = cfg.amplitude * template
    return PingTrain(samples=out, ping_starts=starts, period=period)


def linear_sweep(
    f0: float, f1: float, duration: float, fs: int, amplitude: float, fade: float = 0.01
) -> FloatArray:
    """Long linear sweep with short fades, used to measure the hardware response."""
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    sweep = _scipy_chirp(t, f0=f0, t1=t[-1], f1=f1, method="linear")
    alpha = min(1.0, 2.0 * fade / duration)
    return amplitude * sweep * tukey(n, alpha=alpha)
