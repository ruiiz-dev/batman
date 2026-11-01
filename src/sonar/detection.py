"""Echo detection on a range profile: peaks over an adaptive (OS-CFAR) threshold,
with sidelobe rejection of strong arrivals (direct path included)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks

from sonar.processing import EchoProfile, parabolic_peak

FloatArray = NDArray[np.float64]

_LN2 = math.log(2.0)


@dataclass(frozen=True, slots=True)
class Echo:
    distance_m: float
    snr_db: float
    amplitude: float
    delay_s: float
    index: float


def os_cfar_noise(power: FloatArray, guard: int, train: int) -> FloatArray:
    """Local mean-noise estimate per cell: median of the training cells around it.

    Order-statistic CFAR (median) is not masked by a neighbouring strong echo the
    way cell-averaging CFAR is, which matters for closely spaced reflectors.
    """
    footprint = np.ones(2 * (guard + train) + 1, dtype=bool)
    footprint[train : train + 2 * guard + 1] = False
    return percentile_filter(power, 50, footprint=footprint, mode="reflect") / _LN2


def detect_echoes(profile: EchoProfile, sidelobes: FloatArray) -> list[Echo]:
    """Echoes in ``[min_range, max_range]``, strongest (by SNR) first.

    ``sidelobes`` is the normalised autocorrelation envelope of the template
    (see ``processing.autocorrelation_envelope``).
    """
    cfg = profile.cfg
    env = profile.envelope
    power = env**2
    window = profile.in_range()
    if window.stop - window.start < 3:
        return []

    ml = cfg.mainlobe_samples
    candidates, _ = find_peaks(env[window], distance=max(1, ml // 2))
    candidates = candidates + window.start

    noise = max(profile.noise_power, np.finfo(float).tiny)
    # A wide training window keeps the median noise-dominated even when several
    # echoes cluster within a few tens of centimetres (common at short range).
    local = os_cfar_noise(power, guard=2 * ml, train=24 * ml)
    k_global = 10 ** (cfg.min_snr_db / 10)
    k_local = 10 ** (cfg.cfar_snr_db / 10)
    candidates = [
        int(p) for p in candidates if power[p] >= k_global * noise and power[p] >= k_local * local[p]
    ]
    candidates.sort(key=lambda p: env[p], reverse=True)

    # Greedy sidelobe rejection (strongest first): a peak must stand above the
    # sidelobes that every already-accepted arrival projects onto it.
    margin = 10 ** (cfg.sidelobe_margin_db / 20)
    accepted: list[tuple[int, float]] = [(profile.t0_index, profile.direct_amplitude)]
    echoes: list[Echo] = []
    for p in candidates:
        leak = max(a * _sidelobe(sidelobes, abs(p - i)) for i, a in accepted)
        if env[p] <= margin * leak:
            continue
        accepted.append((p, float(env[p])))
        idx = parabolic_peak(env, p)
        lag = idx - profile.t0_index
        echoes.append(
            Echo(
                distance_m=float(cfg.samples_to_range(lag)),
                snr_db=10 * math.log10(power[p] / noise),
                amplitude=float(env[p]),
                delay_s=lag / cfg.fs,
                index=idx,
            )
        )
    echoes.sort(key=lambda e: e.snr_db, reverse=True)
    return echoes[: cfg.n_echoes]


def _sidelobe(sidelobes: FloatArray, lag: int) -> float:
    return float(sidelobes[lag]) if lag < sidelobes.size else 0.0
