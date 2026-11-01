"""Hardware response estimation and band recommendation (pure functions)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.signal import welch

from sonar.config import AUDIBLE_BAND
from sonar.errors import BandOutOfResponseError

FloatArray = NDArray[np.float64]

BandKind = Literal["near-ultrasonic", "audible", "other"]

NEAR_ULTRASONIC_FLOOR = 17_000.0


@dataclass(frozen=True, slots=True)
class FrequencyResponse:
    """Per-frequency SNR (dB) of a test sweep over the ambient noise."""

    freqs: FloatArray
    snr_db: FloatArray

    def coverage(self, f_lo: float, f_hi: float, min_snr_db: float) -> float:
        """Fraction of ``[f_lo, f_hi]`` with SNR ≥ ``min_snr_db``."""
        mask = (self.freqs >= f_lo) & (self.freqs <= f_hi)
        return float(np.mean(self.snr_db[mask] >= min_snr_db)) if mask.any() else 0.0

    def mean_snr(self, f_lo: float, f_hi: float) -> float:
        mask = (self.freqs >= f_lo) & (self.freqs <= f_hi)
        return float(np.mean(self.snr_db[mask])) if mask.any() else float("-inf")


@dataclass(frozen=True, slots=True)
class BandRecommendation:
    f_start: float
    f_end: float
    kind: BandKind
    mean_snr_db: float

    @property
    def bandwidth(self) -> float:
        return self.f_end - self.f_start

    def range_resolution(self, speed_of_sound: float) -> float:
        return speed_of_sound / (2.0 * self.bandwidth)


def estimate_response(
    sweep_recording: FloatArray,
    noise_recording: FloatArray,
    fs: int,
    nperseg: int = 2048,
    smooth_hz: float = 500.0,
) -> FrequencyResponse:
    """Welch PSD of the recorded sweep against the PSD of ambient noise.

    The noise PSD uses median averaging so a single knock or click while
    recording the ambient reference does not inflate it across the spectrum.
    """
    freqs, p_sweep = welch(sweep_recording, fs=fs, nperseg=nperseg)
    _, p_noise = welch(noise_recording, fs=fs, nperseg=nperseg, average="median")
    snr = 10 * np.log10(np.maximum(p_sweep, 1e-30) / np.maximum(p_noise, 1e-30))
    bins = max(1, int(round(smooth_hz / (freqs[1] - freqs[0]))))
    return FrequencyResponse(freqs=freqs, snr_db=median_filter(snr, size=bins, mode="nearest"))


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Inclusive index ranges of consecutive ``True`` values."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def recommend_band(
    response: FrequencyResponse,
    min_snr_db: float = 15.0,
    min_bandwidth: float = 2_000.0,
    max_bandwidth: float = 6_000.0,
) -> BandRecommendation:
    """Pick the band to use, in order of preference:

    1. the widest usable stretch above 17 kHz (barely audible),
    2. the widest usable stretch inside the audible fallback (8–16 kHz),
    3. the widest usable stretch anywhere.
    """
    f = response.freqs
    usable = response.snr_db >= min_snr_db

    def best_in(lo: float, hi: float) -> tuple[float, float] | None:
        mask = usable & (f >= lo) & (f <= hi)
        runs = [(f[a], f[b]) for a, b in _runs(mask) if f[b] - f[a] >= min_bandwidth]
        if not runs:
            return None
        a, b = max(runs, key=lambda r: r[1] - r[0])
        if b - a > max_bandwidth:  # Keep the top of the run: higher = less audible.
            a = b - max_bandwidth
        return float(a), float(b)

    preferences: list[tuple[BandKind, float, float]] = [
        ("near-ultrasonic", NEAR_ULTRASONIC_FLOOR, float(f[-1])),
        ("audible", *AUDIBLE_BAND),
        ("other", float(f[0]), float(f[-1])),
    ]
    for kind, lo, hi in preferences:
        band = best_in(lo, hi)
        if band is not None:
            # Round inwards to 100 Hz so the CLI suggestion is tidy and stays usable.
            start, end = np.ceil(band[0] / 100) * 100, np.floor(band[1] / 100) * 100
            return BandRecommendation(start, end, kind, response.mean_snr(start, end))
    raise BandOutOfResponseError(
        f"Ninguna banda de ≥{min_bandwidth / 1e3:.0f} kHz supera {min_snr_db:.0f} dB de SNR.",
        "Sube el volumen del sistema o --amplitude, revisa que el micrófono correcto esté "
        "seleccionado (`sonar devices`) y repite en un cuarto silencioso.",
    )
