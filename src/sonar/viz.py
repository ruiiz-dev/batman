"""Plots: A-scan, live A-scan + waterfall, hardware response and polar sweep."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sonar.config import SonarConfig
from sonar.detection import Echo
from sonar.diagnostics import BandRecommendation, FrequencyResponse
from sonar.processing import EchoProfile

_DB_RANGE = (0.0, 50.0)


def _range_grid(cfg: SonarConfig, n: int = 400) -> np.ndarray:
    return np.linspace(0.0, cfg.max_range, n)


def resample_profile(profile: EchoProfile, grid: np.ndarray) -> np.ndarray:
    """SNR (dB) of the profile on a fixed range grid."""
    return np.interp(grid, profile.distances, profile.snr_db)


def plot_ascan(ax: plt.Axes, profile: EchoProfile, echoes: Sequence[Echo]) -> None:
    cfg = profile.cfg
    window = profile.in_range()
    d = profile.distances
    ax.plot(d, profile.snr_db, lw=1.0, color="C0")
    ax.axhline(cfg.min_snr_db, ls="--", lw=0.8, color="0.5", label=f"umbral {cfg.min_snr_db:.0f} dB")
    ax.axvspan(0, d[window.start], color="0.85", label="zona ciega")
    for e in echoes:
        ax.plot(e.distance_m, e.snr_db, "v", color="C3")
        ax.annotate(f"{e.distance_m:.2f} m", (e.distance_m, e.snr_db), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color="C3")
    ax.set_xlim(0, cfg.max_range)
    ax.set_ylim(*_DB_RANGE)
    ax.set_xlabel("distancia (m)")
    ax.set_ylabel("SNR (dB)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def profile_figure(profile: EchoProfile, echoes: Sequence[Echo]) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    plot_ascan(ax, profile, echoes)
    bg = "con" if profile.background_subtracted else "sin"
    ax.set_title(
        f"A-scan · {profile.n_averaged} pings promediados · {bg} sustracción de fondo · "
        f"SNR directo {profile.direct_snr_db:.0f} dB"
    )
    return fig


class LiveView:
    """A-scan on top, scrolling waterfall below."""

    def __init__(self, cfg: SonarConfig, history: int = 120) -> None:
        self.cfg = cfg
        self.grid = _range_grid(cfg)
        self.waterfall = np.zeros((history, self.grid.size))
        plt.ion()
        self.fig, (self.ax_scan, self.ax_fall) = plt.subplots(
            2, 1, figsize=(9, 7), layout="constrained", height_ratios=(1, 2)
        )
        (self.line,) = self.ax_scan.plot(self.grid, np.zeros_like(self.grid), lw=1.0)
        (self.marks,) = self.ax_scan.plot([], [], "v", color="C3")
        self.ax_scan.axhline(cfg.min_snr_db, ls="--", lw=0.8, color="0.5")
        self.ax_scan.set_xlim(0, cfg.max_range)
        self.ax_scan.set_ylim(*_DB_RANGE)
        self.ax_scan.set_ylabel("SNR (dB)")
        self.ax_scan.grid(alpha=0.3)
        self.image = self.ax_fall.imshow(
            self.waterfall, aspect="auto", origin="upper", cmap="magma",
            extent=(0, cfg.max_range, history, 0), vmin=_DB_RANGE[0], vmax=_DB_RANGE[1],
        )
        self.ax_fall.set_xlabel("distancia (m)")
        self.ax_fall.set_ylabel("pings atrás")
        self.fig.colorbar(self.image, ax=self.ax_fall, label="SNR (dB)")

    @property
    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def update(self, profile: EchoProfile, echoes: Sequence[Echo]) -> None:
        row = resample_profile(profile, self.grid)
        self.line.set_ydata(row)
        self.marks.set_data([e.distance_m for e in echoes], [e.snr_db for e in echoes])
        self.waterfall = np.roll(self.waterfall, 1, axis=0)
        self.waterfall[0] = row
        self.image.set_data(self.waterfall)
        best = ", ".join(f"{e.distance_m:.2f} m ({e.snr_db:.0f} dB)" for e in echoes[:3]) or "—"
        self.ax_scan.set_title(f"Ecos: {best}")
        self.fig.canvas.draw_idle()

    def show_status(self, text: str) -> None:
        self.ax_scan.set_title(text, color="C3")
        self.fig.canvas.draw_idle()


def frequency_response_figure(
    response: FrequencyResponse,
    min_snr_db: float,
    recommendation: BandRecommendation | None,
    current_band: tuple[float, float],
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(response.freqs / 1e3, response.snr_db, lw=1.0)
    ax.axhline(min_snr_db, ls="--", color="0.5", lw=0.8, label=f"mínimo {min_snr_db:.0f} dB")
    ax.axvspan(current_band[0] / 1e3, current_band[1] / 1e3, color="C1", alpha=0.15, label="banda actual")
    if recommendation:
        ax.axvspan(recommendation.f_start / 1e3, recommendation.f_end / 1e3, color="C2",
                   alpha=0.25, label=f"recomendada ({recommendation.kind})")
    ax.set_xlabel("frecuencia (kHz)")
    ax.set_ylabel("SNR barrido / ruido (dB)")
    ax.set_title("Respuesta parlante → micrófono")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


@dataclass(frozen=True, slots=True)
class SweepScan:
    angle_deg: float
    profile: EchoProfile
    echoes: tuple[Echo, ...]


def _angle_edges(angles: np.ndarray) -> np.ndarray:
    if angles.size == 1:
        return np.array([angles[0] - 5.0, angles[0] + 5.0])
    mid = (angles[:-1] + angles[1:]) / 2
    return np.concatenate(([2 * angles[0] - mid[0]], mid, [2 * angles[-1] - mid[-1]]))


def polar_sweep_figure(scans: Sequence[SweepScan]) -> Figure:
    """Radar-style plot: angle × range coloured by SNR, strongest echo per angle marked."""
    if not scans:
        raise ValueError("Se necesita al menos un ángulo para el gráfico polar.")
    scans = sorted(scans, key=lambda s: s.angle_deg)
    cfg = scans[0].profile.cfg
    grid = _range_grid(cfg, 300)
    angles = np.array([s.angle_deg for s in scans], dtype=float)
    values = np.array([resample_profile(s.profile, grid) for s in scans])
    r_edges = np.concatenate((grid, [grid[-1] + grid[1] - grid[0]]))

    fig = plt.figure(figsize=(7, 7), layout="constrained")
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    mesh = ax.pcolormesh(np.deg2rad(_angle_edges(angles)), r_edges, values.T, cmap="viridis",
                         vmin=_DB_RANGE[0], vmax=_DB_RANGE[1], shading="flat")
    for s in scans:
        if s.echoes:
            e = s.echoes[0]
            ax.plot(np.deg2rad(s.angle_deg), e.distance_m, "o", color="C3", ms=5)
    ax.set_rmax(cfg.max_range)
    ax.set_title("Barrido sonar (ángulo vs. distancia)")
    fig.colorbar(mesh, ax=ax, label="SNR (dB)", shrink=0.7)
    return fig
