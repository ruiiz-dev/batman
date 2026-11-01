import matplotlib.pyplot as plt

from conftest import make_engine, scene
from sonar.diagnostics import recommend_band
from sonar.simulation import Reflector
from sonar.viz import LiveView, SweepScan, frequency_response_figure, polar_sweep_figure, profile_figure


def test_profile_figure(cfg, tmp_path):
    m = make_engine(cfg, scene(Reflector(1.0, 0.05))).measure()
    profile_figure(m.profile, m.echoes).savefig(tmp_path / "a.png")
    assert (tmp_path / "a.png").stat().st_size > 0
    plt.close("all")


def test_live_view_updates(cfg):
    engine = make_engine(cfg.with_overrides(n_pings=2), scene(Reflector(1.0, 0.05)))
    view = LiveView(cfg, history=10)
    for _ in range(3):
        m = engine.measure()
        view.update(m.profile, m.echoes)
    assert view.waterfall[:3].max() > cfg.min_snr_db
    view.show_status("ruido")
    plt.close("all")


def test_polar_sweep_and_response_figures(cfg, tmp_path):
    scans = []
    for angle, d in [(0, 1.0), (30, 1.5), (60, 2.0)]:
        m = make_engine(cfg.with_overrides(n_pings=2), scene(Reflector(d, 0.05))).measure()
        scans.append(SweepScan(angle, m.profile, tuple(m.echoes)))
    polar_sweep_figure(scans).savefig(tmp_path / "p.png")
    response = make_engine(cfg, scene()).diagnose(duration=0.5)
    frequency_response_figure(response, 15, recommend_band(response), (18e3, 22e3)).savefig(tmp_path / "r.png")
    assert (tmp_path / "p.png").exists() and (tmp_path / "r.png").exists()
    plt.close("all")
