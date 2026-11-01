import numpy as np
import pytest
from scipy.signal import welch

from sonar.config import SonarConfig
from sonar.signal import linear_chirp, linear_sweep, ping_train


def test_chirp_length_peak_and_tapered_edges():
    cfg = SonarConfig()
    x = linear_chirp(cfg)
    assert x.size == cfg.chirp_samples
    assert np.max(np.abs(x)) <= 1.0
    assert abs(x[0]) < 1e-9 and abs(x[-1]) < 1e-9


@pytest.mark.parametrize("band", [(18_000, 22_000), (8_000, 16_000)])
def test_chirp_energy_is_inside_the_band(band):
    cfg = SonarConfig(f_start=band[0], f_end=band[1], window="hann")
    f, p = welch(np.pad(linear_chirp(cfg), 1024), fs=cfg.fs, nperseg=512)
    inside = (f >= band[0] - 1500) & (f <= band[1] + 1500)
    assert p[inside].sum() / p.sum() > 0.97


def test_ping_train_layout_and_amplitude():
    cfg = SonarConfig(n_pings=4, warmup_pings=2, amplitude=0.2)
    train = ping_train(cfg)
    assert len(train.ping_starts) == 6  # warm-up pings are emitted too
    assert train.ping_starts[0] == cfg.pre_silence_samples
    assert np.diff(train.ping_starts).tolist() == [cfg.frame_samples] * 5
    assert np.max(np.abs(train.samples)) == pytest.approx(0.2, rel=1e-3)
    assert not np.any(train.samples[: cfg.pre_silence_samples])
    assert train.samples.size >= train.ping_starts[-1] + cfg.frame_samples + cfg.latency_margin * cfg.fs


def test_sweep_respects_amplitude():
    x = linear_sweep(1000, 20_000, 0.5, 48_000, amplitude=0.1)
    assert x.size == 24_000
    assert np.max(np.abs(x)) <= 0.1 + 1e-12
