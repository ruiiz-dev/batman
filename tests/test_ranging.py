"""End-to-end ranging on synthetic data: chirp + direct path + echoes + noise + random latency."""

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import make_engine, scene
from sonar.simulation import Reflector

TOLERANCE_M = 0.02


def _match(echoes, truth):
    return min(echoes, key=lambda e: abs(e.distance_m - truth))


@pytest.mark.parametrize("distance", [0.3, 0.75, 1.0, 2.2, 3.5, 4.8])
@pytest.mark.parametrize("latency", [0.0, 0.0137, 0.2491])
def test_single_echo_within_2cm(cfg, distance, latency):
    m = make_engine(cfg, scene(Reflector(distance, 0.05), latency_s=latency, noise_std=1e-3)).measure()
    assert m.echoes, "echo not detected"
    assert abs(_match(m.echoes, distance).distance_m - distance) < TOLERANCE_M


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    distance=st.floats(0.3, 4.5),
    gain=st.floats(0.02, 0.2),
    latency=st.floats(0.0, 0.4),
    seed=st.integers(0, 2**16),
)
def test_random_latency_never_biases_distance(cfg, distance, gain, latency, seed):
    """The direct path is the time reference, so the unknown latency cancels out."""
    m = make_engine(cfg, scene(Reflector(distance, gain), latency_s=latency, noise_std=2e-3), seed).measure()
    strongest = m.echoes[0]
    assert abs(strongest.distance_m - distance) < TOLERANCE_M


def test_multiple_echoes_are_all_resolved(cfg):
    truth = [0.6, 1.4, 2.5, 3.9]
    reflectors = [Reflector(d, g) for d, g in zip(truth, [0.08, 0.05, 0.03, 0.02], strict=True)]
    m = make_engine(cfg, scene(*reflectors, latency_s=0.087, noise_std=1e-3)).measure()
    assert len(m.echoes) == len(truth)
    for d in truth:
        assert abs(_match(m.echoes, d).distance_m - d) < TOLERANCE_M


def test_no_echoes_means_no_detections(cfg):
    """Only the direct path and noise: no false alarms (incl. direct-path sidelobes)."""
    for seed in range(5):
        m = make_engine(cfg, scene(latency_s=0.05, noise_std=2e-3), seed).measure()
        assert m.echoes == []


def test_overlapping_echoes_are_separated(cfg):
    """Echoes 15 cm apart overlap almost completely in time (an 8 ms chirp is 2.7 m
    long) but the matched filter compresses them into separate peaks."""
    m = make_engine(cfg, scene(Reflector(1.0, 0.05), Reflector(1.15, 0.04), noise_std=1e-3)).measure()
    assert len(m.echoes) >= 2
    for d in (1.0, 1.15):
        assert abs(_match(m.echoes, d).distance_m - d) < TOLERANCE_M


def test_echoes_below_resolution_merge_into_one(cfg):
    """Below c/2B (~4.3 cm for 4 kHz) two reflectors cannot be told apart."""
    assert cfg.range_resolution == pytest.approx(343 / 8000)
    m = make_engine(cfg, scene(Reflector(2.0, 0.05), Reflector(2.015, 0.05), noise_std=1e-3)).measure()
    near = [e for e in m.echoes if abs(e.distance_m - 2.0) < 0.1]
    assert len(near) == 1
    assert abs(near[0].distance_m - 2.0075) < TOLERANCE_M


def test_coherent_averaging_recovers_weak_echo(cfg):
    weak = scene(Reflector(1.8, 0.015), noise_std=8e-3, latency_s=0.04)
    single = make_engine(cfg.with_overrides(n_pings=1), weak, seed=3).measure()
    averaged = make_engine(cfg.with_overrides(n_pings=32), weak, seed=3).measure()
    assert not any(abs(e.distance_m - 1.8) < TOLERANCE_M for e in single.echoes)
    assert abs(_match(averaged.echoes, 1.8).distance_m - 1.8) < TOLERANCE_M


def test_echo_below_snr_threshold_is_not_reported(cfg):
    m = make_engine(cfg, scene(Reflector(2.0, 2e-4), noise_std=5e-3)).measure()
    assert not any(abs(e.distance_m - 2.0) < 0.05 for e in m.echoes)


def test_temperature_changes_the_estimate(cfg):
    """Wrong speed of sound → proportional range error (sanity of the c·Δt/2 formula)."""
    s = scene(Reflector(3.0, 0.05), noise_std=1e-3)
    hot = make_engine(replace(cfg, speed_of_sound_override=None, temperature_c=40.0), s).measure()
    assert hot.echoes[0].distance_m == pytest.approx(3.0 * 355.1 / 343.0, rel=2e-3)


def test_snr_is_reported_and_sorted(cfg):
    m = make_engine(cfg, scene(Reflector(1.0, 0.1), Reflector(2.0, 0.02), noise_std=1e-3)).measure()
    snrs = [e.snr_db for e in m.echoes]
    assert snrs == sorted(snrs, reverse=True)
    assert all(np.isfinite(snrs)) and snrs[0] > cfg.min_snr_db


@pytest.mark.parametrize("slips", [((0.5, 32),), ((0.4, -32), (0.9, 32), (1.1, 64))])
def test_duplex_buffer_slips_do_not_break_averaging(cfg, slips):
    """Real WASAPI duplex streams drop/repeat whole 32-sample buffers mid-train
    (seen on hardware); each ping must be re-aligned on its own content."""
    s = scene(Reflector(1.7, 0.01), latency_s=0.05, noise_std=2e-3, slips=slips)
    m = make_engine(cfg, s).measure()
    assert abs(_match(m.echoes, 1.7).distance_m - 1.7) < TOLERANCE_M
    # Coherent gain survives: the echo is as strong as without slips.
    clean = make_engine(cfg, scene(Reflector(1.7, 0.01), latency_s=0.05, noise_std=2e-3)).measure()
    assert _match(m.echoes, 1.7).snr_db > _match(clean.echoes, 1.7).snr_db - 3.0
