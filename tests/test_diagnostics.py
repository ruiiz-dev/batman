import numpy as np
import pytest

from conftest import make_engine, scene
from sonar.diagnostics import FrequencyResponse, recommend_band
from sonar.errors import BandOutOfResponseError, MicrophonePermissionError


def _response(usable: list[tuple[float, float]], fs: int = 48_000) -> FrequencyResponse:
    f = np.linspace(0, fs / 2, 1025)
    snr = np.zeros_like(f)
    for lo, hi in usable:
        snr[(f >= lo) & (f <= hi)] = 30.0
    return FrequencyResponse(freqs=f, snr_db=snr)


def test_prefers_near_ultrasonic_when_available():
    r = recommend_band(_response([(1_000, 23_000)]))
    assert r.kind == "near-ultrasonic"
    assert r.f_start >= 17_000 and r.f_end <= 23_000
    assert r.bandwidth >= 2_000


def test_falls_back_to_audible_band():
    r = recommend_band(_response([(1_000, 16_500)]))
    assert r.kind == "audible"
    assert 8_000 <= r.f_start < r.f_end <= 16_000


def test_uses_any_band_as_last_resort():
    r = recommend_band(_response([(3_000, 7_000)]))
    assert r.kind == "other"
    assert r.f_start >= 3_000 and r.f_end <= 7_000


def test_no_usable_band_raises():
    with pytest.raises(BandOutOfResponseError):
        recommend_band(_response([(5_000, 5_500)]))


def test_coverage():
    resp = _response([(18_000, 20_000)])
    assert resp.coverage(18_000, 20_000, 15) == pytest.approx(1.0)
    assert resp.coverage(18_000, 22_000, 15) == pytest.approx(0.5, abs=0.05)


def test_diagnose_flat_hardware(cfg):
    response = make_engine(cfg, scene(noise_std=1e-3)).diagnose(duration=1.0)
    assert recommend_band(response).kind == "near-ultrasonic"


def test_diagnose_laptop_speaker_rolling_off_above_16k(cfg):
    response = make_engine(cfg, scene(noise_std=1e-3, response_band=(300.0, 16_000.0))).diagnose(duration=1.0)
    assert response.coverage(18_500, 22_000, 15) < 0.2
    assert recommend_band(response).kind == "audible"


def test_diagnose_with_blocked_microphone(cfg):
    with pytest.raises(MicrophonePermissionError):
        make_engine(cfg, scene(muted=True)).diagnose(duration=0.5)
