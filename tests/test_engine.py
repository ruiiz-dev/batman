"""Engine-level behaviour: background subtraction and actionable failures."""

import numpy as np
import pytest

from conftest import make_engine, scene
from sonar.audio import FakeAudioDevice
from sonar.engine import SonarEngine
from sonar.errors import (
    BandOutOfResponseError,
    ClippingError,
    ConfigError,
    LowSNRError,
    MicrophonePermissionError,
    UnsupportedSampleRateError,
)
from sonar.simulation import Reflector

TABLE = Reflector(0.25, 0.2)  # Static clutter: e.g. the desk under the laptop.


def test_background_subtraction_removes_static_clutter(cfg):
    static = scene(TABLE, latency_s=0.06, noise_std=1e-3)
    background = make_engine(cfg, static, seed=1).calibrate()

    with_target = static.with_reflectors(Reflector(1.5, 0.02))
    raw = make_engine(cfg, with_target, seed=2).measure()
    clean = make_engine(cfg, with_target, seed=2, background=background).measure()

    assert any(abs(e.distance_m - 0.25) < 0.02 for e in raw.echoes)
    assert not any(abs(e.distance_m - 0.25) < 0.05 for e in clean.echoes)
    assert abs(clean.echoes[0].distance_m - 1.5) < 0.02
    assert clean.profile.background_subtracted


def test_background_survives_latency_and_volume_change(cfg):
    """Latency differs between runs and the user turned the amplitude up after calibrating."""
    background = make_engine(cfg, scene(TABLE, latency_s=0.02, noise_std=1e-3), seed=1).calibrate()
    louder = cfg.with_overrides(amplitude=0.13)
    s = scene(TABLE, Reflector(2.0, 0.03), latency_s=0.31, noise_std=1e-3)
    m = make_engine(louder, s, seed=5, background=background).measure()
    assert abs(m.echoes[0].distance_m - 2.0) < 0.02
    assert not any(abs(e.distance_m - 0.25) < 0.05 for e in m.echoes)
    # The static reflector is cancelled far below the direct path.
    p = m.profile
    table = np.abs(p.distances - 0.25) < 0.05
    assert 20 * np.log10(p.envelope[table].max() / np.abs(p.frame).max()) < -40


def test_muted_microphone_raises_permission_error(cfg):
    with pytest.raises(MicrophonePermissionError) as exc:
        make_engine(cfg, scene(muted=True)).measure()
    assert "Micrófono" in exc.value.hint or "micrófono" in exc.value.hint


def test_clipping_is_reported(cfg):
    with pytest.raises(ClippingError) as exc:
        make_engine(cfg, scene(direct_gain=15.0)).measure()
    assert "--amplitude" in exc.value.hint


def test_low_snr_in_noisy_room(cfg):
    with pytest.raises(LowSNRError) as exc:
        make_engine(cfg, scene(direct_gain=0.02, noise_std=0.05)).measure()
    assert "diagnose" in exc.value.hint


def test_band_outside_hardware_response(cfg):
    """Quiet room, but the speaker cannot reproduce 18–22 kHz."""
    with pytest.raises(BandOutOfResponseError) as exc:
        make_engine(cfg, scene(response_band=(500.0, 9_000.0), noise_std=1e-3)).measure()
    assert "audible" in exc.value.hint


def test_audible_fallback_works_on_limited_hardware(cfg):
    audible = cfg.with_overrides(f_start=8_000, f_end=16_000)
    s = scene(Reflector(1.2, 0.05), response_band=(500.0, 17_000.0), noise_std=1e-3)
    m = make_engine(audible, s).measure()
    assert abs(m.echoes[0].distance_m - 1.2) < 0.02


def test_unsupported_sample_rate(cfg):
    device = FakeAudioDevice(scene=scene(), supported_samplerates=(48_000,))
    with pytest.raises(UnsupportedSampleRateError):
        SonarEngine(device, cfg.with_overrides(fs=96_000)).measure()


def test_device_refuses_amplitude_above_its_cap(cfg):
    device = FakeAudioDevice(scene=scene(), max_amplitude=0.05)
    with pytest.raises(ConfigError):
        SonarEngine(device, cfg).measure()
