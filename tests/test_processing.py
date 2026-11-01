import numpy as np
import pytest

from sonar.config import SonarConfig
from sonar.errors import ClippingError, IncompatibleBackgroundError, MicrophonePermissionError
from sonar.processing import (
    BackgroundProfile,
    autocorrelation_envelope,
    check_recording,
    coherent_average,
    envelope,
    fractional_shift,
    matched_filter,
    parabolic_peak,
)
from sonar.signal import linear_chirp


def test_matched_filter_peak_is_at_the_template_start():
    cfg = SonarConfig()
    t = linear_chirp(cfg)
    x = np.zeros(5000)
    x[1234 : 1234 + t.size] = t
    assert int(np.argmax(envelope(matched_filter(x, t)))) == 1234


@pytest.mark.parametrize("shift", [0.0, 0.3, 7.75, -2.4])
def test_fractional_shift_moves_peak_with_subsample_precision(shift):
    cfg = SonarConfig()
    t = linear_chirp(cfg)
    x = np.zeros(4000)
    x[1000 : 1000 + t.size] = t
    env = envelope(matched_filter(fractional_shift(x, shift), t))
    i = int(np.argmax(env))
    assert parabolic_peak(env, i) == pytest.approx(1000 + shift, abs=0.1)


def test_parabolic_peak_on_exact_parabola():
    y = -((np.arange(10) - 4.3) ** 2)
    assert parabolic_peak(y, 4) == pytest.approx(4.3)
    assert parabolic_peak(y, 0) == 0.0


def test_coherent_average_reduces_noise_by_sqrt_n():
    rng = np.random.default_rng(1)
    x = rng.normal(size=64 * 1000)
    avg = coherent_average(x, [k * 1000 for k in range(64)], 1000)
    assert np.std(avg) == pytest.approx(1 / 8, rel=0.15)


def test_autocorrelation_envelope_is_normalised_and_decays():
    acorr = autocorrelation_envelope(linear_chirp(SonarConfig()))
    assert acorr[0] == pytest.approx(1.0)
    assert acorr[50:].max() < 0.1


def test_digital_silence_means_microphone_blocked():
    with pytest.raises(MicrophonePermissionError) as exc:
        check_recording(np.zeros(1000), SonarConfig())
    assert exc.value.hint


def test_clipping_detected():
    x = np.full(1000, 0.1)
    x[10] = 1.0
    with pytest.raises(ClippingError):
        check_recording(x, SonarConfig())


def test_background_roundtrip_and_mismatch(tmp_path):
    cfg = SonarConfig()
    bg = BackgroundProfile(frame=np.arange(cfg.frame_samples, dtype=float), fingerprint=cfg.fingerprint())
    path = tmp_path / "bg.npz"
    bg.save(path)
    loaded = BackgroundProfile.load(path, cfg)
    assert np.array_equal(loaded.frame, bg.frame)
    with pytest.raises(IncompatibleBackgroundError, match="f_start"):
        BackgroundProfile.load(path, cfg.with_overrides(f_start=17_000))


def test_corrupt_background_file(tmp_path):
    path = tmp_path / "bg.npz"
    path.write_bytes(b"not a numpy file")
    with pytest.raises(IncompatibleBackgroundError):
        BackgroundProfile.load(path, SonarConfig())
