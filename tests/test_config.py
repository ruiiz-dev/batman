import pytest

from sonar.config import SonarConfig, speed_of_sound
from sonar.errors import ConfigError


def test_defaults_are_valid_and_safe():
    cfg = SonarConfig()
    assert (cfg.f_start, cfg.f_end) == (18_000, 22_000)
    assert cfg.amplitude <= cfg.max_amplitude <= 1.0
    assert 0.005 <= cfg.chirp_duration <= 0.010


def test_speed_of_sound_tracks_temperature():
    assert speed_of_sound(20.0) == pytest.approx(343.2, abs=0.2)
    assert speed_of_sound(0.0) == pytest.approx(331.3)
    assert SonarConfig(temperature_c=30).speed_of_sound > SonarConfig(temperature_c=10).speed_of_sound
    assert SonarConfig(speed_of_sound_override=340.0).speed_of_sound == 340.0


def test_range_sample_conversion_roundtrip():
    cfg = SonarConfig(direct_path_m=0.1)
    assert cfg.samples_to_range(cfg.range_to_samples(1.234)) == pytest.approx(1.234)


@pytest.mark.parametrize(
    "overrides",
    [
        {"f_start": 22_000, "f_end": 18_000},
        {"f_end": 23_900},  # too close to Nyquist at 48 kHz
        {"amplitude": 0.8},  # above the default 0.5 safety cap
        {"max_amplitude": 1.5},
        {"n_pings": 0},
        {"window": "boxcar"},
        {"max_range": 20.0},  # does not fit in the ping interval
        {"min_range": 6.0},
    ],
)
def test_invalid_configs_are_rejected_with_hint(overrides):
    with pytest.raises(ConfigError):
        SonarConfig(**overrides)


def test_amplitude_cap_error_is_actionable():
    with pytest.raises(ConfigError) as exc:
        SonarConfig(amplitude=0.9)
    assert "--max-amplitude" in exc.value.hint


def test_with_overrides_ignores_none_and_revalidates():
    cfg = SonarConfig().with_overrides(fs=None, n_pings=4)
    assert cfg.n_pings == 4
    with pytest.raises(ConfigError):
        cfg.with_overrides(n_pings=-1)
