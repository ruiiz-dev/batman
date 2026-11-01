from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from sonar.audio import FakeAudioDevice
from sonar.config import SonarConfig
from sonar.engine import SonarEngine
from sonar.simulation import Reflector, Scene

# The simulator and the estimator share the same speed of sound, so any distance
# error comes from the processing chain alone.
C = 343.0


@pytest.fixture
def cfg() -> SonarConfig:
    return SonarConfig(speed_of_sound_override=C)


def make_engine(cfg: SonarConfig, scene: Scene, seed: int = 0, background=None) -> SonarEngine:
    return SonarEngine(FakeAudioDevice(scene=scene, seed=seed), cfg, background)


def scene(*reflectors: Reflector, **kwargs) -> Scene:
    kwargs.setdefault("speed_of_sound", C)
    return Scene(reflectors=reflectors, **kwargs)
