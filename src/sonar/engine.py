"""Application service: orchestrates the audio port and the pure processing chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sonar.audio import AudioDevice
from sonar.config import SonarConfig
from sonar.detection import Echo, detect_echoes
from sonar.diagnostics import FrequencyResponse, estimate_response
from sonar.processing import (
    BackgroundProfile,
    EchoProfile,
    autocorrelation_envelope,
    check_recording,
    process_recording,
)
from sonar.signal import linear_chirp, linear_sweep, ping_train

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Measurement:
    profile: EchoProfile
    echoes: list[Echo]


class SonarEngine:
    def __init__(
        self,
        device: AudioDevice,
        cfg: SonarConfig,
        background: BackgroundProfile | None = None,
    ) -> None:
        self.device = device
        self.cfg = cfg
        self.background = background
        self._template = linear_chirp(cfg)
        self._train = ping_train(cfg, self._template)
        self._sidelobes = autocorrelation_envelope(self._template)

    def acquire(self) -> FloatArray:
        return self.device.playrec(self._train.samples, self.cfg.fs)

    def analyse(self, recording: FloatArray, use_background: bool = True) -> Measurement:
        background = self.background if use_background else None
        profile = process_recording(recording, self.cfg, self._template, background)
        return Measurement(profile=profile, echoes=detect_echoes(profile, self._sidelobes))

    def measure(self) -> Measurement:
        return self.analyse(self.acquire())

    def calibrate(self) -> BackgroundProfile:
        """Capture the static scene (direct coupling + fixed reflections)."""
        profile = process_recording(self.acquire(), self.cfg, self._template, background=None)
        return BackgroundProfile(frame=profile.frame, fingerprint=self.cfg.fingerprint())

    def diagnose(self, duration: float = 2.0, f_min: float = 1_000.0) -> FrequencyResponse:
        """Play a full-range sweep and compare against the ambient noise floor."""
        fs = self.cfg.fs
        self.device.check_settings(fs)
        noise = self.device.record(duration, fs)
        check_recording(noise, self.cfg)
        sweep = linear_sweep(f_min, 0.97 * fs / 2, duration, fs, self.cfg.amplitude)
        tail = np.zeros(int(round(0.2 * fs)))
        recording = self.device.playrec(np.concatenate([sweep, tail]), fs)
        check_recording(recording, self.cfg)
        return estimate_response(recording, noise, fs)
