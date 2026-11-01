import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from conftest import scene
from sonar import cli
from sonar.audio import FakeAudioDevice
from sonar.simulation import Reflector

runner = CliRunner()
BASE = ["--sound-speed", "343"]


@pytest.fixture
def fake(monkeypatch):
    """Route the CLI to a fake device whose scene each test can swap."""
    state = {"scene": scene(Reflector(1.3, 0.05), noise_std=1e-3), "seed": 0}

    def factory(cfg):
        state["seed"] += 1
        return FakeAudioDevice(scene=state["scene"], seed=state["seed"], max_amplitude=cfg.max_amplitude)

    monkeypatch.setattr(cli, "make_device", factory)
    # Wide consoles so Rich does not wrap table cells mid-word.
    monkeypatch.setattr(cli, "console", Console(width=200))
    monkeypatch.setattr(cli, "err_console", Console(width=200, stderr=True))
    return state


def test_devices(fake):
    result = runner.invoke(cli.app, ["devices"])
    assert result.exit_code == 0, result.output
    assert "Fake Microphone" in result.output


def test_ping_json(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, [*BASE, "ping", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert abs(payload["echoes"][0]["distance_m"] - 1.3) < 0.02


def test_calibrate_then_ping_with_background_and_plot(fake, tmp_path):
    fake["scene"] = scene(Reflector(0.25, 0.2), noise_std=1e-3)
    bg = tmp_path / "bg.npz"
    result = runner.invoke(cli.app, [*BASE, "calibrate", "--out", str(bg)])
    assert result.exit_code == 0, result.output
    assert bg.exists()

    fake["scene"] = fake["scene"].with_reflectors(Reflector(2.0, 0.03))
    plot = tmp_path / "a.png"
    result = runner.invoke(cli.app, [*BASE, "ping", "--background", str(bg), "--plot", str(plot)])
    assert result.exit_code == 0, result.output
    assert "2.00" in result.output and "0.25" not in result.output
    assert plot.exists()


def test_background_from_other_config_is_rejected(fake, tmp_path):
    bg = tmp_path / "bg.npz"
    runner.invoke(cli.app, [*BASE, "calibrate", "--out", str(bg)])
    result = runner.invoke(cli.app, [*BASE, "--band", "audible", "ping", "--background", str(bg)])
    assert result.exit_code == 2
    assert "calibrate" in result.output


def test_permission_error_is_actionable(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake["scene"] = scene(muted=True)
    result = runner.invoke(cli.app, [*BASE, "ping"])
    assert result.exit_code == 2
    assert "Cómo resolverlo" in result.output


def test_invalid_config_exits_cleanly(fake):
    result = runner.invoke(cli.app, ["--amplitude", "0.9", "ping"])
    assert result.exit_code == 2
    assert "tope de seguridad" in result.output


def test_diagnose_recommends_audible_on_limited_hardware(fake, tmp_path):
    fake["scene"] = scene(noise_std=1e-3, response_band=(300.0, 16_000.0))
    result = runner.invoke(cli.app, [*BASE, "diagnose", "--duration", "1", "--plot", str(tmp_path / "r.png")])
    assert result.exit_code == 0, result.output
    assert "audible" in result.output
    assert "fuera de la respuesta" in result.output


def test_sweep_builds_polar_plot(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "radar.png"
    result = runner.invoke(
        cli.app, [*BASE, "--pings", "2", "sweep", "--out", str(out)], input="0\n45\nabc\n90\n\n"
    )
    assert result.exit_code == 0, result.output
    assert "no es un número" in result.output
    assert "3 ángulos" in result.output
    assert out.exists()


def test_ping_saves_raw_recording(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "rec.npy"
    result = runner.invoke(cli.app, [*BASE, "ping", "--save-raw", str(raw)])
    assert result.exit_code == 0, result.output
    import numpy as np

    assert np.load(raw).ndim == 1
