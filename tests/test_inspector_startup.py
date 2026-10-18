from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from robust_industrial_anomaly_detection.prototype import launch
from robust_industrial_anomaly_detection.prototype.engine import PrototypeError


@pytest.fixture
def local_environment(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "artifacts"
    (artifact / "runs").mkdir(parents=True)
    cache = tmp_path / "torch"
    monkeypatch.setenv("ANOMALY_ARTIFACT_ROOT", str(artifact))
    monkeypatch.setenv("TORCH_HOME", str(cache))
    monkeypatch.setattr(launch, "load_dotenv", lambda path: False)
    return artifact, cache


def test_missing_environment_and_storage_are_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "load_dotenv", lambda path: False)
    monkeypatch.delenv("ANOMALY_ARTIFACT_ROOT", raising=False)
    with pytest.raises(launch.StartupError, match=r"Copy \.env\.example"):
        launch.load_inspector(tmp_path)
    monkeypatch.setenv("ANOMALY_ARTIFACT_ROOT", str(tmp_path / "absent"))
    with pytest.raises(launch.StartupError, match="storage is missing"):
        launch.load_inspector(tmp_path)


def test_missing_weights_fail_before_any_model_or_download(
    local_environment, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        launch, "InspectionEngine", lambda **kwargs: pytest.fail("model was loaded")
    )
    with pytest.raises(launch.StartupError, match="Startup does not download"):
        launch.load_inspector(tmp_path)


def test_corrupt_weights_cannot_fall_back(local_environment, tmp_path, monkeypatch):
    _, cache = local_environment
    path = cache / "hub/checkpoints/resnet18-f37072fd.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not the checkpoint")
    monkeypatch.setattr(
        launch, "InspectionEngine", lambda **kwargs: pytest.fail("model was loaded")
    )
    with pytest.raises(launch.StartupError, match="SHA-256"):
        launch.load_inspector(tmp_path)


def test_missing_artifact_error_names_recovery_and_never_falls_back(
    local_environment, tmp_path, monkeypatch
):
    _, cache = local_environment
    path = cache / "hub/checkpoints/resnet18-f37072fd.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture")
    monkeypatch.setattr(launch, "sha256_file", lambda path: launch.ENCODERS["resnet18"].sha256)
    monkeypatch.setattr(launch, "load_prototype_settings", lambda path: "settings")

    def fail(**kwargs):
        raise PrototypeError("no complete full M5 calibration run is available")

    monkeypatch.setattr(launch, "InspectionEngine", fail)
    with pytest.raises(launch.StartupError, match="No fallback model was used"):
        launch.load_inspector(tmp_path)


def test_occupied_port_is_not_reused_and_zero_is_not_an_automatic_fallback():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(launch.StartupError, match="Nothing was stopped"):
            launch.check_port(occupied.getsockname()[1])
        assert occupied.fileno() >= 0
    for port in (0, -1, 65536):
        with pytest.raises(launch.StartupError, match="between 1 and 65535"):
            launch.check_port(port)


def test_readiness_checks_without_serving_or_browser(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(launch, "check_port", lambda port: seen.append(port))
    monkeypatch.setattr(
        launch, "load_inspector", lambda *args, **kwargs: SimpleNamespace(runtime_info=lambda: {})
    )
    monkeypatch.setattr(launch, "serve", lambda *args, **kwargs: pytest.fail("server was started"))
    assert launch.main(["--check", "--port", "8770", "--open-browser"]) == 0
    assert seen == [8770]
    assert "no server was started" in capsys.readouterr().out


def test_bind_race_is_reported_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(launch, "check_port", lambda port: None)
    monkeypatch.setattr(
        launch, "load_inspector", lambda *args, **kwargs: SimpleNamespace(runtime_info=lambda: {})
    )

    def fail(*args, **kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr(launch, "serve", fail)
    assert launch.main([]) == 2
    output = capsys.readouterr().err
    assert "--port 8770" in output and "Traceback" not in output


def test_launcher_preserves_m10_and_historical_scripts():
    root = Path(__file__).parents[1]
    assert (
        "configs/prototype/m10.yaml"
        in (root / "src/robust_industrial_anomaly_detection/prototype/launch.py").read_text()
    )
    for name in ("m6", "m8", "m10", "m12"):
        assert (root / f"scripts/run_{name}_prototype.py").is_file()


def test_local_weight_loader_never_invokes_a_downloader(tmp_path, monkeypatch):
    import torch

    from robust_industrial_anomaly_detection.baselines import multiscale

    calls = []
    state = {"fixture": torch.tensor([1.0])}
    backbone = SimpleNamespace(
        **{
            name: torch.nn.Identity()
            for name in (
                "conv1",
                "bn1",
                "relu",
                "maxpool",
                "layer1",
                "layer2",
                "layer3",
            )
        },
        load_state_dict=lambda value, strict: calls.append((value, strict)),
    )

    def local_backbone(*, weights):
        assert weights is None
        return backbone

    def load(path, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is True
        return state

    monkeypatch.setattr(multiscale, "resnet18", local_backbone)
    monkeypatch.setattr(multiscale.torch, "load", load)
    monkeypatch.setattr(
        multiscale.torch.hub,
        "download_url_to_file",
        lambda *args, **kwargs: pytest.fail("download attempted"),
    )
    multiscale._PyramidModule(tmp_path / "local.pth")
    assert calls == [(state, True)]


def test_local_weight_disappearance_cannot_trigger_fallback(tmp_path, monkeypatch):
    import torch

    from robust_industrial_anomaly_detection.baselines import multiscale

    def local_backbone(*, weights):
        assert weights is None
        return SimpleNamespace(load_state_dict=lambda *args, **kwargs: None)

    monkeypatch.setattr(multiscale, "resnet18", local_backbone)
    monkeypatch.setattr(
        torch.hub, "download_url_to_file", lambda *args, **kwargs: pytest.fail("download attempted")
    )
    with pytest.raises(FileNotFoundError):
        multiscale._PyramidModule(tmp_path / "absent.pth")
