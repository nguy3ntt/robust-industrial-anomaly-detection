from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

import robust_industrial_anomaly_detection

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_and_package_versions_match() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["name"] == "robust-industrial-anomaly-detection"
    assert project["version"] == robust_industrial_anomaly_detection.__version__


def test_visa_configuration_freezes_the_initial_data_contract() -> None:
    with (ROOT / "configs" / "data" / "visa.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["dataset"]["id"] == "visa-20220922"
    assert config["dataset"]["accessed_on"] == "2026-09-16"
    assert config["dataset"]["archive_url"].startswith("https://")
    assert config["integrity"] == {
        "archive_sha256": "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362",
        "expected_images": 10_821,
        "expected_normal": 9_621,
        "expected_anomalous": 1_200,
        "expected_categories": 12,
        "perceptual_max_dhash_distance": 2,
        "perceptual_max_normalized_mae": 0.1,
    }
    assert len(config["categories"]) == len(set(config["categories"])) == 12
    assert config["protocol"]["preserve_official_test"] is True
    assert config["protocol"]["train_labels"] == ["normal"]
    assert config["protocol"]["validation_labels"] == ["normal"]
