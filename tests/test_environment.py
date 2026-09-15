from __future__ import annotations

import json

from robust_deepfake_detection.environment import Check, EnvironmentReport


def test_report_is_ready_when_it_has_no_failures() -> None:
    report = EnvironmentReport(
        (Check("example", "pass", "ok"), Check("optional", "warning", "missing"))
    )

    assert report.ok


def test_report_is_not_ready_when_a_check_fails() -> None:
    report = EnvironmentReport((Check("example", "fail", "bad"),))

    assert not report.ok


def test_report_json_is_machine_readable() -> None:
    report = EnvironmentReport((Check("example", "pass", "ok"),))

    payload = json.loads(report.to_json())

    assert payload == {
        "checks": [{"detail": "ok", "name": "example", "status": "pass"}],
        "ok": True,
    }
