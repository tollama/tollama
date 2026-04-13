"""Tests for the tollama xai CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tollama.cli.main import app


def _new_runner() -> CliRunner:
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


def _result_stdout(result: object) -> str:
    stdout = getattr(result, "stdout", None)
    if isinstance(stdout, str):
        return stdout
    output = getattr(result, "output", None)
    if isinstance(output, str):
        return output
    return ""


# ── explain-decision ──────────────────────────────────────────


class TestXaiExplainDecision:
    def test_success(self, tmp_path: Path):
        payload = {"forecast_result": {"model": "mock", "horizon": 3}}
        input_file = tmp_path / "explain.json"
        input_file.write_text(json.dumps(payload))

        fake_result = {"explanation_id": "test-1", "version": "1.0"}

        runner = _new_runner()
        with patch("tollama.cli.main._make_client") as mock_client_factory:
            mock_client = mock_client_factory.return_value
            mock_client.explain_decision.return_value = fake_result
            result = runner.invoke(app, ["xai", "explain-decision", "--input", str(input_file)])

        assert result.exit_code == 0
        output = json.loads(_result_stdout(result))
        assert output["explanation_id"] == "test-1"

    def test_missing_input_file(self):
        runner = _new_runner()
        result = runner.invoke(app, ["xai", "explain-decision", "--input", "/tmp/nonexistent.json"])
        assert result.exit_code != 0

    def test_daemon_error(self, tmp_path: Path):
        payload = {"forecast_result": {"model": "mock"}}
        input_file = tmp_path / "explain.json"
        input_file.write_text(json.dumps(payload))

        runner = _new_runner()
        with patch("tollama.cli.main._make_client") as mock_client_factory:
            mock_client = mock_client_factory.return_value
            mock_client.explain_decision.side_effect = RuntimeError("daemon down")
            result = runner.invoke(app, ["xai", "explain-decision", "--input", str(input_file)])

        assert result.exit_code != 0


# ── trust-score ───────────────────────────────────────────────


class TestXaiTrustScore:
    def test_success(self, tmp_path: Path):
        payload = {
            "trust_score": 0.75,
            "metrics": {"brier_score": 0.15},
            "source": "polymarket",
        }
        input_file = tmp_path / "trust.json"
        input_file.write_text(json.dumps(payload))

        fake_result = {"overall_score": 0.75, "components": {}}

        runner = _new_runner()
        with patch("tollama.cli.main._make_client") as mock_client_factory:
            mock_client = mock_client_factory.return_value
            mock_client.trust_breakdown.return_value = fake_result
            result = runner.invoke(app, ["xai", "trust-score", "--input", str(input_file)])

        assert result.exit_code == 0
        output = json.loads(_result_stdout(result))
        assert output["overall_score"] == 0.75

    def test_daemon_error(self, tmp_path: Path):
        payload = {"trust_score": 0.5, "metrics": {}}
        input_file = tmp_path / "trust.json"
        input_file.write_text(json.dumps(payload))

        runner = _new_runner()
        with patch("tollama.cli.main._make_client") as mock_client_factory:
            mock_client = mock_client_factory.return_value
            mock_client.trust_breakdown.side_effect = RuntimeError("fail")
            result = runner.invoke(app, ["xai", "trust-score", "--input", str(input_file)])

        assert result.exit_code != 0


# ── model-card ────────────────────────────────────────────────


class TestXaiModelCard:
    def test_json_output(self, tmp_path: Path):
        payload = {"model_info": {"name": "mock", "version": "1.0"}}
        input_file = tmp_path / "model.json"
        input_file.write_text(json.dumps(payload))

        fake_result = {"model_name": "mock", "sections": []}

        runner = _new_runner()
        with patch("tollama.cli.main._make_client") as mock_client_factory:
            mock_client = mock_client_factory.return_value
            mock_client.model_card.return_value = fake_result
            result = runner.invoke(app, ["xai", "model-card", "--input", str(input_file)])

        assert result.exit_code == 0
        output = json.loads(_result_stdout(result))
        assert output["model_name"] == "mock"

    def test_markdown_output(self, tmp_path: Path):
        payload = {"model_info": {"name": "mock"}}
        input_file = tmp_path / "model.json"
        input_file.write_text(json.dumps(payload))

        fake_result = {"content": "# Model Card\n\nMock model."}

        runner = _new_runner()
        with patch("tollama.cli.main._make_client") as mock_client_factory:
            mock_client = mock_client_factory.return_value
            mock_client.model_card.return_value = fake_result
            result = runner.invoke(
                app, ["xai", "model-card", "--input", str(input_file), "--markdown"]
            )

        assert result.exit_code == 0
        stdout = _result_stdout(result)
        assert "# Model Card" in stdout

    def test_daemon_error(self, tmp_path: Path):
        payload = {"model_info": {"name": "mock"}}
        input_file = tmp_path / "model.json"
        input_file.write_text(json.dumps(payload))

        runner = _new_runner()
        with patch("tollama.cli.main._make_client") as mock_client_factory:
            mock_client = mock_client_factory.return_value
            mock_client.model_card.side_effect = RuntimeError("fail")
            result = runner.invoke(app, ["xai", "model-card", "--input", str(input_file)])

        assert result.exit_code != 0


# ── calibration ───────────────────────────────────────────────


class TestXaiCalibration:
    def test_no_data(self, tmp_path: Path):
        runner = _new_runner()
        with patch(
            "tollama.xai.trust_agents.calibration.default_calibration_path",
            return_value=tmp_path / "nonexistent.json",
        ):
            result = runner.invoke(app, ["xai", "calibration"])

        assert result.exit_code == 0
        assert "No calibration data" in _result_stdout(result)

    def test_list_all_agents(self, tmp_path: Path):
        from tollama.xai.trust_agents.calibration import CalibrationTracker

        tracker = CalibrationTracker()
        tracker.record("agent_a", "test", 0.7, 0.6, {})
        tracker.record("agent_b", "test", 0.8, 0.7, {})
        cal_path = tmp_path / "cal.json"
        tracker.save(cal_path)

        runner = _new_runner()
        with patch(
            "tollama.xai.trust_agents.calibration.default_calibration_path",
            return_value=cal_path,
        ):
            result = runner.invoke(app, ["xai", "calibration"])

        assert result.exit_code == 0
        stdout = _result_stdout(result)
        assert "agent_a" in stdout
        assert "agent_b" in stdout

    def test_specific_agent(self, tmp_path: Path):
        from tollama.xai.trust_agents.calibration import CalibrationTracker

        tracker = CalibrationTracker()
        tracker.record("agent_x", "test", 0.7, 0.6, {})
        cal_path = tmp_path / "cal.json"
        tracker.save(cal_path)

        runner = _new_runner()
        with patch(
            "tollama.xai.trust_agents.calibration.default_calibration_path",
            return_value=cal_path,
        ):
            result = runner.invoke(app, ["xai", "calibration", "agent_x"])

        assert result.exit_code == 0
        stdout = _result_stdout(result)
        assert "agent_x" in stdout
