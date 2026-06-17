from typer.testing import CliRunner

from nexrad_picoballoon.cli import app


def test_cli_synthetic_smoke_workflow(tmp_path):
    runner = CliRunner()
    raw = tmp_path / "raw"
    features = tmp_path / "features"
    candidates = tmp_path / "candidates"
    truth = tmp_path / "truth"
    truth.mkdir()

    result = runner.invoke(
        app,
        [
            "fetch",
            "--site",
            "KTEST",
            "--start",
            "2026-01-01T00:00:00Z",
            "--end",
            "2026-01-01T00:10:00Z",
            "--out",
            str(raw),
            "--synthetic",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["features", "--input", str(raw), "--out", str(features)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["detect", "--features", str(features), "--out", str(candidates)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app, ["evaluate", "--candidates", str(candidates), "--truth", str(truth)]
    )
    assert result.exit_code == 0, result.output
    assert (candidates / "evaluation_summary.md").exists()
