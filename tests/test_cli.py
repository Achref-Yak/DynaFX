import io
import json
import sys
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock

from cognitive_engine.core.models import Graph, ReasoningMode


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    p = tmp_path / "input.txt"
    p.write_text("Test text for analysis.")
    return p


@pytest.fixture
def mock_graph() -> Graph:
    g = Graph(
        mode=ReasoningMode.ARGUMENT,
        source_text="test",
    )
    return g


@pytest.fixture(autouse=True)
def mock_orchestrator(mock_graph: Graph) -> MagicMock:
    with patch("cognitive_engine.pipeline.orchestrator.run") as m:
        m.return_value = mock_graph
        yield m


def _run(args: list[str], expect_exit: bool = False) -> tuple[int, str, str]:
    """Run cli.main with sys.argv patched. Returns (exit_code, stdout, stderr)."""
    from cognitive_engine.cli import main

    exit_code: list[int] = [0]
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stderr_buf = io.StringIO()

    def _exit(code: int) -> None:
        exit_code[0] = code
        raise SystemExit(code)

    def _print(*objs, **kwargs) -> None:
        stdout_lines.extend(str(o) for o in objs)

    def _log(level: str, msg: str) -> None:
        stderr_lines.append(f"{level} | {msg}")

    orig_stderr = sys.stderr

    patches = [
        patch("sys.argv", ["cognitive-engine", *args]),
        patch("sys.exit", side_effect=_exit),
        patch("builtins.print", side_effect=_print),
        patch("cognitive_engine.cli.logger.error", side_effect=lambda msg, *a: _log("ERROR", msg % a if a else msg)),
        patch("cognitive_engine.cli.logger.info", side_effect=lambda msg, *a: _log("INFO", msg % a if a else msg)),
    ]

    sys.stderr = stderr_buf
    for p in patches:
        p.start()

    try:
        if expect_exit:
            with pytest.raises(SystemExit):
                main()
        else:
            main()
    except SystemExit:
        if not expect_exit:
            raise
    finally:
        for p in patches:
            p.stop()
        sys.stderr = orig_stderr

    all_stderr = "\n".join(stderr_lines)
    raw_stderr = stderr_buf.getvalue()
    if raw_stderr:
        all_stderr = (all_stderr + "\n" + raw_stderr).strip()

    return exit_code[0], "".join(stdout_lines), all_stderr


class TestCliMinimal:
    def test_minimal_invocation(self, sample_file: Path, mock_orchestrator: MagicMock):
        code, stdout, stderr = _run(["analyze", str(sample_file)])
        assert code == 0
        mock_orchestrator.assert_called_once_with(
            "Test text for analysis.",
            config_path=None,
            mode=None,
            max_tokens=512,
            overlap=128,
        )
        assert "Processing file:" in stderr

    def test_output_to_stdout(self, sample_file: Path, mock_graph: Graph):
        code, stdout, stderr = _run(["analyze", str(sample_file)])
        assert code == 0
        parsed = json.loads(stdout)
        assert parsed["mode"] == "ARGUMENT"
        assert parsed["source_text"] == "test"

    def test_chunk_size_default(self, sample_file: Path, mock_orchestrator: MagicMock):
        _run(["analyze", str(sample_file)])
        kwargs = mock_orchestrator.call_args[1]
        assert kwargs["max_tokens"] == 512

    def test_chunk_overlap_default(self, sample_file: Path, mock_orchestrator: MagicMock):
        _run(["analyze", str(sample_file)])
        kwargs = mock_orchestrator.call_args[1]
        assert kwargs["overlap"] == 128


class TestCliFlags:
    def test_output_to_file(self, sample_file: Path, tmp_path: Path, mock_graph: Graph):
        out = tmp_path / "out.json"
        code, stdout, stderr = _run(["analyze", str(sample_file), "--output", str(out)])
        assert code == 0
        assert stdout == ""
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed["source_text"] == "test"

    def test_config_flag(self, sample_file: Path, mock_orchestrator: MagicMock):
        _run(["analyze", str(sample_file), "--config", "/some/priors.json"])
        kwargs = mock_orchestrator.call_args[1]
        assert kwargs["config_path"] == "/some/priors.json"

    @pytest.mark.parametrize("mode", ["causal", "conditional", "argument", "analogy"])
    def test_mode_flag(self, sample_file: Path, mock_orchestrator: MagicMock, mode: str):
        _run(["analyze", str(sample_file), "--mode", mode])
        kwargs = mock_orchestrator.call_args[1]
        assert kwargs["mode"] == mode

    def test_chunk_size(self, sample_file: Path, mock_orchestrator: MagicMock):
        _run(["analyze", str(sample_file), "--chunk-size", "256"])
        kwargs = mock_orchestrator.call_args[1]
        assert kwargs["max_tokens"] == 256

    def test_chunk_overlap(self, sample_file: Path, mock_orchestrator: MagicMock):
        _run(["analyze", str(sample_file), "--chunk-overlap", "64"])
        kwargs = mock_orchestrator.call_args[1]
        assert kwargs["overlap"] == 64

    def test_all_flags_combined(self, sample_file: Path, tmp_path: Path, mock_orchestrator: MagicMock):
        out = tmp_path / "result.json"
        _run([
            "analyze", str(sample_file),
            "--output", str(out),
            "--config", "/opt/priors.json",
            "--mode", "causal",
            "--chunk-size", "384",
            "--chunk-overlap", "96",
        ])
        kwargs = mock_orchestrator.call_args[1]
        assert kwargs == {
            "config_path": "/opt/priors.json",
            "mode": "causal",
            "max_tokens": 384,
            "overlap": 96,
        }
        assert out.exists()


class TestCliErrors:
    def test_file_not_found(self):
        code, stdout, stderr = _run(["analyze", "/nonexistent/input.txt"], expect_exit=True)
        assert code == 1
        assert "File not found" in stderr

    def test_pipeline_error(self, sample_file: Path, mock_orchestrator: MagicMock):
        mock_orchestrator.side_effect = RuntimeError("model crash")
        code, stdout, stderr = _run(["analyze", str(sample_file)], expect_exit=True)
        assert code == 1
        assert "Pipeline failed" in stderr

    def test_invalid_mode(self, sample_file: Path):
        code, stdout, stderr = _run(["analyze", str(sample_file), "--mode", "bogus"], expect_exit=True)
        assert code == 2
        assert "invalid choice" in stderr
