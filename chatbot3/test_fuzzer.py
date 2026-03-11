"""Tests for crs.fuzzer — all paths exercised with mocks."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crs import config
from crs.fuzzer import (
    _strip_code_fences,
    _minimal_file_reader_poc,
    check_fuzzer_availability,
    generate_fuzzing_harness,
    run_libfuzzer,
    run_afl,
    try_fuzzing,
)
from crs.types import (
    BuildResult,
    CodeContext,
    CodeSnippet,
    CyberGymTask,
    LLMRouter,
    PoCResult,
)


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def task(tmp_path: Path) -> CyberGymTask:
    return CyberGymTask(
        task_id="test_001",
        project_name="libfoo",
        vuln_type="heap-buffer-overflow",
        description="A heap buffer overflow in parse_input()",
        repo_path=tmp_path / "repo",
    )


@pytest.fixture()
def context(task: CyberGymTask) -> CodeContext:
    return CodeContext(
        task=task,
        top_snippets=[
            CodeSnippet("src/parse.c", 10, 25,
                        "void parse_input(const char *buf, int len) { ... }",
                        relevance_score=0.95),
        ],
        build_info={"entry_points": []},
    )


@pytest.fixture()
def build_result(tmp_path: Path) -> BuildResult:
    return BuildResult(
        success=True,
        binary_path=tmp_path / "foobin",
        include_dirs=[str(tmp_path / "include")],
        lib_paths=[str(tmp_path / "lib")],
        object_files=[],
    )


@pytest.fixture()
def router() -> LLMRouter:
    r = MagicMock(spec=LLMRouter)
    r.query.return_value = textwrap.dedent("""\
        #include <stdint.h>
        #include <stddef.h>
        extern void parse_input(const char *buf, int len);
        int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
            parse_input((const char *)data, (int)size);
            return 0;
        }
    """)
    return r


# ── unit tests ─────────────────────────────────────────────────────────────

class TestStripCodeFences:
    def test_plain(self):
        assert _strip_code_fences("int x;") == "int x;"

    def test_with_fences(self):
        src = "```c\nint x;\n```"
        assert _strip_code_fences(src) == "int x;"

    def test_no_language_marker(self):
        src = "```\nint x;\n```"
        assert _strip_code_fences(src) == "int x;"


class TestMinimalFileReaderPoC:
    def test_contains_main(self):
        poc = _minimal_file_reader_poc("int LLVMFuzzerTestOneInput(...){}")
        assert "int main(" in poc
        assert "fopen" in poc
        assert "LLVMFuzzerTestOneInput" in poc


class TestCheckAvailability:
    @patch("crs.fuzzer.shutil.which", return_value=None)
    def test_nothing_available(self, _mock_which):
        result = check_fuzzer_availability()
        assert result["afl++"] is False
        # libfuzzer probe also needs clang which() → None
        assert result["available"] is False

    @patch("crs.fuzzer.shutil.which", side_effect=lambda x: "/usr/bin/afl-fuzz" if x == "afl-fuzz" else None)
    def test_afl_only(self, _mock_which):
        result = check_fuzzer_availability()
        assert result["afl++"] is True
        assert result["libfuzzer"] is False
        assert result["available"] is True


class TestGenerateHarness:
    def test_writes_file(self, context, router, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "WORK_DIR", tmp_path)
        code = generate_fuzzing_harness(context, router)
        assert "LLVMFuzzerTestOneInput" in code
        written = (tmp_path / context.task.task_id / "fuzzer_harness.c").read_text()
        assert written == code


class TestRunLibfuzzer:
    def test_returns_none_on_compile_failure(self, build_result, task,
                                             tmp_path, monkeypatch):
        monkeypatch.setattr(config, "WORK_DIR", tmp_path)
        harness_path = tmp_path / task.task_id / "fuzzer_harness.c"
        harness_path.parent.mkdir(parents=True, exist_ok=True)
        harness_path.write_text("invalid C code !!!")

        # Force clang to fail by patching _run.
        with patch("crs.fuzzer._run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="err")
            result = run_libfuzzer(harness_path, build_result, task)
        assert result is None


class TestRunAfl:
    @patch("crs.fuzzer.shutil.which", return_value=None)
    def test_no_afl(self, _w, build_result, task, context, router):
        assert run_afl(build_result, task, context, router) is None


class TestTryFuzzing:
    def test_disabled(self, context, build_result, router, task, monkeypatch):
        monkeypatch.setattr(config, "FUZZING_ENABLED", False)
        assert try_fuzzing(context, build_result, router, task) is None

    @patch("crs.fuzzer.check_fuzzer_availability",
           return_value={"afl++": False, "libfuzzer": False, "available": False})
    def test_not_available(self, _avail, context, build_result, router, task):
        assert try_fuzzing(context, build_result, router, task) is None

    @patch("crs.fuzzer.check_fuzzer_availability",
           return_value={"afl++": False, "libfuzzer": True, "available": True})
    @patch("crs.fuzzer.run_libfuzzer")
    @patch("crs.fuzzer.generate_fuzzing_harness", return_value="int LLVMFuzzerTestOneInput(...){return 0;}")
    def test_libfuzzer_success(self, _gen, mock_lf, _avail,
                               context, build_result, router, task, tmp_path,
                               monkeypatch):
        monkeypatch.setattr(config, "WORK_DIR", tmp_path)
        # Ensure harness file exists so run_libfuzzer sees it.
        (tmp_path / task.task_id).mkdir(parents=True, exist_ok=True)

        fake_poc = PoCResult(
            poc_code="int main(){}", poc_path=tmp_path / "poc.c",
            strategy_name="libfuzzer", confidence=0.9)
        mock_lf.return_value = fake_poc

        result = try_fuzzing(context, build_result, router, task)
        assert result is not None
        assert result.strategy_name == "libfuzzer"
