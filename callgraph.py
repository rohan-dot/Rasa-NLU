"""
callgraph.py — Lightweight call graph using ctags + regex.

Builds a map of:
1. Function definitions (file, line, signature)
2. Call relationships (who calls whom)
3. Paths from entry points to target functions

This replaces regex guessing with real function resolution.
ctags handles the parsing; we add call relationship extraction.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("gemma-fuzzer.callgraph")


@dataclass
class FunctionDef:
    name: str
    file: str
    line: int
    signature: str = ""
    called_by: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    body: str = ""  # first N lines of the function body


@dataclass 
class CallGraph:
    functions: dict[str, FunctionDef] = field(default_factory=dict)
    files: dict[str, list[str]] = field(default_factory=dict)  # file → [function_names]

    def get_callers(self, func_name: str, depth: int = 3) -> list[list[str]]:
        """Get call chains leading to func_name, up to given depth."""
        paths: list[list[str]] = []
        self._trace_callers(func_name, [], paths, depth)
        return paths

    def _trace_callers(
        self, func: str, current_path: list[str],
        all_paths: list[list[str]], depth: int,
    ) -> None:
        if depth <= 0 or func in current_path:
            return
        current_path = current_path + [func]
        fdef = self.functions.get(func)
        if not fdef or not fdef.called_by:
            all_paths.append(list(reversed(current_path)))
            return
        for caller in fdef.called_by:
            self._trace_callers(caller, current_path, all_paths, depth - 1)

    def get_path_to(self, entry: str, target: str, max_depth: int = 6) -> list[str] | None:
        """Find a call path from entry function to target function."""
        visited: set[str] = set()
        return self._dfs_path(entry, target, visited, max_depth)

    def _dfs_path(
        self, current: str, target: str,
        visited: set[str], depth: int,
    ) -> list[str] | None:
        if depth <= 0 or current in visited:
            return None
        if current == target:
            return [current]
        visited.add(current)
        fdef = self.functions.get(current)
        if not fdef:
            return None
        for callee in fdef.calls:
            path = self._dfs_path(callee, target, visited, depth - 1)
            if path:
                return [current] + path
        return None

    def get_function_context(self, func_name: str, src_dir: str) -> str:
        """Get the source code of a function + its callers' call sites."""
        fdef = self.functions.get(func_name)
        if not fdef:
            return ""

        context = f"// === {func_name} defined in {fdef.file}:{fdef.line} ===\n"
        if fdef.body:
            context += fdef.body + "\n"
        else:
            # Read from file
            try:
                src_path = Path(src_dir) / fdef.file
                lines = src_path.read_text(errors="replace").split("\n")
                start = max(0, fdef.line - 1)
                end = min(len(lines), start + 80)
                context += "\n".join(lines[start:end]) + "\n"
            except Exception:
                pass

        # Add caller context
        for caller_name in fdef.called_by[:3]:
            caller = self.functions.get(caller_name)
            if not caller:
                continue
            context += f"\n// === Called by {caller_name} in {caller.file}:{caller.line} ===\n"
            try:
                src_path = Path(src_dir) / caller.file
                content = src_path.read_text(errors="replace")
                # Find the call site
                idx = content.find(func_name)
                if idx >= 0:
                    start = max(0, idx - 300)
                    end = min(len(content), idx + 300)
                    context += content[start:end] + "\n"
            except Exception:
                pass

        return context

    def to_summary(self) -> str:
        """Human-readable summary of the call graph."""
        lines = [f"Call graph: {len(self.functions)} functions in {len(self.files)} files\n"]
        # Show functions with most callers (likely important internal APIs)
        by_callers = sorted(
            self.functions.values(),
            key=lambda f: len(f.called_by),
            reverse=True,
        )
        lines.append("Most-called functions:")
        for fdef in by_callers[:15]:
            lines.append(
                f"  {fdef.name} ({fdef.file}:{fdef.line}) "
                f"— called by {len(fdef.called_by)}, calls {len(fdef.calls)}"
            )
        return "\n".join(lines)


def build_callgraph(src_dir: str) -> CallGraph:
    """Build a call graph using ctags + regex call extraction."""
    graph = CallGraph()
    src_path = Path(src_dir)

    # Step 1: Run ctags to get all function definitions
    logger.info("[callgraph] Running ctags on %s...", src_dir)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tags", delete=False) as tmp:
            tags_file = tmp.name

        subprocess.run(
            [
                "ctags",
                "-R",
                "--languages=C,C++",
                "--c-kinds=f",  # functions only
                "--fields=+nS",  # include line numbers and signatures
                f"--output-format=json",
                "-f", tags_file,
                str(src_path),
            ],
            capture_output=True,
            timeout=30,
        )

        # Try JSON format first
        if os.path.exists(tags_file) and os.path.getsize(tags_file) > 0:
            _parse_ctags_json(tags_file, src_path, graph)

    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("[callgraph] ctags JSON failed, trying default format...")
    finally:
        try:
            os.unlink(tags_file)
        except Exception:
            pass

    # Fallback: if JSON didn't work, try default ctags format
    if not graph.functions:
        logger.info("[callgraph] Trying ctags default format...")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".tags", delete=False) as tmp:
                tags_file = tmp.name

            subprocess.run(
                [
                    "ctags",
                    "-R",
                    "--languages=C,C++",
                    "--c-kinds=f",
                    "--fields=+n",
                    "-f", tags_file,
                    str(src_path),
                ],
                capture_output=True,
                timeout=30,
            )
            _parse_ctags_default(tags_file, src_path, graph)
        except Exception as exc:
            logger.error("[callgraph] ctags fallback failed: %s", exc)
        finally:
            try:
                os.unlink(tags_file)
            except Exception:
                pass

    # Step 2: Extract call relationships by scanning source
    logger.info("[callgraph] Extracting call relationships...")
    func_names = set(graph.functions.keys())
    _extract_calls(src_path, graph, func_names)

    logger.info(
        "[callgraph] Built graph: %d functions, %d files",
        len(graph.functions), len(graph.files),
    )
    return graph


def _parse_ctags_json(tags_file: str, src_path: Path, graph: CallGraph) -> None:
    """Parse ctags JSON output."""
    with open(tags_file) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                tag = json.loads(line)
                name = tag.get("name", "")
                path = tag.get("path", "")
                line_no = tag.get("line", 0)
                sig = tag.get("signature", "")

                if not name or not path:
                    continue

                try:
                    rel_path = str(Path(path).relative_to(src_path))
                except ValueError:
                    rel_path = path

                fdef = FunctionDef(
                    name=name, file=rel_path,
                    line=line_no, signature=sig,
                )
                graph.functions[name] = fdef
                graph.files.setdefault(rel_path, []).append(name)
            except json.JSONDecodeError:
                continue


def _parse_ctags_default(tags_file: str, src_path: Path, graph: CallGraph) -> None:
    """Parse default ctags tab-separated output."""
    with open(tags_file) as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            name = parts[0]
            path = parts[1]
            # Extract line number from fields
            line_no = 0
            for part in parts[3:]:
                if part.startswith("line:"):
                    try:
                        line_no = int(part.split(":")[1])
                    except (ValueError, IndexError):
                        pass

            try:
                rel_path = str(Path(path).relative_to(src_path))
            except ValueError:
                rel_path = path

            fdef = FunctionDef(name=name, file=rel_path, line=line_no)
            graph.functions[name] = fdef
            graph.files.setdefault(rel_path, []).append(name)


def _extract_calls(src_path: Path, graph: CallGraph, func_names: set[str]) -> None:
    """Scan source files to find call relationships."""
    # Build regex that matches any known function name followed by (
    if not func_names:
        return

    call_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(fn) for fn in func_names) + r')\s*\('
    )

    for fpath in src_path.rglob("*.c"):
        fstr = str(fpath).lower()
        if any(skip in fstr for skip in [".git", "test", "example"]):
            continue

        try:
            content = fpath.read_text(errors="replace")
            rel_path = str(fpath.relative_to(src_path))
        except Exception:
            continue

        # Find which function in this file each line belongs to
        file_funcs = graph.files.get(rel_path, [])
        if not file_funcs:
            continue

        # Simple approach: for each function defined in this file,
        # scan its body for calls to other known functions
        lines = content.split("\n")
        for func_name in file_funcs:
            fdef = graph.functions.get(func_name)
            if not fdef or fdef.line <= 0:
                continue

            # Extract approximate function body (from definition to next function or 200 lines)
            start = fdef.line - 1
            end = min(len(lines), start + 200)

            # Find end of function (crude: matching brace)
            brace_count = 0
            found_open = False
            for i in range(start, end):
                for ch in lines[i]:
                    if ch == '{':
                        brace_count += 1
                        found_open = True
                    elif ch == '}':
                        brace_count -= 1
                if found_open and brace_count == 0:
                    end = i + 1
                    break

            body = "\n".join(lines[start:end])
            fdef.body = body[:2000]  # store first 2000 chars

            # Find calls in this body
            for match in call_pattern.finditer(body):
                callee_name = match.group(1)
                if callee_name != func_name:  # skip recursive self-calls
                    if callee_name not in fdef.calls:
                        fdef.calls.append(callee_name)
                    callee = graph.functions.get(callee_name)
                    if callee and func_name not in callee.called_by:
                        callee.called_by.append(func_name)
