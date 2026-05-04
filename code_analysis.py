"""
code_analysis.py — AST-based code analysis using tree-sitter.

Provides structured understanding of C codebases:
- Function definitions with full parameter types
- Call relationships (who calls whom)
- Include statements per file
- Dangerous pattern detection with context
- Structured queries for security-relevant code

Falls back to regex if tree-sitter is not installed.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("gemma-fuzzer.analysis")

# ── Try to load tree-sitter ───────────────────────────────────────

HAS_TREE_SITTER = False
_parser = None
_c_lang = None

try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser

    _c_lang = Language(tsc.language())
    _parser = Parser(_c_lang)
    HAS_TREE_SITTER = True
    logger.info("[analysis] tree-sitter loaded successfully.")
except ImportError:
    logger.warning("[analysis] tree-sitter not available, using regex fallback.")
except Exception as exc:
    logger.warning("[analysis] tree-sitter init failed (%s), using regex.", exc)


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class ParamInfo:
    """A function parameter with type information."""
    full_text: str          # "const xmlChar *prefix"
    type_text: str          # "const xmlChar *"
    name: str               # "prefix"
    is_pointer: bool        # True
    is_unsigned_int: bool   # False
    is_size: bool           # True if size_t, int size, unsigned int len, etc.


@dataclass
class FunctionInfo:
    """Complete information about a function definition."""
    name: str
    file: str               # relative path
    line: int
    return_type: str
    params: list[ParamInfo]
    body: str               # first 3000 chars of function body
    body_full_len: int
    includes: list[str]     # includes from the file this function is in
    called_by: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    is_static: bool = False

    @property
    def signature(self) -> str:
        params_str = ", ".join(p.full_text for p in self.params)
        return f"{self.return_type} {self.name}({params_str})"


@dataclass
class CallGraph:
    """Call graph with function definitions and relationships."""
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    files: dict[str, list[str]] = field(default_factory=dict)
    file_includes: dict[str, list[str]] = field(default_factory=dict)

    def get_callers(self, func_name: str, depth: int = 3) -> list[list[str]]:
        paths: list[list[str]] = []
        self._trace(func_name, [], paths, depth)
        return paths

    def _trace(self, func, current, all_paths, depth):
        if depth <= 0 or func in current:
            return
        path = current + [func]
        fdef = self.functions.get(func)
        if not fdef or not fdef.called_by:
            all_paths.append(list(reversed(path)))
            return
        for caller in fdef.called_by[:5]:
            self._trace(caller, path, all_paths, depth - 1)

    def get_function_context(self, func_name: str, src_dir: str) -> str:
        """Get function source + caller context for LLM prompts."""
        fdef = self.functions.get(func_name)
        if not fdef:
            return ""

        ctx = f"// === {fdef.name} in {fdef.file}:{fdef.line} ===\n"
        ctx += f"// Signature: {fdef.signature}\n"
        ctx += f"// Static: {fdef.is_static}\n"
        if fdef.body:
            ctx += fdef.body[:3000] + "\n"
        else:
            try:
                content = (Path(src_dir) / fdef.file).read_text(errors="replace")
                lines = content.split("\n")
                start = max(0, fdef.line - 1)
                end = min(len(lines), start + 80)
                ctx += "\n".join(lines[start:end]) + "\n"
            except Exception:
                pass

        # Add caller context (how external input reaches this function)
        for caller_name in fdef.called_by[:3]:
            caller = self.functions.get(caller_name)
            if not caller:
                continue
            ctx += f"\n// === Called by {caller_name} ({caller.file}:{caller.line}) ===\n"
            try:
                content = (Path(src_dir) / caller.file).read_text(errors="replace")
                idx = content.find(func_name + "(")
                if idx < 0:
                    idx = content.find(func_name)
                if idx >= 0:
                    start = max(0, idx - 400)
                    end = min(len(content), idx + 400)
                    ctx += content[start:end] + "\n"
            except Exception:
                pass

        return ctx

    def get_includes_for_file(self, filename: str) -> list[str]:
        """Get #include lines for a source file."""
        return self.file_includes.get(filename, [])

    def to_summary(self) -> str:
        lines = [f"Call graph: {len(self.functions)} functions in {len(self.files)} files"]
        by_callers = sorted(self.functions.values(),
                            key=lambda f: len(f.called_by), reverse=True)
        lines.append("\nMost-called functions:")
        for fdef in by_callers[:20]:
            if fdef.called_by:
                lines.append(
                    f"  {fdef.name} ({fdef.file}:{fdef.line}) "
                    f"← {len(fdef.called_by)} callers, → {len(fdef.calls)} callees"
                )
        return "\n".join(lines)

    def find_dangerous_functions(self, min_score: int = 10) -> list[tuple[str, int]]:
        """Find functions likely to contain vulnerabilities."""
        scored = []
        for name, fdef in self.functions.items():
            score = 0
            body = fdef.body.lower() if fdef.body else ""

            # Score by dangerous operations in body
            for pattern, weight in [
                ("memcpy", 3), ("memmove", 2), ("strcpy", 4), ("strcat", 4),
                ("sprintf", 4), ("malloc", 2), ("realloc", 3), ("free", 2),
                ("atoi", 3), ("sscanf", 3), ("gets", 5), ("strlen", 1),
            ]:
                if pattern in body:
                    score += weight * body.count(pattern)

            # Boost if many callers (important internal API)
            score += len(fdef.called_by) * 2

            # Boost security-relevant function names
            name_lower = name.lower()
            for kw in ["parse", "read", "decode", "add", "copy", "alloc",
                        "create", "append", "concat", "grow", "resize",
                        "buf", "dict", "string", "encode", "header",
                        "cert", "key", "session", "handshake"]:
                if kw in name_lower:
                    score += 5
                    break

            # Boost if has size/length parameters
            for param in fdef.params:
                if param.is_size:
                    score += 5

            if score >= min_score:
                scored.append((name, score))

        scored.sort(key=lambda x: -x[1])
        return scored


# ══════════════════════════════════════════════════════════════════
# BUILD CALL GRAPH (main entry point)
# ══════════════════════════════════════════════════════════════════

def build_callgraph(src_dir: str) -> CallGraph:
    """Build a call graph from a C source directory."""
    graph = CallGraph()
    src_path = Path(src_dir)

    logger.info("[analysis] Building call graph for %s (tree-sitter=%s)...",
                src_dir, HAS_TREE_SITTER)

    # Phase 1: Extract function definitions from all .c files
    for fpath in src_path.rglob("*.c"):
        fstr = str(fpath).lower()
        if any(s in fstr for s in [
            ".git", "/test", "example", "python", "CMakeFiles", "/fuzz/",
            # Docker build tree includes fuzzing infrastructure — skip all of it
            "aflplusplus", "afl-", "honggfuzz", "libqasan", "qemu_mode",
            "libfuzzer", "centipede", "jazzer", "oss-fuzz",
            # Build artifacts
            "/build/", "/obj/", "/.deps/",
        ]):
            continue

        try:
            content = fpath.read_text(errors="replace")
            content_bytes = content.encode("utf-8", errors="replace")
            rel_path = str(fpath.relative_to(src_path))
        except Exception:
            continue

        # Extract includes for this file
        includes = _extract_includes(content)
        graph.file_includes[rel_path] = includes

        # Extract functions
        if HAS_TREE_SITTER:
            funcs = _extract_functions_treesitter(content_bytes, rel_path, includes)
        else:
            funcs = _extract_functions_regex(content, rel_path, includes)

        for func in funcs:
            if func.name not in graph.functions:
                graph.functions[func.name] = func
                graph.files.setdefault(rel_path, []).append(func.name)

    logger.info("[analysis] Found %d functions in %d files.",
                len(graph.functions), len(graph.files))

    # Phase 2: Extract call relationships
    func_names = set(graph.functions.keys())
    if len(func_names) > 5:
        _extract_call_relationships(graph, func_names)

    logger.info("[analysis] Call graph complete.")
    return graph


# ══════════════════════════════════════════════════════════════════
# TREE-SITTER EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _extract_functions_treesitter(
    content_bytes: bytes, filename: str, includes: list[str],
) -> list[FunctionInfo]:
    """Extract functions using tree-sitter AST."""
    if not _parser:
        return []

    tree = _parser.parse(content_bytes)
    functions = []

    for node in _walk(tree.root_node):
        if node.type != "function_definition":
            continue

        # Find function_declarator (may be nested under pointer_declarator)
        func_decl = _find_descendant(node, "function_declarator")
        if not func_decl:
            continue

        # Extract function name
        name_node = _find_child(func_decl, "identifier")
        if not name_node:
            # Try pointer_declarator → identifier
            ptr_decl = _find_child(func_decl, "pointer_declarator")
            if ptr_decl:
                name_node = _find_child(ptr_decl, "identifier")
        if not name_node:
            continue

        name = content_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

        # Skip common non-function identifiers
        if name in {"if", "for", "while", "switch", "return", "sizeof"}:
            continue

        # Extract return type
        return_type = ""
        type_node = _find_child(node, "primitive_type") or \
                     _find_child(node, "type_identifier") or \
                     _find_child(node, "sized_type_specifier")
        if type_node:
            return_type = content_bytes[type_node.start_byte:type_node.end_byte].decode("utf-8", errors="replace")

        # Check if static
        is_static = False
        for child in node.children:
            if child.type == "storage_class_specifier":
                text = content_bytes[child.start_byte:child.end_byte].decode()
                if text == "static":
                    is_static = True

        # Extract parameters
        params_node = _find_child(func_decl, "parameter_list")
        params = _extract_params(params_node, content_bytes) if params_node else []

        # Extract body
        body_node = _find_child(node, "compound_statement")
        body = ""
        body_len = 0
        if body_node:
            raw_body = content_bytes[body_node.start_byte:body_node.end_byte]
            body_len = len(raw_body)
            body = raw_body[:3000].decode("utf-8", errors="replace")

        line = node.start_point[0] + 1

        functions.append(FunctionInfo(
            name=name, file=filename, line=line,
            return_type=return_type, params=params,
            body=body, body_full_len=body_len,
            includes=includes, is_static=is_static,
        ))

    return functions


def _extract_params(params_node, content_bytes: bytes) -> list[ParamInfo]:
    """Extract parameter information from a parameter_list node."""
    params = []
    for child in params_node.children:
        if child.type != "parameter_declaration":
            continue

        full_text = content_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()

        # Extract parameter name (last identifier in the declaration)
        name = ""
        type_text = full_text

        # Find the declarator (contains the parameter name)
        decl = _find_child(child, "identifier")
        if not decl:
            decl = _find_descendant(child, "identifier")
        if decl:
            name = content_bytes[decl.start_byte:decl.end_byte].decode()
            # Type is everything before the name
            type_text = full_text[:full_text.rfind(name)].strip()
            if not type_text:
                type_text = full_text

        is_pointer = "*" in full_text
        full_lower = full_text.lower()
        is_unsigned_int = "unsigned" in full_lower and "int" in full_lower and not is_pointer
        is_size = any(kw in full_lower for kw in [
            "size_t", "unsigned int", "int len", "int size", "int n",
            "int plen", "int nlen", "int count",
        ])

        params.append(ParamInfo(
            full_text=full_text, type_text=type_text, name=name,
            is_pointer=is_pointer, is_unsigned_int=is_unsigned_int,
            is_size=is_size,
        ))

    return params


def _walk(node):
    """Walk all nodes in a tree-sitter tree."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _find_child(node, type_name):
    """Find first direct child of given type."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _find_descendant(node, type_name):
    """Find first descendant of given type (recursive)."""
    for child in node.children:
        if child.type == type_name:
            return child
        found = _find_descendant(child, type_name)
        if found:
            return found
    return None


# ══════════════════════════════════════════════════════════════════
# REGEX FALLBACK
# ══════════════════════════════════════════════════════════════════

def _extract_functions_regex(
    content: str, filename: str, includes: list[str],
) -> list[FunctionInfo]:
    """Fallback: extract functions using brace-matching."""
    functions = []
    lines = content.split("\n")
    IDENT = re.compile(r'^[a-zA-Z_]\w*$')
    SKIP = {"if", "else", "while", "for", "do", "switch", "case", "return",
            "sizeof", "typeof", "goto", "break", "continue", "default",
            "struct", "union", "enum", "typedef"}

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        opens_func = False
        sig_end = -1

        if re.search(r'\)\s*\{', line):
            opens_func = True
            sig_end = i
        elif line.strip() == '{' and i > 0:
            for back in range(i - 1, max(i - 4, -1), -1):
                if lines[back].rstrip().endswith(')'):
                    opens_func = True
                    sig_end = back
                    break
                elif lines[back].rstrip().endswith(',') or lines[back].rstrip().endswith('('):
                    continue
                else:
                    break

        if not opens_func or sig_end < 0:
            i += 1
            continue

        # Gather signature text
        sig_text = ""
        start_line = sig_end
        paren_count = 0
        found_open = False
        for back in range(sig_end, max(sig_end - 10, -1), -1):
            l = lines[back].rstrip()
            sig_text = l + " " + sig_text
            for ch in l:
                if ch == ')': paren_count += 1
                elif ch == '(':
                    paren_count -= 1
                    if paren_count <= 0:
                        found_open = True
            if found_open:
                start_line = back
                for extra in range(1, 3):
                    if back - extra >= 0:
                        prev = lines[back - extra].rstrip()
                        if prev and not any(prev.endswith(c) for c in [';', '}', '#']):
                            sig_text = prev + " " + sig_text
                            start_line = back - extra
                        else:
                            break
                break

        if not found_open:
            i += 1
            continue

        paren_idx = sig_text.find('(')
        if paren_idx < 0:
            i += 1
            continue

        before = sig_text[:paren_idx].strip()
        tokens = re.split(r'[\s\*]+', before)
        func_name = None
        for t in reversed(tokens):
            if IDENT.match(t) and t not in SKIP and len(t) >= 2:
                func_name = t
                break

        if not func_name:
            i += 1
            continue

        is_static = "static" in before.lower()

        # Extract body
        char_offset = sum(len(lines[j]) + 1 for j in range(min(i, sig_end)))
        brace_pos = content.find('{', char_offset)
        if brace_pos < 0:
            i += 1
            continue

        body_end = _find_brace_end(content, brace_pos)
        sig_offset = sum(len(lines[j]) + 1 for j in range(start_line))
        body_text = content[sig_offset:min(body_end, sig_offset + 3000)]

        functions.append(FunctionInfo(
            name=func_name, file=filename, line=start_line + 1,
            return_type="", params=[], body=body_text,
            body_full_len=body_end - sig_offset, includes=includes,
            is_static=is_static,
        ))

        body_end_line = content[:body_end].count('\n')
        i = body_end_line + 1

    return functions


def _find_brace_end(content: str, open_pos: int) -> int:
    """Find matching closing brace."""
    count = 0
    for i in range(open_pos, min(len(content), open_pos + 50000)):
        if content[i] == '{': count += 1
        elif content[i] == '}':
            count -= 1
            if count == 0:
                return i + 1
    return min(len(content), open_pos + 5000)


# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════

def _extract_includes(content: str) -> list[str]:
    """Extract #include lines from source code."""
    includes = []
    for line in content.split("\n")[:100]:
        stripped = line.strip()
        if stripped.startswith("#include"):
            includes.append(stripped)
    return includes


def _extract_call_relationships(graph: CallGraph, func_names: set[str]) -> None:
    """Scan function bodies for calls to other known functions."""
    # Build regex pattern for all known function names
    # Process in chunks to avoid regex-too-large
    name_list = sorted(func_names, key=len, reverse=True)

    for chunk_start in range(0, len(name_list), 200):
        chunk = name_list[chunk_start:chunk_start + 200]
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(n) for n in chunk) + r')\s*\('
        )

        for fname, fdef in graph.functions.items():
            if not fdef.body:
                continue
            for match in pattern.finditer(fdef.body):
                callee = match.group(1)
                if callee != fname and callee in func_names:
                    if callee not in fdef.calls:
                        fdef.calls.append(callee)
                    callee_def = graph.functions.get(callee)
                    if callee_def and fname not in callee_def.called_by:
                        callee_def.called_by.append(fname)


def get_file_includes(src_dir: str, filename: str) -> list[str]:
    """Read #include lines from a source file. Public API for agents."""
    try:
        path = Path(src_dir) / filename
        if not path.exists():
            matches = list(Path(src_dir).rglob(Path(filename).name))
            path = matches[0] if matches else path
        return _extract_includes(path.read_text(errors="replace"))
    except Exception:
        return []


def find_include_dirs(src_dir: str) -> list[str]:
    """Find all directories containing headers. Public API."""
    dirs = set()
    dirs.add(src_dir)
    src_path = Path(src_dir)

    # Add common subdirectories
    for subdir in ["include", "src", "lib"]:
        p = src_path / subdir
        if p.is_dir():
            dirs.add(str(p))

    # Skip patterns for fuzzing infrastructure in Docker builds
    skip = [".git", "CMakeFiles", "aflplusplus", "afl-", "honggfuzz",
            "libqasan", "qemu_mode", "libfuzzer", "centipede"]

    # Add ALL directories containing .h files
    for h in src_path.rglob("*.h"):
        d = str(h.parent)
        if not any(s in d.lower() for s in skip):
            dirs.add(d)

    return list(dirs)


def find_static_lib(src_dir: str) -> str | None:
    """Find the main static library. Public API."""
    for pattern in [".libs/*.a", "**/*.a"]:
        matches = list(Path(src_dir).glob(pattern))
        if matches:
            matches.sort(key=lambda p: p.stat().st_size, reverse=True)
            return str(matches[0])
    return None
