"""
crs/strategy_seed_download.py — Format-aware seed acquisition & targeted mutation.

The problem: LLMs cannot reliably generate valid binary file formats
(MP4, PNG, ELF, TIFF, etc.) from scratch. But they CAN reason about
which specific field in a valid file to corrupt to trigger a vulnerability.

This strategy:
1. DETECT FORMAT: Read source code + description to identify what file
   format the binary expects (MP4, PNG, AAC, PDF, etc.)
2. FIND SEED: Search the repo for test files, or download a minimal
   valid sample from known sources.
3. TARGETED MUTATION: Ask the LLM to analyze the vulnerability and
   write a Python script that takes the valid seed file and corrupts
   the SPECIFIC field/atom/chunk that triggers the bug.
4. TEST + REFINE: If it doesn't crash, feed the error back.

Works for any project — faad2, imagemagick, libpng, ffmpeg, mruby, etc.
"""
from __future__ import annotations
import os, re, subprocess, tempfile, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from crs.config import cfg
from crs.byte_gen import PoCCandidate, _save_poc, _extract_python


# ── Format detection ──────────────────────────────────────────────────────

# Map of format signatures found in source code → format info
# Each entry: (source_keywords, file_extensions, format_name, description)
FORMAT_SIGNATURES = [
    # Audio/Video containers
    (["ftyp", "moov", "mdat", "stbl", "stsz", "stco", "mp4a", "esds"],
     [".mp4", ".m4a", ".m4v", ".mov", ".3gp"], "mp4", "MPEG-4 container"),
    (["ADTS", "adts", "syncword", "0xFFF", "aac_frame"],
     [".aac", ".adts"], "aac", "AAC audio"),
    (["ID3", "MPEG", "mp3_read", "mad_frame"],
     [".mp3"], "mp3", "MP3 audio"),
    (["RIFF", "WAVE", "fmt ", "data"],
     [".wav"], "wav", "WAV audio"),
    (["OggS", "vorbis", "ogg_page"],
     [".ogg", ".oga"], "ogg", "Ogg container"),
    (["fLaC", "FLAC", "flac_decode"],
     [".flac"], "flac", "FLAC audio"),

    # Images
    (["PNG", "IHDR", "IDAT", "IEND", "png_read", "89504E47"],
     [".png"], "png", "PNG image"),
    (["JFIF", "jpeg_read", "SOI", "0xFFD8", "huffman"],
     [".jpg", ".jpeg"], "jpeg", "JPEG image"),
    (["GIF8", "gif_read", "GIF89a"],
     [".gif"], "gif", "GIF image"),
    (["TIFF", "tiff", "IFD", "EXIF", "BigTIFF"],
     [".tiff", ".tif"], "tiff", "TIFF image"),
    (["BMP", "bmp_read", "BITMAPINFOHEADER"],
     [".bmp"], "bmp", "BMP image"),
    (["WEBP", "webp", "VP8", "RIFF"],
     [".webp"], "webp", "WebP image"),
    (["MNG", "mng_read", "MHDR", "mng_LOOP"],
     [".mng"], "mng", "MNG animation"),

    # Documents
    (["PDF", "pdf_read", "%PDF", "xref", "startxref"],
     [".pdf"], "pdf", "PDF document"),
    (["<?xml", "XML", "xml_parse", "libxml"],
     [".xml"], "xml", "XML document"),

    # Executables/Debug
    (["ELF", "elf_read", "Elf32", "Elf64", "7f454c46"],
     [".elf", ".so", ".o"], "elf", "ELF binary"),
    (["DWARF", "dwarf", "debug_info", "debug_line", "DW_TAG"],
     [".dwarf"], "dwarf", "DWARF debug info"),
    (["PE", "IMAGE_DOS", "MZ", "COFF"],
     [".exe", ".dll", ".pe"], "pe", "PE executable"),

    # Archives
    (["ZIP", "PK\x03\x04", "zip_read", "local_file_header"],
     [".zip"], "zip", "ZIP archive"),
    (["gzip", "gz", "inflate", "deflate"],
     [".gz", ".gzip"], "gzip", "GZIP compressed"),

    # Data formats
    (["JSON", "json_parse", "json_read"],
     [".json"], "json", "JSON data"),
    (["YAML", "yaml_parse"],
     [".yaml", ".yml"], "yaml", "YAML data"),

    # Scripting
    (["mrb_", "mruby", "RiteVM", "RITE"],
     [".rb", ".mrb"], "mruby", "mruby bytecode/script"),
    (["njs_", "njs_vm", "javascript"],
     [".js"], "njs", "njs JavaScript"),
    (["lua_", "LuaState", "luaL_"],
     [".lua"], "lua", "Lua script"),
]

# Known sources for minimal valid test files (format → URLs)
# These are well-known sample file repositories
SAMPLE_SOURCES = {
    "mp4":  ["https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
             "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"],
    "aac":  ["https://filesamples.com/samples/audio/aac/sample1.aac"],
    "mp3":  ["https://filesamples.com/samples/audio/mp3/sample1.mp3"],
    "wav":  ["https://filesamples.com/samples/audio/wav/sample1.wav"],
    "ogg":  ["https://filesamples.com/samples/audio/ogg/sample1.ogg"],
    "flac": ["https://filesamples.com/samples/audio/flac/sample1.flac"],
    "png":  ["https://filesamples.com/samples/image/png/sample_640x426.png"],
    "jpeg": ["https://filesamples.com/samples/image/jpeg/sample_640x426.jpeg"],
    "gif":  ["https://filesamples.com/samples/image/gif/sample_640x426.gif"],
    "tiff": ["https://filesamples.com/samples/image/tiff/sample_640x426.tiff"],
    "bmp":  ["https://filesamples.com/samples/image/bmp/sample_640x426.bmp"],
    "webp": ["https://filesamples.com/samples/image/webp/sample_640x426.webp"],
    "pdf":  ["https://filesamples.com/samples/document/pdf/sample1.pdf"],
    "json": [],  # trivial to generate inline
    "xml":  [],  # trivial to generate inline
    "elf":  [],  # can compile a minimal C program
    "zip":  [],  # can create with Python zipfile module
}

# Minimal inline generators for formats that don't need downloading
INLINE_GENERATORS = {
    "json": b'{"key": "value", "array": [1, 2, 3], "nested": {"a": "b"}}',
    "xml":  b'<?xml version="1.0"?>\n<root><item id="1">test</item></root>',
    "yaml": b'key: value\nlist:\n  - item1\n  - item2\n',
    "mruby": b'puts "hello"\na = [1,2,3]\na.each {|x| puts x}\n',
    "njs":  b'var x = {a: 1, b: [1,2,3]};\nfunction f(n) { return n > 0 ? n + f(n-1) : 0; }\nf(10);\n',
    "lua":  b'local t = {1, 2, 3}\nfor i, v in ipairs(t) do print(v) end\n',
}


def detect_format(repo: Path, description: str) -> Optional[dict]:
    """
    Detect what file format the binary expects by scanning source code
    and the vulnerability description.

    Returns: {format_name, description, extensions, confidence}
    """
    desc_lower = description.lower()

    # Score each format
    best_format = None
    best_score = 0

    # Read source files for keyword matching
    source_text = ""
    src_exts = {".c", ".cpp", ".cc", ".h", ".hpp"}
    for f in sorted(repo.rglob("*")):
        if f.is_file() and f.suffix.lower() in src_exts and len(source_text) < 50000:
            try:
                source_text += f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    source_lower = source_text.lower()

    for keywords, extensions, fmt_name, fmt_desc in FORMAT_SIGNATURES:
        score = 0

        # Check description
        for kw in keywords:
            if kw.lower() in desc_lower:
                score += 3

        # Check source code
        for kw in keywords:
            count = source_lower.count(kw.lower())
            if count > 0:
                score += min(count, 5)  # cap per-keyword

        # Check if any matching extensions exist in repo
        for ext in extensions:
            if list(repo.rglob(f"*{ext}"))[:1]:
                score += 2

        if score > best_score:
            best_score = score
            best_format = {
                "format_name": fmt_name,
                "description": fmt_desc,
                "extensions": extensions,
                "confidence": min(score / 15.0, 1.0),
                "keywords_matched": [kw for kw in keywords if kw.lower() in source_lower or kw.lower() in desc_lower],
            }

    if best_format and best_score >= 3:
        return best_format
    return None


# ── Seed acquisition ──────────────────────────────────────────────────────

def find_seed_in_repo(repo: Path, fmt: dict) -> Optional[bytes]:
    """Search the repo for an existing test file of the detected format."""
    search_dirs = ["test", "tests", "testdata", "samples", "examples",
                   "corpus", "fixtures", "test_data", "testcases",
                   "tests/data", "test/data", "fuzz/corpus"]

    for ext in fmt["extensions"]:
        # Search in test directories first
        for d in search_dirs:
            sd = repo / d
            if not sd.is_dir():
                continue
            for f in sorted(sd.rglob(f"*{ext}"))[:5]:
                if f.is_file() and 10 < f.stat().st_size < 2_000_000:
                    try:
                        data = f.read_bytes()
                        print(f"    Found seed in repo: {f} ({len(data)} bytes)")
                        return data
                    except OSError:
                        pass

        # Search entire repo
        for f in sorted(repo.rglob(f"*{ext}"))[:5]:
            if f.is_file() and 10 < f.stat().st_size < 2_000_000:
                # Skip source code and build artifacts
                if any(skip in str(f).lower() for skip in [".git", "build", "cmake"]):
                    continue
                try:
                    data = f.read_bytes()
                    print(f"    Found seed in repo: {f} ({len(data)} bytes)")
                    return data
                except OSError:
                    pass

    return None


def download_seed(fmt: dict, work_dir: Path) -> Optional[bytes]:
    """Download a minimal valid sample file for the detected format."""
    fmt_name = fmt["format_name"]

    # Try inline generation first
    if fmt_name in INLINE_GENERATORS:
        data = INLINE_GENERATORS[fmt_name]
        print(f"    Generated inline {fmt_name} seed ({len(data)} bytes)")
        return data

    # Try to create ELF by compiling minimal C
    if fmt_name == "elf":
        return _create_minimal_elf(work_dir)

    # Try downloading
    urls = SAMPLE_SOURCES.get(fmt_name, [])
    for url in urls:
        try:
            print(f"    Downloading {fmt_name} sample from {url[:60]}...")
            dest = work_dir / f"seed.{fmt_name}"
            r = subprocess.run(
                ["wget", "-q", "--timeout=15", "-O", str(dest), url],
                capture_output=True, timeout=30,
            )
            if r.returncode == 0 and dest.exists() and dest.stat().st_size > 10:
                data = dest.read_bytes()
                print(f"    Downloaded: {len(data)} bytes")
                return data
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"    Download failed: {e}")
            continue

    # Try curl as fallback
    for url in urls:
        try:
            r = subprocess.run(
                ["curl", "-sL", "--max-time", "15", "-o", str(work_dir / f"seed.{fmt_name}"), url],
                capture_output=True, timeout=30,
            )
            dest = work_dir / f"seed.{fmt_name}"
            if r.returncode == 0 and dest.exists() and dest.stat().st_size > 10:
                data = dest.read_bytes()
                print(f"    Downloaded (curl): {len(data)} bytes")
                return data
        except Exception:
            continue

    return None


def _create_minimal_elf(work_dir: Path) -> Optional[bytes]:
    """Compile a minimal C program to get a valid ELF binary."""
    src = work_dir / "minimal.c"
    out = work_dir / "minimal.elf"
    src.write_text("int main(){return 0;}\n")
    try:
        r = subprocess.run(
            ["gcc", "-o", str(out), str(src)],
            capture_output=True, timeout=15,
        )
        if r.returncode == 0 and out.exists():
            data = out.read_bytes()
            print(f"    Created minimal ELF: {len(data)} bytes")
            return data
    except Exception:
        pass
    return None


# ── LLM-guided targeted mutation ──────────────────────────────────────────

SYSTEM_TARGETED_MUTATION = """\
You are an expert at surgically corrupting file formats to trigger
specific C/C++ vulnerabilities.

You will receive:
- A vulnerability description (CVE or ASAN crash info)
- The file format of the input (e.g., MP4, PNG, AAC)
- Source code from the vulnerable project
- A valid seed file is available as a Python bytes literal or file path

YOUR JOB: Write a Python script that reads the valid seed file and
makes the MINIMUM change needed to trigger the vulnerability.

REASONING STEPS:
1. What atom/chunk/field does the vulnerability involve?
   (e.g., "stsz atom in MP4", "IHDR width in PNG", "huffman table in JPEG")
2. How is that field structured in the format?
   (e.g., "stsz: 4-byte size, 'stsz', 1-byte version, 3 flags, 4-byte sample_size, 4-byte sample_count")
3. What value triggers the bug?
   (e.g., "sample_count = 0x7FFFFFFF causes heap overflow when allocating sample_count * 4 bytes")
4. Write the mutation.

CRITICAL RULES:
1. Output ONLY a Python script inside ```python ... ```
2. The script reads the seed file from sys.argv[1]
3. The script writes the mutated file to sys.argv[2]
4. Make the MINIMUM change — keep everything else valid so the parser
   reaches the vulnerable code
5. Use struct.pack for binary fields
6. Comment: "# MUTATION: ..." for every change
7. Search for the relevant atom/chunk/marker by its magic bytes,
   don't hardcode offsets (they vary between files)
"""

_SRC_EXTS = {".c", ".cpp", ".cc", ".h", ".hpp"}

def _read_vuln_sources(repo: Path, description: str, max_chars: int = 6000) -> str:
    """Read source files most relevant to the vulnerability."""
    desc_lower = description.lower()
    desc_words = set(re.findall(r"\b\w{4,}\b", desc_lower))

    scored: list[tuple[Path, float]] = []
    for f in repo.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in _SRC_EXTS:
            continue
        score = 0.0
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            text_lower = text.lower()
            # Score by description word overlap
            for w in desc_words:
                if w in text_lower:
                    score += 1
            # Bonus for filename match
            for w in desc_words:
                if w in f.name.lower():
                    score += 5
            scored.append((f, score))
        except Exception:
            pass

    scored.sort(key=lambda x: x[1], reverse=True)

    parts, total = [], 0
    for path, sc in scored[:8]:
        if total >= max_chars:
            break
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            budget = max_chars - total
            parts.append(f"\n// === {path.name} (relevance={sc:.1f}) ===\n" + content[:budget])
            total += min(len(content), budget)
        except Exception:
            pass
    return "".join(parts)


def generate_targeted_mutation(
    task_id: str,
    description: str,
    repo_path: Path,
    router,
    seed_data: bytes,
    fmt: dict,
    binary_name: str = "",
    binary: Path = None,
    run_func = None,
    max_refine: int = 3,
) -> List[PoCCandidate]:
    """
    Have the LLM write a script that takes a valid seed file and
    makes a targeted mutation to trigger the vulnerability.
    """
    results: list[PoCCandidate] = []
    repo = Path(repo_path)
    work = cfg.task_work_dir(task_id)

    # Save seed file
    seed_path = work / f"seed.{fmt['format_name']}"
    seed_path.write_bytes(seed_data)

    # Read relevant source code
    sources = _read_vuln_sources(repo, description)

    prompt = (
        f"## Vulnerability Description\n{description}\n\n"
        f"## File Format: {fmt['description']} ({fmt['format_name']})\n"
        f"## Format keywords found: {', '.join(fmt.get('keywords_matched', []))}\n\n"
        f"## Seed File\n"
        f"A valid {fmt['format_name']} file ({len(seed_data)} bytes) is available.\n"
        f"The script should read it from sys.argv[1] and write mutated output to sys.argv[2].\n\n"
        f"## Relevant Source Code\n```c\n{sources}\n```\n\n"
    )
    if binary_name:
        prompt += f"## Target binary: {binary_name}\n\n"
    prompt += (
        f"Write a Python script that reads the valid {fmt['format_name']} seed file "
        f"and makes the MINIMUM change to trigger the described vulnerability.\n"
        f"Search for the relevant atom/chunk/field by its magic bytes — "
        f"don't hardcode byte offsets.\n"
    )

    print(f"    Asking LLM for targeted {fmt['format_name']} mutation...")
    raw = router.chat(SYSTEM_TARGETED_MUTATION, prompt, max_tokens=cfg.MAX_TOKENS, temperature=0.15)
    if not raw:
        return results

    script = _extract_python(raw)
    if not script:
        print("    No Python block in LLM response")
        return results

    # Run the mutation script
    mutated = _run_mutation_script(script, seed_path, work)
    if mutated and len(mutated) > 0:
        path = _save_poc(mutated, task_id, "targeted_mutation")
        poc = PoCCandidate(
            "targeted_mutation", mutated, path, 0.88,
            f"Targeted {fmt['format_name']} mutation: {len(mutated)}B "
            f"(seed was {len(seed_data)}B)",
            script,
        )
        results.append(poc)
        print(f"    Generated mutated file: {len(mutated)} bytes")

        # Iterative refinement if we have the binary
        if binary and run_func:
            for attempt in range(1, max_refine + 1):
                result = run_func(poc, binary)
                if result.triggered:
                    poc.confidence = 0.97
                    poc.name = f"targeted_mutation_v{attempt}"
                    print(f"    TRIGGERED on attempt {attempt}!")
                    return results

                print(f"    Attempt {attempt} — not triggered, refining...")
                refined_script = _refine_mutation(
                    router, script, result.sanitizer_output,
                    result.return_code, description, fmt, sources,
                )
                if not refined_script:
                    break

                refined_data = _run_mutation_script(refined_script, seed_path, work)
                if not refined_data or len(refined_data) == 0:
                    break

                path = _save_poc(refined_data, task_id, f"targeted_v{attempt+1}")
                poc = PoCCandidate(
                    f"targeted_v{attempt+1}", refined_data, path,
                    0.88 + attempt * 0.02,
                    f"Refined mutation v{attempt+1}: {len(refined_data)}B",
                    refined_script,
                )
                script = refined_script
                results.append(poc)
    else:
        print("    Mutation script produced no output")

    return results


SYSTEM_REFINE_MUTATION = """\
Your targeted file mutation didn't trigger the vulnerability.
You will see the script, the binary's output, and the vulnerability info.

Analyze:
- Did the binary parse the file? Or reject it?
- Did it reach the vulnerable function? Or take a different path?
- Was the mutation too aggressive (broke the format) or too subtle?

Write a FIXED mutation script. Same rules:
- Read seed from sys.argv[1], write mutated to sys.argv[2]
- Search for atoms/chunks by magic bytes, not hardcoded offsets
- Make MINIMUM changes to keep the format valid

Output ONLY a Python script inside ```python ... ```
Comment every change with "# FIX: ..."
"""


def _refine_mutation(router, prev_script, binary_output, return_code,
                      description, fmt, sources):
    """Ask LLM to fix the mutation based on binary feedback."""
    prompt = (
        f"## Vulnerability\n{description}\n\n"
        f"## Format: {fmt['description']}\n\n"
        f"## Previous Script\n```python\n{prev_script}\n```\n\n"
        f"## Binary Output (exit code {return_code})\n```\n{binary_output[:2500]}\n```\n\n"
        f"## Source Code\n```c\n{sources[:3000]}\n```\n\n"
        f"Fix the mutation.\n"
    )
    raw = router.chat(SYSTEM_REFINE_MUTATION, prompt, max_tokens=cfg.MAX_TOKENS, temperature=0.2)
    return _extract_python(raw) if raw else None


def _run_mutation_script(script: str, seed_path: Path, work: Path) -> Optional[bytes]:
    """Run a mutation script that reads seed_path and writes output."""
    output_path = work / "mutated_output.bin"

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        r = subprocess.run(
            ["python3", tmp_path, str(seed_path), str(output_path)],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            stderr = r.stderr.decode(errors="replace")
            print(f"    Mutation script error: {stderr[:300]}")
            # Maybe the script writes to stdout instead
            if r.stdout and len(r.stdout) > 0:
                return r.stdout
            return None

        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path.read_bytes()

        # Some scripts might write to stdout
        if r.stdout and len(r.stdout) > 0:
            return r.stdout

        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"    Mutation script error: {e}")
        return None
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass


# ── Main entry point ──────────────────────────────────────────────────────

def strategy_seed_and_mutate(
    task_id: str,
    description: str,
    repo_path: Path,
    router,
    binary_name: str = "",
    binary: Path = None,
    run_func = None,
) -> List[PoCCandidate]:
    """
    Complete strategy:
    1. Detect format from source + description
    2. Find or download a valid seed
    3. LLM writes targeted mutation
    4. Test and refine
    """
    results: list[PoCCandidate] = []
    repo = Path(repo_path)
    work = cfg.task_work_dir(task_id)

    # Step 1: Detect format
    print(f"    Detecting input format...")
    fmt = detect_format(repo, description)
    if not fmt:
        print(f"    Could not detect input format")
        return results

    print(f"    Detected: {fmt['description']} (confidence={fmt['confidence']:.2f})")
    print(f"    Keywords: {', '.join(fmt.get('keywords_matched', []))}")

    # Step 2: Get seed file
    print(f"    Looking for seed file...")
    seed_data = find_seed_in_repo(repo, fmt)

    if not seed_data:
        print(f"    No seed in repo, trying download...")
        seed_data = download_seed(fmt, work)

    if not seed_data:
        print(f"    No seed available for {fmt['format_name']}")
        return results

    print(f"    Seed: {len(seed_data)} bytes ({fmt['format_name']})")

    # Step 3 + 4: LLM-guided targeted mutation with refinement
    results = generate_targeted_mutation(
        task_id, description, repo, router,
        seed_data, fmt, binary_name, binary, run_func,
    )

    # Also generate some blind mutations of the seed as fallback
    print(f"    Also generating blind mutations...")
    blind = _blind_mutate_seed(seed_data, task_id, fmt, description)
    results.extend(blind)

    return results


def _blind_mutate_seed(seed: bytes, task_id: str, fmt: dict,
                        description: str) -> List[PoCCandidate]:
    """
    Apply generic but format-informed mutations without LLM.
    Targets common vulnerability patterns in binary formats.
    """
    results: list[PoCCandidate] = []
    desc_lower = description.lower()

    # Find all 4-byte sequences that look like length fields
    # (common in MP4 atoms, PNG chunks, etc.)
    length_offsets = []
    for i in range(0, min(len(seed) - 4, 10000), 4):
        val = int.from_bytes(seed[i:i+4], 'big')
        if 8 < val < len(seed) * 2:
            length_offsets.append(i)

    # Mutation 1: Corrupt length fields to huge values
    for i, offset in enumerate(length_offsets[:5]):
        d = bytearray(seed)
        d[offset:offset+4] = b'\x7f\xff\xff\xff'
        name = f"seed_biglen_{offset}"
        path = _save_poc(bytes(d), task_id, name)
        results.append(PoCCandidate(name, bytes(d), path, 0.45,
            f"Seed with large length at offset {offset}"))

    # Mutation 2: Corrupt length fields to zero
    for i, offset in enumerate(length_offsets[:3]):
        d = bytearray(seed)
        d[offset:offset+4] = b'\x00\x00\x00\x00'
        name = f"seed_zerolen_{offset}"
        path = _save_poc(bytes(d), task_id, name)
        results.append(PoCCandidate(name, bytes(d), path, 0.4,
            f"Seed with zero length at offset {offset}"))

    # Mutation 3: Truncate at various points
    for frac in [0.75, 0.5, 0.25]:
        cutpoint = int(len(seed) * frac)
        if cutpoint > 10:
            d = seed[:cutpoint]
            name = f"seed_trunc_{int(frac*100)}"
            path = _save_poc(d, task_id, name)
            results.append(PoCCandidate(name, d, path, 0.35,
                f"Seed truncated at {int(frac*100)}%"))

    # Mutation 4: If description mentions a specific function/field,
    # search for related magic bytes and corrupt nearby
    vuln_keywords = re.findall(r"\b\w{4,}\b", desc_lower)
    for kw in vuln_keywords[:5]:
        kw_bytes = kw.encode("ascii", errors="ignore")
        idx = seed.find(kw_bytes)
        if idx >= 0 and idx + len(kw_bytes) + 4 < len(seed):
            d = bytearray(seed)
            # Corrupt 4 bytes after the keyword
            corrupt_at = idx + len(kw_bytes)
            d[corrupt_at:corrupt_at+4] = b'\xff\xff\xff\x7f'
            name = f"seed_keyword_{kw}"
            path = _save_poc(bytes(d), task_id, name)
            results.append(PoCCandidate(name, bytes(d), path, 0.5,
                f"Seed corrupted near '{kw}' at offset {corrupt_at}"))

    print(f"    Generated {len(results)} blind mutations")
    return results
