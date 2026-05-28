"""
seed_generator.py — LLM-generated seed corpus and fuzzing dictionary.

LibFuzzer from random bytes is slow to reach deep code. A good seed
corpus (valid format examples) and dictionary (interesting tokens)
dramatically accelerate coverage. This is standard practice and one
of the cheapest wins available.

The LLM, having read the target's source, knows what valid input looks
like — valid JSON, valid XML, valid TIFF headers, etc. We ask it to
generate diverse valid examples plus malformed edge cases.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from llm_client import VLLMClient

logger = logging.getLogger("gemma-fuzzer.seeds")


SEED_PROMPT = """\
You are preparing a fuzzing seed corpus for this library.

Based on the source code, generate diverse example inputs that this
library would parse/process. Include:

1. Minimal valid inputs (smallest thing that parses successfully)
2. Typical valid inputs (realistic examples)
3. Complex valid inputs (nested structures, all features used)
4. Edge cases (empty, deeply nested, very long fields, unusual but legal)

These seeds let the fuzzer start from interesting inputs instead of
random bytes, reaching deep code paths immediately.

Output a JSON array where each element is one seed input as a string.
For binary formats, use a description the harness can interpret, but
prefer real examples. Generate 8-12 diverse seeds.

Start with [ end with ]. Example:
["{}", "{\\"a\\":1}", "{\\"nested\\":{\\"deep\\":[1,2,3]}}", "[]"]"""

DICT_PROMPT = """\
You are building a fuzzing dictionary for this library.

A dictionary is a list of interesting tokens, keywords, magic bytes,
and structural elements that appear in valid inputs. The fuzzer uses
these to construct meaningful mutations instead of random bytes.

Based on the source code, list the important tokens:
- Format keywords (e.g. "true", "null", "<?xml")
- Structural delimiters (e.g. "{", "[", "</")
- Magic bytes / headers (e.g. "\\x89PNG", "II*\\x00")
- Field names the parser looks for
- Special values that trigger different code paths

Output a JSON array of strings. Start with [ end with ].
Example: ["{", "}", "[", "]", "true", "false", "null", "\\""]"""


def generate_seeds(llm: VLLMClient, src_dir: str, seed_dir: str,
                   call_graph=None) -> int:
    """Generate a seed corpus by reading the source. Returns count."""
    if not llm.is_available():
        return 0

    seed_path = Path(seed_dir)
    seed_path.mkdir(parents=True, exist_ok=True)

    # Gather source context (headers + main parser)
    context = _gather_context(src_dir)
    if not context:
        return 0

    logger.info("[seeds] Asking LLM to generate seed corpus...")

    response = llm.chat(
        system="You are a fuzzing expert. Output ONLY a JSON array of strings.",
        user=SEED_PROMPT + f"\n\nSource code:\n{context[:8000]}",
        max_tokens=2000, temperature=0.4,
    )

    if not response:
        return 0

    seeds = _parse_json_array(response)
    count = 0
    for i, seed in enumerate(seeds):
        if not isinstance(seed, str):
            continue
        try:
            # Decode escape sequences for binary-ish seeds
            seed_bytes = seed.encode("utf-8", errors="replace")
            (seed_path / f"llm_seed_{i:03d}").write_bytes(seed_bytes)
            count += 1
        except Exception:
            pass

    logger.info("[seeds] Wrote %d seed inputs to %s", count, seed_dir)
    return count


def generate_dictionary(llm: VLLMClient, src_dir: str,
                        output_dir: str) -> str | None:
    """Generate a LibFuzzer dictionary. Returns path to .dict file."""
    if not llm.is_available():
        return None

    context = _gather_context(src_dir)
    if not context:
        return None

    logger.info("[seeds] Asking LLM to generate fuzzing dictionary...")

    response = llm.chat(
        system="You are a fuzzing expert. Output ONLY a JSON array of strings.",
        user=DICT_PROMPT + f"\n\nSource code:\n{context[:8000]}",
        max_tokens=1500, temperature=0.3,
    )

    if not response:
        return None

    tokens = _parse_json_array(response)
    if not tokens:
        return None

    # Write LibFuzzer dictionary format: one token per line, quoted
    dict_path = Path(output_dir) / "fuzz.dict"
    lines = []
    for i, tok in enumerate(tokens):
        if not isinstance(tok, str) or not tok:
            continue
        # Escape for AFL/LibFuzzer dict format
        escaped = tok.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'kw{i}="{escaped}"')

    if not lines:
        return None

    dict_path.write_text("\n".join(lines))
    logger.info("[seeds] Wrote %d dictionary tokens to %s", len(lines), dict_path)
    return str(dict_path)


def _gather_context(src_dir: str) -> str:
    """Read headers and main source for format understanding."""
    src_path = Path(src_dir)
    context = ""

    for h in sorted(src_path.rglob("*.h"))[:4]:
        if any(s in str(h).lower() for s in [".git", "test", "aflplusplus", "honggfuzz"]):
            continue
        try:
            context += f"// {h.name}\n{h.read_text(errors='replace')[:2500]}\n\n"
        except Exception:
            pass

    for c in sorted(src_path.rglob("*.c"))[:2]:
        if any(s in str(c).lower() for s in [".git", "test", "fuzz", "aflplusplus"]):
            continue
        try:
            context += f"// {c.name}\n{c.read_text(errors='replace')[:2500]}\n\n"
        except Exception:
            pass

    return context


def _parse_json_array(response: str) -> list:
    import json
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    s, e = clean.find("["), clean.rfind("]")
    if s != -1 and e != -1:
        clean = clean[s:e + 1]
    try:
        result = json.loads(clean)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []
