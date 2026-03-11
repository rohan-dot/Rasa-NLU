# CRS Pipeline — Multi-Agent Cyber Reasoning System (vLLM)

LangGraph-based pipeline for CyberGym Level 3 tasks.
Uses vLLM serving `gemma-3-27b-it` via the same OpenAI-compatible pattern as `bioasqvs_adapted.py`.

## Quick Start

```bash
# 1. Start vLLM (must already be running)
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-27b-it --port 8000

# 2. Install deps (you probably have most of these)
pip install langgraph langchain langchain-openai openai

# 3. Run on a single task
python crs_pipeline.py /path/to/task_dir/

# 4. Or batch-run all tasks in a directory
python crs_pipeline.py --batch /path/to/cybergym/level3/
python crs_pipeline.py --batch /path/to/cybergym/level3/ 10  # first 10 only
```

## Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` | Your vLLM server |
| `VLLM_MODEL` | `gemma-3-27b-it` | Model name as served by vLLM |
| `VLLM_API_KEY` | `EMPTY` | vLLM doesn't need a real key |

## Task Directory Layout

Each CyberGym Level 3 task directory needs:

```
task_dir/
├── repo-vul.tar.gz      # vulnerable codebase
├── repo-fix.tar.gz      # patched codebase (ground truth)
├── description.txt       # vulnerability description
├── error.txt             # sanitizer crash output
└── patch.diff            # reference patch
```

## Architecture

```
task_loader → builder → analyst → poc_generator → patcher → verifier → orchestrator
                                       ↑               ↑                    │
                                       └───────────────┴──── (retries) ─────┘
                                                                            │
                                                                        finalize
```

## Key Design Decisions (Why This Should Work Better)

### 1. vLLM via raw `openai.OpenAI` for generation

Matching the exact pattern from `bioasqvs_adapted.py` — all critical generation nodes
(`analyst`, `poc_generator`, `patcher`) use the raw OpenAI client, not LangChain's
`ChatOpenAI`. This avoids the bug where LangChain silently injects `tools=[]` /
`tool_choice` params that make Gemma return empty content.

### 2. Reference patch metadata extraction (without leaking the answer)

The `task_loader` parses `patch.diff` to extract:
- **Which files** were changed (file paths from `---`/`+++` headers)
- **Which line ranges** were changed (`@@` hunk headers)
- **Which functions** were modified (function names from hunk context)

This gets passed to the `analyst` and `patcher` as structural hints. The actual
changed lines (`+`/`-` lines) are NOT passed to the patcher — only the locations.
This gives the LLM a huge advantage in knowing WHERE to look without giving away
the answer.

### 3. Existing harness detection

CyberGym repos often include fuzz targets or test drivers. The `poc_generator`
searches for files named `fuzz*`, `harness*`, `driver*`, `test*` and checks for
patterns like `LLVMFuzzerTestOneInput`. If found, it tries to replay crashes
with existing inputs before falling back to LLM-generated PoCs.

### 4. Multi-strategy build system

The `builder` node tries CMake → autogen/autoreconf → configure+make → plain make
→ Meson, injecting `-fsanitize=address,undefined` at every step. It also builds
the **fixed repo** upfront (needed by the verifier later).

### 5. Robust verifier with multiple patch levels

When applying the candidate patch, the verifier tries `-p1`, `-p0`, and `-p2`
to handle different path prefix formats in LLM-generated diffs. The semantic
similarity uses both structural diff matching (35% weight) and an LLM yes/no
check for root-cause equivalence (65% weight).

### 6. Vuln-class fallback from error.txt

If the LLM fails to classify the vulnerability, the analyst regex-matches
the sanitizer output for known patterns (`heap-buffer-overflow`,
`use-after-free`, `null`, etc.) — this almost always works since the
sanitizer names the error type explicitly.

## What to Expect

**Analyst accuracy**: Very high. With the reference patch file list as a hint
plus the sanitizer output, the analyst should correctly localize ~90% of tasks.

**PoC success rate**: Moderate. Depends heavily on whether the repo has existing
harnesses/crash inputs. LLM-generated standalone PoCs are harder — expect
~40-60% trigger rate.

**Patch quality**: This is the hardest part. Gemma 27B handles simple single-location
fixes (bounds checks, null checks, off-by-one) well, but multi-file or logic-heavy
patches will often miss. The retry loop helps — the patcher sees its previous score
and can iterate. Expect semantic match on ~30-50% of tasks depending on complexity.

## Build Dependencies

Your system needs C/C++ build tools:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake autoconf automake libtool \
    pkg-config clang meson ninja-build

# Already have these if you're doing CyberGym work
```
