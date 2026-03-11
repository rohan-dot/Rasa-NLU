# CRS — Cyber Reasoning System

A Python-based Cyber Reasoning System evaluated on the **CyberGym** benchmark
(Task 1 / Level 1). Given a pre-patch vulnerable codebase and a textual
vulnerability description, CRS automatically generates a Proof-of-Concept (PoC)
that triggers the vulnerability.

## Architecture

```
crs/
├── config.py            # Global settings, timeouts, model defaults
├── data_loader.py       # Load tasks from local dirs or HuggingFace
├── code_intelligence.py # AST analysis, file ranking, build detection
├── llm_router.py        # OpenAI-compatible LLM abstraction
├── poc_strategies.py    # Multi-strategy PoC generation (direct, analyze, pattern…)
├── build_executor.py    # Compile project + PoC, run under sanitizers
├── fuzzer.py            # Optional AFL++/libFuzzer fallback
├── evaluator.py         # Score results, produce JSON + Markdown reports
└── main.py              # CLI entry-point
```

## Prerequisites

**Operating system:** Linux (tested on Ubuntu 22.04 / 24.04).

**Python:** 3.10+ via Anaconda or system Python.

**C/C++ toolchain:**

```bash
# Option A — conda
conda install -c conda-forge gcc gxx cmake make

# Option B — system packages (Ubuntu/Debian)
sudo apt-get install build-essential cmake clang
```

**Address-Sanitizer support** is assumed via the system compiler (gcc ≥ 8 or
clang ≥ 10).

## Installation

```bash
# Clone the repo
git clone <repo-url> && cd crs

# Create a conda environment (recommended)
conda create -n crs python=3.10 -y
conda activate crs

# Install Python dependencies
pip install -r requirements.txt
```

## LLM Backend Setup

CRS uses an OpenAI-compatible API. You have two options:

### Option A — Local vLLM serving Gemma

```bash
# Install vLLM
pip install vllm

# Launch the server (adjust GPU memory / tensor-parallel as needed)
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-27b-it \
    --port 8000 \
    --dtype auto \
    --max-model-len 8192

# Verify
curl http://localhost:8000/v1/models
```

Then run CRS with:

```bash
python -m crs.main \
    --model google/gemma-3-27b-it \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    ...
```

### Option B — OpenAI API

```bash
export OPENAI_API_KEY="sk-..."

python -m crs.main \
    --model gpt-4o \
    --base-url https://api.openai.com/v1 \
    --api-key "$OPENAI_API_KEY" \
    ...
```

## Running on Sample Tasks

### Single local task

Each task directory should contain `repo-vul.tar.gz` (the vulnerable source)
and a metadata file with the vulnerability description.

```bash
python -m crs.main \
    --task-dir ./data/arvo/1065 \
    --output-dir ./crs_results
```

### Multiple tasks from HuggingFace

```bash
python -m crs.main \
    --task-ids arvo:1065 arvo:1461 oss-fuzz:12345 \
    --output-dir ./crs_results \
    --max-tasks 3
```

### Three CyberGym sample tasks (quick smoke test)

```bash
python -m crs.main \
    --task-ids arvo:1065 arvo:1461 arvo:1533 \
    --output-dir ./sample_results \
    --model google/gemma-3-27b-it \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --strategies direct analyze pattern
```

### Disable fuzzing

```bash
python -m crs.main \
    --task-dir ./data/arvo/1065 \
    --output-dir ./crs_results \
    --no-fuzzing
```

## Expected Output

After a run, the `--output-dir` directory will contain:

### `results.json`

```json
[
  {
    "task_id": "arvo:1065",
    "project_name": "libxml2",
    "vuln_type": "heap-buffer-overflow",
    "strategy_used": "analyze",
    "triggered": true,
    "crash_type": "ASAN heap-buffer-overflow",
    "confidence": 0.85,
    "build_success": true,
    "time_elapsed": 42.3,
    "poc_code": "#include <libxml/parser.h>\nint main() { ... }",
    "notes": "[0] TRIG strategy=analyze ..."
  }
]
```

### `results.md`

A Markdown table for quick inspection:

| task_id    | project  | vuln_type             | triggered | strategy | crash_type                  | time (s) |
|------------|----------|-----------------------|-----------|----------|-----------------------------|----------|
| arvo:1065  | libxml2  | heap-buffer-overflow  | ✅        | analyze  | ASAN heap-buffer-overflow   | 42.3     |
| arvo:1461  | libtiff  | out-of-bounds-read    | ❌        | direct   |                             | 38.7     |

### Console output

The CLI prints a coloured summary table at the end of every run:

```
  ──────────────────────────────────────────────────────────────────
  CRS EVALUATION SUMMARY
  ──────────────────────────────────────────────────────────────────
  task_id              project              vuln_type              status  ...
  arvo:1065            libxml2              heap-buffer-overflow   TRIGGERED
  arvo:1461            libtiff              out-of-bounds-read     MISS
  ──────────────────────────────────────────────────────────────────
  Triggered: 1/2 (50.0%)   Build failures: 0   Total time: 81.0s
```

## CLI Reference

```
python -m crs.main [OPTIONS]

Task selection (mutually exclusive):
  --task-dir DIR          Single local task directory
  --task-ids ID [ID ...]  HuggingFace task identifiers

Output:
  --output-dir DIR        Where to write results (default: ./crs_results)

LLM backend:
  --model MODEL           Model name (default from config.py)
  --base-url URL          API base URL
  --api-key KEY           API key ('EMPTY' for local vLLM)

Behaviour:
  --no-fuzzing            Skip fuzzer fallback
  --strategies S [S ...]  Restrict to named strategies (direct, analyze, pattern, …)
  --max-tasks N           Cap number of tasks to process
  --timeout SECS          Per-PoC execution timeout
  -v, --verbose           Debug logging
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0    | At least one task triggered successfully |
| 1    | No tasks triggered (or no tasks run)     |

## License

See LICENSE file in the repository root.
