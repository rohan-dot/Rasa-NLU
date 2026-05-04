#!/bin/bash
# run_standalone.sh — Run gemma-fuzzer WITHOUT Docker or OSS-CRS.
#
# Usage:
#   ./run_standalone.sh <fuzzer_binary> [options]
#
# Examples:
#   # Basic — just point at a LibFuzzer binary
#   ./run_standalone.sh ./xml_fuzzer
#
#   # With source dir for LLM context + custom timeout
#   ./run_standalone.sh ./xml_fuzzer --src-dir ~/oss-fuzz/projects/libxml2 --timeout 600
#
#   # Provide initial seed corpus
#   ./run_standalone.sh ./xml_fuzzer --seed-dir ./my_seeds
#
# Prerequisites:
#   - Python 3.10+  (with pip)
#   - vLLM running on localhost:8000 (optional — fuzzing works without it)
#   - A LibFuzzer binary (built with -fsanitize=address,fuzzer)

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────
FUZZER_BINARY=""
SRC_DIR=""
SEED_DIR=""
TIMEOUT="${FUZZ_TIMEOUT:-3600}"
JOBS="${FUZZ_JOBS:-1}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-gpt-oss-120b}"
LLM_SEED_INTERVAL="${LLM_SEED_INTERVAL:-120}"
OUTPUT_DIR=""

usage() {
    echo "Usage: $0 <fuzzer_binary|auto> [options]"
    echo ""
    echo "  Use 'auto' to have the LLM generate a harness automatically."
    echo "  Requires --src-dir to be set."
    echo ""
    echo "Options:"
    echo "  --src-dir DIR          Source code dir (REQUIRED for auto mode)"
    echo "  --seed-dir DIR         Initial seed corpus directory"
    echo "  --output-dir DIR       Output directory (default: ./output/<binary_name>)"
    echo "  --timeout SECS         Fuzzing duration (default: $TIMEOUT)"
    echo "  --jobs N               Parallel LibFuzzer workers (default: $JOBS)"
    echo "  --vllm-host HOST       vLLM host (default: $VLLM_HOST)"
    echo "  --vllm-port PORT       vLLM port (default: $VLLM_PORT)"
    echo "  --vllm-model NAME      vLLM model name (default: $VLLM_MODEL)"
    echo "  --llm-seed-interval S  Seconds between LLM seed rounds (default: $LLM_SEED_INTERVAL)"
    exit 1
}

# First positional arg is the fuzzer binary
if [ $# -lt 1 ]; then
    usage
fi
FUZZER_BINARY="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --src-dir)          SRC_DIR="$2"; shift 2 ;;
        --seed-dir)         SEED_DIR="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --timeout)          TIMEOUT="$2"; shift 2 ;;
        --jobs)             JOBS="$2"; shift 2 ;;
        --vllm-host)        VLLM_HOST="$2"; shift 2 ;;
        --vllm-port)        VLLM_PORT="$2"; shift 2 ;;
        --vllm-model)       VLLM_MODEL="$2"; shift 2 ;;
        --llm-seed-interval) LLM_SEED_INTERVAL="$2"; shift 2 ;;
        *)                  echo "Unknown option: $1"; usage ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────
if [ "$FUZZER_BINARY" = "auto" ] || [ ! -f "$FUZZER_BINARY" ]; then
    if [ "$FUZZER_BINARY" != "auto" ] && [ -n "$FUZZER_BINARY" ]; then
        echo "NOTE: '$FUZZER_BINARY' not found. Will auto-generate harness via LLM."
    else
        echo "NOTE: Auto-harness mode. LLM will generate and compile a harness."
    fi
    # Create a build dir for the auto-generated harness
    HARNESS_NAME="auto_harness"
    BUILD_DIR="$(pwd)/build"
    mkdir -p "$BUILD_DIR"
else
    FUZZER_BINARY="$(realpath "$FUZZER_BINARY")"
    HARNESS_NAME="$(basename "$FUZZER_BINARY")"
    BUILD_DIR="$(dirname "$FUZZER_BINARY")"
fi

# Default output dir
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./output/${HARNESS_NAME}"
fi

# Default src dir (same as build dir if not specified)
if [ -z "$SRC_DIR" ]; then
    if [ "$HARNESS_NAME" = "auto_harness" ]; then
        echo "ERROR: --src-dir is required when using auto mode."
        exit 1
    fi
    SRC_DIR="$BUILD_DIR"
fi

# ── Set up output directories ─────────────────────────────────────
mkdir -p "$OUTPUT_DIR"/{povs,seeds,bugs,corpus,crashes}
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================="
echo " gemma-fuzzer (standalone mode)"
echo " Binary:   $FUZZER_BINARY"
echo " Harness:  $HARNESS_NAME"
echo " Output:   $OUTPUT_DIR"
echo " Timeout:  ${TIMEOUT}s"
echo " vLLM:     $VLLM_HOST:$VLLM_PORT ($VLLM_MODEL)"
echo "============================================="

# ── Install Python deps if needed ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_CODE_DIR="$SCRIPT_DIR/src"

if ! python3 -c "import openai" 2>/dev/null; then
    echo "[setup] Installing Python dependencies..."
    pip install --quiet -r "$SRC_CODE_DIR/requirements.txt"
fi

# ── Copy seeds if provided ────────────────────────────────────────
if [ -n "$SEED_DIR" ] && [ -d "$SEED_DIR" ]; then
    echo "[setup] Copying seeds from $SEED_DIR..."
    cp -r "$SEED_DIR"/* "$OUTPUT_DIR/seeds/" 2>/dev/null || true
fi

# ── Run the orchestrator ──────────────────────────────────────────
exec python3 "$SRC_CODE_DIR/main.py" \
    --build-dir    "$BUILD_DIR" \
    --src-dir      "$SRC_DIR" \
    --output-dir   "$OUTPUT_DIR" \
    --seed-dir     "$OUTPUT_DIR/seeds" \
    --log-dir      "$LOG_DIR" \
    --harness      "$HARNESS_NAME" \
    --vllm-host    "$VLLM_HOST" \
    --vllm-port    "$VLLM_PORT" \
    --vllm-model   "$VLLM_MODEL" \
    --fuzz-timeout "$TIMEOUT" \
    --fuzz-jobs    "$JOBS" \
    --llm-seed-interval "$LLM_SEED_INTERVAL"
