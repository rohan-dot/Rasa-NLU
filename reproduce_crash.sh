#!/bin/bash
# reproduce_crash.sh — replay a fuzzer crash input and triage the sanitizer output.
#
# Usage:
#   ./reproduce_crash.sh <harness_binary> <crash_input_file> [runs]
#
# Example (OpenSSL cert harness):
#   ./reproduce_crash.sh /opt/target/build/x509 ./crash-abc123 5
#
# It runs the harness on the crash input several times, shows the full
# sanitizer report, and tells you:
#   - the bug type (heap-buffer-overflow, use-after-free, SEGV, ...)
#   - whether the crash is DETERMINISTIC (crashes every run)
#   - the top stack frame, and whether it's in TARGET code or just the
#     harness / sanitizer runtime (which would mean it's not a real bug).

set -u

HARNESS="${1:-}"
CRASH="${2:-}"
RUNS="${3:-5}"

if [ -z "$HARNESS" ] || [ -z "$CRASH" ]; then
    echo "Usage: $0 <harness_binary> <crash_input_file> [runs]"
    exit 1
fi
if [ ! -x "$HARNESS" ]; then
    echo "ERROR: harness '$HARNESS' not found or not executable."
    exit 1
fi
if [ ! -f "$CRASH" ]; then
    echo "ERROR: crash file '$CRASH' not found."
    exit 1
fi

# Symbolized, deterministic sanitizer output
export ASAN_OPTIONS="abort_on_error=1:detect_leaks=0:symbolize=1:print_stacktrace=1"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
if command -v llvm-symbolizer >/dev/null 2>&1; then
    export ASAN_SYMBOLIZER_PATH="$(command -v llvm-symbolizer)"
fi

echo "=================================================================="
echo " Harness : $HARNESS"
echo " Crash   : $CRASH ($(wc -c < "$CRASH") bytes)"
echo " Runs    : $RUNS"
echo "=================================================================="

# ── Determinism check ────────────────────────────────────────────
crash_count=0
LOG="$(mktemp)"
for i in $(seq 1 "$RUNS"); do
    "$HARNESS" "$CRASH" > "$LOG.run" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        crash_count=$((crash_count + 1))
        # keep the first crashing log as the canonical one
        [ -s "$LOG" ] || cp "$LOG.run" "$LOG"
    fi
done

echo ""
echo "Reproducibility: $crash_count / $RUNS runs crashed."
if [ "$crash_count" -eq 0 ]; then
    echo ">> Did NOT crash. Either not a real crash, or the input/harness"
    echo "   doesn't match. Confirm this is the right harness for this input."
    rm -f "$LOG" "$LOG.run"
    exit 0
elif [ "$crash_count" -lt "$RUNS" ]; then
    echo ">> FLAKY — crashes only sometimes. Likely timing/OOM/environment,"
    echo "   not a clean memory-safety bug. Treat with suspicion."
else
    echo ">> DETERMINISTIC — crashes every run. Good sign it's a real defect."
fi

# ── Full sanitizer report ────────────────────────────────────────
echo ""
echo "================== FULL SANITIZER OUTPUT ========================="
cat "$LOG"
echo "=================================================================="

# ── Bug type ─────────────────────────────────────────────────────
echo ""
BUG_TYPE="$(grep -oE 'ERROR: (AddressSanitizer|libFuzzer|UndefinedBehaviorSanitizer): [a-z0-9-]+' "$LOG" | head -1)"
[ -z "$BUG_TYPE" ] && BUG_TYPE="$(grep -oE 'SUMMARY: [A-Za-z]+: [a-zA-Z0-9_-]+' "$LOG" | head -1)"
echo "Bug type   : ${BUG_TYPE:-unknown (see output above)}"

# ── Top meaningful stack frame ───────────────────────────────────
# Skip sanitizer/libfuzzer/libc internal frames; find the first frame
# in real code. If that frame is in the harness file, it's likely a
# harness bug, not a target bug.
TOP_FRAME="$(grep -E '^\s*#[0-9]+ 0x' "$LOG" \
    | grep -viE 'asan|__interceptor|sanitizer|libfuzzer|llvmfuzzer|fuzzer::|__libc|_start|LLVMFuzzerTestOneInput' \
    | head -1)"
echo "Crash site : ${TOP_FRAME:-<only runtime frames found>}"

# Where LLVMFuzzerTestOneInput is (the harness entry)
HARNESS_SRC="$(grep -E '^\s*#[0-9]+ 0x' "$LOG" | grep 'LLVMFuzzerTestOneInput' | head -1)"

echo ""
echo "================== TRIAGE VERDICT ================================"
if echo "$TOP_FRAME" | grep -qiE 'fuzz|harness|target/.*fuzz'; then
    echo ">> Crash site looks like it's in the HARNESS, not the library."
    echo "   This is probably a harness bug, NOT a real target vulnerability."
elif [ -n "$TOP_FRAME" ]; then
    echo ">> Crash site is in library/target code. Worth taking seriously."
    echo "   Next: confirm this code path is reachable through the PUBLIC API,"
    echo "   then check OpenSSL's issue tracker / OSS-Fuzz for this signature."
else
    echo ">> Only sanitizer/runtime frames found — inconclusive."
    echo "   May be a stack overflow (recursion) or a corrupted stack."
fi
echo "=================================================================="

rm -f "$LOG" "$LOG.run"
