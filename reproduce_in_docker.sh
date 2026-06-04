#!/bin/bash
# reproduce_in_docker.sh — reproduce a crash inside a FRESH container built
# from the OSS-CRS image (the original run container is gone, but the harness
# binary still lives inside the image).
#
# Usage:
#   ./reproduce_in_docker.sh <image> <harness_name> <crash_file_on_host>
#
# Example:
#   ./reproduce_in_docker.sh oss-crs-builder:discver-asan-build-17805876269f \
#       acert \
#       ~/llcrs/oss-crs/.oss-crs-workdir/.../SUBMIT_DIR/xml/povs/86d963ce77ed1db33ade70f2c1f0fa87

set -u
IMAGE="${1:-}"
HARNESS="${2:-}"
CRASH="${3:-}"

if [ -z "$IMAGE" ] || [ -z "$HARNESS" ] || [ -z "$CRASH" ]; then
    echo "Usage: $0 <image> <harness_name> <crash_file_on_host>"
    echo "Images available:"; docker images | grep -iE 'openssl|discver|builder'
    exit 1
fi
if [ ! -f "$CRASH" ]; then
    echo "ERROR: crash file '$CRASH' not found on host."; exit 1
fi

CRASH_DIR="$(cd "$(dirname "$CRASH")" && pwd)"
CRASH_NAME="$(basename "$CRASH")"

echo "Image   : $IMAGE"
echo "Harness : $HARNESS"
echo "Crash   : $CRASH_NAME ($(wc -c < "$CRASH") bytes)"
echo "=================================================================="

docker run --rm \
  -v "$CRASH_DIR:/crash:ro" \
  -e ASAN_OPTIONS="abort_on_error=1:detect_leaks=0:symbolize=1:print_stacktrace=1" \
  -e UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
  "$IMAGE" \
  bash -c "
    H=\$(find /out /opt/target/build /src / -name '$HARNESS' -type f -executable 2>/dev/null | head -1)
    if [ -z \"\$H\" ]; then
        echo 'ERROR: harness $HARNESS not found in image. What IS in /out:'
        ls -la /out 2>/dev/null || echo '(no /out)'
        ls -la /opt/target/build 2>/dev/null || echo '(no /opt/target/build)'
        exit 2
    fi
    echo \"Found harness at: \$H\"
    echo '=================== RUN 1 ==================='
    \"\$H\" /crash/$CRASH_NAME; echo \"[exit \$?]\"
    echo '=================== RUN 2 (determinism) ==================='
    \"\$H\" /crash/$CRASH_NAME; echo \"[exit \$?]\"
  "
