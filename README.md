Let me test the parsing logic against a realistic ASAN report so I know the greps actually work:

Parsing works correctly — it skips the `asan_memcpy` interceptor and the harness frame, landing on the real crash site (`asn1_check_tlen` in OpenSSL). Let me make it executable and package it:

Here's `reproduce_crash.sh`. Usage:

```bash
./reproduce_crash.sh <harness_binary> <crash_file> [runs]
```

For your coworker's acert (x509/cert) harness:

```bash
./reproduce_crash.sh /opt/target/build/x509 ./crash-abc123 5
```

(Swap in the real harness binary name and crash file path. The acert harness is likely `x509` or whatever the cert fuzzer is named — use the actual binary he fuzzed with.)

**What it tells you:**

1. **Reproducibility** — runs it 5× and reports if it crashes every time (deterministic = good sign it's real), only sometimes (flaky = suspect, probably timing/OOM), or never (wrong harness/input, or not a real crash).

2. **Full sanitizer report** — the complete ASAN/UBSan output so you see exactly what happened.

3. **Bug type** — heap-buffer-overflow, use-after-free, SEGV, etc.

4. **Crash site** — and this is the key part: it skips the sanitizer/harness frames and finds the first frame in *real* code, then tells you whether that's in **library/target code** (worth taking seriously) or in the **harness itself** (probably a harness bug, not an OpenSSL bug). I tested this against a realistic OpenSSL ASAN trace and it correctly identified the crash in `asn1_check_tlen` rather than the interceptor or harness frame.

**The decision tree after you run it:**
- Crashes every time + frame in OpenSSL code + reachable via public API → real, check OpenSSL issue tracker / OSS-Fuzz history for the signature before claiming it's new.
- Crashes in the harness frame → harness bug, not OpenSSL.
- Flaky or only runtime frames → likely OOM/recursion/stack overflow; treat with suspicion.

Run it and paste me the sanitizer output if you want a second read on whether it's real or a false positive.
