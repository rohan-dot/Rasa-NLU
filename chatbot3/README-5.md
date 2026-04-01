# CRS v2 — Setup & Run Guide

## What This Is

An LLM-powered Cyber Reasoning System that finds and triggers **real
vulnerabilities** in CyberGym project code. It feeds crafted input bytes to
the project's actual fuzz harness and checks for AddressSanitizer crashes
in the project's real code — not in standalone test programs.

## Prerequisites (one-time, needs sudo)

```bash
sudo apt-get install -y build-essential autoconf automake libtool zlib1g-dev g++
```

These are standard C/C++ dev packages. They install headers and libs into
/usr/lib and /usr/include. They do NOT affect Python, conda, or any user
environments.

## Task 1065 (libmagic) — Manual Build Setup

The fuzz target needs to be built once before the CRS can use it.

### Step 1: Build libmagic with ASAN

```bash
cd ~/.crs_workdir/1065/repo/src-vul/file
autoreconf -fi
./configure CFLAGS="-fsanitize=address,undefined -g" LDFLAGS="-fsanitize=address,undefined"
make -j4
cd ..
```

This produces `file/src/.libs/libmagic.a` — the static library with ASAN.

### Step 2: Build the fuzz target

```bash
cat > /tmp/driver.cpp << 'DRIVER_EOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) __attribute__((weak));
int main(int argc, char **argv) {
    if (LLVMFuzzerInitialize) LLVMFuzzerInitialize(&argc, &argv);
    FILE *f = stdin;
    if (argc > 1) { f = fopen(argv[1], "rb"); if (!f) return 1; }
    size_t cap = 65536, len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    while (1) { size_t n = fread(buf+len,1,cap-len,f); len+=n; if(!n)break;
        if(len==cap){cap*=2;buf=(uint8_t*)realloc(buf,cap);} }
    if (f != stdin) fclose(f);
    LLVMFuzzerTestOneInput(buf, len);
    free(buf);
    return 0;
}
DRIVER_EOF
```

```bash
cd ~/.crs_workdir/1065/repo/src-vul
g++ -fsanitize=address,undefined -g -fpermissive \
    -I file/src -I file/src/../include -include string.h \
    /tmp/driver.cpp magic_fuzzer.cc file/src/.libs/libmagic.a \
    -lz -o fuzz_target
```

### Step 3: Copy the magic database

```bash
cp file/magic/magic.mgc .
```

### Step 4: Verify it works

```bash
echo -n "test input" | ./fuzz_target
```

Should run and exit cleanly. No crash = input didn't trigger the bug, but
the fuzz target is working correctly.

## Running the CRS Pipeline

Once the fuzz target is built, the CRS automates byte generation and testing.

### CRS code setup

Put all the .py files in the `crs/` directory:

```
crs/
├── __init__.py
├── config.py
├── data_loader.py
├── code_intelligence.py
├── llm_router.py
├── harness_finder.py
├── harness_synthesizer.py
├── byte_strategies.py
├── harness_runner.py
└── main.py
```

Delete these old files if they exist:
- `poc_strategies.py`
- `build_executor.py`
- `fuzzer.py`
- `evaluator.py`

### Run

```bash
python -m crs.main \
    --task-dir ./data/arvo/1065 \
    --output-dir ./crs_results_1065 \
    --model gemma-3-27b-it \
    --base-url http://g52lambda02.llan.ll.mit.edu:8000/v1 \
    --api-key EMPTY
```

### What the pipeline does

1. Loads task 1065 (libmagic — vuln: regexec/pmatch uninitialized memory)
2. Finds the fuzz harness (`magic_fuzzer.cc` with `LLVMFuzzerTestOneInput`)
3. Finds the already-built `libmagic.a` and compiles the fuzz target
4. Asks the LLM to analyze the vulnerability and generate crafted bytes
5. Feeds each byte payload to the real fuzz target
6. Reports any ASAN/MSAN/UBSAN triggers from libmagic's actual code
7. Saves results to `crs_results_1065/results.json` and `results.md`

## Configuration

Edit `crs/config.py` to change:

```python
LLM_MODEL = "gemma-3-27b-it"
LLM_BASE_URL = "http://g52lambda02.llan.ll.mit.edu:8000/v1"
MAX_TOKENS = 4096
```

## Running on Task 57002 (Mosquitto)

Requires an additional dependency:

```bash
sudo apt-get install -y libcjson-dev
```

## Architecture

```
Old (v1) — WRONG:
  LLM writes C program → compile standalone → ASAN fires in own code → meaningless

New (v2) — CORRECT:
  Find harness → build project with ASAN → LLM crafts input bytes →
  feed to real fuzz target → ASAN fires in project code → real trigger
```

## Troubleshooting

- **"No fuzz harness found"** — check if the project has a `fuzzing/` directory
- **"No static library found"** — run the manual build steps above first
- **"error loading magic file"** — copy magic.mgc: `cp file/magic/magic.mgc ~/.crs_workdir/1065/`
- **"No PoC triggered"** — try `--max-pocs 50` or improve byte strategies
