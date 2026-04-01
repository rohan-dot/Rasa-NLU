# CRS v2 — Cyber Reasoning System

## What This Does

Finds and triggers **real vulnerabilities** in CyberGym project code by:
1. Finding the project's fuzz harness (LLVMFuzzerTestOneInput)
2. Building the project with AddressSanitizer
3. Using an LLM to generate crafted input bytes
4. Feeding those bytes to the real fuzz target
5. Checking for ASAN crashes **in the project's actual code**

## Setup (one-time)

```bash
sudo apt-get install -y build-essential autoconf automake libtool zlib1g-dev
```

## Run

```bash
python -m crs.main --task-dir ./data/arvo/1065 --output-dir ./crs_results_1065
```

## Files

Put all `.py` files in your `crs/` directory. Delete old files:
- DELETE: `poc_strategies.py`, `build_executor.py`, `fuzzer.py`, `evaluator.py`
- KEEP unchanged: `__init__.py`, `data_loader.py`

## Manual Verification (task 1065)

```bash
cd ~/.crs_workdir/1065/repo/src-vul/file
autoreconf -fi
./configure CFLAGS="-fsanitize=address,undefined -g" LDFLAGS="-fsanitize=address,undefined"
make -j4
cd ..
g++ -fsanitize=address,undefined -g -fpermissive -include string.h \
    /tmp/driver.cpp magic_fuzzer.cc file/src/.libs/libmagic.a -lz -o fuzz_target
cp file/magic/magic.mgc .
echo -n "test" | ./fuzz_target
```

If that runs without errors, the pipeline will work.
