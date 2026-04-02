Run these:

```bash
# 1. Get the full ASAN crash report with source lines
./fuzz_target poc_direct_script.bin 2>&1 | head -50

# 2. Hex dump to see what the PoC bytes look like
xxd poc_direct_script.bin

# 3. Find the vulnerable function in the source
grep -n "mng_LOOP\|ReadMNGImage" ~/.crs_workdir/10400/repo/src-vul/graphicsmagick/coders/png.c | head -20

# 4. See the actual validation bug (the chunk size check)
grep -n -B5 -A10 "mng_LOOP" ~/.crs_workdir/10400/repo/src-vul/graphicsmagick/coders/png.c
```

**What to look for in the ASAN output:**

The crash report will show something like:
```
ERROR: AddressSanitizer: heap-buffer-overflow
READ of size N at 0x...
    #0 0x... in ReadMNGImage .../coders/png.c:LINE_NUMBER
    #1 ...
```

That `LINE_NUMBER` is exactly where it crashes. The vulnerability description says the `mng_LOOP` chunk isn't validated to be at least 5 bytes — so the code reads fields from the chunk without checking its length first.

**To validate it's the real bug** (not a false positive):

```bash
# The PoC should crash on the vulnerable version
./fuzz_target poc_direct_script.bin

# If you have the patched repo (repo-fix.tar.gz), build it
# the same way and verify the PoC does NOT crash on the fix
```

If the ASAN report points to `ReadMNGImage` in `png.c` reading past a `mng_LOOP` chunk, that confirms you've triggered the exact vulnerability from the description.
