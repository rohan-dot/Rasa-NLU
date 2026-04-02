Now I see it. `FUZZ_GRAPHICSMAGICK_CODER` is a **compile-time** macro, not a runtime env var. Look at the code:

```c
#define FUZZ_CODER FUZZ_CODER_STRING_LITERAL_X(FUZZ_GRAPHICSMAGICK_CODER)
```

It gets baked into the binary at compile time via `-D`. Your fuzz_target was compiled without it, so it literally uses the string "FUZZ_GRAPHICSMAGICK_CODER" as the codec name.

Quick manual fix — recompile with the define:

```bash
cd ~/.crs_workdir/10400

g++ -fsanitize=address,undefined -g -O1 -fpermissive \
    -DFUZZ_GRAPHICSMAGICK_CODER=MNG \
    -include string.h \
    standalone_driver.cpp \
    ~/.crs_workdir/10400/repo/src-vul/graphicsmagick/fuzzing/coder_fuzzer.cc \
    -I ~/.crs_workdir/10400/repo/src-vul/graphicsmagick \
    $(find ~/.crs_workdir/10400/repo/src-vul -name "*.a" -size +10k | sort -r | tr '\n' ' ') \
    -lz -lm -lpthread -ldl -lbz2 -lstdc++ \
    -o fuzz_target_mng

# Now test
./fuzz_target_mng poc_mng_loop_1byte.bin 2>&1 | head -40
```

This should finally show `AddressSanitizer: heap-buffer-overflow` at the `mng_LOOP` chunk read.
