The `.a` files exist. The pipeline just isn't finding them all because they're in `.libs/` directories. Forget debugging the pipeline for now — build the fuzz target manually right now:

```bash
cd ~/.crs_workdir/10400/repo/src-vul
g++ -fsanitize=address,undefined -g -O1 -fpermissive \
    -include string.h \
    /tmp/driver.cpp \
    graphicsmagick/fuzzing/coder_fuzzer.cc \
    graphicsmagick/magick/.libs/libGraphicsMagick.a \
    graphicsmagick/magick/.libs/libGraphicsMagick++.a \
    build/libpng.a \
    -I graphicsmagick \
    -I graphicsmagick/Magick++/lib \
    -I build \
    -lz -lm -lpthread -lgomp -fopenmp \
    -o ~/.crs_workdir/10400/fuzz_target
```

Then test:

```bash
echo -n "AAAA" | ~/.crs_workdir/10400/fuzz_target
```

If that compiles, you have a working fuzz target and the CRS byte generation will work against it.
