Now I see the exact paths. Run this:

```bash
cd ~/.crs_workdir/10400/repo/src-vul
g++ -fsanitize=address,undefined -g -O1 -fpermissive -include string.h \
    /tmp/driver.cpp \
    graphicsmagick/fuzzing/coder_fuzzer.cc \
    graphicsmagick/magick/.libs/libGraphicsMagick.a \
    graphicsmagick/Magick++/.libs/libGraphicsMagick++.a \
    graphicsmagick/wand/.libs/libGraphicsMagickWand.a \
    ~/.crs_workdir/10400/build/libpng.a \
    -I graphicsmagick \
    -I graphicsmagick/Magick++/lib \
    -I ~/.crs_workdir/10400/build \
    -lz -lm -lpthread -lgomp -fopenmp -lbz2 \
    -o ~/.crs_workdir/10400/fuzz_target
```

Then:

```bash
echo -n "AAAA" | ~/.crs_workdir/10400/fuzz_target
```
