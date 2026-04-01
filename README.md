cd ~/.crs_workdir/10400/repo/src-vul/graphicsmagick
./configure CFLAGS="-fsanitize=address,undefined -g" CXXFLAGS="-fsanitize=address,undefined -g" LDFLAGS="-fsanitize=address,undefined" --enable-static --disable-shared
make -j4
