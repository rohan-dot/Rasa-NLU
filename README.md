cd ~/.crs_workdir/1065/repo/src-vul/file
autoreconf -fi
./configure --enable-static --disable-shared
make -j4
