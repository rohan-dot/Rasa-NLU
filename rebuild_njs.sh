#!/bin/bash
cd secbench_tasks/njs.cve-2022-32414/repo
make clean
NJS_CFLAGS="-fsanitize=address -fno-omit-frame-pointer" ./configure
make -j4
cd ../../..
echo "Done. Now run:"
echo "ASAN_OPTIONS=detect_leaks=0 ./secbench_tasks/njs.cve-2022-32414/repo/build/njs poc_njs.js"
