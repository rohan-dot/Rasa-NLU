cd ~/.crs_workdir/1065/repo/src-vul
g++ -fsanitize=address,undefined -g -I file/src -I file/src/../include magic_fuzzer.cc file/src/.libs/libmagic.a -lz -o fuzz_target
