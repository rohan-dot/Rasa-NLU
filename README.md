mkdir -p build && clang -g -O1 -fsanitize=address,fuzzer -I/usr/include/libxml2 xml_fuzzer.c -lxml2 -lz -llzma -o build/xml_fuzzer


# 1. Extract
tar -xzf gemma-fuzzer.tar.gz && cd gemma-fuzzer

# 2. Build a target
sudo apt install clang libxml2-dev
./build_target.sh ~/oss-fuzz/projects/libxml2

# 3. Run (vLLM should already be running)
./run_standalone.sh ./build/libxml2/xml_read_memory_fuzzer \
    --src-dir ~/oss-fuzz/projects/libxml2 \
    --timeout 600

# 4. Results
ls output/xml_read_memory_fuzzer/povs/   # crashes
ls output/xml_read_memory_fuzzer/bugs/   # LLM reports
