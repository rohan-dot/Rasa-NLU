cd ~/libxml2-vuln
git checkout e41032ae~1


autoreconf -fi && ./configure CC=clang CFLAGS="-g -O1 -fsanitize=address -fsanitize=fuzzer-no-link" --without-python && make clean && make -j$(nproc)




cd ~/gemma-fuzzer
clang -g -O1 -fsanitize=address,fuzzer -I ~/libxml2-vuln/include -I ~/libxml2-vuln ~/xml_fuzzer.c ~/libxml2-vuln/.libs/libxml2.a -lz -llzma -lm -o build/xml_fuzzer_vuln



./run_standalone.sh ./build/xml_fuzzer_vuln --src-dir ~/libxml2-vuln --timeout 600 --vllm-model gemma-4-31b-it
xx

git clone https://gitlab.gnome.org/GNOME/libxml2.git ~/libxml2-vuln
cd ~/libxml2-vuln

git log --oneline --all | head -50


cd ~/libxml2-vuln
./autogen.sh
./configure CC=clang CFLAGS="-g -O1 -fsanitize=address -fsanitize=fuzzer-no-link"
make -j$(nproc)


cd ~/gemma-fuzzer
clang -g -O1 -fsanitize=address,fuzzer \
    -I~/libxml2-vuln/include \
    xml_fuzzer.c \
    ~/libxml2-vuln/.libs/libxml2.a -lz -llzma -lm \
    -o build/xml_fuzzer_vuln

./run_standalone.sh ./build/xml_fuzzer_vuln --src-dir ~/libxml2-vuln --timeout 600 --vllm-model gemma-4-31b-it




old commit

./run_standalone.sh ./build/xml_fuzzer --src-dir ~/libxml2-src --timeout 300 --vllm-model gemma-4-31b-it





apt source libxml2 2>/dev/null || git clone https://gitlab.gnome.org/GNOME/libxml2.git ~/libxml2-src


./run_standalone.sh ./build/xml_fuzzer --src-dir /usr/include/libxml2 --timeout 300 --vllm-model gpt-oss-120b






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
