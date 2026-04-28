cat > /tmp/xml_fuzzer.c << 'EOF'
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <stdint.h>
#include <stddef.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    xmlDocPtr doc = xmlReadMemory((const char *)data, size,
                                  "noname.xml", NULL, 0);
    if (doc != NULL) {
        xmlFreeDoc(doc);
    }
    return 0;
}
EOF

mkdir -p build

clang -g -O1 -fsanitize=address,fuzzer \
    -I/usr/include/libxml2 \
    /tmp/xml_fuzzer.c \
    -lxml2 -lz -llzma \
    -o build/xml_fuzzer

pip install openai

./run_standalone.sh ./build/xml_fuzzer \
    --src-dir /usr/include/libxml2 \
    --timeout 300 \
    --vllm-model gpt-oss-120b




git clone https://github.com/google/oss-fuzz.git ~/oss-fuzz


mkdir -p build

clang -g -O1 -fsanitize=address,fuzzer \
    -I/usr/include/libxml2 \
    ~/oss-fuzz/projects/libxml2/xml_read_memory_fuzzer.c \
    -lxml2 -lz -llzma \
    -o build/xml_fuzzer



xxx
mkdir -p build
clang -g -O1 -fsanitize=address,fuzzer \
    -I/usr/include/libxml2 \
    ~/oss-fuzz/projects/libxml2/xml_read_memory_fuzzer.c \
    -lxml2 -lz -llzma \
    -o build/xml_fuzzer


pip install openai

./run_standalone.sh ./build/xml_fuzzer \
    --src-dir ~/oss-fuzz/projects/libxml2 \
    --timeout 300 \
    --vllm-model gpt-oss-120b
    




conda install -c conda-forge clang clangxx compiler-rt libxml2


which clang
which gcc
clang --version
module avail 2>&1 | grep -i clang
module avail 2>&1 | grep -i llvm
conda list | grep clang
pkg-config --libs libxml-2.0
find /usr -name "libxml2" -type d 2>/dev/null






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
