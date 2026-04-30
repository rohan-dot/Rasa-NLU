cd ~ && tar -xzf gemma-fuzzer-v4-final.tar.gz && cd gemma-fuzzer

pip install tree-sitter tree-sitter-c openai


mkdir -p build && LIB=$(find ~/libxml2-vuln -name "libxml2.a" 2>/dev/null | head -1) && clang -g -O1 -fsanitize=address,fuzzer -I ~/libxml2-vuln/include -I ~/libxml2-vuln xml_fuzzer.c "$LIB" -lz -llzma -lm -o build/xml_fuzzer_vuln && echo "BUILD OK"


./run_standalone.sh ./build/xml_fuzzer_vuln --src-dir ~/libxml2-vuln --timeout 1200 --vllm-model gemma-4-31b-it
