cd ~/oss-crs

uv run oss-crs prepare \
    --compose-file /path/to/gemma-fuzzer/compose.yaml

uv run oss-crs build-target \
    --compose-file /path/to/gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/cjson

uv run oss-crs run \
    --compose-file /path/to/gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/cjson \
    --target-harness cjson_read_fuzzer \
    --timeout 1200

uv run oss-crs artifacts \
    --compose-file /path/to/gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/cjson \
    --target-harness cjson_read_fuzzer

    



git clone https://github.com/DaveGamble/cJSON.git cJSON-vuln

cd cJSON-vuln && git checkout d6d5449~1 && cd ..
chmod +x run_standalone.sh && ./run_standalone.sh auto --src-dir /panfs/g52-panfs/exp/FY26/aim/ro31337/AIxCC/cJSON-vuln --output-dir /panfs/g52-panfs/exp/FY26/aim/ro31337/AIxCC/results-cjson --timeout 600 --vllm-model gemma-4-31b-it
