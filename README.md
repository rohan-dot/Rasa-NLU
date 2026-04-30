uv run oss-crs prepare --compose-file ../gemma-fuzzer/compose.yaml

uv run oss-crs build-target \
    --compose-file ../gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/cjson
