uv run oss-crs build-target --compose-file example/discver/compose.yaml \
    --fuzz-proj-path ../oss-fuzz-aixcc/projects/libxml2

uv run oss-crs run --compose-file example/discver/compose.yaml \
    --fuzz-proj-path ../oss-fuzz-aixcc/projects/libxml2 \
    --target-harness xml --timeout 900
