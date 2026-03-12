python -m crs.main \
    --task-dir ./data/arvo/1065 \
    --output-dir ./crs_results \
    --model Qwen2.5-32B-Instruct \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --no-fuzzing
