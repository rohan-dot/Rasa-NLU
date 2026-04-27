python fetch_task.py --id "njs.cve-2022-32414"
cd secbench_tasks/njs.cve-2022-32414/repo
git fetch --unshallow
git checkout f65981b0b8fcf02d69a40bc934803c25c9f607ab
cd ../../..

ASAN_OPTIONS=detect_leaks=0 python run_task.py \
    --task-dir ./secbench_tasks/njs.cve-2022-32414 \
    --run-mode file \
    --run-args "{poc}" \
    --base-url http://localhost:8000/v1 \
    --model gpt-oss-120b
