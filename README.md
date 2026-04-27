cd secbench_tasks/njs.cve-2022-32414/repo
./configure
CFLAGS="-fsanitize=address -g" make -j4
cd ../../..


ASAN_OPTIONS=detect_leaks=0 python run_task.py \
    --task-dir ./secbench_tasks/njs.cve-2022-32414 \
    --binary ./secbench_tasks/njs.cve-2022-32414/repo/build/njs \
    --skip-build \
    --run-mode file \
    --run-args "{poc}" \
    --base-url http://localhost:8000/v1 \
    --model gemma-4-31b-it
