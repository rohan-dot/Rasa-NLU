python fetch_task.py --list --project mruby
python fetch_task.py --id "PICK_ONE_WITH_HEAP_OVERFLOW"
python run_task.py --task-dir ./secbench_tasks/THAT_ID \
    --run-mode file --run-args "{poc}" \
    --base-url http://g52lambda02.llan.ll.mit.edu:8000/v1 \
    --model gemma-3-27b-it
