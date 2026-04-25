# 1. Unpack
tar xzf crs_secbench_nodock.tar.gz
cd crs_nodock

# 2. Install deps
pip install datasets openai

# 3. See available ASAN tasks
python fetch_task.py --list --project faad2

# 4. Pick one and fetch it (clones the repo at the vulnerable commit)
python fetch_task.py --id "WHATEVER_ID_YOU_SEE_FROM_STEP_3"

# 5. Read the harness to know how the binary runs
cat secbench_tasks/WHATEVER_ID/harness_commands.txt

# 6. Run it
python run_task.py \
    --task-dir ./secbench_tasks/WHATEVER_ID \
    --run-mode file \
    --run-args "{poc}" \
    --base-url http://g52lambda02.llan.ll.mit.edu:8000/v1 \
    --model gemma-3-27b-it


pt2
# Build manually
cd secbench_tasks/WHATEVER_ID/repo
# ... follow harness_commands.txt but add ASAN flags ...
cd ../../..

# Point at your binary directly
python run_task.py \
    --task-dir ./secbench_tasks/WHATEVER_ID \
    --binary ./secbench_tasks/WHATEVER_ID/repo/path/to/binary \
    --skip-build \
    --run-mode file \
    --run-args "{poc}" \
    --base-url http://g52lambda02.llan.ll.mit.edu:8000/v1
    
