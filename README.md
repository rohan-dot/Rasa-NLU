Same shallow clone issue. Fix it:

```bash
cd secbench_tasks/mruby.cve-2022-1286/repo
git fetch --unshallow
git checkout c30e6ebe1200
cd ../../..
```

Then run:

```bash
ASAN_OPTIONS=detect_leaks=0 python run_task.py \
    --task-dir ./secbench_tasks/mruby.cve-2022-1286 \
    --run-mode file \
    --run-args "{poc}" \
    --base-url http://g52lambda02.llan.ll.mit.edu:8000/v1 \
    --model gemma-3-27b-it
```

Also — to stop this from happening on every fetch, update `fetch_task.py` line that does the clone. Change `--depth 50` to just a full clone:

```bash
# In fetch_task.py, find this line:
["git", "clone", "--depth", "50", repo_url, str(repo_dir)]

# Change to:
["git", "clone", repo_url, str(repo_dir)]
```

Slower but won't break on checkout.





xxx

python fetch_task.py --list --project mruby
python fetch_task.py --id "PICK_ONE_WITH_HEAP_OVERFLOW"
python run_task.py --task-dir ./secbench_tasks/THAT_ID \
    --run-mode file --run-args "{poc}" \
    --base-url http://g52lambda02.llan.ll.mit.edu:8000/v1 \
    --model gemma-3-27b-it
