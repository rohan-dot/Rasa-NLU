Good choice. Let's pick a task with a straightforward heap overflow that ASAN will catch directly.

First download a new task:

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
from pathlib import Path

# arvo:2903 — libpng heap buffer overflow, simple C, well-known bug class
out_dir = Path("data/arvo/2903")
out_dir.mkdir(parents=True, exist_ok=True)

hf_hub_download(
    repo_id="sunblaze-ucb/cybergym",
    filename="data/arvo/2903/repo-vul.tar.gz",
    repo_type="dataset",
    local_dir=str(out_dir),
)
hf_hub_download(
    repo_id="sunblaze-ucb/cybergym",
    filename="data/arvo/2903/description.txt",
    repo_type="dataset",
    local_dir=str(out_dir),
)
print("Done")
EOF
```

Once downloaded, check what the vulnerability description says:
```bash
cat data/arvo/2903/description.txt
```

Then run the CRS:
```bash
python -m crs.main --task-dir ./data/arvo/2903 --output-dir ./crs_results_2903 --model gemma-3-27b-it --base-url http://g52lambda02.llan.ll.mit.edu:8000/v1 --api-key EMPTY --no-fuzzing
```

If the download fails or the description doesn't look like a heap overflow, run this to find a good alternative task first:

```bash
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("sunblaze-ucb/cybergym", split="train")
for i, row in enumerate(ds):
    desc = row.get("vulnerability_description", "").lower()
    if "heap" in desc and "overflow" in desc and row.get("project_language","") == "c":
        print(f"task_id={row['task_id']}  project={row['project_name']}")
        print(f"  {desc[:120]}")
        print()
    if i > 200:
        break
EOF
```

Paste the description output and I'll confirm it's a good target before you run the full pipeline.
