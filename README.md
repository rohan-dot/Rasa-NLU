cat > list_easy.py << 'EOF'
from datasets import load_dataset
ds = load_dataset("SEC-bench/SEC-bench", split="cve")
skip = ["faad2", "mruby", "php", "cpython"]
for row in ds:
    iid = row["instance_id"]
    if any(s in iid for s in skip):
        continue
    build = row.get("build_sh", "")
    if "configure" in build or "cmake" in build:
        desc = row.get("bug_description", "")
        print(f"{iid}")
        print(f"  {desc[:120]}")
        print()
EOF

python list_easy.py | head -80
