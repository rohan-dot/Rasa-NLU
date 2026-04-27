python3 -c "
import json
from datasets import load_dataset
ds = load_dataset('SEC-bench/SEC-bench', split='cve')
for row in ds:
    iid = row['instance_id']
    desc = row.get('bug_description','')
    build = row.get('build_sh','')
    # Skip projects we know are painful
    skip = ['faad2','mruby','php','cpython']
    if any(s in iid for s in skip):
        continue
    # Show ones that use configure/make/cmake (not rake/cargo/etc)
    if 'configure' in build or 'cmake' in build or 'make' in build:
        lang = row.get('lang','')
        print(f'{iid}')
        print(f'  {desc[:100]}')
        print()
" 2>&1 | head -80
