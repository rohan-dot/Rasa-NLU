The error is clear — `mosquitto.h: No such file or directory`. The header exists somewhere but not in the paths we're passing. Find it:

```bash
find /home/ro31337/.crs_workdir/57002/repo -name "mosquitto.h" 2>/dev/null
```

Then add that directory to the compile command. Also find `db.h`:

```bash
find /home/ro31337/.crs_workdir/57002/repo -name "db.h" 2>/dev/null
```

Paste those two outputs and I'll give you the exact working compile command. Once we get it compiling manually we know exactly which `-I` paths to hardcode in `build_executor.py`'s fallback.

The broader fix for the CRS is to make the fallback smarter — instead of hardcoding `src` and `lib`, it should find all directories containing `.h` files and add them all as `-I` paths:

```python
# In build_executor.py fallback — collect ALL include dirs
include_dirs = set()
for h in Path(task.repo_path).rglob("*.h"):
    include_dirs.add(str(h.parent))
src_cmd += [f"-I{d}" for d in sorted(include_dirs)]
```

This works universally for any project. But first let's confirm what `find` returns for `mosquitto.h`.
