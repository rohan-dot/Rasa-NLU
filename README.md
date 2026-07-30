git am --abort
git checkout -b toolcalling-reliability
git apply --3way discver-full.patch
git add -A && git commit -m "tool-calling reliability + instrumentation"
