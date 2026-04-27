
# 1. Rebuild njs with ASAN
bash rebuild_njs.sh

# 2. Test the known PoC
ASAN_OPTIONS=detect_leaks=0 ./secbench_tasks/njs.cve-2022-32414/repo/build/njs poc_njs.js
