**Everything your coworker needs, start to finish:**

**Step 1: Clone repos**
```bash
git clone https://github.com/ossf/oss-crs.git ~/oss-crs
git clone https://github.com/google/oss-fuzz.git ~/oss-fuzz
pip install uv
```

**Step 2: Get gemma-fuzzer onto their machine** (USB, scp, git, whatever you use)

**Step 3: Start vLLM**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model /panfs/g52-panfs/exp/FY25/models/gpt-oss-120b \
    --served-model-name gpt-oss-120b \
    --dtype bfloat16 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --host 0.0.0.0 --port 8000
```

**Step 4: Check Docker bridge IP**
```bash
ip addr show docker0 | grep inet
```
If it's not `172.17.0.1`, edit `gemma-fuzzer/oss-crs/crs.yaml` and change `VLLM_HOST` to whatever it shows.

**Step 5: Pin the vulnerable commit**
```bash
nano ~/oss-fuzz/projects/libxml2/Dockerfile
```
Find the `git clone` line for libxml2 and change it to:
```
RUN git clone https://gitlab.gnome.org/GNOME/libxml2.git && \
    cd libxml2 && \
    git checkout a7511af0
```

**Step 6: Run the whole thing**
```bash
cd ~/oss-crs

uv run oss-crs prepare \
    --compose-file /path/to/gemma-fuzzer/compose.yaml

uv run oss-crs build-target \
    --compose-file /path/to/gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/libxml2

uv run oss-crs run \
    --compose-file /path/to/gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/libxml2 \
    --target-harness xml \
    --timeout 3600

uv run oss-crs artifacts \
    --compose-file /path/to/gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/libxml2 \
    --target-harness xml
```

**To run on latest (no bug pinned):** Remove the `git checkout a7511af0` line from the Dockerfile.

**To run on a different project entirely:** Swap the project path and harness name. No other changes:
```bash
uv run oss-crs build-target --compose-file /path/to/gemma-fuzzer/compose.yaml --fuzz-proj-path ~/oss-fuzz/projects/curl
uv run oss-crs run --compose-file /path/to/gemma-fuzzer/compose.yaml --fuzz-proj-path ~/oss-fuzz/projects/curl --target-harness curl_fuzzer --timeout 3600
```

No manual `clang`, no writing harness files, no building libraries. OSS-CRS does all of that inside Docker using the `asan-builder.Dockerfile` and `compile_target` script that are already in gemma-fuzzer.








xxx
1. Clone repos:
   git clone https://github.com/ossf/oss-crs.git ~/oss-crs
   git clone https://github.com/google/oss-fuzz.git ~/oss-fuzz

2. Start vLLM (MUST use 0.0.0.0):
   CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
       --model /panfs/g52-panfs/exp/FY25/models/gpt-oss-120b \
       --served-model-name gpt-oss-120b \
       --dtype bfloat16 --tensor-parallel-size 2 \
       --gpu-memory-utilization 0.85 --max-model-len 8192 \
       --host 0.0.0.0 --port 8000

3. Check docker bridge IP:
   ip addr show docker0 | grep inet
   (if not 172.17.0.1, edit oss-crs/crs.yaml VLLM_HOST)

4. Run on ANY project (just swap the project name and harness):

   cd ~/oss-crs

   uv run oss-crs prepare \
       --compose-file /path/to/gemma-fuzzer/compose.yaml

   uv run oss-crs build-target \
       --compose-file /path/to/gemma-fuzzer/compose.yaml \
       --fuzz-proj-path ~/oss-fuzz/projects/libxml2

   uv run oss-crs run \
       --compose-file /path/to/gemma-fuzzer/compose.yaml \
       --fuzz-proj-path ~/oss-fuzz/projects/libxml2 \
       --target-harness xml \
       --timeout 3600

   uv run oss-crs artifacts \
       --compose-file /path/to/gemma-fuzzer/compose.yaml \
       --fuzz-proj-path ~/oss-fuzz/projects/libxml2 \
       --target-harness xml

5. To fuzz a different project, just change the path:
   --fuzz-proj-path ~/oss-fuzz/projects/curl
   --target-harness curl_fuzzer

   That's it. No manual building, no writing harnesses,
   no clang commands. OSS-CRS handles all of it.


