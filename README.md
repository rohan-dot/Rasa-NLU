git clone https://github.com/ossf/oss-crs.git ~/oss-crs
git clone https://github.com/google/oss-fuzz.git ~/oss-fuzz
cd ~/oss-crs && pip install uv
tar -xzf gemma-fuzzer-v4-final.tar.gz



nano ~/oss-fuzz/projects/cjson/Dockerfile


RUN git clone --depth 1 https://github.com/DaveGamble/cJSON.git cjson && \
    cd cjson && \
    git fetch --unshallow && \
    git checkout d6d5449~1


   CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model /panfs/g52-panfs/exp/FY25/models/gpt-oss-120b \
    --served-model-name gpt-oss-120b \
    --dtype bfloat16 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --host 0.0.0.0 --port 8000

cd ~/oss-crs

uv run oss-crs prepare \
    --compose-file /path/to/gemma-fuzzer/compose.yaml

uv run oss-crs build-target \
    --compose-file /path/to/gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/cjson

uv run oss-crs run \
    --compose-file /path/to/gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/cjson \
    --target-harness cjson_read_fuzzer \
    --timeout 1200

uv run oss-crs artifacts \
    --compose-file /path/to/gemma-fuzzer/compose.yaml \
    --fuzz-proj-path ~/oss-fuzz/projects/cjson \
    --target-harness cjson_read_fuzzer


    
    
