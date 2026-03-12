CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
    --model /panfs/g52-panfs/exp/FY25/models/Qwen2.5-32B-Instruct \
    --served-model-name Qwen2.5-32B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --host 127.0.0.1 --port 8000
