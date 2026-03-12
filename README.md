Now I can see the problem clearly. Your squeue shows **only one job** — a `bash` session on `g52lambda02`. That's your current terminal. vLLM is **not showing as a running job at all**, which means it's running in a separate terminal session on a **different node** that you can't reach via `127.0.0.1`.

Run this to find the node vLLM is actually on:

```bash
squeue -u ro31337 --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %N"
```

Also check if there are other sessions:
```bash
who | grep ro31337
```

And find the actual IP of where vLLM is listening:
```bash
ss -tlnp | grep 8000
```

The real fix is to **run both vLLM and CRS in the same SLURM job on the same GPU node**. Create a job script like this:

```bash
# run_crs.sh
#!/bin/bash
#SBATCH --partition=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=crs_run

# Start vLLM in background
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model /panfs/g52-panfs/exp/FY25/models/Qwen3.5-27B --served-model-name Qwen3.5-27B --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 8192 --host 127.0.0.1 --port 8000 &

# Wait for vLLM to be ready
echo "Waiting for vLLM..."
sleep 120

# Run CRS
cd /exp/FY26/aim/ro31337/AIxCC
python -m crs.main --task-dir ./data/arvo/1065 --output-dir ./crs_results --model Qwen3.5-27B --base-url http://127.0.0.1:8000/v1 --api-key EMPTY --no-fuzzing
```

Then submit it:
```bash
sbatch run_crs.sh
```

This guarantees vLLM and CRS are on the **same node** so `127.0.0.1` works correctly.
