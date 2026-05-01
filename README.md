edit gemma-fuzzer/oss-crs/docker-compose/run-fuzzer.sh and change the last section to:


# ── 4. Launch the Python orchestrator ──
echo "[init] Starting orchestrator..."

# Inside OSS-CRS Docker, LiteLLM proxy runs as a sidecar.
# Use it instead of direct vLLM connection.
LITELLM_HOST="${LITELLM_HOST:-oss-crs-litellm-1}"
LITELLM_PORT="${LITELLM_PORT:-4000}"

exec /opt/gemma-fuzzer/venv/bin/python3 /opt/gemma-fuzzer/src/main.py \
    --build-dir    /opt/target/build \
    --src-dir      /opt/target/src \
    --output-dir   /output \
    --seed-dir     /input/seeds \
    --log-dir      /var/log/gemma-fuzzer \
    --harness      "${OSS_CRS_TARGET_HARNESS}" \
    --vllm-host    "${LITELLM_HOST}" \
    --vllm-port    "${LITELLM_PORT}" \
    --vllm-model   "${VLLM_MODEL:-hnode02-Deepseekv4-vllm}" \
    --fuzz-timeout "${FUZZ_TIMEOUT:-3600}" \
    --fuzz-jobs    "${FUZZ_JOBS:-1}" \
    --llm-seed-interval "${LLM_SEED_INTERVAL:-120}"
