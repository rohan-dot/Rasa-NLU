The issue is clear: Docker bridge network can't reach the external LLM. OSS-CRS uses LiteLLM as a proxy specifically for this. The fix is configuring LiteLLM correctly.

Tell him to edit the `litellm-config.yaml` in the gemma-fuzzer directory:

```yaml
model_list:
  - model_name: "gpt-oss-120b"
    litellm_params:
      model: "openai/gpt-oss-120b"
      api_base: "http://EXTERNAL_LLM_HOST:8000/v1"
      api_key: "dummy"
```

Replace `EXTERNAL_LLM_HOST` with the actual IP/hostname of the machine running vLLM (not `localhost`, not `127.0.0.1` — the real network IP).

Then in `run-fuzzer.sh`, make sure it talks to the LiteLLM sidecar (which IS on the Docker bridge network):

```bash
export VLLM_HOST="oss-crs-litellm-1"
export VLLM_PORT="4000"
```

The flow is: gemma-fuzzer container → LiteLLM sidecar (same Docker network) → external LLM host (real network). LiteLLM bridges the gap between Docker's bridge network and the external world.

If he still can't reach the external LLM from LiteLLM, the Docker host might need `--add-host` or the external host's firewall might be blocking. He should test from the Docker host first:

```bash
curl http://EXTERNAL_LLM_HOST:8000/v1/models
```

If that works from the host but not from inside Docker, it's a Docker networking issue he needs to discuss with IT.
