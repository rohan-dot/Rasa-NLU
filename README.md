It worked! Tell him yes, the LiteLLM config should use `host.docker.internal`:

```yaml
model_list:
  - model_name: "gpt-oss-120b"
    litellm_params:
      model: "openai/gpt-oss-120b"
      api_base: "http://host.docker.internal:8000/v1"
      api_key: "dummy"
```

And in `run-fuzzer.sh`:

```bash
export VLLM_HOST="oss-crs-litellm-1"
export VLLM_PORT="4000"
```

The chain: gemma-fuzzer container → LiteLLM sidecar (`oss-crs-litellm-1:4000`) → `host.docker.internal:8000` → vLLM on the host machine.

Then `oss-crs prepare` to rebuild, and `oss-crs run`.
