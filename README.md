He doesn't need it. The `--add-host` was only for the test container. OSS-CRS uses docker-compose, and on Docker 20.10+, `host.docker.internal` works automatically in compose services on Linux.

Tell him: just run it. The LiteLLM config already points to `host.docker.internal:8000`, and since the test proved that resolves correctly, the LiteLLM sidecar will too.

If it somehow doesn't work, he can add this to `gemma-fuzzer/compose.yaml` under the CRS service:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

But try without it first — it should just work.
