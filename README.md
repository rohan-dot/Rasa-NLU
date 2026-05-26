Give him this test script. It launches a container on the same Docker network OSS-CRS uses and tests LLM connectivity:

```bash
docker run --rm --network bridge \
  curlimages/curl:latest \
  curl -s -o /dev/null -w "%{http_code}" \
  http://EXTERNAL_LLM_HOST:8000/v1/models
```

If that returns `200`, networking works. If it times out, Docker can't reach the external host.

For testing with the actual LiteLLM sidecar setup, tell him to create a file called `test_llm.sh`:

```bash
#!/bin/bash
# Test LLM connectivity from inside a Docker container
# Usage: bash test_llm.sh <LLM_HOST> <LLM_PORT>

LLM_HOST="${1:-host.docker.internal}"
LLM_PORT="${2:-8000}"

echo "Testing from Docker container → $LLM_HOST:$LLM_PORT"

docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  curlimages/curl:latest \
  curl -sv "http://$LLM_HOST:$LLM_PORT/v1/models" 2>&1

echo ""
echo "If you see model list JSON above, networking works."
echo "If connection refused/timeout, the container can't reach the LLM."
```

Run it with: `bash test_llm.sh <his-llm-host-ip> 8000`

The `--add-host=host.docker.internal:host-gateway` flag maps `host.docker.internal` to the Docker host's IP. If his LLM runs on the Docker host machine itself, he can use `host.docker.internal` as the hostname in the LiteLLM config.
