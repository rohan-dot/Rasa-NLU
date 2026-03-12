Let's diagnose exactly where it's failing. Run these two tests right now:

**Test 1 — Can your terminal actually reach vLLM?**
```bash
curl http://127.0.0.1:8000/v1/models
```

**Test 2 — Send a real completion request manually:**
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-27B", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 50}'
```

Paste the output of both here. This tells us immediately whether the problem is:

- **curl works → problem is in llm_router.py** (timeout setting didn't save or wrong location)
- **curl hangs/times out → network/firewall issue** between your terminal and vLLM
- **curl gives connection refused → vLLM isn't actually listening** on that address
- **curl gives model not found → served-model-name mismatch**

Also run this to confirm the timeout edit actually landed:
```bash
grep -n "timeout" crs/llm_router.py
```

Don't try fixing anything else until we see these outputs — we need to know what's actually failing before guessing at more fixes.
