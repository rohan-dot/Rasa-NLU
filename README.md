Yes — and quick suspicion first: if you launched both planners against vLLM at once, a likely reason it died is an out-of-memory crash from concurrent requests on a 31B model. Check vLLM's last log lines for "CUDA out of memory"; restarting it (one planner at a time) may be the fastest fix, and vLLM will always be faster than in-script loading.

But direct loading absolutely works and removes the server dependency. Here's a drop-in backend — it mimics the OpenAI client interface your scripts already use, so the change is **one import line per file**:

Drop `llm_backend.py` into your folder, then three edits total:

**1.** In `llm_backend.py`: set `MODEL_PATH` to the local directory holding your Gemma weights (the same path you gave vLLM's `--model`).

**2.** In `planner.py`, change the import: `from openai import OpenAI` → `from llm_backend import OpenAI`

**3.** Same one-line change in `agentic_planner.py`.

Run exactly as before. First call loads the model (a few minutes, it prints progress), then everything works — and switching back to vLLM later is just flipping `BACKEND = "vllm"` in this one file, no script edits.

Honest tradeoffs so nothing surprises you:

**Memory:** a 31B model needs ~65 GB GPU memory in bf16. `LOAD_4BIT = True` (the default I set) cuts it to ~20 GB via bitsandbytes with mild quality loss — fine for coordinates and route JSON. If vLLM ran on this box, 4-bit will fit.

**Packages:** needs `torch`, `transformers`, `accelerate` (+ `bitsandbytes` for 4-bit). If your air-gapped box only had vLLM's stack, vLLM installs pull in torch and transformers anyway, so you likely have most of it — check with `python -c "import transformers, torch"`.

**JSON discipline:** vLLM's `response_format` *forced* valid JSON at the token level; transformers can't, so the adapter compensates by appending "respond with ONLY a single JSON object" and your scripts' parsers already strip fences and hunt for the first `{`. Expect the occasional malformed reply — the agentic loop's retry handles those, but if a specific call fails repeatedly, tell me which and I'll harden that parser.

**Speed:** slower per call than vLLM (no batching/paged attention), but your workload is sequential calls anyway, so it's usable — and honestly, the "is it working" problem improves, since everything runs in one visible process with the load progress printed.

Still worth checking vLLM's crash log when you get a minute — if it was OOM from the two concurrent runs, it'll work fine restarted, and it remains the faster runtime for the coordinate bootstrap.
