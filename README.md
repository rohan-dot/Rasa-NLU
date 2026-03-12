CyberGym Task (repo-vul.tar.gz + description.txt)
         │
         ▼
┌─────────────────┐
│  data_loader.py  │  Extract codebase, read description
└────────┬────────┘
         │ CyberGymTask
         ▼
┌──────────────────────┐
│ code_intelligence.py │  Rank files, extract snippets,
│                      │  detect build system, classify vuln
└────────┬─────────────┘
         │ CodeContext
         ▼
┌────────────────────────────────────────────────────┐
│                poc_strategies.py                    │
│  Strategy 1: AnalyzeFirst    (2-shot LLM)          │
│  Strategy 2: CallPathTargeted (function-focused)    │
│  Strategy 3: PatternReplay   (template-seeded)     │
│  Strategy 4: IterativeRefine (compile-loop)        │
│  Strategy 5: Direct          (fallback)            │
│                 ↓ via llm_router.py                │
│              vLLM/Gemma or GPT                     │
└────────────────┬───────────────────────────────────┘
                 │ list[PoCResult]
                 ▼
┌───────────────────────────┐
│    build_executor.py       │  Build project → compile PoC
│                            │  → run under ASAN/UBSAN
│    crash detected? ──YES──►│  triggered=True
│                            │
│    crash detected? ──NO───►│  try fuzzer.py (optional)
└────────────────────────────┘
                 │ list[RunResult]
                 ▼
┌──────────────────┐
│  evaluator.py    │  Score, record, save JSON + Markdown






Totally understandable. Here's everything you need:

**Start vLLM with Gemma:**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model /panfs/g52-panfs/exp/FY25/models/gemma-3-27b-it --served-model-name gemma-3-27b-it --dtype bfloat16 --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 8192 --host 0.0.0.0 --port 8000
```

**Wait for "Application startup complete", then run CRS:**
```bash
python -m crs.main --task-dir ./data/arvo/1065 --output-dir ./crs_results --model gemma-3-27b-it --base-url http://g52lambda02.llan.ll.mit.edu:8000/v1 --api-key EMPTY --no-fuzzing
```

**Also update config.py so defaults match:**
```python
# Line 43:
default_factory=lambda: os.environ.get("LLM_MODEL", "gemma-3-27b-it")

# Line 35:
"OPENAI_BASE_URL", "http://g52lambda02.llan.ll.mit.edu:8000/v1"
```

This is exactly what worked before, just with `--host 0.0.0.0` added so you don't have the node routing problem again.
