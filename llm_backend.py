"""
llm_backend.py - run planner.py / agentic_planner.py WITHOUT a vLLM server by
loading the model directly with HuggingFace transformers.

Setup (one time, on a connected machine or from your local wheel cache):
    pip install torch transformers accelerate
    pip install bitsandbytes          # only if LOAD_4BIT = True

Use: in planner.py AND agentic_planner.py change ONE line:
    from openai import OpenAI
        ->
    from llm_backend import OpenAI

Set BACKEND below:
  "local" -> loads MODEL_PATH in this process (no server needed)
  "vllm"  -> passthrough to the real openai client (undo without editing scripts)

Honest expectations for local mode with a ~31B model:
  - bf16 needs ~65 GB of GPU memory; LOAD_4BIT=True cuts that to ~20 GB
    at slightly reduced quality (fine for coords/crossings/route JSON).
  - First call is slow (model load: several minutes). After that, each call
    is slower than vLLM but perfectly usable for this workload.
  - No server to crash, no port config, one process.
"""

BACKEND    = "local"                       # "local" or "vllm"
MODEL_PATH = "/path/to/your/gemma-model"   # local directory with the weights
LOAD_4BIT  = True                          # False if you have the VRAM for bf16
MAX_INPUT_TOKENS = 8192

# ------------------------------------------------------------------
if BACKEND == "vllm":
    from openai import OpenAI              # passthrough, nothing else runs
else:
    from types import SimpleNamespace

    _model, _tok = None, None

    def _load():
        global _model, _tok
        if _model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  loading model from {MODEL_PATH} (first time only, be patient)...",
              flush=True)
        _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if LOAD_4BIT:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
            kwargs.pop("torch_dtype")
        _model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
        _model.eval()
        print("  model loaded.", flush=True)

    class _Completions:
        def create(self, model=None, messages=None, temperature=0.0,
                   max_tokens=1500, response_format=None, **kw):
            import torch
            _load()
            # Gemma chat templates reject a system role: fold it into user turn
            msgs = list(messages or [])
            if msgs and msgs[0]["role"] == "system":
                sys_txt = msgs[0]["content"]
                msgs = msgs[1:]
                if msgs and msgs[0]["role"] == "user":
                    msgs[0] = {"role": "user",
                               "content": sys_txt + "\n\n" + msgs[0]["content"]}
                else:
                    msgs.insert(0, {"role": "user", "content": sys_txt})
            if response_format is not None:
                # no grammar enforcement locally - reinforce via instruction
                msgs[-1] = {"role": msgs[-1]["role"],
                            "content": msgs[-1]["content"]
                            + "\n\nRespond with ONLY a single JSON object. "
                              "No prose, no markdown fences."}
            ids = _tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt",
                truncation=True, max_length=MAX_INPUT_TOKENS).to(_model.device)
            with torch.no_grad():
                out = _model.generate(
                    ids, max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5) if temperature > 0 else None,
                    pad_token_id=_tok.eos_token_id)
            text = _tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            msg = SimpleNamespace(content=text)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class OpenAI:                            # drop-in constructor signature
        def __init__(self, base_url=None, api_key=None, timeout=None, **kw):
            self.chat = SimpleNamespace(completions=_Completions())
