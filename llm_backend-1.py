"""
llm_backend.py - run planner.py / agentic_planner.py WITHOUT a vLLM server by
loading the model directly with HuggingFace transformers.

Use: in planner.py AND agentic_planner.py change ONE line:
    from openai import OpenAI   ->   from llm_backend import OpenAI

BACKEND "local" loads MODEL_PATH in-process; "vllm" is a passthrough to the
real openai client so you can switch back without editing the scripts.

v2 fix: apply_chat_template can return a BatchEncoding (dict) instead of a
tensor depending on transformers version -> generate() crashed with
KeyError 'shape'. Now we request return_dict=True and pass input_ids +
attention_mask explicitly. Also prints which device the model landed on.
"""

BACKEND    = "local"                       # "local" or "vllm"
MODEL_PATH = "/panfs/g52-panfs/exp/FY26/models/gemma-4-31B-it"
LOAD_4BIT  = True                          # False if you have VRAM for bf16
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
        if _tok.pad_token_id is None:
            _tok.pad_token = _tok.eos_token
        kwargs = {"device_map": "auto"}
        if LOAD_4BIT:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        _model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
        _model.eval()
        dev = next(_model.parameters()).device
        print(f"  model loaded on: {dev}"
              + ("  (WARNING: CPU - generation will be very slow; check CUDA)"
                 if dev.type == "cpu" else ""), flush=True)

    class _Completions:
        def create(self, model=None, messages=None, temperature=0.0,
                   max_tokens=1500, response_format=None, **kw):
            import torch
            _load()
            # Gemma chat templates reject a system role: fold it into user turn
            msgs = [dict(m) for m in (messages or [])]
            if msgs and msgs[0]["role"] == "system":
                sys_txt = msgs[0]["content"]
                msgs = msgs[1:]
                if msgs and msgs[0]["role"] == "user":
                    msgs[0]["content"] = sys_txt + "\n\n" + msgs[0]["content"]
                else:
                    msgs.insert(0, {"role": "user", "content": sys_txt})
            if response_format is not None:
                msgs[-1]["content"] += ("\n\nRespond with ONLY a single JSON "
                                        "object. No prose, no markdown fences.")

            enc = _tok.apply_chat_template(
                msgs, add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
            # enc is a BatchEncoding; extract tensors explicitly (v2 fix)
            input_ids = enc["input_ids"]
            attn = enc.get("attention_mask")
            if input_ids.shape[1] > MAX_INPUT_TOKENS:      # left-truncate history
                input_ids = input_ids[:, -MAX_INPUT_TOKENS:]
                if attn is not None:
                    attn = attn[:, -MAX_INPUT_TOKENS:]
            dev = next(_model.parameters()).device
            input_ids = input_ids.to(dev)
            if attn is not None:
                attn = attn.to(dev)

            gen_kwargs = {"input_ids": input_ids,
                          "max_new_tokens": max_tokens,
                          "pad_token_id": _tok.pad_token_id}
            if attn is not None:
                gen_kwargs["attention_mask"] = attn
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)
            else:
                gen_kwargs.update(do_sample=False)

            with torch.no_grad():
                out = _model.generate(**gen_kwargs)
            text = _tok.decode(out[0][input_ids.shape[1]:],
                               skip_special_tokens=True)
            msg = SimpleNamespace(content=text)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class OpenAI:                            # drop-in constructor signature
        def __init__(self, base_url=None, api_key=None, timeout=None, **kw):
            self.chat = SimpleNamespace(completions=_Completions())
