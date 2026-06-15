choice = resp.choices[0]
raw = choice.message.content or ""
try:
    return json.loads(raw)
except json.JSONDecodeError:
    print(f"[DEBUG] finish_reason={choice.finish_reason}  len={len(raw)} chars")
    print("[DEBUG] raw output >>>")
    print(raw[:1200])
    print("[DEBUG] <<<")
    raise
