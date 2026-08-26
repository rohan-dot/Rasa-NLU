time curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"gemma-4-31B-it","max_tokens":10,"messages":[{"role":"user","content":"say ok"}]}'
