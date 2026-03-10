from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
r = c.chat.completions.create(model="gemma-3-27b-it", 
    messages=[{"role":"user","content":"Is water wet? Answer in one word."}])
print(r.choices[0].message.content)
