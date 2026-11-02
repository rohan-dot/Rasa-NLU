# LiteLLM from Scratch: Chat, Then Code

Oct 26, 2026 · @Jo bell

You get one file and a model name. Everything else you build yourself, and the model you are talking to is allowed to help you build it. Two rounds of problems: first turn the file into something you would use to talk to a model, then make it write and fix code.

## Setup (10 min)

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install litellm
export LITELLM_PROXY_API_BASE=https://<gateway-host>   # Windows PowerShell: $env:NAME="value"
export LITELLM_PROXY_API_KEY=sk-<your key>
export MODEL=litellm_proxy/<a name from the gateway's /v1/models>
```

No key or no network: install Ollama, run `ollama pull llama3.2`, and use `export MODEL=ollama_chat/llama3.2` instead of the three gateway lines.

## The one file

`talk.py`:

```python
import os
from litellm import completion

response = completion(
    model=os.environ["MODEL"],
    messages=[{"role": "user", "content": "Say hello and tell me which model you are."}],
)
print(response.choices[0].message.content)
```

`python talk.py` prints a greeting. That is the whole API: one function, a list of `{role, content}` messages, the reply at `choices[0].message.content`, the token counts at `response.usage`. Change the `MODEL` string and the same file talks to a different model. Everything below is this file plus code you write.

## How the problems work

Each problem says what to build, how you know it is done, and one hint: the only new fact you need. Work alone or in pairs. Stuck for more than five minutes? Paste your code into the chat you built in A1 and ask it. Nobody hands out solutions; after most of the room has finished a problem, one volunteer shows theirs on the projector and everyone moves on.

## Part A — Talk to it (30 min)

1. **Make it a conversation.** Loop on `input()`, keep every message in a list, send the whole list on every call.
   - Done when: you tell it your name, ask two unrelated things, then ask "what's my name?" and it knows.
   - Hint: the model remembers nothing between calls. The list is its memory, and its own replies go in it too, as `{"role": "assistant", "content": ...}`.
2. **Give it a house style.** Add a system message.
   - Done when: every reply follows the style (two sentences max, or always a haiku) even when you ask it to stop.
   - Hint: `{"role": "system", "content": ...}`, first in the list, sent every time.
3. **Stream it.**
   - Done when: words appear as they are generated, and the full reply still lands in the history.
   - Hint: `completion(..., stream=True)` returns chunks; each has `chunk.choices[0].delta.content`, which is sometimes `None`.
4. **Swap models mid-conversation.** Add a `/model <name>` command.
   - Done when: you switch models and the new one answers a follow-up that depends on earlier turns.
   - Hint: the model string is the only thing that changes; nothing else in your code does. That is the point of `litellm`. Anyone with Ollama: `/model ollama_chat/llama3.2`.
5. **Stretch: show the bill.** Print tokens after each reply.
   - Hint: `response.usage.prompt_tokens` and `.completion_tokens`; when streaming, pass `stream_options={"include_usage": True}` and read the last chunk.

## Part B — Make it write and fix code (45 min)

Start from your Part A file or from `talk.py` again; either works.

1. **Make it write a file you can run.** Ask for a function that tests itself, save the code to `answer.py`, run it.
   - Done when: `python answer.py` prints `ALL TESTS PASSED` and you never copy-pasted anything.
   - Hint: the reply is text. Tell the model to answer with exactly one \`\`\`python block, cut the block out with a regex, then `subprocess.run([sys.executable, "answer.py"])`.
2. **Make it fix a bug.** Save this as `buggy.py`:

   ```python
   def average(numbers):
       return sum(numbers) / (len(numbers) - 1)
   
   assert average([2, 4, 6]) == 4, "average is wrong"
   print("fixed")
   ```

   Send the model the file and the instruction "make the assert pass, change as little as possible", and write back what it returns.
   - Done when: `python buggy.py` prints `fixed` and only one line changed.
   - Hint: put the file's text in the prompt inside a code fence and ask for the whole file back. `difflib.unified_diff` shows you the change before you overwrite anything.
3. **Let it read files itself.** Give it one tool, `read_file(path)`.
   - Done when: you ask "what does buggy.py do?" without pasting anything, and it reads the file, then answers.
   - Hint: `completion(..., tools=[...])` takes a JSON schema per function. The reply then has `message.tool_calls` (each with `.id`, `.function.name`, `.function.arguments` as a JSON string) instead of text. Append that message to the list, run the function, append `{"role": "tool", "tool_call_id": call.id, "content": result}`, and call `completion` again.
4. **Make it an agent.** Put B3 in a loop and add `write_file(path, content)`, `edit_file(path, old, new)` and `run_command(command)`.
   - Done when: `python agent.py "buggy.py fails, fix it"` reads the file, edits one line, runs it, and reports that it printed `fixed`, with no help from you.
   - Hint: the picture. Stop when a reply has no tool calls; cap it at 20 turns. Make `edit_file` refuse unless `old` matches exactly once and return the refusal as text; the model will try again with a longer snippet.

&#91;embedded content: the agent loop · the hint for B4\]

5. **Stretch: make it safe enough to leave alone.**
   - Done when: it cannot read or write outside a `workspace/` folder, asks you before any write, and only runs commands that start with `python`.
   - Hint: `Path(path).resolve()` and check it sits under the folder; `input("ok? [y/N] ")` before writing; a tuple of allowed command prefixes.

## Facilitator sheet

Hand out the setup lines and `talk.py`. Nothing else, not even A1.

- Rhythm per problem: read it aloud, let them work, and when about two thirds are done have one volunteer show theirs on the projector for two minutes, then move on. The rest adopt the volunteer's approach, not their code.
- The first hint is in the doc. The second hint is you at their desk, never the projector.
- Timings for 10 to 15 people: setup 10 min, A1 to A4 30 min, B1 to B2 15 min, B3 to B4 30 min. B3 is where it gets hard; ending a short session after B3 is fine, and B4 makes a good second session.
- Push them to use the model: "ask the thing you just built" is the habit the session is really teaching.
- Your reference solutions are in `litellm-workshop.zip`: `03_chat.py` (A1, A2), `04_stream.py` (A3, A4), `10_write_code.py` (B1), `11_edit_file.py` (B2), `12_one_tool.py` (B3), `agent.py` (B4, B5). They read `MODEL` from the environment too; they also expect `pip install python-dotenv pytest`.
- Before the day: run the zip's `12_one_tool.py` against the gateway model you will name in the setup lines. Tool calling is the one thing that varies by model, and B3 and B4 depend on it.

## When something breaks

| What they see | Fix |
| --- | --- |
| `AuthenticationError` | The `LITELLM_PROXY_API_KEY` export is missing in this terminal, or has quotes or spaces around it |
| `BadRequestError: LLM Provider NOT provided` | `MODEL` needs the prefix: `litellm_proxy/<name>` |
| `NotFoundError` or 404 | The name is not one the gateway serves; copy it exactly from `/v1/models` |
| `APIConnectionError` | Wrong `LITELLM_PROXY_API_BASE`, no VPN, or for Ollama `ollama serve` is not running |
| `ModuleNotFoundError: litellm` | The venv is not activated in this terminal |
| In B3 `tool_calls` is `None` and it guesses | That model does not support tools; switch to the one you tested |
| 400 about `temperature` or another parameter | `litellm.drop_params = True` after the import |

To see exactly what went over the wire: `LITELLM_LOG=DEBUG python talk.py`.

Sources: [function calling](https://docs.litellm.ai/docs/completion/function_call.md), [LiteLLM proxy as a provider](https://litellm.vercel.app/docs/providers/litellm_proxy), [litellm README](https://github.com/BerriAI/litellm).
