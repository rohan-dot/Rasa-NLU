def build_sample(example, add_labels: bool):
    # …  (same code that builds `chat` and gets `tokens`)
    tokens = tokenizer(chat,
                       truncation=True,
                       max_length=4096,
                       add_special_tokens=False)["input_ids"]

    if tokens[-1] != tokenizer.eos_token_id:
        tokens.append(tokenizer.eos_token_id)

    if add_labels:
        # length of everything before the assistant’s answer
        pre_len = len(
            tokenizer(
                "<|begin_of_text|>\n"
                + chat_wrap("system", SYSTEM_MSG)
                + chat_wrap("user",
                            f"Instruction: Inject CWE ID {example['cwe_id']} in the clean code\n"
                            f"Input:\n{example['func_after']}\nOutput:")
            , add_special_tokens=False)["input_ids"]
        )

        # build labels ***the safe way***: copy then mask
        labels = tokens.copy()              # same length by construction
        labels[:pre_len] = [-100] * pre_len # ignore system+user part
    else:
        labels = [-100] * len(tokens)

    return {"input_ids": tokens, "labels": labels}

---

### 1. Expand the **Chain-of-Thought** Steps

Add an explicit instruction to **exhaustively examine the context** for all relevant facts. For example:

> **Examine the Provided Context:**  
> - **Read through the entire context thoroughly** and identify **all** facts or data points that **directly or indirectly** address the hateful speech.  
> - Include **all** potentially relevant details, 

# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


