If your final output from the chain-of-thought prompt is “too short” or missing important details, there are a few adjustments you can make to encourage the model to reference more information from the context. Here are some practical tips:

---

### 1. Expand the **Chain-of-Thought** Steps

Add an explicit instruction to **exhaustively examine the context** for all relevant facts. For example:

> **Examine the Provided Context:**  
> - **Read through the entire context thoroughly** and identify **all** facts or data points that **directly or indirectly** address the hateful speech.  
> - Include **all** potentially relevant details, even if they don’t perfectly match the hateful statement but still offer background or related insights.

This ensures the model knows it should err on the side of including more detail rather than less.

---

### 2. Request a **Comprehensive** Answer

At the end of your chain-of-thought instructions or prompt, add language like:

> “Provide a **comprehensive** answer, **including** any relevant context details you find. Be as **thorough** as possible in identifying and citing these details.”

The word “comprehensive” signals that you want all relevant facts, not just a short summary.

---

### 3. Ask for **Specific References** to the Context

Encourage references (like mini-quotes or paraphrases) to the original context. For example:

> “For each fact you use from the context, **include a brief reference** (e.g., a short paraphrase or statement about where it came from). This will ensure clarity and completeness.”

Reinforcing references to the text itself helps the model recall and re-present all the important points from the source.

---

### 4. Provide a **Detailed Outline** Template

You can include a prompt structure like this:

1. **Identify the hateful statements.** (One or more bullet points of what is being claimed.)  
2. **List every potentially relevant fact from the context.**  
   - Provide bullet points or short paragraphs for each fact.  
   - Explain why each fact contradicts or weakens the hateful statements.  
3. **Summarize how these facts collectively refute the hateful message.**

By literally telling the model, “List **every** relevant fact,” you reduce the chance of it leaving out important details.

---

### 5. Incorporate a “**Re-check the Context**” Step

In your chain-of-thought prompt, add a final step:

> 4. **Re-check the Context**  
>    - Revisit the context to confirm no relevant detail was missed.  
>    - If any additional facts or clarifications are found, incorporate them before delivering the final answer.

This acts as a failsafe to prompt the model to scan the context again for completeness.

---

### 6. Use a Higher **Temperature** (Optional)

If you have control over generation parameters (like temperature), increasing the temperature **can** lead to more expansive answers. However, it can also introduce a bit more “creative” wording or tangential details. If your main goal is thoroughness rather than creativity, you might rely more on step-by-step instructions than temperature.

---

## Example Revised Prompt

Putting this all together, here’s an example of how you might revise your chain-of-thought prompt:

> **System/Instruction:**  
> You are an AI fact-checking model. You will be given hateful speech and a context containing factual information. **Your job is to comprehensively identify all relevant facts in the context** that contradict or refute the hateful claims. **Do not add any information beyond what the context provides.**
>
> Follow these steps in detail (chain-of-thought):
> 1. **Identify the Hateful Claims**  
>    - Summarize the main points or assertions in the hateful speech.  
> 2. **Examine the Provided Context Thoroughly**  
>    - Read through the entire context carefully, listing **all** facts or statements that could refute, oppose, or weaken the hateful claims.  
>    - If no fact directly contradicts the hate speech, gather the closest relevant information that undermines or challenges it.  
> 3. **Refutation Construction**  
>    - For each fact you identified, provide a brief explanation of how it opposes the hateful claim.  
>    - Use bullet points or a concise paragraph structure.  
> 4. **Re-check the Context**  
>    - Quickly revisit the context to ensure no relevant detail is missed.  
>    - Add any final facts or clarifications if you find something you skipped initially.  
> 5. **Final Answer**  
>    - Present your final refutation with all supporting evidence from the context.  
>    - Keep it factual, thorough, and directly relevant to the hateful speech.
>
> **Context:**  
> ```
> {Context}
> ```
>
> **Hateful Speech:**  
> ```
> “...”
> ```

With these revisions, you’re explicitly guiding the model to be as exhaustive as possible, which should yield more detailed, longer answers grounded in your context.







xxxxxx
[System/Instruction to Model]

You are an AI fact-checking model. You will be given a piece of hateful speech and a contextual reference containing factual information. Your objective is to refute the hateful speech by identifying factual evidence from the context. You must not invent any details or rely on information outside the context. Present your final answer in bullet points or a short paragraph.

Follow this chain-of-thought process step by step:

1. **Identify the Hateful Claims**  
   - Read the hateful speech carefully.  
   - Summarize the main hateful points or assertions.

2. **Examine the Provided Context**  
   - Look for any facts, data, or statements relevant to those hateful claims.  
   - Determine which specific parts of the context directly contradict or debunk the hateful assertions.  
   - If there are no directly opposing facts, find the closest relevant information that challenges the hateful content.

3. **Construct the Refutation**  
   - Organize the discovered facts in bullet points or a coherent paragraph.  
   - Explain how these facts counter the hateful statements, demonstrating that the hateful speech is either incorrect, misleading, or unfounded.

4. **Deliver the Final Answer**  
   - Present only the information found within the context.  
   - Ensure no extraneous or fabricated details are included.  
   - The final answer should be a clear, direct response that addresses the hateful content specifically.

[End of Chain-of-Thought Instructions]

---

**Context**:

# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


