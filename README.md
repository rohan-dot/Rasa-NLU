# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic

### Methodology

We employed the Mistral model with a Retrieval-Augmented Generation (RAG) framework for authorship impersonation. Our approach involved capturing and replicating the target author's writing style using prompt engineering techniques.

**1. Style Extraction:**  
We queried the model using prompts to extract the target author's writing style based on specific criteria: voice, tone, diction, sentence structure, imagery, metaphor, dialogue, and emotional expression. The extracted styles were encoded into embeddings and stored in a vector database.

**2. RAG Pipeline Setup:**  
The RAG pipeline was configured to query the vector database for the target author's stylistic elements. This setup allowed efficient retrieval and integration of these elements into the generation process.

**3. Style Application:**  
Another Mistral model instance was used to incorporate the retrieved stylistic elements into the source documents, effectively transforming them to mimic the target author's style.

**Framework Components:**
- **Original Input (OI):** Original sample text and metadata.
- **Style Criteria (SC):** Specific stylistic elements to extract and replicate.
- **Style Guidance (SG):** Instructions for perturbing the original text to align with the target author's style.

An example of OI, SC, and SG is provided in the appendix. This concise framework enabled effective and accurate authorship impersonation.

\documentclass{article}
\usepackage{amsmath}

\begin{document}

\[
S' = \text{ApplyStyle}(M, \text{OI}, \text{RetrieveStyle}(V, \text{SC}, \text{SG}))
\]

\end{document}


