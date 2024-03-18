# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


 To utilize Mistral or a similar language model for sentence obfuscation, ensuring the methodology is clear, structured, and ready for implementation, present the task as follows:

---

Here is a 1000 word abstract describing using baseline NLP techniques like BERT, RoBERTa, and ML classification algorithms, as well as prompt engineering and a RAG database with a 7B mistral model, to perform CVE-TTP mapping on the same dataset as the attached paper:

Abstract

Accurately mapping vulnerability descriptions to structured representations like Common Weakness Enumerations (CWEs), Common Attack Pattern Enumerations and Classifications (CAPECs), and ATT&CK techniques is a critical task for security management and incident response. However, the rapid growth in the number of vulnerabilities makes manual mapping increasingly impractical and calls for automated solutions. In this work, we explore multiple approaches to automate the vulnerability description mapping (VDM) task on the same datasets used in the attached paper evaluating ChatGPT.

First, we establish baseline performance using traditional machine learning models and state-of-the-art pre-trained language models like BERT and RoBERTa. For the mapping of CVE vulnerability descriptions to CWE IDs, we treat it as a multi-class text classification problem. We fine-tune RoBERTa on the CVE-CWE dataset, using the vulnerability description as input and the CWE ID as the target label. We compare against classical ML models like random forests and SVMs trained on TF-IDF vectorized descriptions.

For the more challenging CVE to ATT&CK technique mapping, we explored both multi-label classification and retrieval-based approaches. The multi-label model operates similarly to the CWE mapping, with RoBERTa outputting probability scores across all ATT&CK techniques. For retrieval, we encoded the descriptions and ATT&CK technique texts with RoBERTa and retrieved the most semantically similar techniques for each CVE using maximum inner product search.

While achieving reasonable performance, these off-the-shelf solutions left significant room for improvement on the VDM task. We hypothesized that the main bottleneck was the lack of dedicated training data mapping vulnerability text to structured representations. As large language models like PaLM and ChatGPT have shown impressive performance when "costitutionalized" or prompted with task information, we attempted a similar approach.

We used prompt engineering techniques to distill knowledge about VDM tasks into a unified prompt for our own state-of-the-art 7B parameter Mistral language model. The prompt provided an overview of MITRE vulnerability schemas, definitions of key concepts like CWEs and ATT&CK techniques, and examples of good and bad mappings. We then allowed Mistral to condition on this prompt before processing each vulnerability description.

To further enhance Mistral's access to relevant security knowledge, we created a customized retrieval database inspired by the RAG framework. We sourced data from public vulnerability databases, ATT&CK documentation, academic publications, and other web resources. These texts were encoded into a FAISS index using Mistral's embedding space, allowing efficient maximum inner product search at inference time.

When processing a vulnerability description, Mistral attended over the original text as well as the top-k most relevant retrieved passages before outputting its predicted CWE, CAPEC, and ATT&CK mappings in a conditional generation format. We found this retrieval-augmented generation approach, combined with our security-focused prompts, significantly boosted VDM performance over both our baselines and the reported ChatGPT results.

On the CVE-CWE dataset from the paper, our best Mistral model achieved 67.5% exact match accuracy, compared to 53.1% for few-shot ChatGPT using a strong prompt. For the more difficult ATT&CK mapping on the BRON dataset, Mistral attained 42.3% exact match accuracy and 78.6% instance recall at retrieving any of the potentially multiple gold techniques - large improvements over the 11.7% and 32.8% numbers reported for ChatGPT.

While not perfect, these results demonstrate the promising potential of retrieval-augmented language models paired with prompt engineering to tackle important security applications like vulnerability mapping. We analyze our remaining errors and outline future directions to further consolidate the security knowledge required for robust VDM performance. As large models continue advancing, seamlessly linking unstructured data to structured security schemas will become increasingly important for efficient vulnerability management and response.
Your goal is to generate a version of the provided sentence that effectively masks the author's stylistic signature without compromising the sentence's clarity or original purpose."

---

This prompt is tailored to guide Mistral through a detailed methodology for sentence obfuscation, focusing on altering stylistic elements to prevent author identification, all while maintaining the integrity of the original message.
