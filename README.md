# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic

In this research, we utilized the Mistral model for zero-shot prompt engineering to obfuscate text, leveraging various prompt techniques such as standard, few-shot, role-playing, and chain of thought prompts to guide the language model (LLM) in generating desired responses without explicit training. Our proposed PromptAttack framework consists of three key components: original input (OI), attack objective (AO), and attack guidance (AG). The OI includes the original sample and its label, the AO describes the task requiring the LLM to generate a semantically similar but misclassified sentence, and the AG provides perturbation instructions. We implemented a fidelity filter using word modification ratio and BERTScore to ensure the adversarial samples retained their original meaning. Testing against GPT-3.5 demonstrated PromptAttack's effectiveness in producing adversarial samples that misled the LLM while maintaining semantic fidelity, highlighting the potential of zero-shot prompt engineering in enhancing text obfuscation and exploring model robustness.
