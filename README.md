# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load your fine-tuned BERT model
model_path = "path/to/your/fine-tuned/model"
model = BertForSequenceClassification.from_pretrained(model_path)

# Remove the last layer (classification head)
model.classifier = torch.nn.Identity()

# Save the modified model
model.save_pretrained("path/to/save/modified/model")

# Optionally, you can also save the tokenizer if needed
tokenizer = BertTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained("path/to/save/modified/model")
