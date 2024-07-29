# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


from transformers import BertModel, BertTokenizer

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
bert_model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

import torch.nn as nn
import torch

class BertWithAdditionalEmbedding(nn.Module):
    def __init__(self, bert_model, embedding_dim):
        super(BertWithAdditionalEmbedding, self).__init__()
        self.bert = bert_model
        self.additional_embedding = nn.Embedding(bert_model.config.vocab_size, embedding_dim)
        self.linear = nn.Linear(bert_model.config.hidden_size + embedding_dim, 2)  # Assuming binary classification

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls_output = bert_outputs.last_hidden_state[:, 0, :]  # CLS token output
        additional_embeds = self.additional_embedding(input_ids)
        additional_cls_embeds = additional_embeds[:, 0, :]  # CLS token additional embedding
        combined_output = torch.cat((bert_cls_output, additional_cls_embeds), dim=1)
        logits = self.linear(combined_output)
        return logits

# Instantiate the model
embedding_dim = 128  # Example embedding dimension
model = BertWithAdditionalEmbedding(bert_model, embedding_dim)
