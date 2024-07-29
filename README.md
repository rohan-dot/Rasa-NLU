# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


 import torch
from transformers import BertModel, BertTokenizer

class BertWithCustomHead(torch.nn.Module):
    def __init__(self, num_classes):
        super(BertWithCustomHead, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Assuming the tensor_dim is the size of BERT's last hidden state
        tensor_dim = self.bert.config.hidden_size 
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(tensor_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output (corresponding to [CLS] token)
        cls_output = outputs[1]
        # Pass through custom linear layers
        logits = self.linear_relu_stack(cls_output)
        return logits

# Example usage
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertWithCustomHead(num_classes=10)
    
    # Dummy input
    input_text = ["Hello, this is a test sentence."]
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    # Forward pass
    output_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    print(output_logits)
