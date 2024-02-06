# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


from sklearn.metrics import f1_score
import numpy as np

# Original columns
C1 = ['t1056', 't1256']
C2 = ['t1006', 't1256']

# Combine the values to find unique labels
all_labels = set(C1 + C2)

# Create a mapping from label to integer
label_to_int = {label: i for i, label in enumerate(all_labels)}

# Encode the original labels using the mapping
C1_encoded = [label_to_int[label] for label in C1]
C2_encoded = [label_to_int[label] for label in C2]

# Calculate F1 score
# Since F1 score is typically calculated for binary classification, we need to ensure it's applicable here.
# If there are more than 2 classes, we may need to calculate F1 score in a different manner (e.g., micro, macro)

# Verify if binary or multiclass
is_binary = len(all_labels) <= 2

if is_binary:
    # For binary classification, we directly calculate F1 score
    f1 = f1_score(C1_encoded, C2_encoded)
else:
    # For multiclass, calculate F1 score with 'micro' to aggregate the contributions of all classes
    f1 = f1_score(C1_encoded, C2_encoded, average='micro')

f1, label_to_int





