# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


 import pandas as pd

# Assuming C1 and C2 are pandas Series objects
C1_series = pd.Series(C1)
C2_series = pd.Series(C2)

# Combine the series to ensure consistent encoding
combined_series = pd.concat([C1_series, C2_series], ignore_index=True)

# Factorize the combined series to get unique codes for each label
_, unique_codes = pd.factorize(combined_series)

# Encode the original series using the factorized values
C1_encoded_series = pd.factorize(C1_series)[0]
C2_encoded_series = pd.factorize(C2_series)[0]

# Calculate F1 score (using 'micro' if necessary)
# We check again for binary or multiclass
is_binary = len(np.unique(unique_codes)) <= 2

if is_binary:
    f1_pandas = f1_score(C1_encoded_series, C2_encoded_series)
else:
    f1_pandas = f1_score(C1_encoded_series, C2_encoded_series, average='micro')

f1_pandas






