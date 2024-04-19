# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic



 
import pandas as pd

# Assuming df1 and df2 are already loaded and have the 'story_pair' column

# Get the unique 'story_pair' values from df1
unique_pairs_df1 = df1['story_pair'].unique()

# Filter df2 to only include rows where 'story_pair' values are in df1
filtered_df2 = df2[df2['story_pair'].isin(unique_pairs_df1)]

# Optionally, you can drop extra rows if both dataframes should be of equal length and completely matched
# This will remove the extra unmatched rows from df2 to make the lengths identical
if len(filtered_df2) > len(df1):
    filtered_df2 = filtered_df2.iloc[:len(df1)]

# Now you can check the result
print(f"Length of df1: {len(df1)}")
print(f"Filtered length of df2: {len(filtered_df2)}")
