# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic



 
import pandas as pd

# Assuming df1 and df2 are already loaded DataFrames and 'story_pair' contains lists

# Convert the list in 'story_pair' to a tuple for both DataFrames to make them hashable and comparable
df1['story_pair_tuple'] = df1['story_pair'].apply(tuple)
df2['story_pair_tuple'] = df2['story_pair'].apply(tuple)

# Get the unique 'story_pair_tuple' from df1
unique_pairs_df1 = df1['story_pair_tuple'].unique()

# Filter df2 to only include rows where 'story_pair_tuple' values are in df1
filtered_df2 = df2[df2['story_pair_tuple'].isin(unique_pairs_df1)]

# Optionally, you can drop extra rows if both dataframes should be of equal length and completely matched
if len(filtered_df2) > len(df1):
    filtered_df2 = filtered_df2.iloc[:len(df1)]

# Check the result
print(f"Length of df1: {len(df1)}")
print(f"Filtered length of df2: {len(filtered_df2)}")

# If you no longer need the tuple columns, you can drop them
df1.drop(columns=['
