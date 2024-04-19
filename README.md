# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic



 
import json

# Load the JSON files
with open('file1.json', 'r') as file:
    data1 = json.load(file)

with open('file2.json', 'r') as file:
    data2 = json.load(file)

# Assuming data1 and data2 are lists of dictionaries
story_pairs_in_data2 = {item['story_pair'] for item in data2}  # Collect all story_pairs from data2

# Filter to keep items in data1 that have a story_pair that appears in data2
filtered_data1 = [item for item in data1 if item['story_pair'] in story_pairs_in_data2]

# Save the filtered data back to a new JSON file
with open('filtered_file1.json', 'w') as file:
    json.dump(filtered_data1, file, indent=4)

print(f"Kept {len(filtered
