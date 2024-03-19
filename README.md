# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic



def filter_lines(input_text):
    # Split the input text into lines
    lines = input_text.split('\n')
    
    # Initialize an empty list to hold the filtered lines
    filtered_lines = []
    
    # Loop through each line
    for line in lines:
        # Check if the line starts with a number followed by a period and a space
        if len(line) > 0 and line[0].isdigit() and line.find('. ') != -1:
            # Extract the part of the line after the number and period
            part_after_number = line.split('. ', 1)[1]
            # Add this part to the list of filtered lines
            filtered_lines.append(part_after_number)
    
    # Return the filtered lines
    return filtered_lines

# Example input
input_text = """The outlet is fkan heie gisndi and the tone fkcks gifha
1. Qwerty
2. Bfisnf
3. fjvirb
Dodneis"""

# Filter the lines
filtered_lines = filter_lines(input_text)

# Print the filtered lines
for line in filtered_lines:
    print(line)
