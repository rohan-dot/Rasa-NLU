# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic



    

    
    import os

# Specify the directory
directory = 'A'

# Initialize an empty list to store file contents
file_contents = []

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .txt file
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        # Open and read the file
        with open(filepath, 'r') as file:
            contents = file.read()
            # Append the contents to the list
            file_contents.append(contents)

# Print the list of file contents
print(file_contents)









import os

# Specify the directory you want to work with
directory = 'A'  # Replace 'A' with your actual directory path if needed

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Construct full file path
    filepath = os.path.join(directory, filename)
    
    # Check if it's a file (and not a directory)
    if os.path.isfile(filepath):
        # Skip files that already end with '.txt'
        if not filename.endswith('.txt'):
            # Create new file name by adding '.txt'
            new_filename = filename + '.txt'
            new_filepath = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(filepath, new_filepath)
            print(f'Renamed "{filename}" to "{new_filename}"')
