# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic

You are an expert C programmer with deep knowledge of software security vulnerabilities. Your task is to analyze the following C code and insert a vulnerability corresponding to a specific CWE and CVE. Please follow these steps:

1. Identify a location in the code where the vulnerability can be logically inserted.
2. Insert a vulnerability corresponding to the provided CWE.
3. Ensure the code compiles correctly after inserting the vulnerability.

### Provided CWE: [CWE-ID]
### Provided CVE: [CVE-ID]

Here is the original C code:
```c
#include <stdio.h>
#include <string.h>

void vulnerable_function(char *input) {
    char buffer[50];
    // Secure code
    strcpy(buffer, input);
    printf("Buffer content: %s\n", buffer);
}

int main() {
    char user_input[100];
    printf("Enter input: ");
    fgets(user_input, sizeof(user_input), stdin);
    user_input[strcspn(user_input, "\n")] =
