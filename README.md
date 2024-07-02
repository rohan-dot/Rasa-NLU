I'm# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic

To calculate the weighted F1 score, we need to take into account the F1 scores of each class and their respective supports. The formula for the weighted F1 score is:

\[ \text{Weighted F1 Score} = \frac{\sum (\text{support}_i \times \text{F1 score}_i)}{\sum \text{support}_i} \]

From the images, the F1 scores and supports are as follows:

- T1006: F1 score = 0.8739495798319329, support = 64
- T1040: F1 score = 0.0, support = 6
- T1055: F1 score = 0.1111111111111111, support = 17
- T1059: F1 score = 0.8127659574468086, support = 197
- T1068: F1 score = 0.5586592178770949, support = 68
- T1078: F1 score = 0.0, support = 3
- T1083: F1 score = 0.0, support = 2
- T1110: F1 score = 0.15384615384615385, support = 11
- T1202: F1 score = 0.7500000000000001, support = 33
- T1204: F1 score = 0.9090909090909091, support = 6
- T1222: F1 score = 0.5185185185185185, support = 20
- T1499: F1 score = 0.7529411764705882, support = 44
- T1552: F1 score = 0.16666666666666669, support = 11
- T1588: F1 score = 0.0, support = 1
- T1592: F1 score = 0.464, support = 58
- T1600: F1 score = 0.0, support = 8
- T1608: F1 score = 0.7599999999999999, support = 29
- T1611: F1 score = 0.0, support = 3

Let's compute the weighted F1 score using the provided data.

First, let's sum the product of the F1 score and support for each class:

\[ \sum (\text{support}_i \times \text{F1 score}_i) \]

Then, sum all the supports:

\[ \sum \text{support}_i \]

Finally, compute the weighted F1 score:

\[ \text{Weighted F1 Score} = \frac{\sum (\text{support}_i \times \text{F1 score}_i)}{\sum \text{support}_i} \]

Let's perform these calculations.

The weighted F1 score is approximately **0.6690**.
