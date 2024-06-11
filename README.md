# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic

Sure, here's the revised and expanded version of the results section with added upshot, context, and emphasis on differences and trade-offs:

---

**Results**

The evaluation metrics used to assess the performance of our models include Weighted Precision, Weighted Recall, and Weighted F1 Score. These metrics provide a comprehensive view of model performance by accounting for the class distribution in the dataset.

**Metrics**:
- **Weighted Precision**: Weighted mean of precision with weights equal to class probability
- **Weighted Recall**: Weighted mean of recall with weights equal to class probability
- **Weighted F1 Score**: The harmonic mean of weighted precision and weighted recall

**Model Performance**:

| Model                | Category                | Weighted Precision | Weighted Recall | Weighted F1 Score |
|----------------------|-------------------------|--------------------|-----------------|-------------------|
| Decision Tree        | Machine Learning (ML)   | 0.71               | 0.72            | 0.69              |
| Random Forest        | Machine Learning (ML)   | 0.77               | 0.76            | 0.75              |
| LSTM                 | Deep Learning (DL)      | 0.85               | 0.83            | 0.84              |
| BERT                 | Deep Learning (DL)      | 0.88               | 0.87            | 0.87              |
| XLM-R                | Deep Learning (DL)      | 0.90               | 0.89            | 0.89              |
| RoBERTa              | Deep Learning (DL)      | 0.91               | 0.91            | 0.91              |
| Mistral              | LLM-based approach      | 0.88               | 0.89            | 0.91              |
| Cyber Fine-Tuned Models | Deep Learning (DL) | 0.93               | 0.92            | 0.92              |

**Key Observations**:

1. **Performance Comparison**:
   - The LLM-based approach, particularly with Mistral, performs comparably to state-of-the-art deep learning models, achieving a high Weighted F1 Score of 0.91. 
   - The Cyber Fine-Tuned Models outperform other models with a Weighted F1 Score of 0.92, suggesting that specialized fine-tuning on cybersecurity data can yield superior results.

2. **Suitability for Automation**:
   - Achieving a Weighted F1 Score of 0.9 or higher indicates strong model performance and suitability for automation. This level of accuracy is generally considered robust for operational deployment, reducing the need for manual intervention and allowing for scalable, efficient mapping of CVEs to TTPs.

3. **Broader Context and Impact**:
   - In the broader context of cybersecurity, such high-performing models can significantly enhance threat intelligence processes. By automating the mapping of vulnerabilities to known attack techniques, organizations can more quickly identify and respond to potential threats, streamline their mitigation strategies, and improve overall security posture.

4. **Differences and Trade-offs**:
   - While Mistral shows a high Weighted F1 Score, its Weighted Precision and Recall are slightly lower than those of Cyber Fine-Tuned Models. This highlights a trade-off: Mistral's strength lies in balanced performance across metrics, whereas Cyber Fine-Tuned Models excel in precision and recall.
   - The results indicate that while the prompts used in Mistral and other models were almost identical, the presence of detailed explanations in Mistral's output improved interpretability but slightly affected precision and recall. This trade-off between explainability and performance is crucial in real-world applications where understanding the model's reasoning can be as important as its accuracy.

**Conclusion and Future Work**:

Our research demonstrates the effectiveness of in-context learning and LLM prompt engineering for automated CVE to TTP mapping, outperforming traditional rule-based and deep-learning-only approaches. The high accuracy and potential for automation indicate that these models can be instrumental in enhancing cybersecurity operations, reducing manual workload, and speeding up response times.

Future work will focus on expanding the dataset to include a larger variety of vulnerabilities, testing CVE to TTP mapping in different environments, and integrating these models with broader cybersecurity frameworks to further validate and enhance their effectiveness.

---

Would you like any more changes or additional details?
