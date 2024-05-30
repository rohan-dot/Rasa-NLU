# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic

### Suggested Title
"Automating CVE to TTP Mapping with Advanced NLP for Government Cybersecurity"

### National Need
- Strengthen cybersecurity defenses against evolving threats.
- Enhance efficiency and accuracy in vulnerability assessments for government systems.

### Problem
- Manual assessments are time-consuming and resource-intensive.
- Incomplete and inaccurate CVE to TTP mapping due to outdated methods.
- Red Teams struggle to keep up with evolving cyber threats, leading to security gaps.
- Traditional methods lack scalability, affecting proactive defense capabilities.

### Methodology in Steps
1. **Data Collection:** Compile CVE datasets with ATT&CK Technique IDs.
2. **Model Integration:** Use Mistral-Instruct with RAG indexing and prompt engineering.
3. **Model Training:** Compare with LLMs and traditional ML models.
4. **Evaluation:** Validate on datasets for accurate CVE to TTP mapping.
5. **Implementation:** Automate the process for efficient Red Team assessments.

### Dataset Description for Poster

We utilize three datasets constructed from the Common Vulnerabilities and Exposures (CVE) database for our research:

- **Dataset I:** Comprising 13,513 CVEs from the year 2021, each linked to a specific Common Weakness Enumeration (CWE) ID, with exclusions for entries without CWE IDs.
- **Dataset II:** Includes 7,013 CVEs from 2021, each associated with a specific ATT&CK Technique ID, sourced from third-party databases such as VulDB.
- **Dataset III:** Consists of 25,439 CVEs annotated with lists of ATT&CK Technique IDs according to the BRON framework, accounting for multiple techniques per vulnerability.

These datasets provide comprehensive and structured data essential for evaluating the effectiveness of our proposed NLP-based CVE to TTP mapping methodology.
