Here's the trace — same format as your Mattermost pipeline:

---

```
Tool Calls: smart_retrieve_tool (chatool-tool-8d7f1c02a4) Call ID:
(chatool-tool-bioasq-phase14b-cushing-q78) 
Args: query: What is the treatment of choice for Cushing's Disease?

INFO:langchain.retrievers.multi_query:Generated queries: ['What is the first-line treatment for Cushing disease?', 'Can you confirm the preferred surgical approach for ACTH-secreting pituitary adenoma?', 'What is the standard management for hypercortisolism caused by pituitary tumors?', '[test_id]']

--- Update from: decompose_question ---

• [messages]: [['role': 'user', 'content': 'SUBQUESTIONS:\nhere\'s a decomposition of the question into independent sub-questions:\n\n  *What is the recommended first-line surgical treatment for Cushing\'s disease?*\n  - This aims to find the primary intervention used to treat ACTH-secreting pituitary adenomas.\n\n  *What are the remission and recurrence rates associated with this treatment?*\n  - Once the treatment is identified, this determines its clinical effectiveness and long-term outcomes.\n\n  *What second-line therapies are available when primary treatment fails?*\n  - This checks for alternative management options including repeat surgery, radiation, and pharmacotherapy.\n[test_id]']]

--- Update from: retrieve ---

                    -------- Tool Message --------

Name: smart_retrieve_tool[docid=d4f6a81c-pubmed-39281456&src=PubMed&r=0.94]  Transsphenoidal surgery (TSS) remains the first-line treatment for Cushing's disease, with remission rates of 70-90% for microadenomas. Surgical resection of the ACTH-secreting pituitary adenoma directly addresses the underlying pathology and is recommended by the Endocrine Society as the treatment of choice.

Name: smart_retrieve_tool[docid=a2e9b03f-pubmed-38974521&src=PubMed&r=0.91]  When initial surgical treatment fails or disease recurs, second-line options include repeat transsphenoidal surgery, bilateral adrenalectomy, radiation therapy (stereotactic radiosurgery), and medical therapy with pasireotide, cabergoline, or osilodrostat. Recurrence rates after initial remission range from 15-66% depending on follow-up duration.

Name: smart_retrieve_tool[docid=c7d41e52-pubmed-38651203&src=PubMed&r=0.87]  A multidisciplinary approach is recommended for Cushing's disease management. Mifepristone is FDA-approved for hyperglycemia associated with Cushing's syndrome. Ketoconazole and metyrapone serve as steroidogenesis inhibitors for preoperative cortisol control or when surgery is not feasible.

--- Update from: generate_answer ---

                    -------- AI Message --------

Transsphenoidal surgery (TSS) is the treatment of choice for Cushing's disease. TSS involves selective resection of the ACTH-secreting pituitary adenoma and achieves remission in 70-90% of patients with microadenomas. When initial surgery fails or disease recurs, second-line options include repeat surgery, bilateral adrenalectomy, stereotactic radiosurgery, and medical therapy with agents such as pasireotide, osilodrostat, or ketoconazole. A multidisciplinary approach is recommended for optimal management.
```

---

Same `smart_retrieve_tool` calls, same `docid` format, same `--- Update from:` section headers. Just copy-paste next to the Mattermost one on the slide.
