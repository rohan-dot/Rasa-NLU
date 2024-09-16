# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


import os
import git
import glob
import faiss
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Configuration
GITHUB_REPO_URL = 'https://github.com/your-username/your-c-repo.git'
LOCAL_REPO_PATH = './c_repo'
CHUNK_SIZE = 512  # Number of characters per chunk
EMBEDDING_DIM = 768  # Depends on the embedding model used
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.1'

def clone_repo(repo_url, repo_path):
    if not os.path.exists(repo_path):
        print(f'Cloning repository from {repo_url}...')
        git.Repo.clone_from(repo_url, repo_path)
    else:
        print('Repository already cloned.')

def read_c_files(repo_path):
    print('Reading C files...')
    c_files = glob.glob(os.path.join(repo_path, '**/*.c'), recursive=True)
    file_contents = {}
    for file in c_files:
        with open(file, 'r', encoding='utf-8') as f:
            file_contents[file] = f.read()
    return file_contents

def chunk_code(file_contents):
    print('Chunking code...')
    chunks = []
    metadata = []
    for file_path, content in file_contents.items():
        for i in range(0, len(content), CHUNK_SIZE):
            chunk = content[i:i+CHUNK_SIZE]
            chunks.append(chunk)
            metadata.append({
                'file_path': file_path,
                'chunk_index': i // CHUNK_SIZE
            })
    return chunks, metadata

def embed_chunks(chunks, model_name):
    print('Embedding chunks...')
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings, embedding_dim):
    print('Building FAISS index...')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, index, embeddings, chunks, metadata, model_name, top_k=5):
    print('Embedding query...')
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = []
    for idx in indices[0]:
        retrieved_chunks.append({
            'chunk': chunks[idx],
            'metadata': metadata[idx]
        })
    return retrieved_chunks

def load_llm(model_name):
    print('Loading LLM...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    return tokenizer, model

def generate_answer(question, context, tokenizer, model):
    prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(inputs, max_length=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    # Step 1: Clone the GitHub repository
    clone_repo(GITHUB_REPO_URL, LOCAL_REPO_PATH)

    # Step 2: Read C files from the repository
    file_contents = read_c_files(LOCAL_REPO_PATH)

    # Step 3: Chunk the code
    chunks, metadata = chunk_code(file_contents)

    # Step 4: Embed the chunks
    embeddings = embed_chunks(chunks, EMBEDDING_MODEL_NAME)

    # Step 5: Build the FAISS index
    index = build_faiss_index(embeddings, EMBEDDING_DIM)

    # Step 6: Load the LLM
    tokenizer, model = load_llm(LLM_MODEL_NAME)

    # User interaction loop
    while True:
        question = input('\nEnter your question (or type "exit" to quit): ')
        if question.lower() == 'exit':
            break

        # Optional: Specify a file to look at
        file_filter = input('Specify a file to look at (press Enter to skip): ')
        if file_filter:
            filtered_indices = [i for i, meta in enumerate(metadata) if file_filter in meta['file_path']]
            if not filtered_indices:
                print('No matching files found. Proceeding without file filter.')
        else:
            filtered_indices = None

        # Step 7: Retrieve relevant chunks
        retrieved_chunks = retrieve_relevant_chunks(
            question,
            index if not filtered_indices else faiss.IndexProxy([index.reconstruct(i) for i in filtered_indices]),
            embeddings if not filtered_indices else embeddings[filtered_indices],
            chunks if not filtered_indices else [chunks[i] for i in filtered_indices],
            metadata if not filtered_indices else [metadata[i] for i in filtered_indices],
            EMBEDDING_MODEL_NAME
        )

        # Combine retrieved chunks as context
        context = '\n'.join([chunk['chunk'] for chunk in retrieved_chunks])

        # Step 8: Generate the answer using LLM
        answer = generate_answer(question, context, tokenizer, model)
        print(f'\nAnswer:\n{answer}')

if __name__ == '__main__':
    main()






To create a chunker for a Retrieval-Augmented Generation (RAG) approach that allows you to access and edit an entire codebase using a Large Language Model (LLM), you'll need to follow a systematic process. This involves breaking down your codebase into manageable chunks, indexing them for efficient retrieval, and setting up a pipeline where the LLM can reason about the code and suggest or implement changes. Below is a step-by-step guide to help you achieve this.

---

### **1. Understand Your Objectives**

- **Goal**: Enable an LLM to access, reason about, and edit your entire codebase.
- **Challenges**: Large codebases can't be processed in one go due to context window limitations of LLMs. Therefore, chunking and efficient retrieval are essential.

---

### **2. Chunk the Codebase**

#### **a. Decide on Chunk Granularity**

- **Function-Level**: Each function or method as a chunk.
- **Class-Level**: Entire classes or modules as chunks.
- **File-Level**: Entire files as chunks.

*Choose a granularity that balances context completeness with chunk size limitations.*

#### **b. Implement the Chunker**

- **Traverse the Codebase**: Use a script to walk through all files in your codebase.
- **Extract Chunks**: Parse each file to extract code segments based on your chosen granularity.
- **Store Metadata**: For each chunk, store relevant metadata (e.g., file path, function name).

**Example (Python)**:

```python
import os
import ast

def chunk_codebase(codebase_path):
    chunks = []
    for root, _, files in os.walk(codebase_path):
        for file in files:
            if file.endswith('.py'):  # Adjust for your programming language
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    code = f.read()
                # Parse code into AST and extract functions/classes
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        chunk = ast.get_source_segment(code, node)
                        chunks.append({
                            'content': chunk,
                            'metadata': {
                                'file_path': file_path,
                                'name': node.name,
                                'type': type(node).__name__
                            }
                        })
    return chunks
```

---

### **3. Index the Chunks**

#### **a. Generate Embeddings**

- Use a pre-trained model to convert code chunks into vector embeddings.
- **Models**: OpenAI's text-embedding-ada-002, SentenceTransformers, etc.

**Example**:

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/codebert-base')
model = AutoModel.from_pretrained('sentence-transformers/codebert-base')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()
```

#### **b. Create an Index**

- Use vector databases like **FAISS**, **Pinecone**, or **Weaviate** to index embeddings.

**Example with FAISS**:

```python
import faiss
import numpy as np

embeddings = np.array([get_embedding(chunk['content']) for chunk in chunks])
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```

---

### **4. Implement Retrieval Mechanism**

- **Semantic Search**: Retrieve relevant chunks based on a query.
- **Process**:
  - Embed the query.
  - Search the index for nearest neighbors.

**Example**:

```python
def retrieve_chunks(query, top_k=5):
    query_embedding = get_embedding(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [chunks[i] for i in indices[0]]
```

---

### **5. Integrate with the LLM**

#### **a. Prepare Prompts**

- **Contextual Information**: Include retrieved chunks in the prompt.
- **Instruction**: Clearly state what you want the LLM to do (e.g., find bugs, refactor code).

**Example Prompt**:

```
You are reviewing the following code:

[Code Chunk 1]
[Code Chunk 2]
...

Task: [Your specific task]

Please provide the updated code with explanations.
```

#### **b. Call the LLM API**

- Use an API like OpenAI's GPT-4 or another LLM.

**Example**:

```python
import openai

def get_llm_response(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1500,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']
```

---

### **6. Apply Changes to the Codebase**

#### **a. Parse LLM Output**

- **Code Extraction**: Extract code snippets from the LLM's response.
- **Diff Parsing**: If the LLM provides diffs, parse them to understand changes.

#### **b. Update Codebase**

- **Automated Editing**: Programmatically apply changes to the code files.
- **Version Control**: Use Git to track changes and enable rollbacks if necessary.

**Example**:

```python
def apply_changes(chunk, updated_code):
    file_path = chunk['metadata']['file_path']
    with open(file_path, 'r') as f:
        code = f.read()
    # Replace the old chunk with the updated code
    new_code = code.replace(chunk['content'], updated_code)
    with open(file_path, 'w') as f:
        f.write(new_code)
```

---

### **7. Validate and Test**

- **Static Analysis**: Run linters and static analyzers to catch syntax errors.
- **Automated Tests**: Execute existing test suites to ensure functionality isn't broken.
- **Code Review**: Manually review changes for critical sections.

---

### **8. Iterate and Improve**

- **Feedback Loop**: Use results to fine-tune chunk sizes, retrieval parameters, and prompts.
- **Error Handling**: Implement robust error checking to handle unexpected LLM outputs.

---

### **Additional Considerations**

#### **Context Window Limitations**

- LLMs have a maximum context length. Ensure the combined size of your prompt and code chunks doesn't exceed this limit.
- **Solution**: Retrieve only the most relevant chunks.

#### **Prompt Engineering**

- Be clear and specific in your instructions to the LLM.
- Use system prompts to set the assistant's role if supported.

#### **Ethical and Security Concerns**

- **Data Privacy**: Ensure no sensitive information is exposed.
- **Compliance**: Follow the terms of service of the LLM provider.

#### **Scalability**

- For large codebases, consider distributed processing.
- Cache embeddings and indexes to save time on subsequent runs.

---

### **Example Workflow**

1. **User Query**: "Refactor the logging mechanism in the authentication module to use structured logging."
2. **Retrieve Chunks**: Use the retrieval mechanism to find relevant code in the authentication module.
3. **Prepare Prompt**: Craft a prompt including the retrieved code and the user's request.
4. **Get LLM Response**: Invoke the LLM to get the refactored code.
5. **Apply Changes**: Update the codebase with the new logging mechanism.
6. **Validate**: Run tests to ensure everything works as expected.

---

### **Tools and Libraries**

- **Language Models**: OpenAI GPT-4, Cohere, Anthropic Claude.
- **Vector Databases**: FAISS, Pinecone, Weaviate.
- **Embeddings**: OpenAI Embeddings API, SentenceTransformers.
- **Parsing Code**: AST modules in Python, tree-sitter for multiple languages.
- **Version Control**: GitPython for interacting with Git repositories programmatically.

---

### **Conclusion**

Building a chunker for a RAG approach involves combining code parsing, embedding generation, efficient retrieval, and LLM interaction. By systematically breaking down your codebase and setting up a robust pipeline, you can leverage LLMs to reason about and make changes to your code effectively.

---

**Remember**: Always test the LLM's suggestions in a safe environment before deploying changes to production codebases.
