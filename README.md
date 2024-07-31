# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic


from langchain.text_splitter import TextSplitter

class DocumentTextSplitter(TextSplitter):
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def split_documents(self, documents: list) -> list:
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def split_text(self, text: str) -> list:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size
        return chunks

# Example usage
splitter = DocumentTextSplitter(chunk_size=1000)

# Assuming `docs` is a list of 72 documents, each containing different C code
docs = ["C code document 1...", "C code document 2...", ..., "C code document 72..."]

chunks = splitter.split_documents(docs)

# Print out chunks for verification
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")
