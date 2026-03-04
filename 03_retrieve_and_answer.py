"""
BioASQ 13b Pipeline — Step 3: Retrieval + Answer Generation
============================================================
For each test question:
  1. Encode query with MedCPT Query Encoder
  2. Retrieve top-K from papers index + snippets index (parallel)
  3. Hybrid rerank: fuse dense FAISS scores + BM25 scores
  4. Boost Results/Discussion chunks
  5. Pull top-5 most similar training questions as few-shot examples
  6. Format prompt by question type (yes/no, factoid, list, summary)
  7. Call LLM (OpenAI-compatible) and parse structured answer
  8. Write BioASQ-format submission JSON

Usage:
  python 03_retrieve_and_answer.py \
    --test  BioASQ-13b-testset.json \
    --index index/ \
    --train_questions data/training_questions.json \
    --out   submissions/submission.json \
    --model gpt-4o          # or any OpenAI-compatible model
"""

import json
import pickle
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI


# ── Hybrid Retrieval ──────────────────────────────────────────────────────────
DENSE_WEIGHT   = 0.7   # weight for FAISS cosine score
BM25_WEIGHT    = 0.3   # weight for BM25 score
SECTION_BOOST  = {5: 0.15, 4: 0.10, 3: 0.0, 2: 0.0, 1: -0.05}  # added to final score
TOP_K_DENSE    = 50    # candidates from FAISS per source
TOP_K_FINAL    = 8     # chunks passed to LLM


class HybridRetriever:
    def __init__(self, faiss_index, chunks: List[dict], bm25: BM25Okapi, name: str):
        self.index  = faiss_index
        self.chunks = chunks
        self.bm25   = bm25
        self.name   = name
        self.index.hnsw.efSearch = 128   # higher = better recall

    def retrieve(self, query_embedding: np.ndarray, query_text: str, k: int = TOP_K_DENSE) -> List[Tuple[dict, float]]:
        """
        Returns list of (chunk, score) sorted by hybrid score descending.
        """
        # Dense retrieval
        distances, indices = self.index.search(query_embedding, k)
        dense_scores = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:
                dense_scores[idx] = float(dist)   # already cosine similarity (normalised)

        # BM25 retrieval — score all chunks then take top-k
        tokenized_query = query_text.lower().split()
        bm25_raw = self.bm25.get_scores(tokenized_query)

        # Normalise BM25 to [0, 1]
        bm25_max = bm25_raw.max()
        if bm25_max > 0:
            bm25_norm = bm25_raw / bm25_max
        else:
            bm25_norm = bm25_raw

        # Merge scores for dense candidates
        results = []
        for idx, d_score in dense_scores.items():
            b_score   = float(bm25_norm[idx])
            sec_prio  = self.chunks[idx].get("section_priority", 2)
            sec_boost = SECTION_BOOST.get(sec_prio, 0.0)

            hybrid = (DENSE_WEIGHT * d_score) + (BM25_WEIGHT * b_score) + sec_boost
            results.append((self.chunks[idx], hybrid))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


def merge_results(
    paper_results   : List[Tuple[dict, float]],
    snippet_results : List[Tuple[dict, float]],
    top_k           : int = TOP_K_FINAL,
) -> List[dict]:
    """
    Merge results from both indexes, deduplicate by PMID+text prefix,
    return top_k chunks sorted by score.
    """
    all_results = paper_results + snippet_results
    all_results.sort(key=lambda x: x[1], reverse=True)

    seen     = set()
    merged   = []
    for chunk, score in all_results:
        key = (chunk["pmid"], chunk["text"][:80])
        if key not in seen:
            seen.add(key)
            chunk["retrieval_score"] = round(score, 4)
            merged.append(chunk)
        if len(merged) >= top_k:
            break

    return merged


# ── Few-Shot Retrieval ────────────────────────────────────────────────────────
class FewShotRetriever:
    """
    Retrieves the most similar training questions to use as few-shot examples.
    Uses a separate FAISS index built over training question embeddings.
    """
    def __init__(self, questions: List[dict], query_model: SentenceTransformer):
        self.questions = questions
        self.model     = query_model
        self._build_index()

    def _build_index(self):
        texts = [q["body"] for q in self.questions]
        print("  Building few-shot question index...")
        embeddings = self.model.encode(texts, batch_size=128, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=True).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
        self.index.add(embeddings)

    def get_few_shots(self, query: str, q_type: str, k: int = 3) -> List[dict]:
        """Get k most similar training questions of the same type."""
        emb = self.model.encode([query], normalize_embeddings=True,
                                convert_to_numpy=True).astype("float32")
        _, indices = self.index.search(emb, k * 4)   # over-fetch then filter by type

        results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            q = self.questions[idx]
            if q.get("type") == q_type:
                results.append(q)
            if len(results) >= k:
                break

        # If not enough same-type, fill with any type
        if len(results) < k:
            for idx in indices[0]:
                if idx < 0 or self.questions[idx] in results:
                    continue
                results.append(self.questions[idx])
                if len(results) >= k:
                    break

        return results


# ── Prompt Building ───────────────────────────────────────────────────────────
def format_few_shot(q: dict) -> str:
    """Format a training question as a few-shot example."""
    q_type = q.get("type", "summary")
    answer = q.get("ideal_answer", "")
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    exact = q.get("exact_answer", "")

    lines = [f"Question: {q['body']}"]
    if q_type == "yesno":
        lines.append(f"Answer: {exact if exact else answer}")
    elif q_type == "factoid":
        if isinstance(exact, list):
            exact_str = ", ".join(exact[0]) if exact and isinstance(exact[0], list) else str(exact)
        else:
            exact_str = str(exact)
        lines.append(f"Exact answer: {exact_str}")
        lines.append(f"Ideal answer: {answer}")
    elif q_type == "list":
        if isinstance(exact, list):
            items = [i[0] if isinstance(i, list) else i for i in exact]
            exact_str = ", ".join(items)
        else:
            exact_str = str(exact)
        lines.append(f"List: {exact_str}")
        lines.append(f"Ideal answer: {answer}")
    else:  # summary
        lines.append(f"Ideal answer: {answer}")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are a biomedical question answering expert trained on PubMed literature.
You answer questions based ONLY on the provided context passages.
If the context does not contain the answer, say so honestly.
Always be precise and concise. Use medical terminology correctly."""


def build_prompt(question: dict, context_chunks: List[dict], few_shots: List[dict]) -> str:
    q_type = question.get("type", "summary")
    q_body = question["body"]

    # Format context
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        src   = chunk.get("source", "paper")
        sec   = chunk.get("section", "")
        pmid  = chunk.get("pmid", "")
        context_parts.append(f"[{i+1}] (PMID:{pmid}, section:{sec}, source:{src})\n{chunk['text']}")
    context_str = "\n\n".join(context_parts)

    # Format few-shots
    few_shot_str = ""
    if few_shots:
        examples = "\n\n---\n".join(format_few_shot(q) for q in few_shots)
        few_shot_str = f"\n\nHere are similar questions with their answers as examples:\n{examples}\n\nNow answer the following:\n"

    # Type-specific instructions
    type_instructions = {
        "yesno": (
            'Answer with EXACTLY "yes" or "no" first, then provide a brief explanation.\n'
            'Respond in JSON: {"exact_answer": "yes" or "no", "ideal_answer": "..."}'
        ),
        "factoid": (
            "Provide the single most specific answer entity (gene, drug, disease, number, etc.).\n"
            'Respond in JSON: {"exact_answer": ["answer1", "answer2"], "ideal_answer": "..."}\n'
            "exact_answer should be a list of 1-5 synonyms/variants of the answer."
        ),
        "list": (
            "Provide a complete list of all relevant entities that answer the question.\n"
            'Respond in JSON: {"exact_answer": [["item1"], ["item2"], ["item3"]], "ideal_answer": "..."}\n'
            "Each item in exact_answer is a list containing the entity and its synonyms."
        ),
        "summary": (
            "Provide a comprehensive paragraph summary answer (3-5 sentences).\n"
            'Respond in JSON: {"ideal_answer": "..."}'
        ),
    }
    instruction = type_instructions.get(q_type, type_instructions["summary"])

    prompt = f"""{few_shot_str}
CONTEXT PASSAGES:
{context_str}

QUESTION ({q_type.upper()}): {q_body}

INSTRUCTIONS: {instruction}

Respond ONLY with valid JSON. No preamble, no markdown, no explanation outside the JSON."""

    return prompt


# ── LLM Call ──────────────────────────────────────────────────────────────────
def call_llm(system: str, user: str, client: OpenAI, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def parse_answer(raw: str, q_type: str) -> dict:
    """Parse LLM JSON response into BioASQ answer format."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from the string
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except Exception:
                parsed = {}
        else:
            parsed = {}

    ideal  = parsed.get("ideal_answer", parsed.get("answer", "Unable to answer from context."))
    exact  = parsed.get("exact_answer", "")

    result = {"ideal_answer": [ideal] if not isinstance(ideal, list) else ideal}

    if q_type == "yesno":
        result["exact_answer"] = str(exact).lower() if exact else "no"
    elif q_type == "factoid":
        if isinstance(exact, list):
            result["exact_answer"] = [exact] if exact and not isinstance(exact[0], list) else exact
        else:
            result["exact_answer"] = [[str(exact)]] if exact else [["unknown"]]
    elif q_type == "list":
        if isinstance(exact, list):
            result["exact_answer"] = [[i] if not isinstance(i, list) else i for i in exact]
        else:
            result["exact_answer"] = [[str(exact)]] if exact else []

    return result


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",             required=True)
    parser.add_argument("--index",            default="index")
    parser.add_argument("--train_questions",  default="data/training_questions.json")
    parser.add_argument("--out",              default="submissions/submission.json")
    parser.add_argument("--model",            default="gpt-4o")
    parser.add_argument("--api_key",          default=os.getenv("OPENAI_API_KEY", ""))
    args = parser.parse_args()

    print("=" * 60)
    print("BioASQ 13b — Step 3: Retrieval + Answer Generation")
    print("=" * 60)

    idx_dir = Path(args.index)

    # ── Load indexes ──────────────────────────────────────────────────────────
    print("\nLoading indexes...")
    paper_index   = faiss.read_index(str(idx_dir / "papers_hnsw.faiss"))
    snippet_index = faiss.read_index(str(idx_dir / "snippets_hnsw.faiss"))

    with open(idx_dir / "papers_chunks.pkl",   "rb") as f: paper_chunks   = pickle.load(f)
    with open(idx_dir / "snippets_chunks.pkl", "rb") as f: snippet_chunks = pickle.load(f)
    with open(idx_dir / "papers_bm25.pkl",     "rb") as f: paper_bm25     = pickle.load(f)
    with open(idx_dir / "snippets_bm25.pkl",   "rb") as f: snippet_bm25   = pickle.load(f)

    print(f"  Paper chunks   : {len(paper_chunks)}")
    print(f"  Snippet chunks : {len(snippet_chunks)}")

    # ── Load query encoder (DIFFERENT from article encoder!) ─────────────────
    print("\nLoading MedCPT Query Encoder...")
    query_model = SentenceTransformer("ncats/MedCPT-Query-Encoder")

    paper_retriever   = HybridRetriever(paper_index,   paper_chunks,   paper_bm25,   "papers")
    snippet_retriever = HybridRetriever(snippet_index, snippet_chunks, snippet_bm25, "snippets")

    # ── Load training questions for few-shot ──────────────────────────────────
    print("\nLoading training questions for few-shot retrieval...")
    with open(args.train_questions) as f:
        train_questions = json.load(f)
    few_shot_retriever = FewShotRetriever(train_questions, query_model)

    # ── Load test questions ───────────────────────────────────────────────────
    with open(args.test) as f:
        test_data = json.load(f)
    test_questions = test_data["questions"]
    print(f"\nTest questions: {len(test_questions)}")

    # ── LLM client ───────────────────────────────────────────────────────────
    client = OpenAI(api_key=args.api_key)

    # ── Answer each question ──────────────────────────────────────────────────
    answers = []
    errors  = []

    for q in tqdm(test_questions, desc="Answering"):
        try:
            q_text = q["body"]
            q_type = q.get("type", "summary")

            # 1. Encode query
            q_emb = query_model.encode(
                [q_text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

            # 2. Retrieve from both sources
            paper_results   = paper_retriever.retrieve(q_emb, q_text)
            snippet_results = snippet_retriever.retrieve(q_emb, q_text)

            # 3. Merge and deduplicate
            top_chunks = merge_results(paper_results, snippet_results)

            # 4. Get few-shot examples from training
            few_shots = few_shot_retriever.get_few_shots(q_text, q_type, k=3)

            # 5. Build prompt and call LLM
            prompt = build_prompt(q, top_chunks, few_shots)
            raw    = call_llm(SYSTEM_PROMPT, prompt, client, args.model)

            # 6. Parse answer
            parsed = parse_answer(raw, q_type)

            # 7. Build BioASQ-format answer with retrieved document URLs
            doc_urls = list({
                f"http://www.ncbi.nlm.nih.gov/pubmed/{c['pmid']}"
                for c in top_chunks
                if c.get("pmid")
            })

            answer = {
                "id"           : q["id"],
                "type"         : q_type,
                "ideal_answer" : parsed.get("ideal_answer", [""]),
                "documents"    : doc_urls[:10],
                "snippets"     : [
                    {
                        "text"               : c["text"][:500],
                        "document"           : f"http://www.ncbi.nlm.nih.gov/pubmed/{c['pmid']}",
                        "offsetInBeginSection": 0,
                        "offsetInEndSection" : len(c["text"][:500]),
                        "beginSection"       : c.get("section", "abstract"),
                        "endSection"         : c.get("section", "abstract"),
                    }
                    for c in top_chunks[:5]
                    if c.get("pmid")
                ],
            }

            if q_type == "yesno":
                answer["exact_answer"] = parsed.get("exact_answer", "no")
            elif q_type in ("factoid", "list"):
                answer["exact_answer"] = parsed.get("exact_answer", [])

            answers.append(answer)

        except Exception as e:
            errors.append({"id": q.get("id"), "error": str(e)})
            print(f"\n  [ERROR] Question {q.get('id')}: {e}")
            # Add minimal fallback answer so submission is complete
            answers.append({
                "id"           : q["id"],
                "type"         : q.get("type", "summary"),
                "ideal_answer" : ["Unable to retrieve answer."],
                "documents"    : [],
                "snippets"     : [],
            })

    # ── Save submission ───────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission = {"questions": answers}
    with open(args.out, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"\nSubmission saved → {args.out}")
    print(f"  Answered : {len(answers)}")
    print(f"  Errors   : {len(errors)}")

    if errors:
        err_path = Path(args.out).parent / "errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"  Errors logged → {err_path}")


if __name__ == "__main__":
    main()
