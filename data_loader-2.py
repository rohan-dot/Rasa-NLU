"""
BioASQ 14b - Data Loader
========================
Loads training/test data and builds TWO FAISS indexes over training data:
  1. Question-level index — for few-shot example retrieval
  2. Snippet-level index  — for context retrieval (replaces PubMed entirely)

Your training set has ~5700 questions with ~50k+ snippets from PubMed
abstracts — that's a massive biomedical knowledge base already.
"""

import json
import logging
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Snippet:
    text: str
    document: str
    begin_section: str = ""
    end_section: str = ""
    offset_begin: int = 0
    offset_end: int = 0


@dataclass
class BioASQQuestion:
    """Unified question representation for all phases."""
    id: str
    body: str
    qtype: str  # "factoid", "list", "yesno", "summary"

    # Gold data (available in training and Phase B)
    documents: list = field(default_factory=list)
    snippets: list = field(default_factory=list)
    ideal_answer: Optional[str] = None
    exact_answer: Optional[list] = None
    concepts: list = field(default_factory=list)

    # System-generated fields
    retrieved_snippets: list = field(default_factory=list)
    generated_ideal: Optional[str] = None
    generated_exact: Optional[list] = None


def load_questions(filepath: str, is_training: bool = False) -> list[BioASQQuestion]:
    """Load questions from a BioASQ JSON file."""
    logger.info(f"Loading questions from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for q in data["questions"]:
        bq = BioASQQuestion(
            id=q["id"],
            body=q["body"],
            qtype=q["type"],
        )

        if is_training or "documents" in q:
            bq.documents = q.get("documents", [])
            bq.concepts = q.get("concepts", [])

            raw_snippets = q.get("snippets", [])
            bq.snippets = [
                Snippet(
                    text=s.get("text", ""),
                    document=s.get("document", ""),
                    begin_section=s.get("beginSection", ""),
                    end_section=s.get("endSection", ""),
                    offset_begin=s.get("offsetInBeginSection", 0),
                    offset_end=s.get("offsetInEndSection", 0),
                )
                for s in raw_snippets
            ]

            ideal = q.get("ideal_answer", None)
            if isinstance(ideal, list) and len(ideal) > 0:
                bq.ideal_answer = ideal[0]
            elif isinstance(ideal, str):
                bq.ideal_answer = ideal

            bq.exact_answer = q.get("exact_answer", None)

        questions.append(bq)

    logger.info(f"Loaded {len(questions)} questions (training={is_training})")
    return questions


class TrainingIndex:
    """Dual FAISS indexes over training data.

    1. Question index  — retrieves similar training Qs for few-shot demos
    2. Snippet index   — retrieves relevant snippets as context (NO PubMed needed)
    """

    def __init__(self, questions: list[BioASQQuestion], embedding_model_name: str):
        self.questions = questions
        self.questions_by_type: dict[str, list[BioASQQuestion]] = {}
        for q in questions:
            self.questions_by_type.setdefault(q.qtype, []).append(q)

        # Collect ALL snippets from training data
        self.all_snippets: list[Snippet] = []
        self.snippet_to_question: list[BioASQQuestion] = []
        for q in questions:
            for s in q.snippets:
                if s.text and len(s.text.strip()) > 20:
                    self.all_snippets.append(s)
                    self.snippet_to_question.append(q)

        logger.info(
            f"Training corpus: {len(questions)} questions, "
            f"{len(self.all_snippets)} snippets"
        )

        self._q_index = None
        self._s_index = None
        self._model = None
        self._model_name = embedding_model_name

    def build(self):
        """Build both FAISS indexes. Call once after construction."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "faiss-cpu or sentence-transformers not installed. "
                "Falling back to random selection."
            )
            return

        logger.info(f"Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)

        # ── Question Index ──────────────────────────────────────────
        logger.info("Building question-level FAISS index...")
        q_texts = [f"query: {q.body}" for q in self.questions]
        q_embs = self._model.encode(
            q_texts, show_progress_bar=True, normalize_embeddings=True,
            batch_size=256,
        ).astype("float32")

        dim = q_embs.shape[1]
        self._q_index = faiss.IndexFlatIP(dim)
        self._q_index.add(q_embs)
        logger.info(f"Question index: {self._q_index.ntotal} vectors")

        # ── Snippet Index ───────────────────────────────────────────
        logger.info(f"Building snippet-level FAISS index ({len(self.all_snippets)} snippets)...")
        s_texts = [f"passage: {s.text}" for s in self.all_snippets]

        batch_size = 512
        all_embs = []
        for i in range(0, len(s_texts), batch_size):
            batch = s_texts[i:i + batch_size]
            emb = self._model.encode(
                batch, show_progress_bar=False, normalize_embeddings=True,
            )
            all_embs.append(emb)
            if (i // batch_size) % 20 == 0:
                logger.info(f"  Encoded {i + len(batch)}/{len(s_texts)} snippets...")

        s_embs = np.vstack(all_embs).astype("float32")
        self._s_index = faiss.IndexFlatIP(dim)
        self._s_index.add(s_embs)
        logger.info(f"Snippet index: {self._s_index.ntotal} vectors")

    def retrieve_similar_questions(
        self, question: BioASQQuestion, top_k: int = 3, same_type: bool = True
    ) -> list[BioASQQuestion]:
        """Find similar training questions for few-shot demonstrations."""
        if self._q_index is None or self._model is None:
            pool = self.questions_by_type.get(question.qtype, self.questions)
            import random
            return random.sample(pool, min(top_k, len(pool)))

        query_emb = self._model.encode(
            [f"query: {question.body}"], normalize_embeddings=True
        ).astype("float32")

        search_k = top_k * 10 if same_type else top_k
        scores, indices = self._q_index.search(
            query_emb, min(search_k, self._q_index.ntotal)
        )

        results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            candidate = self.questions[idx]
            if same_type and candidate.qtype != question.qtype:
                continue
            if candidate.id == question.id:
                continue
            results.append(candidate)
            if len(results) >= top_k:
                break
        return results

    def retrieve_snippets(
        self, question: BioASQQuestion, top_k: int = 10
    ) -> list[Snippet]:
        """Retrieve relevant training snippets for a question.

        This is the core retrieval — uses the training data's ~50k+ snippets
        as the knowledge base instead of PubMed.
        """
        if self._s_index is None or self._model is None:
            # Fallback: get snippets from similar questions
            similar_qs = self.retrieve_similar_questions(
                question, top_k=5, same_type=False
            )
            snippets = []
            for q in similar_qs:
                snippets.extend(q.snippets[:3])
            return snippets[:top_k]

        query_emb = self._model.encode(
            [f"query: {question.body}"], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self._s_index.search(query_emb, top_k * 2)

        results = []
        seen = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            snippet = self.all_snippets[idx]
            text_key = snippet.text[:100].lower()
            if text_key in seen:
                continue
            seen.add(text_key)
            results.append(snippet)
            if len(results) >= top_k:
                break

        logger.info(
            f"Retrieved {len(results)} snippets (top score: {scores[0][0]:.3f})"
        )
        return results


def format_submission(questions: list[BioASQQuestion]) -> dict:
    """Format system output for BioASQ Phase A+ / Phase B submission."""
    out_questions = []
    for q in questions:
        entry = {
            "id": q.id,
            "type": q.qtype,
            "body": q.body,
        }

        if q.generated_ideal:
            entry["ideal_answer"] = q.generated_ideal

        if q.qtype == "yesno" and q.generated_exact is not None:
            entry["exact_answer"] = q.generated_exact

        elif q.qtype == "factoid" and q.generated_exact is not None:
            entry["exact_answer"] = [
                [ans] if isinstance(ans, str) else ans
                for ans in q.generated_exact[:5]
            ]

        elif q.qtype == "list" and q.generated_exact is not None:
            entry["exact_answer"] = [
                [ans] if isinstance(ans, str) else ans
                for ans in q.generated_exact[:100]
            ]

        out_questions.append(entry)
    return {"questions": out_questions}
