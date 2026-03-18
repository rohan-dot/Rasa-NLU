"""
BioASQ 14b - Data Loader
========================
Loads training data (with gold answers) and test data (questions only).
Builds a FAISS index over training snippets for few-shot retrieval.
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
    snippets: list = field(default_factory=list)      # List[Snippet]
    ideal_answer: Optional[str] = None
    exact_answer: Optional[list] = None
    concepts: list = field(default_factory=list)

    # System-generated fields (populated during pipeline)
    retrieved_docs: list = field(default_factory=list)
    retrieved_snippets: list = field(default_factory=list)
    generated_ideal: Optional[str] = None
    generated_exact: Optional[list] = None


def load_questions(filepath: str, is_training: bool = False) -> list[BioASQQuestion]:
    """Load questions from a BioASQ JSON file.

    Args:
        filepath: Path to the JSON file.
        is_training: If True, also load gold answers, snippets, documents.

    Returns:
        List of BioASQQuestion objects.
    """
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

            # Parse snippets
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

            # Parse ideal answer (training data has it as a list)
            ideal = q.get("ideal_answer", None)
            if isinstance(ideal, list) and len(ideal) > 0:
                bq.ideal_answer = ideal[0]
            elif isinstance(ideal, str):
                bq.ideal_answer = ideal

            # Parse exact answer
            bq.exact_answer = q.get("exact_answer", None)

        questions.append(bq)

    logger.info(f"Loaded {len(questions)} questions (training={is_training})")
    return questions


class TrainingIndex:
    """FAISS-based index over training questions for few-shot retrieval.

    Embeds training question bodies and retrieves the most similar
    training examples for a given test question, providing few-shot
    demonstrations with gold answers.
    """

    def __init__(self, questions: list[BioASQQuestion], embedding_model_name: str):
        self.questions = questions
        self.questions_by_type: dict[str, list[BioASQQuestion]] = {}
        for q in questions:
            self.questions_by_type.setdefault(q.qtype, []).append(q)

        self._index = None
        self._embeddings = None
        self._model = None
        self._model_name = embedding_model_name

    def build(self):
        """Build the FAISS index. Call this once after construction."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "faiss-cpu or sentence-transformers not installed. "
                "Few-shot retrieval will fall back to random selection."
            )
            return

        logger.info(f"Building FAISS index over {len(self.questions)} training questions...")
        self._model = SentenceTransformer(self._model_name)

        texts = [f"query: {q.body}" for q in self.questions]
        self._embeddings = self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        self._embeddings = np.array(self._embeddings, dtype="float32")

        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vecs)
        self._index.add(self._embeddings)
        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    def retrieve_similar(
        self, question: BioASQQuestion, top_k: int = 3, same_type: bool = True
    ) -> list[BioASQQuestion]:
        """Find the most similar training questions for few-shot demos.

        Args:
            question: The test question to find examples for.
            top_k: Number of examples to retrieve.
            same_type: If True, only retrieve examples of the same question type.

        Returns:
            List of similar training questions with gold answers.
        """
        if self._index is None or self._model is None:
            # Fallback: random selection from same type
            pool = self.questions_by_type.get(question.qtype, self.questions)
            import random
            return random.sample(pool, min(top_k, len(pool)))

        query_emb = self._model.encode(
            [f"query: {question.body}"], normalize_embeddings=True
        ).astype("float32")

        # Search wider if we need to filter by type
        search_k = top_k * 10 if same_type else top_k
        scores, indices = self._index.search(query_emb, min(search_k, self._index.ntotal))

        results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            candidate = self.questions[idx]
            if same_type and candidate.qtype != question.qtype:
                continue
            # Skip if it's the exact same question (by ID)
            if candidate.id == question.id:
                continue
            results.append(candidate)
            if len(results) >= top_k:
                break

        return results


def format_submission_phaseA(questions: list[BioASQQuestion]) -> dict:
    """Format system output for Phase A submission."""
    out_questions = []
    for q in questions:
        entry = {
            "id": q.id,
            "type": q.qtype,
            "body": q.body,
            "documents": q.retrieved_docs[:10],
            "snippets": [
                {
                    "document": s.document,
                    "text": s.text,
                    "offsetInBeginSection": s.offset_begin,
                    "offsetInEndSection": s.offset_end,
                    "beginSection": s.begin_section,
                    "endSection": s.end_section,
                }
                for s in q.retrieved_snippets[:10]
            ],
        }
        out_questions.append(entry)
    return {"questions": out_questions}


def format_submission_phaseB(questions: list[BioASQQuestion]) -> dict:
    """Format system output for Phase B (or A+) submission."""
    out_questions = []
    for q in questions:
        entry = {
            "id": q.id,
            "type": q.qtype,
            "body": q.body,
        }

        # Ideal answer (all types)
        if q.generated_ideal:
            entry["ideal_answer"] = q.generated_ideal

        # Exact answer (type-dependent)
        if q.qtype == "yesno" and q.generated_exact is not None:
            entry["exact_answer"] = q.generated_exact  # "yes" or "no"

        elif q.qtype == "factoid" and q.generated_exact is not None:
            # List of lists, each inner list has 1 element, up to 5
            entry["exact_answer"] = [
                [ans] if isinstance(ans, str) else ans
                for ans in q.generated_exact[:5]
            ]

        elif q.qtype == "list" and q.generated_exact is not None:
            # List of lists, each inner list has 1 element
            entry["exact_answer"] = [
                [ans] if isinstance(ans, str) else ans
                for ans in q.generated_exact[:100]
            ]

        # summary type: no exact_answer field

        out_questions.append(entry)
    return {"questions": out_questions}
