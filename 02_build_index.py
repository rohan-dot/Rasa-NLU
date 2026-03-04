"""
BioASQ 13b Pipeline — Step 2: Index Building
=============================================
Chunks all fetched papers (section-aware for full-text, sliding window
for abstracts), embeds with MedCPT, and builds two FAISS indexes:
  1. papers_index  — chunked paper text (abstract + full-text)
  2. snippets_index — gold training snippets as a separate retrieval source

Also builds a BM25 index over the same chunks for hybrid retrieval.

Outputs:
  - index/papers_hnsw.faiss
  - index/papers_chunks.pkl
  - index/snippets_hnsw.faiss
  - index/snippets_chunks.pkl
  - index/papers_bm25.pkl
  - index/snippets_bm25.pkl

Usage:
  python 02_build_index.py --data data/ --out index/
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ── Section Priority ─────────────────────────────────────────────────────────
# Higher = more likely to contain BioASQ answers
SECTION_PRIORITY = {
    "results"         : 5,
    "discussion"      : 4,
    "conclusions"     : 4,
    "conclusion"      : 4,
    "abstract"        : 3,
    "intro"           : 2,
    "introduction"    : 2,
    "methods"         : 1,
    "materials"       : 1,
    "supplementary"   : 0,
    "references"      : -1,  # excluded
    "body"            : 2,
    "unknown"         : 2,
}

EXCLUDED_SECTIONS = {"references", "acknowledgements", "acknowledgments", "competing interests"}

CHUNK_SIZE    = 180   # tokens approx (we split by words as proxy)
CHUNK_OVERLAP = 30
WORDS_PER_TOKEN = 0.75  # rough conversion


# ── Chunking ─────────────────────────────────────────────────────────────────
def words_to_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sliding window chunker operating on whitespace-split words."""
    words = text.split()
    step  = size - overlap
    chunks = []
    for i in range(0, max(1, len(words) - overlap), step):
        chunk = " ".join(words[i : i + size])
        if len(chunk.strip()) > 30:   # skip tiny fragments
            chunks.append(chunk)
    return chunks


def chunk_paper(paper: dict) -> List[dict]:
    """
    Chunk a paper into retrievable units.
    - Full text: section-aware, exclude reference sections
    - Abstract-only: sliding window over title + abstract
    """
    chunks = []
    pmid   = paper["pmid"]
    title  = paper.get("title", "")

    if paper.get("full_text"):
        # Section-aware chunking
        for section_name, section_text in paper["full_text"].items():
            sec_lower = section_name.lower()

            # Skip irrelevant sections
            if any(excl in sec_lower for excl in EXCLUDED_SECTIONS):
                continue

            priority = SECTION_PRIORITY.get(sec_lower, 2)
            if priority < 0:
                continue

            # Prepend title to first chunk of each section for context
            full_section = f"{title}. {section_text}" if title else section_text
            for chunk_text in words_to_chunks(full_section):
                chunks.append({
                    "text"            : chunk_text,
                    "pmid"            : pmid,
                    "section"         : sec_lower,
                    "section_priority": priority,
                    "year"            : paper.get("year", 0),
                    "mesh_terms"      : paper.get("mesh_terms", []),
                    "source"          : "pmc_fulltext",
                    "title"           : title,
                })
    else:
        # Abstract-only fallback
        abstract = paper.get("abstract", "")
        full_text = f"{title}. {abstract}" if title else abstract
        if not full_text.strip():
            return chunks

        for chunk_text in words_to_chunks(full_text):
            chunks.append({
                "text"            : chunk_text,
                "pmid"            : pmid,
                "section"         : "abstract",
                "section_priority": 3,
                "year"            : paper.get("year", 0),
                "mesh_terms"      : paper.get("mesh_terms", []),
                "source"          : "abstract",
                "title"           : title,
            })

    return chunks


def chunk_all_papers(papers_path: str) -> List[dict]:
    """Load papers.jsonl and chunk all."""
    all_chunks = []
    with open(papers_path) as f:
        for line in tqdm(f, desc="Chunking papers"):
            paper = json.loads(line)
            all_chunks.extend(chunk_paper(paper))
    print(f"  Total paper chunks: {len(all_chunks)}")
    return all_chunks


def load_snippets(snippets_path: str) -> List[dict]:
    """Load gold training snippets — already perfect spans, no rechunking needed."""
    snippets = []
    with open(snippets_path) as f:
        for line in f:
            s = json.loads(line)
            if s.get("text", "").strip():
                snippets.append({
                    "text"            : s["text"].strip(),
                    "pmid"            : s["pmid"],
                    "section"         : s.get("section", "unknown"),
                    "section_priority": SECTION_PRIORITY.get(s.get("section", "unknown").lower(), 2),
                    "year"            : 0,
                    "mesh_terms"      : [],
                    "source"          : "training_snippet",
                    "title"           : "",
                    "source_question" : s.get("source_question", ""),
                    "question_type"   : s.get("question_type", ""),
                    "ideal_answer"    : s.get("ideal_answer", ""),
                })
    print(f"  Total training snippets: {len(snippets)}")
    return snippets


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_chunks(chunks: List[dict], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Embed chunk texts using MedCPT Article Encoder."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via inner product
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


# ── FAISS Index ───────────────────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    """
    Build HNSW index.
    - M=32: neighbors per node (higher = better recall, more memory)
    - efConstruction=200: build-time beam width
    - Inner product (IP) because embeddings are L2-normalised → equivalent to cosine
    """
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    return index


# ── BM25 Index ────────────────────────────────────────────────────────────────
def build_bm25_index(chunks: List[dict]) -> BM25Okapi:
    """Build BM25 index over tokenized chunk texts."""
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


# ── Save ──────────────────────────────────────────────────────────────────────
def save_index(index, chunks, bm25, name: str, out_dir: Path):
    faiss.write_index(index, str(out_dir / f"{name}_hnsw.faiss"))
    with open(out_dir / f"{name}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(out_dir / f"{name}_bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"  Saved {name} index ({len(chunks)} chunks)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="Directory with papers.jsonl etc.")
    parser.add_argument("--out",  default="index", help="Output directory for indexes")
    parser.add_argument("--model", default="ncats/MedCPT-Article-Encoder",
                        help="HuggingFace model for article embedding")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BioASQ 13b — Step 2: Building FAISS + BM25 Indexes")
    print("=" * 60)

    # Load embedding model
    print(f"\nLoading embedding model: {args.model}")
    article_model = SentenceTransformer(args.model)

    # ── Papers Index ──────────────────────────────────────────────────────────
    print("\n[ Papers Index ]")
    paper_chunks = chunk_all_papers(str(Path(args.data) / "papers.jsonl"))

    print("Embedding paper chunks...")
    paper_embeddings = embed_chunks(paper_chunks, article_model)

    print("Building FAISS HNSW index...")
    paper_index = build_faiss_index(paper_embeddings)

    print("Building BM25 index...")
    paper_bm25 = build_bm25_index(paper_chunks)

    save_index(paper_index, paper_chunks, paper_bm25, "papers", out_dir)

    # ── Snippets Index ────────────────────────────────────────────────────────
    print("\n[ Training Snippets Index ]")
    snippet_chunks = load_snippets(str(Path(args.data) / "training_snippets.jsonl"))

    print("Embedding snippet chunks...")
    snippet_embeddings = embed_chunks(snippet_chunks, article_model)

    print("Building FAISS HNSW index...")
    snippet_index = build_faiss_index(snippet_embeddings)

    print("Building BM25 index...")
    snippet_bm25 = build_bm25_index(snippet_chunks)

    save_index(snippet_index, snippet_chunks, snippet_bm25, "snippets", out_dir)

    print("\nStep 2 complete. Run 03_retrieve_and_answer.py next.")


if __name__ == "__main__":
    main()
