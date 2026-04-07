#!/usr/bin/env python3
"""
BioASQ Task 14b Phase B — Agentic Answer Generator
====================================================
Uses vLLM + Gemma (or any HF model) in a multi-pass agentic loop:

  1. RETRIEVE  — gather gold snippets + optionally fetch from local
                 PubMed baseline index
  2. GENERATE  — produce candidate exact + ideal answers
  3. VERIFY    — self-check answers against evidence
  4. REFINE    — improve answers based on verification feedback
  5. CONSENSUS — pick the best answer across passes

Usage:
    python bioasq_phase_b.py \
        --test-input   BioASQ-task14bPhaseB-testset1.json \
        --training     training13b.json \
        --output       submission_phaseB.json \
        --model        google/gemma-2-9b-it \
        --passes       3

Requirements:
    pip install vllm transformers torch
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# Stop-words for query building
STOP_WORDS = set(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in on at to for with by from "
    "and or not but if then so that this these those it its what which who "
    "whom how when where why all each every any some no most more less very "
    "also into about between through during before after above below up down "
    "out off over under again further once here there than too as please "
    "list describe common".split()
)


# =====================================================================
# Baseline Retriever — searches the local PubMed SQLite index
# =====================================================================

class BaselineRetriever:
    """Searches the local PubMed annual baseline index for relevant
    articles, extracts snippet-like passages from their abstracts."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Verify the DB exists and has articles
        import sqlite3
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
        log.info("BaselineRetriever: %s (%d articles indexed)", db_path, count)
        self.article_count = count

    def retrieve(self, question: str, max_articles: int = 10,
                 max_snippets_per_article: int = 3) -> tuple[list[str], list[str]]:
        """
        Search the baseline for articles relevant to the question.

        Returns:
            (document_urls, snippet_texts)
        """
        from pubmed_baseline import search_articles

        # Build search queries from the question
        queries = self._build_queries(question)

        seen_pmids = set()
        all_articles = []

        for q in queries:
            if len(all_articles) >= max_articles:
                break
            results = search_articles(self.db_path, q,
                                      max_results=max_articles)
            for art in results:
                if art["pmid"] not in seen_pmids:
                    seen_pmids.add(art["pmid"])
                    all_articles.append(art)

        all_articles = all_articles[:max_articles]

        doc_urls = []
        snippet_texts = []

        for art in all_articles:
            doc_urls.append(art["url"])
            # Extract relevant sentences from abstract
            if art.get("abstract"):
                sents = self._extract_relevant_sentences(
                    art["abstract"], question,
                    max_sentences=max_snippets_per_article)
                snippet_texts.extend(sents)

        log.info("    Baseline: %d articles, %d snippets for: %.50s...",
                 len(doc_urls), len(snippet_texts), question)
        return doc_urls, snippet_texts

    def fetch_pmid_abstracts(self, pmids: list[str]) -> dict[str, str]:
        """Fetch abstracts for specific PMIDs from the local index."""
        from pubmed_baseline import fetch_by_pmids
        articles = fetch_by_pmids(self.db_path, pmids)
        return {a["pmid"]: a.get("abstract", "") for a in articles}

    def _build_queries(self, question: str) -> list[str]:
        """Turn a question into 1-3 search queries."""
        text = question.strip().rstrip("?").strip()
        tokens = re.findall(r"[A-Za-z0-9α-ωΑ-Ω/\-']+", text)
        keywords = [t for t in tokens
                    if t.lower() not in STOP_WORDS and len(t) > 1]

        queries = []
        if keywords:
            queries.append(" ".join(keywords))
        if len(keywords) > 6:
            queries.append(" ".join(keywords[:6]))
        if len(keywords) > 3:
            queries.append(" ".join(keywords[:3]))
        if not queries:
            queries.append(question)
        return queries

    def _extract_relevant_sentences(self, abstract: str, question: str,
                                     max_sentences: int = 3) -> list[str]:
        """Score and return the most relevant sentences from an abstract."""
        q_tokens = set(
            t.lower() for t in re.findall(r"[A-Za-z0-9\-']+", question)
            if t.lower() not in STOP_WORDS and len(t) > 2
        )

        sentences = re.split(r'(?<=[.!?])\s+', abstract)
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 40:
                continue
            s_tokens = set(t.lower() for t in re.findall(r"[A-Za-z0-9\-']+", sent))
            overlap = len(q_tokens & s_tokens)
            score = overlap / max(len(q_tokens), 1)
            if score > 0:
                scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:max_sentences]]


# =====================================================================
# vLLM Model Wrapper
# =====================================================================

class LLMEngine:
    """Wraps vLLM for batched / single inference."""

    def __init__(self, model_name: str, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.90,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto"):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams

        log.info("Loading model '%s' via vLLM...", model_name)
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model_name = model_name
        log.info("Model loaded.")

    def generate(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.3, top_p: float = 0.95,
                 stop: list[str] | None = None) -> str:
        params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(self, prompts: list[str], max_tokens: int = 1024,
                       temperature: float = 0.3, top_p: float = 0.95,
                       stop: list[str] | None = None) -> list[str]:
        params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text.strip() for o in outputs]


# =====================================================================
# Few-Shot Example Bank — mined from training data
# =====================================================================

class FewShotBank:
    """Stores training examples by question type for few-shot prompting."""

    def __init__(self):
        self.examples: dict[str, list[dict]] = {
            "factoid": [],
            "list": [],
            "yesno": [],
            "summary": [],
        }

    def load_training(self, path: str, max_per_type: int = 50):
        """Load training data and store the best examples per type."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        log.info("Loading training examples from %s (%d questions)",
                 path, len(questions))

        for q in questions:
            qtype = q.get("type", "").lower()
            if qtype not in self.examples:
                continue

            # Only keep questions that have snippets and answers
            snippets = q.get("snippets", [])
            if not snippets:
                continue

            exact = q.get("exact_answer")
            ideal = q.get("ideal_answer")
            if qtype != "summary" and not exact:
                continue
            if not ideal:
                continue

            entry = {
                "body": q["body"],
                "type": qtype,
                "snippets": [s.get("text", "") for s in snippets[:5]],
                "exact_answer": exact,
                "ideal_answer": ideal if isinstance(ideal, str)
                                else ideal[0] if isinstance(ideal, list) and ideal
                                else "",
            }
            self.examples[qtype].append(entry)

        for qtype, exs in self.examples.items():
            # Keep diverse, shorter examples for prompt efficiency
            exs.sort(key=lambda x: len(x["body"]))
            self.examples[qtype] = exs[:max_per_type]
            log.info("  %s: %d examples", qtype, len(self.examples[qtype]))

    def get_few_shot(self, qtype: str, n: int = 2) -> list[dict]:
        """Return n few-shot examples for the given question type."""
        exs = self.examples.get(qtype, [])
        # Pick examples that aren't too long
        short_exs = [e for e in exs if len(" ".join(e["snippets"])) < 1500]
        if not short_exs:
            short_exs = exs
        return short_exs[:n]


# =====================================================================
# Prompt Templates — one per question type
# =====================================================================

def _snippets_block(snippets: list[str], max_chars: int = 4000) -> str:
    """Join snippet texts, truncating to fit context window."""
    block = ""
    for i, s in enumerate(snippets, 1):
        line = f"[{i}] {s.strip()}\n"
        if len(block) + len(line) > max_chars:
            break
        block += line
    return block.strip()


def _format_exact(exact) -> str:
    """Format an exact answer for display in few-shot examples."""
    if isinstance(exact, str):
        return exact
    if isinstance(exact, list):
        if exact and isinstance(exact[0], list):
            # list-of-lists (list questions)
            return "; ".join([", ".join(x) if isinstance(x, list) else str(x)
                              for x in exact])
        return ", ".join(str(x) for x in exact)
    return str(exact)


def build_prompt_factoid(question: str, snippets: list[str],
                         few_shot: list[dict]) -> str:
    prompt = (
        "You are an expert biomedical question answering system. "
        "Given the evidence snippets below, provide the exact answer "
        "to the question. The exact answer should be a specific entity, "
        "number, or short phrase — as concise as possible.\n\n"
        "Then provide an ideal answer: a 2-4 sentence paragraph that "
        "answers the question using the evidence.\n\n"
    )

    # Few-shot examples
    for ex in few_shot:
        prompt += "---\n"
        prompt += f"QUESTION: {ex['body']}\n"
        prompt += f"EVIDENCE:\n{_snippets_block(ex['snippets'], 800)}\n"
        prompt += f"EXACT_ANSWER: {_format_exact(ex['exact_answer'])}\n"
        prompt += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"

    # Target question
    prompt += "---\n"
    prompt += f"QUESTION: {question}\n"
    prompt += f"EVIDENCE:\n{_snippets_block(snippets)}\n"
    prompt += "EXACT_ANSWER:"
    return prompt


def build_prompt_yesno(question: str, snippets: list[str],
                       few_shot: list[dict]) -> str:
    prompt = (
        "You are an expert biomedical question answering system. "
        "Given the evidence snippets below, answer the yes/no question. "
        "First give the exact answer: exactly 'yes' or 'no'. "
        "Then provide an ideal answer: a 2-4 sentence explanation.\n\n"
    )

    for ex in few_shot:
        prompt += "---\n"
        prompt += f"QUESTION: {ex['body']}\n"
        prompt += f"EVIDENCE:\n{_snippets_block(ex['snippets'], 800)}\n"
        exact_str = ex['exact_answer']
        if isinstance(exact_str, list):
            exact_str = exact_str[0] if exact_str else "yes"
        prompt += f"EXACT_ANSWER: {exact_str}\n"
        prompt += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"

    prompt += "---\n"
    prompt += f"QUESTION: {question}\n"
    prompt += f"EVIDENCE:\n{_snippets_block(snippets)}\n"
    prompt += "EXACT_ANSWER:"
    return prompt


def build_prompt_list(question: str, snippets: list[str],
                      few_shot: list[dict]) -> str:
    prompt = (
        "You are an expert biomedical question answering system. "
        "Given the evidence snippets below, list all correct answers "
        "to the question. Provide each answer item on its own line, "
        "prefixed with '- '. Be comprehensive but precise.\n\n"
        "Then provide an ideal answer: a 2-4 sentence paragraph.\n\n"
    )

    for ex in few_shot:
        prompt += "---\n"
        prompt += f"QUESTION: {ex['body']}\n"
        prompt += f"EVIDENCE:\n{_snippets_block(ex['snippets'], 800)}\n"
        prompt += f"EXACT_ANSWER:\n"
        exact = ex['exact_answer']
        if isinstance(exact, list):
            for item in exact:
                if isinstance(item, list):
                    prompt += f"- {', '.join(item)}\n"
                else:
                    prompt += f"- {item}\n"
        else:
            prompt += f"- {exact}\n"
        prompt += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"

    prompt += "---\n"
    prompt += f"QUESTION: {question}\n"
    prompt += f"EVIDENCE:\n{_snippets_block(snippets)}\n"
    prompt += "EXACT_ANSWER:\n"
    return prompt


def build_prompt_summary(question: str, snippets: list[str],
                         few_shot: list[dict]) -> str:
    prompt = (
        "You are an expert biomedical question answering system. "
        "Given the evidence snippets below, write a comprehensive "
        "ideal answer to the question. The answer should be a well-written "
        "paragraph of 3-6 sentences that directly addresses the question "
        "using information from the evidence. Be precise and cite specific "
        "findings from the evidence.\n\n"
    )

    for ex in few_shot:
        prompt += "---\n"
        prompt += f"QUESTION: {ex['body']}\n"
        prompt += f"EVIDENCE:\n{_snippets_block(ex['snippets'], 800)}\n"
        prompt += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"

    prompt += "---\n"
    prompt += f"QUESTION: {question}\n"
    prompt += f"EVIDENCE:\n{_snippets_block(snippets)}\n"
    prompt += "IDEAL_ANSWER:"
    return prompt


# =====================================================================
# Answer Parsers — extract structured answers from LLM output
# =====================================================================

def parse_factoid_response(response: str) -> tuple[list[str], str]:
    """Parse a factoid response into (exact_answers, ideal_answer)."""
    # Split at IDEAL_ANSWER marker if present
    parts = re.split(r'IDEAL_ANSWER\s*:', response, maxsplit=1,
                     flags=re.IGNORECASE)

    if len(parts) == 2:
        exact_raw = parts[0].strip()
        ideal = parts[1].strip()
    else:
        # Try to find the first line as exact, rest as ideal
        lines = response.strip().split("\n")
        exact_raw = lines[0].strip()
        ideal = " ".join(lines[1:]).strip() if len(lines) > 1 else exact_raw

    # Clean exact answer
    exact_raw = exact_raw.strip().strip('"').strip("'").rstrip(".")
    # Remove common prefixes
    for prefix in ["The answer is", "The exact answer is", "Answer:"]:
        if exact_raw.lower().startswith(prefix.lower()):
            exact_raw = exact_raw[len(prefix):].strip()

    exact_list = [exact_raw] if exact_raw else ["unknown"]

    if not ideal:
        ideal = exact_raw

    return exact_list, ideal


def parse_yesno_response(response: str) -> tuple[str, str]:
    """Parse a yes/no response."""
    parts = re.split(r'IDEAL_ANSWER\s*:', response, maxsplit=1,
                     flags=re.IGNORECASE)

    if len(parts) == 2:
        exact_raw = parts[0].strip().lower()
        ideal = parts[1].strip()
    else:
        lines = response.strip().split("\n")
        exact_raw = lines[0].strip().lower()
        ideal = " ".join(lines[1:]).strip() if len(lines) > 1 else ""

    # Determine yes/no
    exact_raw = exact_raw.strip().strip('"').strip("'").rstrip(".")
    if "yes" in exact_raw[:20].lower():
        exact = "yes"
    elif "no" in exact_raw[:20].lower():
        exact = "no"
    else:
        # Fallback: scan the ideal answer
        if ideal and "yes" in ideal[:50].lower():
            exact = "yes"
        else:
            exact = "no"

    if not ideal:
        ideal = f"The answer is {exact}."

    return exact, ideal


def parse_list_response(response: str) -> tuple[list[list[str]], str]:
    """Parse a list response into (list_of_lists, ideal_answer)."""
    parts = re.split(r'IDEAL_ANSWER\s*:', response, maxsplit=1,
                     flags=re.IGNORECASE)

    if len(parts) == 2:
        list_raw = parts[0].strip()
        ideal = parts[1].strip()
    else:
        # Try to split at first paragraph break
        chunks = response.strip().split("\n\n")
        list_raw = chunks[0]
        ideal = " ".join(chunks[1:]).strip() if len(chunks) > 1 else ""

    # Parse list items
    items = []
    for line in list_raw.split("\n"):
        line = line.strip()
        # Remove bullet prefixes
        line = re.sub(r'^[\-\*•]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = line.strip().strip('"').strip("'").rstrip(".")
        if line and len(line) > 1:
            # Each item is a list of synonyms (BioASQ format)
            items.append([line])

    if not items:
        items = [["unknown"]]
    if not ideal:
        ideal = f"The answer includes: {', '.join(i[0] for i in items)}."

    return items, ideal


def parse_summary_response(response: str) -> str:
    """Parse a summary response."""
    # Remove any stray markers
    response = re.sub(r'^(IDEAL_ANSWER\s*:?\s*)', '', response,
                      flags=re.IGNORECASE)
    ideal = response.strip()
    if not ideal:
        ideal = "No answer could be generated from the provided evidence."
    return ideal


# =====================================================================
# Verification Agent — self-checks answers against evidence
# =====================================================================

def build_verification_prompt(question: str, qtype: str,
                              snippets: list[str],
                              candidate_answer: str) -> str:
    """Build a prompt that asks the model to verify its own answer."""
    prompt = (
        "You are a biomedical expert reviewer. Check the following "
        "candidate answer against the evidence and the question. "
        "Identify any errors, unsupported claims, or missing information.\n\n"
        f"QUESTION: {question}\n"
        f"QUESTION TYPE: {qtype}\n"
        f"EVIDENCE:\n{_snippets_block(snippets, 3000)}\n\n"
        f"CANDIDATE ANSWER:\n{candidate_answer}\n\n"
        "VERIFICATION:\n"
        "1. Is the answer factually supported by the evidence? (yes/no)\n"
        "2. Is anything important missing? If so, what?\n"
        "3. Are there any errors or unsupported claims? If so, what?\n"
        "4. Provide a corrected/improved version of the answer.\n\n"
        "CORRECTED_ANSWER:"
    )
    return prompt


def build_refinement_prompt(question: str, qtype: str,
                            snippets: list[str],
                            original_answer: str,
                            verification_feedback: str) -> str:
    """Build a prompt to refine the answer based on verification."""
    prompt = (
        "You are an expert biomedical question answering system. "
        "Improve the following answer based on the reviewer feedback. "
        "Keep the same format. Be precise and evidence-based.\n\n"
        f"QUESTION: {question}\n"
        f"QUESTION TYPE: {qtype}\n"
        f"EVIDENCE:\n{_snippets_block(snippets, 3000)}\n\n"
        f"ORIGINAL ANSWER:\n{original_answer}\n\n"
        f"REVIEWER FEEDBACK:\n{verification_feedback}\n\n"
        "IMPROVED ANSWER:"
    )
    return prompt


# =====================================================================
# Consensus — pick the best answer from multiple passes
# =====================================================================

def consensus_yesno(candidates: list[str]) -> str:
    """Majority vote for yes/no."""
    counter = Counter(candidates)
    return counter.most_common(1)[0][0]


def consensus_factoid(candidates: list[list[str]]) -> list[str]:
    """Pick the most common factoid answer."""
    flat = []
    for c in candidates:
        if c:
            flat.append(c[0].lower().strip())
    if not flat:
        return ["unknown"]
    counter = Counter(flat)
    best = counter.most_common(1)[0][0]
    # Return the original-cased version
    for c in candidates:
        if c and c[0].lower().strip() == best:
            return c
    return [best]


def consensus_list(candidates: list[list[list[str]]]) -> list[list[str]]:
    """Merge list answers — union of items found in majority of passes."""
    if not candidates:
        return [["unknown"]]

    # Count how often each item appears
    item_counts: dict[str, int] = {}
    item_original: dict[str, list[str]] = {}
    for cand in candidates:
        seen = set()
        for item_syns in cand:
            key = item_syns[0].lower().strip() if item_syns else ""
            if key and key not in seen:
                seen.add(key)
                item_counts[key] = item_counts.get(key, 0) + 1
                item_original[key] = item_syns

    # Keep items found in at least 1 pass (union for recall)
    threshold = 1
    result = []
    for key, count in sorted(item_counts.items(), key=lambda x: -x[1]):
        if count >= threshold:
            result.append(item_original[key])

    return result if result else [["unknown"]]


def consensus_ideal(candidates: list[str]) -> str:
    """Pick the longest ideal answer (usually the most comprehensive)."""
    if not candidates:
        return ""
    # Pick the one with medium length (not too short, not too verbose)
    candidates_sorted = sorted(candidates, key=len)
    mid = len(candidates_sorted) // 2
    return candidates_sorted[mid]


# =====================================================================
# Main Agentic Pipeline
# =====================================================================

class BioASQAgent:
    """
    Agentic pipeline for BioASQ Phase B:
      Pass 1: Generate candidate answer
      Pass 2: Verify and refine
      Pass 3: Generate alternative answer (higher temperature)
      Final:  Consensus across passes
    """

    def __init__(self, llm: LLMEngine, few_shot_bank: FewShotBank,
                 num_passes: int = 3,
                 baseline: BaselineRetriever | None = None,
                 baseline_articles: int = 10):
        self.llm = llm
        self.bank = few_shot_bank
        self.num_passes = num_passes
        self.baseline = baseline
        self.baseline_articles = baseline_articles

    def solve_question(self, question: dict) -> dict:
        """Process a single question through the agentic pipeline."""
        qid = question["id"]
        body = question["body"]
        qtype = question.get("type", "summary").lower()
        snippets_raw = question.get("snippets", [])

        # Extract gold snippet texts
        snippet_texts = []
        for s in snippets_raw:
            text = s.get("text", "").strip()
            if text:
                snippet_texts.append(text)

        # --- BASELINE AUGMENTATION ---
        # Also pull PMIDs from gold documents and fetch their abstracts
        gold_pmids = []
        for doc_url in question.get("documents", []):
            m = re.search(r'/pubmed/(\d+)', doc_url)
            if m:
                gold_pmids.append(m.group(1))

        if self.baseline:
            # 1. Fetch abstracts for gold PMIDs from local index
            if gold_pmids:
                pmid_abstracts = self.baseline.fetch_pmid_abstracts(gold_pmids)
                for pmid, abstract in pmid_abstracts.items():
                    if abstract:
                        sents = self.baseline._extract_relevant_sentences(
                            abstract, body, max_sentences=2)
                        for s in sents:
                            if s not in snippet_texts:
                                snippet_texts.append(s)

            # 2. Search baseline for additional relevant articles
            extra_urls, extra_snippets = self.baseline.retrieve(
                body,
                max_articles=self.baseline_articles,
                max_snippets_per_article=2,
            )
            for s in extra_snippets:
                if s not in snippet_texts:
                    snippet_texts.append(s)

        if not snippet_texts:
            log.warning("  No snippets for question %s — using question only",
                        qid)
            snippet_texts = [body]

        log.info("  Question %s [%s]: %.60s... (%d snippets)",
                 qid, qtype, body, len(snippet_texts))

        few_shot = self.bank.get_few_shot(qtype, n=2)

        # ---- Multi-pass generation ----
        exact_candidates = []
        ideal_candidates = []

        for pass_idx in range(self.num_passes):
            temp = 0.2 + (pass_idx * 0.15)  # 0.2, 0.35, 0.5, ...
            log.info("    Pass %d/%d (temp=%.2f)",
                     pass_idx + 1, self.num_passes, temp)

            # Step 1: Generate
            if qtype == "factoid":
                prompt = build_prompt_factoid(body, snippet_texts, few_shot)
                raw = self.llm.generate(prompt, max_tokens=512,
                                        temperature=temp)
                exact, ideal = parse_factoid_response(raw)
                exact_candidates.append(exact)

            elif qtype == "yesno":
                prompt = build_prompt_yesno(body, snippet_texts, few_shot)
                raw = self.llm.generate(prompt, max_tokens=512,
                                        temperature=temp)
                exact, ideal = parse_yesno_response(raw)
                exact_candidates.append(exact)

            elif qtype == "list":
                prompt = build_prompt_list(body, snippet_texts, few_shot)
                raw = self.llm.generate(prompt, max_tokens=768,
                                        temperature=temp)
                exact, ideal = parse_list_response(raw)
                exact_candidates.append(exact)

            elif qtype == "summary":
                prompt = build_prompt_summary(body, snippet_texts, few_shot)
                raw = self.llm.generate(prompt, max_tokens=768,
                                        temperature=temp)
                ideal = parse_summary_response(raw)

            else:
                log.warning("    Unknown question type '%s', treating as "
                            "summary", qtype)
                prompt = build_prompt_summary(body, snippet_texts, few_shot)
                raw = self.llm.generate(prompt, max_tokens=768,
                                        temperature=temp)
                ideal = parse_summary_response(raw)

            # Step 2: Verify (on first pass)
            if pass_idx == 0:
                ver_prompt = build_verification_prompt(
                    body, qtype, snippet_texts, raw)
                feedback = self.llm.generate(ver_prompt, max_tokens=512,
                                             temperature=0.2)

                # Step 3: Refine based on feedback
                ref_prompt = build_refinement_prompt(
                    body, qtype, snippet_texts, raw, feedback)
                refined = self.llm.generate(ref_prompt, max_tokens=768,
                                            temperature=0.2)

                # Parse the refined answer too
                if qtype == "factoid":
                    ref_exact, ref_ideal = parse_factoid_response(refined)
                    exact_candidates.append(ref_exact)
                    ideal_candidates.append(ref_ideal)
                elif qtype == "yesno":
                    ref_exact, ref_ideal = parse_yesno_response(refined)
                    exact_candidates.append(ref_exact)
                    ideal_candidates.append(ref_ideal)
                elif qtype == "list":
                    ref_exact, ref_ideal = parse_list_response(refined)
                    exact_candidates.append(ref_exact)
                    ideal_candidates.append(ref_ideal)
                else:
                    ref_ideal = parse_summary_response(refined)
                    ideal_candidates.append(ref_ideal)

            ideal_candidates.append(ideal)

        # ---- Consensus ----
        result = {"id": qid}

        if qtype == "factoid":
            result["exact_answer"] = consensus_factoid(exact_candidates)
        elif qtype == "yesno":
            result["exact_answer"] = consensus_yesno(exact_candidates)
        elif qtype == "list":
            result["exact_answer"] = consensus_list(exact_candidates)
        # summary has no exact_answer

        result["ideal_answer"] = consensus_ideal(ideal_candidates)

        log.info("    → exact: %s", str(result.get("exact_answer", "N/A"))[:80])
        log.info("    → ideal: %.80s...", result.get("ideal_answer", ""))

        return result


# =====================================================================
# Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BioASQ Task 14b Phase B — Agentic Answer Generator"
    )
    parser.add_argument(
        "--test-input", "-t", required=True,
        help="Phase B test set JSON (questions + gold Phase A responses)"
    )
    parser.add_argument(
        "--training", "-tr", default=None,
        help="Training data JSON (e.g. training13b.json) for few-shot examples"
    )
    parser.add_argument(
        "--output", "-o", default="submission_phaseB.json",
        help="Output submission JSON"
    )
    parser.add_argument(
        "--model", "-m", default="google/gemma-2-9b-it",
        help="HuggingFace model name or local path for vLLM"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Max context length for the model"
    )
    parser.add_argument(
        "--gpu-memory", type=float, default=0.90,
        help="GPU memory utilization (0-1)"
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--dtype", default="auto",
        help="Model dtype (auto, float16, bfloat16)"
    )
    parser.add_argument(
        "--passes", type=int, default=3,
        help="Number of generation passes per question"
    )
    parser.add_argument(
        "--question-ids", nargs="*", default=None,
        help="Process only these question IDs"
    )
    parser.add_argument(
        "--pubmed-db", default=None,
        help="Path to PubMed baseline SQLite index (from pubmed_baseline.py). "
             "If provided, the solver will search the baseline for additional "
             "evidence beyond the gold snippets."
    )
    parser.add_argument(
        "--baseline-articles", type=int, default=10,
        help="Max additional articles to retrieve from baseline per question"
    )
    args = parser.parse_args()

    # Load training data for few-shot
    bank = FewShotBank()
    if args.training:
        bank.load_training(args.training)
    else:
        log.warning("No training data provided — running zero-shot")

    # Initialize baseline retriever if DB provided
    baseline = None
    if args.pubmed_db:
        if not Path(args.pubmed_db).exists():
            log.error("PubMed baseline DB not found: %s", args.pubmed_db)
            log.error("Run pubmed_baseline.py download first!")
            sys.exit(1)
        baseline = BaselineRetriever(args.pubmed_db)

    # Load test set
    with open(args.test_input, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    questions = test_data.get("questions", [])
    log.info("Loaded %d test questions", len(questions))

    # Filter by ID if requested
    if args.question_ids:
        id_set = set(args.question_ids)
        questions = [q for q in questions if q["id"] in id_set]
        log.info("Filtered to %d questions", len(questions))

    # Initialize LLM
    llm = LLMEngine(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=args.tensor_parallel,
        dtype=args.dtype,
    )

    # Create agent
    agent = BioASQAgent(
        llm=llm,
        few_shot_bank=bank,
        num_passes=args.passes,
        baseline=baseline,
        baseline_articles=args.baseline_articles,
    )

    # Process questions
    results = []
    for i, q in enumerate(questions):
        log.info("=== Question %d / %d ===", i + 1, len(questions))
        try:
            result = agent.solve_question(q)
            results.append(result)
        except Exception as e:
            log.error("Failed on question %s: %s", q.get("id"), e)
            # Write a fallback answer
            results.append({
                "id": q["id"],
                "ideal_answer": "Unable to generate answer.",
                "exact_answer": "unknown"
                    if q.get("type") != "summary" else None,
            })

    # Clean up None exact_answers for summary questions
    for r in results:
        if r.get("exact_answer") is None:
            r.pop("exact_answer", None)

    # Write output
    submission = {"questions": results}
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    log.info("Wrote submission to %s", out_path)

    # Summary stats
    types = Counter(q.get("type", "unknown") for q in questions)
    print(f"\nDone! Processed {len(results)} questions")
    print(f"Question types: {dict(types)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
