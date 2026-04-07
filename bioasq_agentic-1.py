#!/usr/bin/env python3
"""
BioASQ Task 14b Phase B — Agentic RAG System
=============================================

The LLM is the AGENT. It doesn't just answer — it controls the entire
retrieval-and-reasoning loop:

  ┌─────────────────────────────────────────────────────────┐
  │                    AGENTIC LOOP                         │
  │                                                         │
  │  1. PLAN    — LLM reads question, plans search strategy │
  │  2. SEARCH  — LLM generates PubMed queries              │
  │  3. RETRIEVE — E-utilities fetches articles              │
  │  4. EVALUATE — LLM judges: enough evidence? relevant?   │
  │  5. REFINE  — if not enough, LLM generates new queries  │
  │       └──→ loop back to step 3 (max 3 iterations)       │
  │  6. ANSWER  — LLM generates answer from all evidence    │
  │  7. VERIFY  — LLM self-checks answer against evidence   │
  │  8. CORRECT — LLM fixes any issues found                │
  │  9. CONSENSUS — best answer from multiple temperatures  │
  └─────────────────────────────────────────────────────────┘

vs simple RAG (what you had):
  search once → dump snippets into prompt → generate answer

Usage:
    # Terminal 1 — start vLLM server:
    vllm serve google/gemma-3-27b-it \
        --port 8000 \
        --max-model-len 8192 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16

    # Terminal 2 — run the agent:
    python bioasq_agentic.py \
        --test-input  BioASQ-task14bPhaseB-testset1.json \
        --training    training13b.json \
        --output      submission_phaseB.json \
        --model       google/gemma-3-27b-it

Requirements:
    pip install vllm transformers torch requests rouge-score
"""

import argparse
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    url: str
    relevance_score: float = 0.0

@dataclass
class Snippet:
    text: str
    document_url: str
    section: str = "abstract"
    offset_begin: int = 0
    offset_end: int = 0
    relevance_score: float = 0.0

@dataclass
class EvidencePool:
    """All evidence gathered for a single question."""
    articles: list[Article] = field(default_factory=list)
    snippets: list[Snippet] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    search_iterations: int = 0

    @property
    def seen_pmids(self) -> set[str]:
        return {a.pmid for a in self.articles}

    def top_snippets(self, n: int = 15) -> list[Snippet]:
        """Return top N snippets by relevance score."""
        ranked = sorted(self.snippets, key=lambda s: s.relevance_score,
                        reverse=True)
        return ranked[:n]

    def snippet_texts(self, n: int = 15) -> list[str]:
        return [s.text for s in self.top_snippets(n)]

    def doc_urls(self) -> list[str]:
        seen = set()
        urls = []
        for a in self.articles:
            if a.url not in seen:
                seen.add(a.url)
                urls.append(a.url)
        return urls


# =====================================================================
# PubMed E-utilities Client (from your code, cleaned up)
# =====================================================================

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedClient:
    """Thin wrapper around NCBI E-utilities."""

    def __init__(self, api_key: str | None = None):
        self.session = requests.Session()
        self.api_key = api_key
        self._last_request = 0.0
        # Rate limit: 3/sec without key, 10/sec with key
        self._min_interval = 0.15 if api_key else 0.35

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def search(self, query: str, max_results: int = 15) -> list[str]:
        """Search PubMed, return PMIDs."""
        self._rate_limit()
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            resp = self.session.get(ESEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            log.warning("PubMed search failed: %s", e)
            return []

    def fetch(self, pmids: list[str]) -> list[Article]:
        """Fetch articles by PMID."""
        if not pmids:
            return []
        self._rate_limit()
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            resp = self.session.get(EFETCH_URL, params=params, timeout=60)
            resp.raise_for_status()
            return self._parse_xml(resp.text)
        except Exception as e:
            log.warning("PubMed fetch failed: %s", e)
            return []

    def _parse_xml(self, xml_text: str) -> list[Article]:
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for elem in root.findall(".//PubmedArticle"):
                pmid_el = elem.find(".//PMID")
                pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

                title_el = elem.find(".//ArticleTitle")
                title = "".join(title_el.itertext()).strip() if title_el is not None else ""

                abstract_parts = []
                for abs_el in elem.findall(".//AbstractText"):
                    label = abs_el.get("Label", "")
                    text = "".join(abs_el.itertext()).strip()
                    if label and text:
                        abstract_parts.append(f"{label}: {text}")
                    elif text:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                if pmid and (title or abstract):
                    articles.append(Article(
                        pmid=pmid, title=title, abstract=abstract,
                        url=f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
                    ))
        except ET.ParseError as e:
            log.warning("XML parse error: %s", e)
        return articles


# =====================================================================
# vLLM Server Client — hits localhost:8000/v1/chat/completions
# =====================================================================
# Start the server first:
#   vllm serve google/gemma-3-27b-it \
#       --port 8000 \
#       --max-model-len 8192 \
#       --tensor-parallel-size 2 \
#       --gpu-memory-utilization 0.90 \
#       --dtype bfloat16
# =====================================================================

class LLMEngine:
    """Calls vLLM's OpenAI-compatible server on localhost."""

    def __init__(self, base_url: str = "http://localhost:8000",
                 model: str = "google/gemma-3-27b-it"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()

        # Verify server is up
        try:
            resp = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            models = resp.json()
            available = [m["id"] for m in models.get("data", [])]
            log.info("vLLM server at %s — models: %s", self.base_url, available)
            if available and self.model not in available:
                log.warning("Model '%s' not found on server, available: %s",
                            self.model, available)
                # Use whatever is available
                self.model = available[0]
                log.info("Using model: %s", self.model)
        except Exception as e:
            log.error("Cannot reach vLLM server at %s: %s", self.base_url, e)
            log.error("Start it with: vllm serve %s --port 8000", self.model)
            sys.exit(1)

    def generate(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.3, stop: list[str] | None = None) -> str:
        """Send a chat completion request to the vLLM server."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
        }
        if stop:
            payload["stop"] = stop

        try:
            resp = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            log.warning("LLM request timed out — retrying...")
            time.sleep(2)
            try:
                resp = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.error("LLM retry failed: %s", e)
                return ""
        except Exception as e:
            log.error("LLM request failed: %s", e)
            return ""


# =====================================================================
# Few-Shot Bank (from training data)
# =====================================================================

class FewShotBank:
    def __init__(self):
        self.examples: dict[str, list[dict]] = {
            "factoid": [], "list": [], "yesno": [], "summary": [],
        }

    def load(self, path: str, max_per_type: int = 40):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("questions", []):
            qtype = q.get("type", "").lower()
            if qtype not in self.examples:
                continue
            snippets = q.get("snippets", [])
            if not snippets:
                continue
            exact = q.get("exact_answer")
            ideal = q.get("ideal_answer")
            if qtype != "summary" and not exact:
                continue
            if not ideal:
                continue

            self.examples[qtype].append({
                "body": q["body"],
                "snippets": [s.get("text", "") for s in snippets[:5]],
                "exact_answer": exact,
                "ideal_answer": ideal if isinstance(ideal, str)
                                else ideal[0] if isinstance(ideal, list) and ideal
                                else "",
            })

        for qtype in self.examples:
            self.examples[qtype] = sorted(
                self.examples[qtype], key=lambda x: len(x["body"])
            )[:max_per_type]
            log.info("  Few-shot %s: %d examples", qtype,
                     len(self.examples[qtype]))

    def get(self, qtype: str, n: int = 2) -> list[dict]:
        exs = self.examples.get(qtype, [])
        short = [e for e in exs if len(" ".join(e["snippets"])) < 1500]
        return (short or exs)[:n]


# =====================================================================
# AGENT PROMPTS — these are the "tools" the LLM uses
# =====================================================================

def prompt_generate_queries(question: str, qtype: str,
                            previous_queries: list[str] | None = None) -> str:
    """Ask the LLM to generate PubMed search queries."""
    prompt = (
        "You are a biomedical information retrieval expert. "
        "Generate 3 PubMed search queries to find articles that can "
        "answer the following question. Each query should target different "
        "aspects or use different terminology (synonyms, related terms). "
        "Use MeSH terms where appropriate.\n\n"
        f"QUESTION TYPE: {qtype}\n"
        f"QUESTION: {question}\n\n"
    )
    if previous_queries:
        prompt += (
            "These queries were already tried but didn't find enough "
            "relevant evidence. Generate DIFFERENT queries:\n"
            + "\n".join(f"  - {q}" for q in previous_queries)
            + "\n\n"
        )
    prompt += (
        "Return exactly 3 queries, one per line, numbered 1-3. "
        "No other text.\n\n"
        "QUERIES:\n"
    )
    return prompt


def prompt_evaluate_evidence(question: str, qtype: str,
                              snippets: list[str]) -> str:
    """Ask the LLM: is this evidence sufficient to answer the question?"""
    snippet_block = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets[:10]))
    prompt = (
        "You are a biomedical expert evaluating search results.\n\n"
        f"QUESTION: {question}\n"
        f"TYPE: {qtype}\n\n"
        f"EVIDENCE FOUND:\n{snippet_block}\n\n"
        "Evaluate:\n"
        "1. Can this question be answered with the evidence above? "
        "(SUFFICIENT / INSUFFICIENT)\n"
        "2. If INSUFFICIENT, what specific information is missing?\n"
        "3. If INSUFFICIENT, suggest 2 alternative search terms.\n\n"
        "EVALUATION:"
    )
    return prompt


def prompt_score_snippets(question: str, snippets: list[str]) -> str:
    """Ask the LLM to score snippet relevance."""
    snippet_block = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets[:20]))
    prompt = (
        "Rate each snippet's relevance to the question on a scale of 0-2:\n"
        "  0 = irrelevant\n"
        "  1 = somewhat relevant\n"
        "  2 = directly answers the question\n\n"
        f"QUESTION: {question}\n\n"
        f"SNIPPETS:\n{snippet_block}\n\n"
        "Return one score per line in format: [N] SCORE\n"
        "SCORES:\n"
    )
    return prompt


def _snippets_block(snippets: list[str], max_chars: int = 4000) -> str:
    block = ""
    for i, s in enumerate(snippets, 1):
        line = f"[{i}] {s.strip()}\n"
        if len(block) + len(line) > max_chars:
            break
        block += line
    return block.strip()


def _format_exact(exact) -> str:
    if isinstance(exact, str):
        return exact
    if isinstance(exact, list):
        if exact and isinstance(exact[0], list):
            return "; ".join([", ".join(x) if isinstance(x, list) else str(x)
                              for x in exact])
        return ", ".join(str(x) for x in exact)
    return str(exact)


def prompt_answer(question: str, qtype: str, snippets: list[str],
                  few_shot: list[dict]) -> str:
    """Build the final answer-generation prompt with few-shot examples."""

    type_instructions = {
        "factoid": (
            "Provide the EXACT ANSWER: a specific entity, number, or short "
            "phrase. Then provide an IDEAL ANSWER: a 2-4 sentence paragraph."
        ),
        "yesno": (
            "Provide the EXACT ANSWER: exactly 'yes' or 'no'. "
            "Then provide an IDEAL ANSWER: a 2-4 sentence explanation."
        ),
        "list": (
            "Provide the EXACT ANSWER: list all correct items, one per line "
            "prefixed with '- '. Be comprehensive. "
            "Then provide an IDEAL ANSWER: a 2-4 sentence paragraph."
        ),
        "summary": (
            "Provide an IDEAL ANSWER: a comprehensive 3-6 sentence paragraph "
            "that directly addresses the question using the evidence."
        ),
    }

    prompt = (
        "You are an expert biomedical question answering system. "
        "Given the evidence snippets below, answer the question precisely. "
        f"{type_instructions.get(qtype, type_instructions['summary'])}\n\n"
    )

    # Few-shot examples
    for ex in few_shot[:2]:
        prompt += "---\n"
        prompt += f"QUESTION: {ex['body']}\n"
        prompt += f"EVIDENCE:\n{_snippets_block(ex['snippets'], 800)}\n"
        if qtype != "summary":
            prompt += f"EXACT_ANSWER: {_format_exact(ex['exact_answer'])}\n"
        prompt += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"

    # Target question
    prompt += "---\n"
    prompt += f"QUESTION: {question}\n"
    prompt += f"EVIDENCE:\n{_snippets_block(snippets)}\n"
    if qtype != "summary":
        prompt += "EXACT_ANSWER:"
    else:
        prompt += "IDEAL_ANSWER:"
    return prompt


def prompt_verify(question: str, qtype: str, snippets: list[str],
                  candidate: str) -> str:
    prompt = (
        "You are a biomedical reviewer. Check this answer against the "
        "evidence. Identify errors, unsupported claims, or missing info.\n\n"
        f"QUESTION: {question}\n"
        f"TYPE: {qtype}\n"
        f"EVIDENCE:\n{_snippets_block(snippets, 3000)}\n\n"
        f"CANDIDATE:\n{candidate}\n\n"
        "Is the answer correct and supported? If not, provide a corrected "
        "version.\n\nCORRECTED_ANSWER:"
    )
    return prompt


# =====================================================================
# Answer Parsers
# =====================================================================

def parse_factoid(response: str) -> tuple[list[str], str]:
    parts = re.split(r'IDEAL_ANSWER\s*:', response, maxsplit=1,
                     flags=re.IGNORECASE)
    if len(parts) == 2:
        exact_raw, ideal = parts[0].strip(), parts[1].strip()
    else:
        lines = response.strip().split("\n")
        exact_raw = lines[0].strip()
        ideal = " ".join(lines[1:]).strip() if len(lines) > 1 else exact_raw

    exact_raw = exact_raw.strip().strip('"\'').rstrip(".")
    for prefix in ["The answer is", "The exact answer is", "Answer:"]:
        if exact_raw.lower().startswith(prefix.lower()):
            exact_raw = exact_raw[len(prefix):].strip()
    return [exact_raw] if exact_raw else ["unknown"], ideal or exact_raw


def parse_yesno(response: str) -> tuple[str, str]:
    parts = re.split(r'IDEAL_ANSWER\s*:', response, maxsplit=1,
                     flags=re.IGNORECASE)
    if len(parts) == 2:
        exact_raw, ideal = parts[0].strip().lower(), parts[1].strip()
    else:
        lines = response.strip().split("\n")
        exact_raw = lines[0].strip().lower()
        ideal = " ".join(lines[1:]).strip() if len(lines) > 1 else ""

    exact_raw = exact_raw.strip().strip('"\'').rstrip(".")
    if "yes" in exact_raw[:20]:
        exact = "yes"
    elif "no" in exact_raw[:20]:
        exact = "no"
    else:
        exact = "yes" if ideal and "yes" in ideal[:50].lower() else "no"
    return exact, ideal or f"The answer is {exact}."


def parse_list(response: str) -> tuple[list[list[str]], str]:
    parts = re.split(r'IDEAL_ANSWER\s*:', response, maxsplit=1,
                     flags=re.IGNORECASE)
    if len(parts) == 2:
        list_raw, ideal = parts[0].strip(), parts[1].strip()
    else:
        chunks = response.strip().split("\n\n")
        list_raw = chunks[0]
        ideal = " ".join(chunks[1:]).strip() if len(chunks) > 1 else ""

    items = []
    for line in list_raw.split("\n"):
        line = re.sub(r'^[\-\*•]\s*', '', line.strip())
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = line.strip().strip('"\'').rstrip(".")
        if line and len(line) > 1:
            items.append([line])
    if not items:
        items = [["unknown"]]
    if not ideal:
        ideal = f"The answer includes: {', '.join(i[0] for i in items)}."
    return items, ideal


def parse_summary(response: str) -> str:
    response = re.sub(r'^(IDEAL_ANSWER\s*:?\s*)', '', response,
                      flags=re.IGNORECASE)
    return response.strip() or "No answer could be generated."


# =====================================================================
# Consensus Functions
# =====================================================================

def consensus_factoid(candidates: list[list[str]]) -> list[str]:
    flat = [c[0].lower().strip() for c in candidates if c]
    if not flat:
        return ["unknown"]
    best = Counter(flat).most_common(1)[0][0]
    for c in candidates:
        if c and c[0].lower().strip() == best:
            return c
    return [best]

def consensus_yesno(candidates: list[str]) -> str:
    return Counter(candidates).most_common(1)[0][0]

def consensus_list(candidates: list[list[list[str]]]) -> list[list[str]]:
    counts: dict[str, list[str]] = {}
    for cand in candidates:
        seen = set()
        for syns in cand:
            key = syns[0].lower().strip() if syns else ""
            if key and key not in seen:
                seen.add(key)
                counts.setdefault(key, syns)
    return list(counts.values()) or [["unknown"]]

def consensus_ideal(candidates: list[str]) -> str:
    if not candidates:
        return ""
    candidates_sorted = sorted(candidates, key=len)
    return candidates_sorted[len(candidates_sorted) // 2]


# =====================================================================
# THE AGENT — orchestrates the full agentic loop
# =====================================================================

class BioASQAgent:
    """
    The core agent. For each question:

    Phase 1 — AGENTIC RETRIEVAL:
      - LLM generates search queries
      - PubMed E-utilities retrieves articles
      - LLM evaluates evidence sufficiency
      - If insufficient: LLM generates refined queries → loop

    Phase 2 — AGENTIC ANSWERING:
      - Multiple generation passes (varying temperature)
      - Pass 1 includes verify → refine cycle
      - Consensus across passes

    This is fundamentally different from simple RAG because the LLM
    CONTROLS the retrieval process — it decides what to search for,
    evaluates what it found, and decides when to stop searching.
    """

    def __init__(self, llm: LLMEngine, pubmed: PubMedClient,
                 few_shot_bank: FewShotBank,
                 num_answer_passes: int = 3,
                 max_retrieval_iterations: int = 3,
                 max_articles_per_search: int = 10):
        self.llm = llm
        self.pubmed = pubmed
        self.bank = few_shot_bank
        self.num_answer_passes = num_answer_passes
        self.max_retrieval_iters = max_retrieval_iterations
        self.max_articles = max_articles_per_search

    def solve(self, question: dict) -> dict:
        """Full agentic pipeline for one question."""
        qid = question["id"]
        body = question["body"]
        qtype = question.get("type", "summary").lower()

        log.info("━━━ Question %s [%s]: %.70s", qid, qtype, body)

        # ── Phase 1: AGENTIC RETRIEVAL ──
        evidence = self._agentic_retrieve(body, qtype, question)

        # ── Phase 2: AGENTIC ANSWERING ──
        result = self._agentic_answer(body, qtype, evidence)
        result["id"] = qid

        # Attach documents and snippets for Phase A format compatibility
        result["documents"] = evidence.doc_urls()[:10]

        log.info("  ✓ exact=%s  ideal=%.60s...",
                 str(result.get("exact_answer", "N/A"))[:50],
                 result.get("ideal_answer", "")[:60])
        return result

    # -----------------------------------------------------------------
    # Phase 1: Agentic Retrieval
    # -----------------------------------------------------------------

    def _agentic_retrieve(self, question: str, qtype: str,
                          raw_question: dict) -> EvidencePool:
        """LLM-driven retrieval loop."""
        pool = EvidencePool()

        # Step 0: Ingest gold snippets if present (Phase B input)
        gold_snippets = raw_question.get("snippets", [])
        for gs in gold_snippets:
            text = gs.get("text", "").strip()
            if text:
                pool.snippets.append(Snippet(
                    text=text,
                    document_url=gs.get("document", ""),
                    section=gs.get("beginSection", "abstract"),
                    offset_begin=gs.get("offsetInBeginSection", 0),
                    offset_end=gs.get("offsetInEndSection", 0),
                    relevance_score=1.0,  # gold = max relevance
                ))

        # Also ingest gold document PMIDs
        for doc_url in raw_question.get("documents", []):
            m = re.search(r'/pubmed/(\d+)', doc_url)
            if m:
                pmid = m.group(1)
                if pmid not in pool.seen_pmids:
                    arts = self.pubmed.fetch([pmid])
                    for a in arts:
                        pool.articles.append(a)
                        self._extract_snippets(a, question, pool)

        log.info("  Gold evidence: %d snippets, %d articles",
                 len(pool.snippets), len(pool.articles))

        # Step 1: LLM generates initial search queries
        for iteration in range(self.max_retrieval_iters):
            pool.search_iterations += 1

            log.info("  Retrieval iteration %d/%d",
                     iteration + 1, self.max_retrieval_iters)

            # Ask LLM to generate queries
            query_prompt = prompt_generate_queries(
                question, qtype, pool.queries_used or None)
            query_response = self.llm.generate(
                query_prompt, max_tokens=256, temperature=0.3)

            # Parse queries from response
            new_queries = self._parse_queries(query_response)
            log.info("    LLM generated queries: %s", new_queries)

            # Execute searches
            for q in new_queries:
                if q in pool.queries_used:
                    continue
                pool.queries_used.append(q)

                pmids = self.pubmed.search(q, self.max_articles)
                new_pmids = [p for p in pmids if p not in pool.seen_pmids]

                if new_pmids:
                    articles = self.pubmed.fetch(new_pmids)
                    for a in articles:
                        pool.articles.append(a)
                        self._extract_snippets(a, question, pool)

            log.info("    Pool: %d articles, %d snippets",
                     len(pool.articles), len(pool.snippets))

            # Ask LLM: is evidence sufficient?
            if len(pool.snippets) >= 5:
                eval_prompt = prompt_evaluate_evidence(
                    question, qtype, pool.snippet_texts(10))
                eval_response = self.llm.generate(
                    eval_prompt, max_tokens=256, temperature=0.2)

                if "SUFFICIENT" in eval_response.upper().split("\n")[0]:
                    log.info("    LLM says: SUFFICIENT — stopping retrieval")
                    break
                else:
                    log.info("    LLM says: INSUFFICIENT — searching more")
            elif iteration == self.max_retrieval_iters - 1:
                log.warning("    Max iterations reached with %d snippets",
                            len(pool.snippets))

        # Score snippets using the LLM
        if len(pool.snippets) > 5:
            self._score_snippets(question, pool)

        log.info("  Final evidence: %d articles, %d snippets, %d searches",
                 len(pool.articles), len(pool.snippets),
                 len(pool.queries_used))
        return pool

    def _extract_snippets(self, article: Article, question: str,
                          pool: EvidencePool):
        """Extract relevant sentences from an article's abstract."""
        if not article.abstract:
            return

        q_tokens = set(
            t.lower() for t in re.findall(r"[A-Za-z0-9\-']+", question)
            if len(t) > 2
        )

        sentences = re.split(r'(?<=[.!?])\s+', article.abstract)
        offset = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 40:
                continue
            begin = article.abstract.find(sent, offset)
            if begin == -1:
                begin = offset
            end = begin + len(sent)
            offset = end

            s_tokens = set(t.lower() for t in re.findall(r"[A-Za-z0-9\-']+", sent))
            overlap = len(q_tokens & s_tokens)
            score = overlap / max(len(q_tokens), 1)

            if score > 0.05:  # At least some keyword overlap
                pool.snippets.append(Snippet(
                    text=sent,
                    document_url=article.url,
                    section="abstract",
                    offset_begin=begin,
                    offset_end=end,
                    relevance_score=score,
                ))

    def _score_snippets(self, question: str, pool: EvidencePool):
        """Use the LLM to re-score snippet relevance."""
        texts = [s.text for s in pool.snippets[:20]]
        score_prompt = prompt_score_snippets(question, texts)
        response = self.llm.generate(score_prompt, max_tokens=256,
                                     temperature=0.1)

        # Parse scores
        for line in response.strip().split("\n"):
            m = re.match(r'\[(\d+)\]\s*(\d)', line)
            if m:
                idx = int(m.group(1)) - 1
                score = int(m.group(2))
                if 0 <= idx < len(pool.snippets):
                    # Blend LLM score with keyword score
                    pool.snippets[idx].relevance_score = (
                        pool.snippets[idx].relevance_score * 0.3 +
                        (score / 2.0) * 0.7
                    )

    def _parse_queries(self, response: str) -> list[str]:
        """Parse numbered queries from LLM response."""
        queries = []
        for line in response.strip().split("\n"):
            line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
            line = line.strip().strip('"')
            if line and len(line) > 5 and len(line) < 200:
                queries.append(line)
        return queries[:3]

    # -----------------------------------------------------------------
    # Phase 2: Agentic Answering
    # -----------------------------------------------------------------

    def _agentic_answer(self, question: str, qtype: str,
                        evidence: EvidencePool) -> dict:
        """Multi-pass answer generation with verification."""
        few_shot = self.bank.get(qtype, n=2)
        snippet_texts = evidence.snippet_texts(15)

        exact_candidates = []
        ideal_candidates = []

        for pass_idx in range(self.num_answer_passes):
            temp = 0.15 + (pass_idx * 0.15)
            log.info("  Answer pass %d/%d (temp=%.2f)",
                     pass_idx + 1, self.num_answer_passes, temp)

            # Generate
            answer_prompt = prompt_answer(question, qtype, snippet_texts,
                                          few_shot)
            raw = self.llm.generate(answer_prompt, max_tokens=768,
                                    temperature=temp)

            # Parse
            if qtype == "factoid":
                exact, ideal = parse_factoid(raw)
                exact_candidates.append(exact)
            elif qtype == "yesno":
                exact, ideal = parse_yesno(raw)
                exact_candidates.append(exact)
            elif qtype == "list":
                exact, ideal = parse_list(raw)
                exact_candidates.append(exact)
            else:
                ideal = parse_summary(raw)

            ideal_candidates.append(ideal)

            # Verify + refine on first pass only
            if pass_idx == 0:
                ver_prompt = prompt_verify(question, qtype, snippet_texts, raw)
                corrected = self.llm.generate(ver_prompt, max_tokens=768,
                                              temperature=0.2)

                if qtype == "factoid":
                    ref_exact, ref_ideal = parse_factoid(corrected)
                    exact_candidates.append(ref_exact)
                    ideal_candidates.append(ref_ideal)
                elif qtype == "yesno":
                    ref_exact, ref_ideal = parse_yesno(corrected)
                    exact_candidates.append(ref_exact)
                    ideal_candidates.append(ref_ideal)
                elif qtype == "list":
                    ref_exact, ref_ideal = parse_list(corrected)
                    exact_candidates.append(ref_exact)
                    ideal_candidates.append(ref_ideal)
                else:
                    ideal_candidates.append(parse_summary(corrected))

        # Consensus
        result = {}
        if qtype == "factoid":
            result["exact_answer"] = consensus_factoid(exact_candidates)
        elif qtype == "yesno":
            result["exact_answer"] = consensus_yesno(exact_candidates)
        elif qtype == "list":
            result["exact_answer"] = consensus_list(exact_candidates)

        result["ideal_answer"] = consensus_ideal(ideal_candidates)

        # Build snippets for submission
        result["snippets"] = []
        for s in evidence.top_snippets(10):
            result["snippets"].append({
                "text": s.text,
                "document": s.document_url,
                "beginSection": s.section,
                "endSection": s.section,
                "offsetInBeginSection": s.offset_begin,
                "offsetInEndSection": s.offset_end,
            })

        return result


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BioASQ 14b Phase B — Agentic RAG System"
    )
    parser.add_argument("--test-input", "-t", required=True,
                        help="Phase B test set JSON")
    parser.add_argument("--training", "-tr", default=None,
                        help="Training JSON for few-shot examples")
    parser.add_argument("--output", "-o", default="submission_phaseB.json")
    parser.add_argument("--vllm-url", default="http://localhost:8000",
                        help="vLLM server URL (default: http://localhost:8000)")
    parser.add_argument("--model", "-m", default="google/gemma-3-27b-it",
                        help="Model name served by vLLM")
    parser.add_argument("--passes", type=int, default=3,
                        help="Answer generation passes")
    parser.add_argument("--retrieval-iterations", type=int, default=3,
                        help="Max agentic retrieval iterations per question")
    parser.add_argument("--articles-per-search", type=int, default=10)
    parser.add_argument("--api-key", default=None,
                        help="NCBI API key (optional, faster rate limit)")
    parser.add_argument("--question-ids", nargs="*", default=None)
    args = parser.parse_args()

    # Load few-shot examples
    bank = FewShotBank()
    if args.training:
        bank.load(args.training)

    # Load test set
    with open(args.test_input) as f:
        test_data = json.load(f)
    questions = test_data.get("questions", [])
    log.info("Loaded %d test questions", len(questions))

    if args.question_ids:
        id_set = set(args.question_ids)
        questions = [q for q in questions if q["id"] in id_set]

    # Initialize components
    llm = LLMEngine(base_url=args.vllm_url, model=args.model)
    pubmed = PubMedClient(api_key=args.api_key)

    agent = BioASQAgent(
        llm=llm,
        pubmed=pubmed,
        few_shot_bank=bank,
        num_answer_passes=args.passes,
        max_retrieval_iterations=args.retrieval_iterations,
        max_articles_per_search=args.articles_per_search,
    )

    # Process
    results = []
    for i, q in enumerate(questions):
        log.info("═══ Question %d / %d ═══", i + 1, len(questions))
        try:
            result = agent.solve(q)
            results.append(result)
        except Exception as e:
            log.error("Failed on %s: %s", q.get("id"), e, exc_info=True)
            results.append({
                "id": q["id"],
                "ideal_answer": "Unable to generate answer.",
            })

    # Clean submission
    for r in results:
        if r.get("exact_answer") is None:
            r.pop("exact_answer", None)

    submission = {"questions": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    types = Counter(q.get("type", "?") for q in questions)
    print(f"\nDone! {len(results)} questions processed")
    print(f"Types: {dict(types)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
