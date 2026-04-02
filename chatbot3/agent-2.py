"""
BioASQ 14b - Agentic QA Pipeline (LangGraph)
=============================================
Hybrid retrieval strategy:
  - Phase A+: BioASQ PubMed service (live search) + FAISS training snippets
  - Phase B:  Gold snippets (provided in test set)

Both phases use FAISS question-level index for few-shot demos.

Pipeline:
  1. retrieve_context  — hybrid PubMed + FAISS (A+) or gold snippets (B)
  2. build_prompt      — few-shot examples + context + type instructions
  3. generate_answer   — Gemma via vLLM (raw OpenAI client)
  4. parse_answer      — extract exact + ideal answers
"""

import re
import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from data_loader import BioASQQuestion, Snippet, TrainingIndex
from llm_client import LLMClient
from bioasq_retriever import BioASQPubMedRetriever
from config import FEW_SHOT_EXAMPLES, SNIPPET_TOP_K

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: BioASQQuestion
    context_text: str
    few_shot_prompt: str
    raw_answer: str
    phase: str
    error: Optional[str]


SYSTEM_PROMPT = """You are a biomedical expert answering questions from the BioASQ challenge.
You provide accurate, evidence-based answers using the provided PubMed snippets.

Rules:
- Base answers strictly on the provided context snippets.
- For ideal answers: concise paragraph, max 200 words.
- For exact answers: precise short answers as instructed per question type.
- Use proper biomedical terminology.
- Do not invent facts not in the context."""


TYPE_INSTRUCTIONS = {
    "yesno": """This is a YES/NO question.
Provide:
1. EXACT: Either "yes" or "no" (lowercase)
2. IDEAL: A paragraph (max 200 words) explaining the evidence

Format:
EXACT: [yes or no]
IDEAL: [paragraph]""",

    "factoid": """This is a FACTOID question seeking a specific entity/number/short expression.
Provide:
1. EXACT: Up to 5 candidate answers separated by |, most confident first
2. IDEAL: A paragraph (max 200 words) with evidence

Format:
EXACT: [answer1] | [answer2] | [answer3]
IDEAL: [paragraph]""",

    "list": """This is a LIST question seeking multiple entities/items.
Provide:
1. EXACT: All answer items separated by |
2. IDEAL: A paragraph (max 200 words) summarizing the information

Format:
EXACT: [item1] | [item2] | [item3] | ...
IDEAL: [paragraph]""",

    "summary": """This is a SUMMARY question requiring a paragraph answer.
Provide:
IDEAL: A paragraph (max 200 words) that comprehensively answers the question

Format:
IDEAL: [paragraph]""",
}


class BioASQAgent:
    """LangGraph agent with hybrid retrieval for BioASQ QA."""

    def __init__(self, llm: LLMClient,
                 training_index: Optional[TrainingIndex] = None,
                 pubmed_retriever: Optional[BioASQPubMedRetriever] = None):
        self.llm = llm
        self.training_index = training_index
        self.pubmed_retriever = pubmed_retriever
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("retrieve_context", self.node_retrieve_context)
        builder.add_node("build_prompt", self.node_build_prompt)
        builder.add_node("generate_answer", self.node_generate_answer)
        builder.add_node("parse_answer", self.node_parse_answer)

        builder.set_entry_point("retrieve_context")
        builder.add_edge("retrieve_context", "build_prompt")
        builder.add_edge("build_prompt", "generate_answer")
        builder.add_edge("generate_answer", "parse_answer")
        builder.add_edge("parse_answer", END)
        return builder.compile()

    # ── Node 1: Hybrid Retrieval ────────────────────────────────────

    def node_retrieve_context(self, state: AgentState) -> dict:
        """Get relevant snippets using hybrid strategy.

        Phase B: gold snippets from test set
        Phase A+: 
          1. BioASQ PubMed service (live search for fresh articles)
          2. FAISS training snippets (backup / supplement)
          3. Merge + deduplicate, keep top-10
        """
        question = state["question"]
        snippets = []

        if state["phase"] == "B" and question.snippets:
            # ── Phase B: gold snippets ──────────────────────────
            snippets = question.snippets
            logger.info(f"Using {len(snippets)} gold snippets")

        else:
            # ── Phase A+: hybrid retrieval ──────────────────────

            # 1) BioASQ PubMed service (live search)
            pubmed_snippets = []
            if self.pubmed_retriever:
                try:
                    doc_urls, raw_snippets = self.pubmed_retriever.search_and_get_snippets(
                        question.body, max_articles=10
                    )
                    question.retrieved_docs = doc_urls
                    # Convert raw dicts to Snippet objects
                    for s in raw_snippets:
                        pubmed_snippets.append(Snippet(
                            text=s["text"],
                            document=s["document"],
                            begin_section=s.get("beginSection", "abstract"),
                            end_section=s.get("endSection", "abstract"),
                            offset_begin=s.get("offsetInBeginSection", 0),
                            offset_end=s.get("offsetInEndSection", 0),
                        ))
                    logger.info(f"PubMed: {len(pubmed_snippets)} snippets from {len(doc_urls)} articles")
                except Exception as e:
                    logger.warning(f"BioASQ PubMed retrieval failed: {e}")

            # 2) FAISS training snippets (supplement)
            faiss_snippets = []
            if self.training_index:
                faiss_snippets = self.training_index.retrieve_snippets(
                    question, top_k=SNIPPET_TOP_K
                )
                logger.info(f"FAISS: {len(faiss_snippets)} training snippets")

            # 3) Merge: PubMed first (fresh & relevant), then FAISS to fill gaps
            seen_texts = set()
            for s in pubmed_snippets + faiss_snippets:
                key = s.text[:80].lower().strip()
                if key not in seen_texts:
                    seen_texts.add(key)
                    snippets.append(s)
                if len(snippets) >= SNIPPET_TOP_K:
                    break

            question.retrieved_snippets = snippets
            logger.info(f"Merged: {len(snippets)} total snippets for Q: {question.id}")

        # Build context string
        parts = []
        for i, s in enumerate(snippets[:10]):
            parts.append(f"[{i+1}] {s.text}")
        context = "\n\n".join(parts) if parts else "(No relevant snippets found.)"

        return {"context_text": context}

    # ── Node 2: Build Prompt ────────────────────────────────────────

    def node_build_prompt(self, state: AgentState) -> dict:
        question = state["question"]
        few_shot = ""

        if self.training_index:
            examples = self.training_index.retrieve_similar_questions(
                question, top_k=FEW_SHOT_EXAMPLES, same_type=True
            )
            if examples:
                few_shot = "--- EXAMPLES OF SIMILAR QUESTIONS WITH ANSWERS ---\n"
                for ex in examples:
                    few_shot += f"\nQ ({ex.qtype}): {ex.body}\n"
                    for s in ex.snippets[:2]:
                        few_shot += f"  Snippet: {s.text[:200]}...\n"
                    if ex.exact_answer is not None:
                        few_shot += f"  EXACT: {_format_exact(ex)}\n"
                    if ex.ideal_answer:
                        few_shot += f"  IDEAL: {ex.ideal_answer[:300]}...\n"
                few_shot += "--- END EXAMPLES ---\n\n"

        return {"few_shot_prompt": few_shot}

    # ── Node 3: Generate Answer ─────────────────────────────────────

    def node_generate_answer(self, state: AgentState) -> dict:
        question = state["question"]
        context = state["context_text"]
        few_shot = state.get("few_shot_prompt", "")

        instructions = TYPE_INSTRUCTIONS.get(question.qtype, TYPE_INSTRUCTIONS["summary"])

        prompt = f"""{few_shot}--- CONTEXT (PubMed snippets) ---
{context}
--- END CONTEXT ---

QUESTION ({question.qtype}): {question.body}

{instructions}

YOUR ANSWER:"""

        try:
            raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT,
                                     max_tokens=1024, temperature=0.3)
            logger.info(f"Generated answer for Q: {question.id} ({len(raw)} chars)")
            return {"raw_answer": raw}
        except Exception as e:
            logger.error(f"Generation failed for Q {question.id}: {e}")
            return {"raw_answer": "", "error": str(e)}

    # ── Node 4: Parse Answer ────────────────────────────────────────

    def node_parse_answer(self, state: AgentState) -> dict:
        question = state["question"]
        raw = state.get("raw_answer", "")

        if not raw:
            question.generated_ideal = "No answer could be generated."
            if question.qtype == "yesno":
                question.generated_exact = "yes"
            elif question.qtype in ("factoid", "list"):
                question.generated_exact = []
            return {}

        # Parse EXACT
        exact_match = re.search(r'EXACT:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
        if exact_match:
            exact_raw = exact_match.group(1).strip()
            if question.qtype == "yesno":
                question.generated_exact = "yes" if "yes" in exact_raw.lower() else "no"
            elif question.qtype in ("factoid", "list"):
                items = re.split(r'\s*\|\s*', exact_raw)
                items = [i.strip().strip('"\'') for i in items if i.strip()]
                if not items:
                    items = re.split(r'\s*,\s*', exact_raw)
                    items = [i.strip().strip('"\'') for i in items if i.strip()]
                question.generated_exact = items
        else:
            if question.qtype == "yesno":
                question.generated_exact = "yes" if "yes" in raw.lower()[:100] else "no"
            elif question.qtype in ("factoid", "list"):
                question.generated_exact = []

        # Parse IDEAL
        ideal_match = re.search(r'IDEAL:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
        ideal = ideal_match.group(1).strip() if ideal_match else raw.strip()
        words = ideal.split()
        if len(words) > 200:
            ideal = " ".join(words[:200])
        question.generated_ideal = ideal

        logger.info(f"Parsed Q {question.id}: exact={'set' if question.generated_exact else 'none'}, "
                     f"ideal={len(question.generated_ideal)} chars")
        return {}

    # ── Public Interface ────────────────────────────────────────────

    def answer_question(self, question, phase="A+"):
        initial_state = {
            "question": question, "context_text": "", "few_shot_prompt": "",
            "raw_answer": "", "phase": phase, "error": None,
        }
        try:
            self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"Pipeline failed for Q {question.id}: {e}")
            question.generated_ideal = "An error occurred during answer generation."
            if question.qtype == "yesno":
                question.generated_exact = "yes"
            elif question.qtype in ("factoid", "list"):
                question.generated_exact = []
        return question

    def answer_batch(self, questions, phase="A+"):
        total = len(questions)
        for i, q in enumerate(questions, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{total}] Q: {q.body[:80]}... ({q.qtype})")
            self.answer_question(q, phase=phase)
        return questions


def _format_exact(q):
    ea = q.exact_answer
    if ea is None:
        return "N/A"
    if isinstance(ea, str):
        return ea
    if isinstance(ea, list):
        flat = []
        for item in ea:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(str(item))
        return " | ".join(flat[:5])
    return str(ea)
