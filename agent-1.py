"""
BioASQ 14b - Agentic QA Pipeline (LangGraph)
=============================================
No PubMed API needed. All context comes from:
  - Phase A+: FAISS retrieval over training data snippets
  - Phase B:  Gold snippets provided in test set

Pipeline nodes:
  1. retrieve_context  — FAISS snippet retrieval OR use gold snippets
  2. build_prompt      — assemble few-shot examples + context + instructions
  3. generate_answer   — Gemma via vLLM (raw OpenAI client)
  4. parse_answer      — extract exact + ideal answers from LLM output
"""

import re
import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from data_loader import BioASQQuestion, Snippet, TrainingIndex
from llm_client import LLMClient
from config import FEW_SHOT_EXAMPLES, SNIPPET_TOP_K

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: BioASQQuestion
    context_text: str
    few_shot_prompt: str
    raw_answer: str
    phase: str           # "A+" or "B"
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
    """LangGraph agent for BioASQ QA. No PubMed needed."""

    def __init__(self, llm: LLMClient, training_index: Optional[TrainingIndex] = None):
        self.llm = llm
        self.training_index = training_index
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
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

    # ── Node 1: Retrieve Context ────────────────────────────────────

    def node_retrieve_context(self, state: AgentState) -> dict:
        """Get relevant snippets — from gold (Phase B) or FAISS (Phase A+)."""
        question = state["question"]

        if state["phase"] == "B" and question.snippets:
            # Phase B: use gold snippets directly
            snippets = question.snippets
            logger.info(f"Using {len(snippets)} gold snippets for Q: {question.id}")
        elif self.training_index:
            # Phase A+: FAISS retrieval from training snippets
            snippets = self.training_index.retrieve_snippets(
                question, top_k=SNIPPET_TOP_K
            )
            question.retrieved_snippets = snippets
            logger.info(f"Retrieved {len(snippets)} training snippets for Q: {question.id}")
        else:
            snippets = []
            logger.warning(f"No retrieval available for Q: {question.id}")

        # Build context string
        parts = []
        for i, s in enumerate(snippets[:10]):
            parts.append(f"[{i+1}] {s.text}")
        context = "\n\n".join(parts) if parts else "(No relevant snippets found.)"

        return {"context_text": context}

    # ── Node 2: Build Prompt ────────────────────────────────────────

    def node_build_prompt(self, state: AgentState) -> dict:
        """Assemble few-shot examples + context + type-specific instructions."""
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
                    # Show a couple of snippets for context
                    for s in ex.snippets[:2]:
                        few_shot += f"  Snippet: {s.text[:200]}...\n"
                    # Show gold answers
                    if ex.exact_answer is not None:
                        few_shot += f"  EXACT: {_format_exact(ex)}\n"
                    if ex.ideal_answer:
                        ideal_preview = ex.ideal_answer[:300]
                        few_shot += f"  IDEAL: {ideal_preview}...\n"
                few_shot += "--- END EXAMPLES ---\n\n"

        return {"few_shot_prompt": few_shot}

    # ── Node 3: Generate Answer ─────────────────────────────────────

    def node_generate_answer(self, state: AgentState) -> dict:
        """Generate answer using Gemma via vLLM."""
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
            raw = self.llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=1024,
                temperature=0.3,
            )
            logger.info(f"Generated answer for Q: {question.id} ({len(raw)} chars)")
            return {"raw_answer": raw}
        except Exception as e:
            logger.error(f"Generation failed for Q {question.id}: {e}")
            return {"raw_answer": "", "error": str(e)}

    # ── Node 4: Parse Answer ────────────────────────────────────────

    def node_parse_answer(self, state: AgentState) -> dict:
        """Parse LLM output into exact + ideal answers."""
        question = state["question"]
        raw = state.get("raw_answer", "")

        if not raw:
            question.generated_ideal = "No answer could be generated."
            if question.qtype == "yesno":
                question.generated_exact = "yes"
            elif question.qtype in ("factoid", "list"):
                question.generated_exact = []
            return {}

        # ── Parse EXACT ─────────────────────────────────────────
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
                # Scan first 100 chars for yes/no signal
                question.generated_exact = "yes" if "yes" in raw.lower()[:100] else "no"
            elif question.qtype in ("factoid", "list"):
                question.generated_exact = []

        # ── Parse IDEAL ─────────────────────────────────────────
        ideal_match = re.search(r'IDEAL:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
        if ideal_match:
            ideal = ideal_match.group(1).strip()
        else:
            # Use full output as ideal answer
            ideal = raw.strip()

        # Truncate to 200 words
        words = ideal.split()
        if len(words) > 200:
            ideal = " ".join(words[:200])
        question.generated_ideal = ideal

        logger.info(
            f"Parsed Q {question.id}: "
            f"exact={'set' if question.generated_exact else 'none'}, "
            f"ideal={len(question.generated_ideal)} chars"
        )
        return {}

    # ── Public Interface ────────────────────────────────────────────

    def answer_question(self, question: BioASQQuestion, phase: str = "A+") -> BioASQQuestion:
        """Run the full pipeline for a single question."""
        initial_state: AgentState = {
            "question": question,
            "context_text": "",
            "few_shot_prompt": "",
            "raw_answer": "",
            "phase": phase,
            "error": None,
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

    def answer_batch(self, questions: list[BioASQQuestion], phase: str = "A+") -> list[BioASQQuestion]:
        """Process a batch of questions sequentially."""
        total = len(questions)
        for i, q in enumerate(questions, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{total}] Q: {q.body[:80]}... ({q.qtype})")
            self.answer_question(q, phase=phase)
        return questions


def _format_exact(q: BioASQQuestion) -> str:
    """Format exact answer for few-shot display."""
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
