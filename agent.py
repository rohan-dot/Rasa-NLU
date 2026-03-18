"""
BioASQ 14b - Agentic QA Pipeline (LangGraph)
=============================================
A multi-node LangGraph StateGraph that orchestrates:

1. Query Formulation  — reformulates the question for PubMed search
2. Document Retrieval  — retrieves PubMed articles (Phase A only)
3. Snippet Extraction   — extracts relevant snippets from articles
4. Context Assembly     — builds context from snippets + few-shot examples
5. Answer Generation    — generates exact + ideal answers via Gemma
6. Answer Validation    — validates and fixes formatting

Nodes use the raw OpenAI client (NOT LangChain ChatOpenAI) to avoid
the Gemma/vLLM empty-content bug from tool_choice injection.
"""

import re
import json
import logging
from typing import TypedDict, Optional, Annotated

from langgraph.graph import StateGraph, END

from data_loader import BioASQQuestion, Snippet, TrainingIndex
from llm_client import LLMClient
from retriever import PubMedRetriever, extract_snippets_from_articles, build_search_query
from config import FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)


# ── State Definition ────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State passed between LangGraph nodes."""
    question: BioASQQuestion
    search_query: str
    articles: list            # List of PubMedArticle
    context_text: str         # Assembled context for LLM
    few_shot_prompt: str      # Few-shot examples
    raw_answer: str           # Raw LLM output
    phase: str                # "A", "A+", or "B"
    error: Optional[str]


# ── Prompt Templates ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a biomedical expert assistant specializing in answering questions
from the BioASQ challenge. You provide accurate, evidence-based answers using
the provided context from PubMed articles.

Rules:
- Base your answers strictly on the provided context/snippets.
- For ideal answers: Write a concise paragraph (max 200 words) summarizing the key information.
- For exact answers: Provide precise, short answers as instructed per question type.
- If the context is insufficient, provide the best answer you can based on available evidence.
- Use proper biomedical terminology.
- Do not hallucinate facts not supported by the context."""

QUERY_REFORMULATION_PROMPT = """Given the following biomedical question, generate an optimized
PubMed search query. Extract the key biomedical concepts, gene names, disease names,
drug names, and other important terms. Return ONLY the search query, nothing else.

Question: {question}

Search query:"""


def _build_answer_prompt(question: BioASQQuestion, context: str, few_shot: str) -> str:
    """Build the answer generation prompt based on question type."""

    type_instructions = {
        "yesno": """This is a YES/NO question.
You must provide:
1. EXACT ANSWER: Either "yes" or "no" (lowercase).
2. IDEAL ANSWER: A paragraph (max 200 words) explaining the evidence.

Format your response as:
EXACT: [yes or no]
IDEAL: [your paragraph]""",

        "factoid": """This is a FACTOID question seeking a specific entity name, number, or short expression.
You must provide:
1. EXACT ANSWER: Up to 5 candidate answers, most confident first. Each should be a short
   entity name, number, or expression.
2. IDEAL ANSWER: A paragraph (max 200 words) explaining the answer with evidence.

Format your response as:
EXACT: [answer1] | [answer2] | [answer3]
IDEAL: [your paragraph]""",

        "list": """This is a LIST question seeking multiple entity names or short expressions.
You must provide:
1. EXACT ANSWER: A list of entity names, numbers, or short expressions that together
   constitute the complete answer. Each item should be concise.
2. IDEAL ANSWER: A paragraph (max 200 words) summarizing the information.

Format your response as:
EXACT: [item1] | [item2] | [item3] | ...
IDEAL: [your paragraph]""",

        "summary": """This is a SUMMARY question requiring a paragraph-level answer.
You must provide:
1. IDEAL ANSWER: A paragraph (max 200 words) that comprehensively answers the question
   using information from the provided context.

Format your response as:
IDEAL: [your paragraph]""",
    }

    prompt = f"""{few_shot}

--- CONTEXT (relevant PubMed snippets) ---
{context}
--- END CONTEXT ---

QUESTION ({question.qtype}): {question.body}

INSTRUCTIONS:
{type_instructions.get(question.qtype, type_instructions["summary"])}

YOUR ANSWER:"""

    return prompt


# ── Graph Nodes ─────────────────────────────────────────────────────────

class BioASQAgent:
    """LangGraph-based agent for BioASQ question answering."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Optional[PubMedRetriever] = None,
        training_index: Optional[TrainingIndex] = None,
    ):
        self.llm = llm
        self.retriever = retriever or PubMedRetriever()
        self.training_index = training_index
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph StateGraph."""
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("reformulate_query", self.node_reformulate_query)
        builder.add_node("retrieve_documents", self.node_retrieve_documents)
        builder.add_node("extract_snippets", self.node_extract_snippets)
        builder.add_node("build_context", self.node_build_context)
        builder.add_node("generate_answer", self.node_generate_answer)
        builder.add_node("validate_answer", self.node_validate_answer)

        # Set entry point
        builder.set_entry_point("reformulate_query")

        # Add edges — conditional on phase
        builder.add_conditional_edges(
            "reformulate_query",
            self._route_after_reformulate,
            {
                "retrieve": "retrieve_documents",
                "skip_retrieval": "build_context",
            },
        )
        builder.add_edge("retrieve_documents", "extract_snippets")
        builder.add_edge("extract_snippets", "build_context")
        builder.add_edge("build_context", "generate_answer")
        builder.add_edge("generate_answer", "validate_answer")
        builder.add_edge("validate_answer", END)

        return builder.compile()

    @staticmethod
    def _route_after_reformulate(state: AgentState) -> str:
        """Route based on phase: Phase B already has gold snippets."""
        if state["phase"] == "B":
            return "skip_retrieval"
        return "retrieve"

    # ── Node: Query Reformulation ───────────────────────────────────

    def node_reformulate_query(self, state: AgentState) -> dict:
        """Reformulate the question into a PubMed-optimized search query."""
        question = state["question"]

        if state["phase"] == "B":
            # No retrieval needed; just use the original question for context
            return {"search_query": question.body}

        try:
            reformulated = self.llm.generate(
                prompt=QUERY_REFORMULATION_PROMPT.format(question=question.body),
                max_tokens=100,
                temperature=0.1,
            )
            # Clean up: remove quotes, newlines, take first line
            reformulated = reformulated.strip().strip('"').split('\n')[0]
            logger.info(f"Reformulated query: {reformulated}")
            return {"search_query": reformulated}
        except Exception as e:
            logger.warning(f"Query reformulation failed, using keyword extraction: {e}")
            return {"search_query": build_search_query(question.body)}

    # ── Node: Document Retrieval ────────────────────────────────────

    def node_retrieve_documents(self, state: AgentState) -> dict:
        """Retrieve PubMed articles for the search query."""
        query = state["search_query"]
        question = state["question"]

        articles = self.retriever.search_and_fetch(query)

        # Store document URLs on the question object
        question.retrieved_docs = [a.url for a in articles]

        logger.info(f"Retrieved {len(articles)} articles for Q: {question.id}")
        return {"articles": articles}

    # ── Node: Snippet Extraction ────────────────────────────────────

    def node_extract_snippets(self, state: AgentState) -> dict:
        """Extract relevant snippets from retrieved articles."""
        question = state["question"]
        articles = state.get("articles", [])

        snippets = extract_snippets_from_articles(question.body, articles)
        question.retrieved_snippets = snippets

        logger.info(f"Extracted {len(snippets)} snippets for Q: {question.id}")
        return {}

    # ── Node: Context Assembly ──────────────────────────────────────

    def node_build_context(self, state: AgentState) -> dict:
        """Assemble context from snippets + few-shot examples."""
        question = state["question"]

        # --- Build context from snippets ---
        if state["phase"] == "B" and question.snippets:
            # Use gold snippets for Phase B
            snippets = question.snippets
        else:
            # Use retrieved snippets for Phase A/A+
            snippets = question.retrieved_snippets

        context_parts = []
        for i, s in enumerate(snippets[:10]):
            text = s.text if isinstance(s, Snippet) else str(s)
            context_parts.append(f"[{i+1}] {text}")

        context_text = "\n\n".join(context_parts) if context_parts else "(No relevant snippets found.)"

        # --- Build few-shot examples ---
        few_shot = ""
        if self.training_index:
            examples = self.training_index.retrieve_similar(
                question, top_k=FEW_SHOT_EXAMPLES, same_type=True
            )
            if examples:
                few_shot = "--- FEW-SHOT EXAMPLES ---\n"
                for ex in examples:
                    few_shot += f"\nExample Question ({ex.qtype}): {ex.body}\n"
                    # Show snippets context (first 2)
                    if ex.snippets:
                        for s in ex.snippets[:2]:
                            few_shot += f"  Context snippet: {s.text[:200]}...\n"
                    # Show gold answer
                    if ex.exact_answer is not None:
                        few_shot += f"  Exact answer: {_format_exact_for_display(ex)}\n"
                    if ex.ideal_answer:
                        few_shot += f"  Ideal answer: {ex.ideal_answer[:300]}...\n"
                few_shot += "--- END EXAMPLES ---\n"

        return {"context_text": context_text, "few_shot_prompt": few_shot}

    # ── Node: Answer Generation ─────────────────────────────────────

    def node_generate_answer(self, state: AgentState) -> dict:
        """Generate exact + ideal answers using Gemma via vLLM."""
        question = state["question"]
        context = state["context_text"]
        few_shot = state.get("few_shot_prompt", "")

        prompt = _build_answer_prompt(question, context, few_shot)

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
            logger.error(f"Answer generation failed for Q {question.id}: {e}")
            return {"raw_answer": "", "error": str(e)}

    # ── Node: Answer Validation & Parsing ───────────────────────────

    def node_validate_answer(self, state: AgentState) -> dict:
        """Parse and validate the raw LLM output into structured answers."""
        question = state["question"]
        raw = state.get("raw_answer", "")

        if not raw:
            question.generated_ideal = "No answer could be generated."
            if question.qtype == "yesno":
                question.generated_exact = "yes"
            elif question.qtype in ("factoid", "list"):
                question.generated_exact = []
            return {}

        # Parse EXACT answer
        exact_match = re.search(r'EXACT:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
        if exact_match:
            exact_raw = exact_match.group(1).strip()
            if question.qtype == "yesno":
                question.generated_exact = "yes" if "yes" in exact_raw.lower() else "no"
            elif question.qtype in ("factoid", "list"):
                # Split by pipe or comma
                items = re.split(r'\s*\|\s*', exact_raw)
                items = [i.strip().strip('"').strip("'") for i in items if i.strip()]
                if not items:
                    items = re.split(r'\s*,\s*', exact_raw)
                    items = [i.strip().strip('"').strip("'") for i in items if i.strip()]
                question.generated_exact = items
        else:
            # Fallback: try to extract from the full text
            if question.qtype == "yesno":
                question.generated_exact = "yes" if "yes" in raw.lower()[:50] else "no"
            elif question.qtype in ("factoid", "list"):
                question.generated_exact = []

        # Parse IDEAL answer
        ideal_match = re.search(r'IDEAL:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
        if ideal_match:
            ideal = ideal_match.group(1).strip()
            # Truncate to ~200 words
            words = ideal.split()
            if len(words) > 200:
                ideal = " ".join(words[:200])
            question.generated_ideal = ideal
        else:
            # Use the full response as ideal if no IDEAL tag found
            words = raw.split()
            if len(words) > 200:
                raw = " ".join(words[:200])
            question.generated_ideal = raw

        logger.info(
            f"Validated Q {question.id}: "
            f"exact={'set' if question.generated_exact else 'none'}, "
            f"ideal={len(question.generated_ideal)} chars"
        )
        return {}

    # ── Public Interface ────────────────────────────────────────────

    def answer_question(
        self, question: BioASQQuestion, phase: str = "A+"
    ) -> BioASQQuestion:
        """Run the full pipeline for a single question.

        Args:
            question: The BioASQ question to answer.
            phase: One of "A", "A+", or "B".

        Returns:
            The same question object, populated with generated answers.
        """
        initial_state: AgentState = {
            "question": question,
            "search_query": "",
            "articles": [],
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

    def answer_batch(
        self,
        questions: list[BioASQQuestion],
        phase: str = "A+",
    ) -> list[BioASQQuestion]:
        """Process a batch of questions sequentially.

        Args:
            questions: List of BioASQ questions.
            phase: One of "A", "A+", or "B".

        Returns:
            List of questions with generated answers.
        """
        total = len(questions)
        for i, q in enumerate(questions, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing question {i}/{total}: {q.id} ({q.qtype})")
            logger.info(f"Q: {q.body[:100]}...")
            self.answer_question(q, phase=phase)
            logger.info(f"Done: {q.id}")

        return questions


# ── Helpers ─────────────────────────────────────────────────────────────

def _format_exact_for_display(q: BioASQQuestion) -> str:
    """Format exact answer for few-shot display."""
    ea = q.exact_answer
    if ea is None:
        return "N/A"
    if isinstance(ea, str):
        return ea
    if isinstance(ea, list):
        # Handle nested lists
        flat = []
        for item in ea:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(str(item))
        return " | ".join(flat[:5])
    return str(ea)
