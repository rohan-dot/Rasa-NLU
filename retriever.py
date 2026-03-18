"""
BioASQ 14b - PubMed Retriever
==============================
Retrieves relevant PubMed articles and extracts snippets for Phase A.
Uses NCBI E-utilities API (esearch + efetch).
"""

import re
import time
import logging
import xml.etree.ElementTree as ET
from typing import Optional
from dataclasses import dataclass

import requests

from config import (
    PUBMED_BASE_URL,
    PUBMED_EMAIL,
    PUBMED_API_KEY,
    PUBMED_MAX_RESULTS,
    PUBMED_MAX_SNIPPETS,
)
from data_loader import Snippet

logger = logging.getLogger(__name__)


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    url: str


class PubMedRetriever:
    """Retrieve documents and snippets from PubMed via E-utilities."""

    def __init__(self):
        self.session = requests.Session()
        self.base_params = {"email": PUBMED_EMAIL, "tool": "bioasq_agent"}
        if PUBMED_API_KEY:
            self.base_params["api_key"] = PUBMED_API_KEY

    def search(self, query: str, max_results: int = PUBMED_MAX_RESULTS) -> list[str]:
        """Search PubMed and return a list of PMIDs.

        Args:
            query: The biomedical question or search query.
            max_results: Maximum number of results to return.

        Returns:
            List of PMID strings, ordered by relevance.
        """
        params = {
            **self.base_params,
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
        }

        try:
            resp = self.session.get(f"{PUBMED_BASE_URL}/esearch.fcgi", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"PubMed search returned {len(pmids)} results for: {query[:80]}...")
            return pmids
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def fetch_articles(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch article details (title + abstract) for given PMIDs.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of PubMedArticle objects with title and abstract.
        """
        if not pmids:
            return []

        params = {
            **self.base_params,
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }

        try:
            resp = self.session.get(f"{PUBMED_BASE_URL}/efetch.fcgi", params=params, timeout=60)
            resp.raise_for_status()
            return self._parse_xml(resp.text)
        except Exception as e:
            logger.error(f"PubMed fetch failed: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> list[PubMedArticle]:
        """Parse PubMed XML response into PubMedArticle objects."""
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for article_elem in root.findall(".//PubmedArticle"):
                pmid_elem = article_elem.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""

                title_elem = article_elem.find(".//ArticleTitle")
                title = self._get_text_content(title_elem)

                abstract_texts = []
                for abs_text in article_elem.findall(".//AbstractText"):
                    label = abs_text.get("Label", "")
                    text = self._get_text_content(abs_text)
                    if label:
                        abstract_texts.append(f"{label}: {text}")
                    else:
                        abstract_texts.append(text)

                abstract = " ".join(abstract_texts)

                articles.append(PubMedArticle(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    url=f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
                ))
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")

        return articles

    @staticmethod
    def _get_text_content(elem) -> str:
        """Extract all text content from an XML element, including nested tags."""
        if elem is None:
            return ""
        return "".join(elem.itertext()).strip()

    def search_and_fetch(self, query: str, max_results: int = PUBMED_MAX_RESULTS) -> list[PubMedArticle]:
        """Combined search + fetch in one call.

        Args:
            query: The biomedical question.
            max_results: Maximum number of articles to retrieve.

        Returns:
            List of PubMedArticle objects.
        """
        pmids = self.search(query, max_results)
        if not pmids:
            return []

        # Respect NCBI rate limit (3 req/sec without API key, 10 with)
        time.sleep(0.35)
        return self.fetch_articles(pmids)


def extract_snippets_from_articles(
    question_text: str,
    articles: list[PubMedArticle],
    max_snippets: int = PUBMED_MAX_SNIPPETS,
) -> list[Snippet]:
    """Extract relevant snippets from retrieved articles using sentence-level matching.

    This is a lightweight extraction approach:
    1. Split abstracts into sentences.
    2. Score each sentence by keyword overlap with the question.
    3. Return the top-k sentences as snippets.

    For better quality, the LLM agent can refine these later.

    Args:
        question_text: The biomedical question.
        articles: Retrieved PubMed articles.
        max_snippets: Maximum number of snippets to return.

    Returns:
        List of Snippet objects sorted by relevance score.
    """
    # Tokenize question into keywords
    question_tokens = set(
        re.findall(r'\b[a-zA-Z]{3,}\b', question_text.lower())
    )
    # Remove common stop words
    stop_words = {
        "the", "and", "are", "was", "were", "been", "being", "have", "has",
        "had", "does", "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "need", "dare", "ought", "used", "for",
        "with", "from", "that", "this", "these", "those", "which", "what",
        "where", "when", "who", "whom", "how", "not", "nor", "but", "yet",
        "also", "just", "than", "then", "only", "very", "more", "most",
        "some", "any", "all", "each", "every", "both", "few", "several",
    }
    question_tokens -= stop_words

    scored_snippets = []
    for article in articles:
        # Process both title and abstract
        for section_name, text in [("title", article.title), ("abstract", article.abstract)]:
            if not text:
                continue
            sentences = _split_sentences(text)
            for sent in sentences:
                sent_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower()))
                overlap = len(question_tokens & sent_tokens)
                if overlap < 1:
                    continue

                # Score: overlap count, weighted by sentence length (prefer informative)
                score = overlap / max(len(question_tokens), 1)

                offset_begin = text.find(sent)
                offset_end = offset_begin + len(sent) if offset_begin >= 0 else 0

                scored_snippets.append((
                    score,
                    Snippet(
                        text=sent,
                        document=article.url,
                        begin_section=section_name,
                        end_section=section_name,
                        offset_begin=max(offset_begin, 0),
                        offset_end=offset_end,
                    ),
                ))

    # Sort by score descending, take top-k
    scored_snippets.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored_snippets[:max_snippets]]


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter for biomedical text."""
    # Handle abbreviations common in biomedical text
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def build_search_query(question_body: str) -> str:
    """Transform a natural language question into a PubMed search query.

    Extracts key biomedical terms and constructs a query string.

    Args:
        question_body: The natural language question.

    Returns:
        Optimized PubMed search query string.
    """
    # Remove question words and common filler
    remove_patterns = [
        r'^(what|which|who|where|when|why|how|is|are|was|were|do|does|did|can|could)\s+',
        r'\b(the|a|an|of|in|on|at|to|for|with|by|from|as|into|through)\b',
        r'[?!.,;:]',
    ]
    query = question_body
    for pattern in remove_patterns:
        query = re.sub(pattern, ' ', query, flags=re.IGNORECASE)

    # Clean up whitespace
    query = re.sub(r'\s+', ' ', query).strip()

    return query
