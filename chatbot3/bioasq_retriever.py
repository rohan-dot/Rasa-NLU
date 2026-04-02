"""
BioASQ 14b - BioASQ PubMed Search Service
==========================================
Uses BioASQ's own PubMed search endpoint (http://bioasq.org:8000/pubmed).
NO API key, NO email needed — just HTTP POST with JSON.

This is the designated resource for BioASQ Task B Phase A.

Workflow:
  1. Open a session (GET the base URL → get a session URL back)
  2. POST search queries to the session URL
  3. Parse results (title, abstract, pmid)
"""

import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

BIOASQ_PUBMED_URL = "http://bioasq.org:8000/pubmed"


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    url: str


class BioASQPubMedRetriever:
    """Client for BioASQ's PubMed search service."""

    def __init__(self):
        self.session = requests.Session()
        self.session_url = None

    def _open_session(self):
        """Open a session with the BioASQ PubMed service.

        GET the base URL — it returns a session-specific URL
        that we use for subsequent search requests.
        """
        try:
            resp = self.session.get(BIOASQ_PUBMED_URL, timeout=30)
            # The response body IS the session URL (plain text)
            self.session_url = resp.text.strip().strip('"')
            if not self.session_url.startswith("http"):
                # Some versions return just the path
                self.session_url = resp.url
            logger.info(f"BioASQ PubMed session opened: {self.session_url}")
        except Exception as e:
            logger.error(f"Failed to open BioASQ PubMed session: {e}")
            self.session_url = None

    def search(self, query: str, max_results: int = 10) -> list[PubMedArticle]:
        """Search PubMed via BioASQ's service.

        Args:
            query: Search query (supports full PubMed query syntax).
            max_results: Number of articles to retrieve.

        Returns:
            List of PubMedArticle objects with title and abstract.
        """
        # Open session if we don't have one
        if self.session_url is None:
            self._open_session()
        if self.session_url is None:
            logger.warning("No BioASQ PubMed session — skipping search")
            return []

        # Build the search request
        search_payload = {
            "findPubMedCitations": [
                query,           # keywords
                0,               # page (0-indexed)
                max_results,     # articlesPerPage
            ]
        }

        articles = []
        try:
            resp = self.session.post(
                self.session_url,
                data={"json": str(search_payload).replace("'", '"')},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            if "exception" in data:
                logger.warning(f"BioASQ PubMed exception: {data['exception']}")
                # Session expired — reset and retry once
                self.session_url = None
                self._open_session()
                if self.session_url:
                    resp = self.session.post(
                        self.session_url,
                        data={"json": str(search_payload).replace("'", '"')},
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=60,
                    )
                    data = resp.json()
                    if "exception" in data:
                        return []

            result = data.get("result", {})
            docs = result.get("documents", [])

            for doc in docs:
                pmid = doc.get("pmid", "")
                title = doc.get("title", "")
                abstract = doc.get("documentAbstract", "")
                if pmid and (title or abstract):
                    articles.append(PubMedArticle(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        url=f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
                    ))

            logger.info(
                f"BioASQ PubMed: {len(articles)} articles for: {query[:60]}..."
            )

        except requests.exceptions.Timeout:
            logger.warning("BioASQ PubMed search timed out")
        except Exception as e:
            logger.error(f"BioASQ PubMed search failed: {e}")
            # Reset session on error
            self.session_url = None

        return articles

    def search_and_get_snippets(
        self, question_body: str, max_articles: int = 10
    ) -> tuple[list[str], list[dict]]:
        """Search and extract snippets from retrieved articles.

        Args:
            question_body: The biomedical question.
            max_articles: Max articles to retrieve.

        Returns:
            Tuple of (document_urls, snippets) where snippets are dicts
            with text, document, beginSection, endSection, offsets.
        """
        articles = self.search(question_body, max_articles)
        if not articles:
            return [], []

        doc_urls = [a.url for a in articles]

        # Extract snippets: use title and abstract sentences
        snippets = []
        for article in articles:
            # Title as a snippet
            if article.title:
                snippets.append({
                    "text": article.title,
                    "document": article.url,
                    "beginSection": "title",
                    "endSection": "title",
                    "offsetInBeginSection": 0,
                    "offsetInEndSection": len(article.title),
                })

            # Abstract sentences as snippets
            if article.abstract:
                # Split on sentence boundaries
                import re
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', article.abstract)
                offset = 0
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 30:  # Skip very short fragments
                        real_offset = article.abstract.find(sent, offset)
                        if real_offset == -1:
                            real_offset = offset
                        snippets.append({
                            "text": sent,
                            "document": article.url,
                            "beginSection": "abstract",
                            "endSection": "abstract",
                            "offsetInBeginSection": real_offset,
                            "offsetInEndSection": real_offset + len(sent),
                        })
                        offset = real_offset + len(sent)

        # Rate limit: be nice to BioASQ's server
        time.sleep(0.5)

        return doc_urls, snippets
