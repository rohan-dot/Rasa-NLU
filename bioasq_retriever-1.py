"""
BioASQ 14b - NCBI PubMed Retriever (E-utilities)
=================================================
Uses standard NCBI E-utilities over HTTPS (port 443).
NO API key, NO email required. Rate limited to 3 req/sec.

Two-step process:
  1. esearch — search PubMed, get PMIDs
  2. efetch  — fetch title + abstract XML for those PMIDs
"""

import re
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    url: str


class BioASQPubMedRetriever:
    """NCBI E-utilities client for PubMed retrieval. No API key needed."""

    def __init__(self):
        self.session = requests.Session()

    def _search_pmids(self, query: str, max_results: int = 10) -> list[str]:
        """Search PubMed and return PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
        }
        try:
            resp = self.session.get(ESEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"PubMed search: {len(pmids)} PMIDs for: {query[:60]}...")
            return pmids
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def _fetch_articles(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch title + abstract for given PMIDs."""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }
        try:
            resp = self.session.get(EFETCH_URL, params=params, timeout=60)
            resp.raise_for_status()
            return self._parse_xml(resp.text)
        except Exception as e:
            logger.error(f"PubMed fetch failed: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> list[PubMedArticle]:
        """Parse efetch XML into PubMedArticle objects."""
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for elem in root.findall(".//PubmedArticle"):
                pmid_el = elem.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                title_el = elem.find(".//ArticleTitle")
                title = "".join(title_el.itertext()).strip() if title_el is not None else ""

                abstract_parts = []
                for abs_el in elem.findall(".//AbstractText"):
                    label = abs_el.get("Label", "")
                    text = "".join(abs_el.itertext()).strip()
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                if pmid and (title or abstract):
                    articles.append(PubMedArticle(
                        pmid=pmid, title=title, abstract=abstract,
                        url=f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
                    ))
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
        return articles

    def search_and_get_snippets(
        self, question_body: str, max_articles: int = 10
    ) -> tuple[list[str], list[dict]]:
        """Search PubMed and extract snippets from results.

        Returns:
            Tuple of (document_urls, snippets).
        """
        # Step 1: search
        pmids = self._search_pmids(question_body, max_articles)
        if not pmids:
            return [], []

        # Rate limit: 3 req/sec without API key
        time.sleep(0.35)

        # Step 2: fetch articles
        articles = self._fetch_articles(pmids)
        if not articles:
            return [], []

        doc_urls = [a.url for a in articles]

        # Step 3: extract snippets from titles + abstracts
        snippets = []
        for article in articles:
            if article.title:
                snippets.append({
                    "text": article.title,
                    "document": article.url,
                    "beginSection": "title",
                    "endSection": "title",
                    "offsetInBeginSection": 0,
                    "offsetInEndSection": len(article.title),
                })

            if article.abstract:
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', article.abstract)
                offset = 0
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 30:
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

        # Rate limit between questions
        time.sleep(0.35)

        logger.info(f"PubMed: {len(snippets)} snippets from {len(articles)} articles")
        return doc_urls, snippets
