"""
BioASQ 13b Pipeline — Step 1: Corpus Fetching
==============================================
Extracts all unique PMIDs from BioASQ 13b training data,
fetches abstracts via PubMed E-utilities, attempts PMC full-text
for each paper, falls back to abstract if unavailable.

Outputs:
  - data/papers.jsonl         (one paper per line, with full text or abstract)
  - data/training_snippets.jsonl  (gold snippets from training set)
  - data/fetch_stats.json     (coverage report)

Usage:
  python 01_fetch_corpus.py --training BioASQ-training13b.json --out data/
"""

import json
import time
import argparse
import os
import re
import pickle
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_LINK_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
PMC_OA_URL        = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/ascii"
UNPAYWALL_URL     = "https://api.unpaywall.org/v2/{doi}?email={email}"
NCBI_RATE_SLEEP   = 0.34   # 3 requests/sec max without API key; set 0.1 with key
BATCH_SIZE        = 200


# ── PMID Extraction ──────────────────────────────────────────────────────────
def extract_pmids_and_snippets(training_path: str):
    """
    Parse BioASQ training JSON.
    Returns:
      pmids    : set of PMID strings
      snippets : list of dicts (gold evidence spans)
      questions: list of question dicts (for few-shot later)
    """
    with open(training_path) as f:
        data = json.load(f)

    pmids    = set()
    snippets = []
    questions = data["questions"]

    for q in questions:
        # Collect PMIDs from document URLs
        for doc_url in q.get("documents", []):
            pmid = doc_url.strip("/").split("/")[-1]
            if pmid.isdigit():
                pmids.add(pmid)

        # Collect gold snippets
        for snip in q.get("snippets", []):
            pmid = snip.get("document", "").strip("/").split("/")[-1]
            snippets.append({
                "text"            : snip.get("text", "").strip(),
                "pmid"            : pmid,
                "section"         : snip.get("section", "unknown"),
                "begin_section"   : snip.get("beginSection", ""),
                "end_section"     : snip.get("endSection", ""),
                "offset_begin"    : snip.get("offsetInBeginSection", 0),
                "offset_end"      : snip.get("offsetInEndSection", 0),
                "source_question" : q["body"],
                "question_id"     : q["id"],
                "question_type"   : q["type"],
                "ideal_answer"    : q.get("ideal_answer", [""])[0] if isinstance(q.get("ideal_answer"), list) else q.get("ideal_answer", ""),
                "exact_answer"    : q.get("exact_answer", ""),
            })

    print(f"  Unique PMIDs      : {len(pmids)}")
    print(f"  Gold snippets     : {len(snippets)}")
    print(f"  Training questions: {len(questions)}")
    return pmids, snippets, questions


# ── PubMed Abstract Fetching ─────────────────────────────────────────────────
def parse_pubmed_xml(xml_bytes: bytes) -> dict:
    """Parse PubMed efetch XML → dict of {pmid: paper_dict}"""
    papers = {}
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return papers

    for article in root.findall(".//PubmedArticle"):
        try:
            pmid_el = article.find(".//PMID")
            if pmid_el is None:
                continue
            pmid = pmid_el.text.strip()

            # Title
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            # Abstract (may have multiple structured sections)
            abstract_texts = []
            for ab in article.findall(".//AbstractText"):
                label = ab.get("Label", "")
                text  = "".join(ab.itertext()).strip()
                if label:
                    abstract_texts.append(f"{label}: {text}")
                else:
                    abstract_texts.append(text)
            abstract = " ".join(abstract_texts)

            # Year
            year_el = article.find(".//PubDate/Year")
            year = int(year_el.text) if year_el is not None else 0

            # MeSH terms
            mesh_terms = [
                m.text.strip()
                for m in article.findall(".//MeshHeading/DescriptorName")
                if m.text
            ]

            # DOI
            doi = ""
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text.strip() if id_el.text else ""
                    break

            # PMC ID
            pmc_id = ""
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType") == "pmc":
                    pmc_id = id_el.text.strip() if id_el.text else ""
                    break

            papers[pmid] = {
                "pmid"       : pmid,
                "title"      : title,
                "abstract"   : abstract,
                "year"       : year,
                "mesh_terms" : mesh_terms,
                "doi"        : doi,
                "pmc_id"     : pmc_id,
                "full_text"  : None,   # filled in later
                "source"     : "abstract",
            }
        except Exception:
            continue
    return papers


def fetch_pubmed_batch(pmid_batch: list, session: requests.Session) -> dict:
    """Fetch a batch of abstracts from PubMed."""
    params = {
        "db"      : "pubmed",
        "id"      : ",".join(pmid_batch),
        "rettype" : "abstract",
        "retmode" : "xml",
    }
    try:
        r = session.get(PUBMED_FETCH_URL, params=params, timeout=30)
        r.raise_for_status()
        return parse_pubmed_xml(r.content)
    except Exception as e:
        print(f"  [WARN] PubMed batch failed: {e}")
        return {}


def fetch_all_abstracts(pmids: set, session: requests.Session) -> dict:
    """Fetch all abstracts in batches."""
    pmid_list = list(pmids)
    papers    = {}

    print(f"\nFetching {len(pmid_list)} abstracts from PubMed...")
    for i in tqdm(range(0, len(pmid_list), BATCH_SIZE)):
        batch   = pmid_list[i : i + BATCH_SIZE]
        results = fetch_pubmed_batch(batch, session)
        papers.update(results)
        time.sleep(NCBI_RATE_SLEEP)

    print(f"  Fetched {len(papers)} abstracts successfully.")
    return papers


# ── PMC Full-Text Fetching ───────────────────────────────────────────────────
def fetch_pmc_fulltext(pmc_id: str, session: requests.Session) -> Optional[dict]:
    """
    Fetch full text from PMC Open Access via BioC JSON API.
    Returns dict of {section_name: text} or None if unavailable.
    """
    # Normalise — PMC IDs look like "PMC1234567" or just "1234567"
    pmc_id_clean = pmc_id.replace("PMC", "").strip()
    url = PMC_OA_URL.format(pmcid=f"PMC{pmc_id_clean}")
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None

        data = r.json()
        sections = {}

        # BioC JSON structure: documents → passages
        for document in data.get("documents", []):
            for passage in document.get("passages", []):
                infons   = passage.get("infons", {})
                sec_type = infons.get("section_type", infons.get("type", "body")).lower()
                text     = passage.get("text", "").strip()
                if not text:
                    continue
                if sec_type not in sections:
                    sections[sec_type] = []
                sections[sec_type].append(text)

        # Flatten each section to string
        return {k: " ".join(v) for k, v in sections.items() if v}

    except Exception:
        return None


def enrich_with_fulltext(papers: dict, session: requests.Session) -> dict:
    """
    For each paper with a PMC ID, attempt to fetch full text.
    Updates papers in-place.
    """
    pmc_candidates = {
        pmid: p for pmid, p in papers.items() if p.get("pmc_id")
    }
    print(f"\nAttempting PMC full-text for {len(pmc_candidates)} papers...")

    success = 0
    for pmid, paper in tqdm(pmc_candidates.items()):
        sections = fetch_pmc_fulltext(paper["pmc_id"], session)
        if sections:
            paper["full_text"] = sections
            paper["source"]    = "pmc_fulltext"
            success += 1
        time.sleep(0.2)  # be gentle with PMC

    print(f"  PMC full-text fetched: {success}/{len(pmc_candidates)}")
    return papers


# ── Saving ───────────────────────────────────────────────────────────────────
def save_outputs(papers: dict, snippets: list, questions: list, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Papers — one JSON per line
    papers_path = out / "papers.jsonl"
    with open(papers_path, "w") as f:
        for paper in papers.values():
            f.write(json.dumps(paper) + "\n")
    print(f"\nSaved {len(papers)} papers → {papers_path}")

    # Gold snippets
    snip_path = out / "training_snippets.jsonl"
    with open(snip_path, "w") as f:
        for s in snippets:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(snippets)} snippets → {snip_path}")

    # Questions (for few-shot)
    q_path = out / "training_questions.json"
    with open(q_path, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} questions → {q_path}")

    # Stats
    abstract_only  = sum(1 for p in papers.values() if p["source"] == "abstract")
    fulltext_count = sum(1 for p in papers.values() if p["source"] == "pmc_fulltext")
    stats = {
        "total_papers"    : len(papers),
        "abstract_only"   : abstract_only,
        "pmc_fulltext"    : fulltext_count,
        "fulltext_pct"    : round(fulltext_count / max(len(papers), 1) * 100, 1),
        "total_snippets"  : len(snippets),
        "total_questions" : len(questions),
    }
    with open(out / "fetch_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nCoverage stats: {stats}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True, help="Path to BioASQ-training13b.json")
    parser.add_argument("--out",      default="data", help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("BioASQ 13b — Step 1: Corpus Fetching")
    print("=" * 60)

    # Parse training data
    print("\nParsing training data...")
    pmids, snippets, questions = extract_pmids_and_snippets(args.training)

    session = requests.Session()
    session.headers.update({"User-Agent": "BioASQ-Research/1.0"})

    # Fetch abstracts
    papers = fetch_all_abstracts(pmids, session)

    # Enrich with PMC full text
    papers = enrich_with_fulltext(papers, session)

    # Save everything
    save_outputs(papers, snippets, questions, args.out)
    print("\nStep 1 complete. Run 02_build_index.py next.")


if __name__ == "__main__":
    main()
