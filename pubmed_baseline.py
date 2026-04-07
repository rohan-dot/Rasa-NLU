#!/usr/bin/env python3
"""
PubMed Annual Baseline — Download, Parse & Index
=================================================
Downloads PubMed baseline XML files from NCBI FTP, parses them,
and loads articles into a SQLite database with FTS5 full-text search.

The baseline is ~1,100 gzipped XML files (~35 GB compressed,
~300 GB uncompressed). Each file contains ~30,000 articles.
Total: ~36 million articles.

You can download ALL files or a subset (e.g., first 100 files for
testing, or specific file ranges).

Usage:
    # Download + parse + index (first 10 files for quick test)
    python pubmed_baseline.py download \
        --output-dir ./pubmed_baseline \
        --db pubmed_index.db \
        --start 1 --end 10

    # Download everything (will take hours)
    python pubmed_baseline.py download \
        --output-dir ./pubmed_baseline \
        --db pubmed_index.db

    # Parse already-downloaded XML files into the DB
    python pubmed_baseline.py parse \
        --xml-dir ./pubmed_baseline \
        --db pubmed_index.db

    # Search the index
    python pubmed_baseline.py search \
        --db pubmed_index.db \
        --query "Cushing syndrome cortisol"

    # Get stats
    python pubmed_baseline.py stats --db pubmed_index.db

Requirements:
    pip install lxml requests
"""

import argparse
import ftplib
import gzip
import json
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Generator
from xml.etree import ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# NCBI FTP details
FTP_HOST = "ftp.ncbi.nlm.nih.gov"
FTP_BASELINE_DIR = "/pubmed/baseline/"
# As of 2024/2025, files are named: pubmed25n0001.xml.gz ... pubmed25n1219.xml.gz
# The prefix changes yearly — we auto-detect it.


# =====================================================================
# SQLite Schema + FTS5 Index
# =====================================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    pmid         TEXT PRIMARY KEY,
    title        TEXT,
    abstract     TEXT,
    journal      TEXT,
    year         TEXT,
    mesh_terms   TEXT,
    authors      TEXT
);

-- FTS5 virtual table for full-text search over title + abstract
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
    pmid,
    title,
    abstract,
    mesh_terms,
    content='articles',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
    INSERT INTO articles_fts(rowid, pmid, title, abstract, mesh_terms)
    VALUES (new.rowid, new.pmid, new.title, new.abstract, new.mesh_terms);
END;

CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
    INSERT INTO articles_fts(articles_fts, rowid, pmid, title, abstract, mesh_terms)
    VALUES ('delete', old.rowid, old.pmid, old.title, old.abstract, old.mesh_terms);
END;

CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
    INSERT INTO articles_fts(articles_fts, rowid, pmid, title, abstract, mesh_terms)
    VALUES ('delete', old.rowid, old.pmid, old.title, old.abstract, old.mesh_terms);
    INSERT INTO articles_fts(rowid, pmid, title, abstract, mesh_terms)
    VALUES (new.rowid, new.pmid, new.title, new.abstract, new.mesh_terms);
END;

-- Track which baseline files have been processed
CREATE TABLE IF NOT EXISTS processed_files (
    filename TEXT PRIMARY KEY,
    num_articles INTEGER,
    processed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Create / open the SQLite database and ensure schema exists."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64 MB cache
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# =====================================================================
# XML Parsing — extract articles from PubMed baseline XML
# =====================================================================

def parse_pubmed_xml(xml_path: str) -> Generator[dict, None, None]:
    """
    Parse a PubMed baseline XML file (plain or .gz) and yield article
    dicts with keys: pmid, title, abstract, journal, year, mesh_terms, authors.

    Uses iterparse for memory efficiency on large files.
    """
    if xml_path.endswith(".gz"):
        f = gzip.open(xml_path, "rb")
    else:
        f = open(xml_path, "rb")

    try:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag == "PubmedArticle":
                article = _extract_article(elem)
                if article and article.get("pmid"):
                    yield article
                elem.clear()
    except ET.ParseError as e:
        log.warning("XML parse error in %s: %s", xml_path, e)
    finally:
        f.close()


def _extract_article(elem) -> dict | None:
    """Extract article data from a PubmedArticle XML element."""
    try:
        citation = elem.find(".//MedlineCitation")
        if citation is None:
            return None

        # PMID
        pmid_elem = citation.find("PMID")
        pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else ""
        if not pmid:
            return None

        article = citation.find("Article")
        if article is None:
            return None

        # Title
        title_elem = article.find("ArticleTitle")
        title = _get_text(title_elem)

        # Abstract
        abstract_elem = article.find("Abstract")
        abstract = ""
        if abstract_elem is not None:
            parts = []
            for at in abstract_elem.findall("AbstractText"):
                label = at.get("Label", "")
                text = _get_text(at)
                if label and text:
                    parts.append(f"{label}: {text}")
                elif text:
                    parts.append(text)
            abstract = " ".join(parts)

        # Journal
        journal_elem = article.find(".//Journal/Title")
        journal = _get_text(journal_elem)

        # Year
        year = ""
        pub_date = article.find(".//Journal/JournalIssue/PubDate")
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None and year_elem.text:
                year = year_elem.text.strip()
            else:
                medline_date = pub_date.find("MedlineDate")
                if medline_date is not None and medline_date.text:
                    m = re.search(r"(\d{4})", medline_date.text)
                    if m:
                        year = m.group(1)

        # MeSH Terms
        mesh_list = citation.find("MeshHeadingList")
        mesh_terms = []
        if mesh_list is not None:
            for mh in mesh_list.findall("MeshHeading"):
                desc = mh.find("DescriptorName")
                if desc is not None and desc.text:
                    mesh_terms.append(desc.text.strip())

        # Authors
        author_list = article.find("AuthorList")
        authors = []
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.find("LastName")
                fore = author.find("ForeName")
                parts = []
                if last is not None and last.text:
                    parts.append(last.text)
                if fore is not None and fore.text:
                    parts.append(fore.text)
                if parts:
                    authors.append(" ".join(parts))

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "mesh_terms": "; ".join(mesh_terms),
            "authors": "; ".join(authors[:10]),  # Cap at 10 authors
        }

    except Exception as e:
        log.debug("Error extracting article: %s", e)
        return None


def _get_text(elem) -> str:
    """Get all text content from an element, including mixed content."""
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


# =====================================================================
# FTP Download
# =====================================================================

def list_baseline_files(ftp: ftplib.FTP) -> list[str]:
    """List all .xml.gz files in the baseline directory."""
    files = []
    ftp.cwd(FTP_BASELINE_DIR)
    items = ftp.nlst()
    for item in items:
        if item.endswith(".xml.gz"):
            files.append(item)
    files.sort()
    return files


def download_file(ftp: ftplib.FTP, filename: str, output_dir: str) -> str:
    """Download a single file from FTP. Returns local path."""
    local_path = os.path.join(output_dir, filename)
    if os.path.exists(local_path):
        log.info("  Already downloaded: %s", filename)
        return local_path

    temp_path = local_path + ".tmp"
    log.info("  Downloading %s ...", filename)
    try:
        with open(temp_path, "wb") as f:
            ftp.retrbinary(f"RETR {filename}", f.write)
        os.rename(temp_path, local_path)
    except Exception as e:
        log.error("  Download failed for %s: %s", filename, e)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    log.info("  Downloaded: %s (%.1f MB)", filename, size_mb)
    return local_path


def download_and_index(output_dir: str, db_path: str,
                       start: int = 1, end: int | None = None,
                       keep_xml: bool = False):
    """Download baseline files and index them into SQLite."""
    os.makedirs(output_dir, exist_ok=True)
    conn = init_db(db_path)

    # Get list of already-processed files
    cursor = conn.execute("SELECT filename FROM processed_files")
    already_done = {row[0] for row in cursor.fetchall()}

    # Connect to FTP
    log.info("Connecting to %s ...", FTP_HOST)
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()  # anonymous login
    ftp.cwd(FTP_BASELINE_DIR)

    all_files = list_baseline_files(ftp)
    log.info("Found %d baseline files on FTP", len(all_files))

    # Apply range filter
    if end is None:
        end = len(all_files)
    target_files = all_files[start - 1 : end]
    log.info("Processing files %d to %d (%d files)",
             start, min(end, len(all_files)), len(target_files))

    for i, filename in enumerate(target_files):
        if filename in already_done:
            log.info("[%d/%d] Skipping %s (already processed)",
                     i + 1, len(target_files), filename)
            continue

        log.info("[%d/%d] Processing %s",
                 i + 1, len(target_files), filename)

        try:
            # Download
            local_path = download_file(ftp, filename, output_dir)

            # Parse and index
            count = index_xml_file(conn, local_path, filename)
            log.info("  Indexed %d articles from %s", count, filename)

            # Optionally remove XML to save disk space
            if not keep_xml and os.path.exists(local_path):
                os.remove(local_path)
                log.info("  Removed %s to save space", filename)

        except Exception as e:
            log.error("  Failed on %s: %s", filename, e)
            # Reconnect FTP if needed
            try:
                ftp.voidcmd("NOOP")
            except Exception:
                log.info("  Reconnecting to FTP...")
                ftp = ftplib.FTP(FTP_HOST)
                ftp.login()
                ftp.cwd(FTP_BASELINE_DIR)

    ftp.quit()
    conn.close()
    log.info("Done! Database: %s", db_path)


def index_xml_file(conn: sqlite3.Connection, xml_path: str,
                   filename: str) -> int:
    """Parse an XML file and insert articles into the database."""
    count = 0
    batch = []
    batch_size = 5000

    for article in parse_pubmed_xml(xml_path):
        batch.append((
            article["pmid"],
            article["title"],
            article["abstract"],
            article["journal"],
            article["year"],
            article["mesh_terms"],
            article["authors"],
        ))
        count += 1

        if len(batch) >= batch_size:
            _insert_batch(conn, batch)
            batch = []

    if batch:
        _insert_batch(conn, batch)

    # Record that this file has been processed
    conn.execute(
        "INSERT OR REPLACE INTO processed_files (filename, num_articles) "
        "VALUES (?, ?)",
        (filename, count),
    )
    conn.commit()
    return count


def _insert_batch(conn: sqlite3.Connection, batch: list[tuple]):
    """Insert a batch of articles, ignoring duplicates."""
    conn.executemany(
        "INSERT OR IGNORE INTO articles "
        "(pmid, title, abstract, journal, year, mesh_terms, authors) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        batch,
    )
    conn.commit()


# =====================================================================
# Parse-only mode (for already-downloaded files)
# =====================================================================

def parse_local_files(xml_dir: str, db_path: str):
    """Parse all .xml.gz files in a directory into the database."""
    conn = init_db(db_path)

    cursor = conn.execute("SELECT filename FROM processed_files")
    already_done = {row[0] for row in cursor.fetchall()}

    xml_files = sorted(Path(xml_dir).glob("*.xml.gz"))
    log.info("Found %d .xml.gz files in %s", len(xml_files), xml_dir)

    for i, xml_path in enumerate(xml_files):
        filename = xml_path.name
        if filename in already_done:
            log.info("[%d/%d] Skipping %s (already processed)",
                     i + 1, len(xml_files), filename)
            continue

        log.info("[%d/%d] Parsing %s", i + 1, len(xml_files), filename)
        count = index_xml_file(conn, str(xml_path), filename)
        log.info("  Indexed %d articles", count)

    conn.close()


# =====================================================================
# Search
# =====================================================================

def search_articles(db_path: str, query: str,
                    max_results: int = 10,
                    require_abstract: bool = True) -> list[dict]:
    """
    Search the FTS5 index using BM25 ranking.
    Returns a list of article dicts sorted by relevance.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Build FTS5 query — quote terms for safety
    fts_query = " OR ".join(
        f'"{term}"' for term in query.split()
        if len(term) > 1
    )

    sql = """
        SELECT
            a.pmid, a.title, a.abstract, a.journal, a.year,
            a.mesh_terms, a.authors,
            articles_fts.rank AS score
        FROM articles_fts
        JOIN articles a ON a.pmid = articles_fts.pmid
        WHERE articles_fts MATCH ?
    """

    if require_abstract:
        sql += " AND length(a.abstract) > 50"

    sql += " ORDER BY articles_fts.rank LIMIT ?"

    try:
        cursor = conn.execute(sql, (fts_query, max_results))
        results = []
        for row in cursor.fetchall():
            results.append({
                "pmid": row["pmid"],
                "title": row["title"],
                "abstract": row["abstract"],
                "journal": row["journal"],
                "year": row["year"],
                "mesh_terms": row["mesh_terms"],
                "score": row["score"],
                "url": f"http://www.ncbi.nlm.nih.gov/pubmed/{row['pmid']}",
            })
        return results
    except sqlite3.OperationalError as e:
        log.warning("Search failed: %s", e)
        return []
    finally:
        conn.close()


def fetch_by_pmids(db_path: str, pmids: list[str]) -> list[dict]:
    """Fetch specific articles by PMID from the local index."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    placeholders = ",".join("?" for _ in pmids)
    sql = f"""
        SELECT pmid, title, abstract, journal, year, mesh_terms, authors
        FROM articles
        WHERE pmid IN ({placeholders})
    """

    cursor = conn.execute(sql, pmids)
    results = []
    for row in cursor.fetchall():
        results.append(dict(row))
    conn.close()
    return results


def get_stats(db_path: str):
    """Print database statistics."""
    conn = sqlite3.connect(db_path)

    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    with_abstract = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE length(abstract) > 50"
    ).fetchone()[0]
    files = conn.execute("SELECT COUNT(*) FROM processed_files").fetchone()[0]

    year_dist = conn.execute(
        "SELECT year, COUNT(*) as cnt FROM articles "
        "WHERE year != '' GROUP BY year ORDER BY year DESC LIMIT 10"
    ).fetchall()

    print(f"\nPubMed Baseline Index: {db_path}")
    print(f"  Total articles:       {count:,}")
    print(f"  With abstract (>50c): {with_abstract:,}")
    print(f"  Baseline files done:  {files}")
    print(f"\n  Recent years:")
    for year, cnt in year_dist:
        print(f"    {year}: {cnt:,}")

    db_size = os.path.getsize(db_path) / (1024 * 1024 * 1024)
    print(f"\n  Database size: {db_size:.2f} GB")
    conn.close()


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PubMed Annual Baseline — Download, Parse & Index"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = subparsers.add_parser("download",
        help="Download baseline from NCBI FTP and index into SQLite")
    p_dl.add_argument("--output-dir", "-d", default="./pubmed_baseline",
                      help="Directory to store downloaded XML files")
    p_dl.add_argument("--db", default="pubmed_index.db",
                      help="SQLite database path")
    p_dl.add_argument("--start", type=int, default=1,
                      help="First file number to download (1-based)")
    p_dl.add_argument("--end", type=int, default=None,
                      help="Last file number (default: all)")
    p_dl.add_argument("--keep-xml", action="store_true",
                      help="Keep XML files after indexing (uses ~300GB)")

    # parse
    p_parse = subparsers.add_parser("parse",
        help="Parse already-downloaded XML files into SQLite")
    p_parse.add_argument("--xml-dir", required=True)
    p_parse.add_argument("--db", default="pubmed_index.db")

    # search
    p_search = subparsers.add_parser("search",
        help="Search the local PubMed index")
    p_search.add_argument("--db", default="pubmed_index.db")
    p_search.add_argument("--query", "-q", required=True)
    p_search.add_argument("--max-results", "-n", type=int, default=10)

    # stats
    p_stats = subparsers.add_parser("stats",
        help="Show database statistics")
    p_stats.add_argument("--db", default="pubmed_index.db")

    args = parser.parse_args()

    if args.command == "download":
        download_and_index(
            output_dir=args.output_dir,
            db_path=args.db,
            start=args.start,
            end=args.end,
            keep_xml=args.keep_xml,
        )

    elif args.command == "parse":
        parse_local_files(args.xml_dir, args.db)

    elif args.command == "search":
        results = search_articles(args.db, args.query, args.max_results)
        print(f"\nSearch: '{args.query}' — {len(results)} results\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['pmid']}] {r['title'][:100]}")
            print(f"   Journal: {r['journal']}, Year: {r['year']}")
            print(f"   Abstract: {r['abstract'][:200]}...")
            print()

    elif args.command == "stats":
        get_stats(args.db)


if __name__ == "__main__":
    main()
