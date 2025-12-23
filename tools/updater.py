from __future__ import annotations

from datetime import datetime
from typing import List

import json
from config import README_FILE, INDEX_FILE, logger
from statics import README_HEADER


def _parse_date(text: str, fmt: str) -> datetime:
    return datetime.strptime(text, fmt)


def format_relevant_paper(paper: dict) -> str:
    """Format a single relevant paper entry for README."""
    tags = paper.get("tags", []) or []
    title = paper["title"]
    link = paper["link"]

    if tags:
        entry = "* " + " ".join(f"`{t}`" for t in tags) + f" [{title}]({link})\n"
    else:
        entry = f"* [{title}]({link})\n"

    if tldr := paper.get("tldr", ""):
        entry += f"  > **TL;DR**: {tldr}\n"

    return entry


def update_daily_arxiv(papers: List[dict], date_str: str) -> None:
    """Prepend a new section into README for the *current execution date*.

        Args:
            papers: List of paper dicts that are NEW and RELEVANT.
            date_str: The date string to display in header (e.g. "2025-10-11").
        """
    logger.info(f"Updating README with {len(papers)} new papers under date {date_str}")

    lines: list[str] = []
    if README_FILE.exists():
        lines = README_FILE.read_text(encoding="utf-8").splitlines(keepends=True)

    # 1) Drop the previous header block (everything before first ###)
    #    This ensures we always use the latest README_HEADER template from statics.py
    body_lines = []
    found_first_header = False
    for line in lines:
        if line.startswith("###"):
            found_first_header = True

        if found_first_header:
            body_lines.append(line)

    # 2) Build new section
    new_section: list[str] = []
    if papers:
        section_lines: list[str] = [f"### {date_str}\n"]
        for p in papers:
            # Double check relevance just in case, though main.py handles it
            if not p.get("relevant", False):
                continue
            section_lines.append(format_relevant_paper(p))
        # Only add the section if we actually added paper lines
        if len(section_lines) > 1:
            new_section = section_lines + ["\n"]

    # 3) Rebuild Content
    #    Header -> New Section -> Existing Body
    paper_count = sum(1 for l in (new_section + body_lines) if l.strip().startswith("* "))
    header = README_HEADER.format(papers=paper_count, update=date_str.replace("-", "."))
    full_content = header + "".join(new_section + body_lines)
    README_FILE.write_text(full_content, encoding="utf-8")

    logger.info(f"README updated: Added {len(new_section) - 2 if new_section else 0} entries. Total: {paper_count}")


def load_index() -> dict[str, dict]:
    """Load the paper index from JSON file.
    Returns: dict mapping arxiv_id -> paper_info_dict
    """
    if not INDEX_FILE.exists():
        logger.warning(f"Index file {INDEX_FILE} not found. Creating a new one.")
        return {}

    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        # Ensure it's a dict (in case it was saved as list previously, though we design it as dict)
        if isinstance(data, dict):
            return data
        else:
            logger.error("Index file format incorrect (expected dict). Resetting index.")
            return {}
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        return {}


def save_index(index_data: dict[str, dict]) -> None:
    """Save the paper index to JSON file."""
    try:
        INDEX_FILE.write_text(json.dumps(index_data, indent=4, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Index saved. Total entries: {len(index_data)}")
    except Exception as e:
        logger.error(f"Failed to save index: {e}")
