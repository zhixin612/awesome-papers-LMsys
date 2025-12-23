from __future__ import annotations

from datetime import datetime, timedelta

from arxiv_fetcher import fetch_arxiv_papers
from config import ARXIV_CATEGORIES, LOOKBACK_DAYS, logger
from llm_filter import llm_filter
from updater import update_daily_arxiv, load_index, save_index


def run() -> None:
    # 1. Calculate Date Range (Past 7 days to Today)
    #    We check a wider window to catch papers that were submitted earlier but published recently.
    today = datetime.now()
    start_date = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")
    display_date = today.strftime("%Y-%m-%d")

    logger.info(f"Fetching papers from {start_date} to {end_date} (Lookback: {LOOKBACK_DAYS} days)")

    # 2. Load Index
    paper_index = load_index()

    # 3. Fetch Papers from Arxiv
    fetched_papers = fetch_arxiv_papers(
        categories=ARXIV_CATEGORIES,
        start_date=start_date,
        end_date=end_date
    )

    # 4. Filter out papers already present in the index
    new_papers = []
    for p in fetched_papers:
        p_id = p.get("id")  # arxiv id link
        if p_id in paper_index:
            logger.debug(f"Skip existing paper: {p['title']}")
            continue
        new_papers.append(p)

    if not new_papers:
        logger.info("No new papers found after index check.")
        return

    logger.info(f"Found {len(new_papers)} new papers not in index. Starting LLM processing...")

    # 5. Process new papers with LLM (Relevance + Tags + TLDR)
    #    Note: llm_filter modifies the dictionary in-place
    processed_papers = llm_filter(new_papers)

    # 6. Update Index with processed results
    #    We store ALL processed papers in index, even irrelevant ones,
    #    to prevent reprocessing them in the future.
    for p in processed_papers:
        p_id = p.get("id")
        # Add a timestamp for when we indexed it
        p["indexed_date"] = display_date
        paper_index[p_id] = p

    save_index(paper_index)

    # 7. Collect Relevant Papers for README
    relevant_papers = [p for p in processed_papers if p.get("relevant", False)]

    # 8. Update README
    if relevant_papers:
        update_daily_arxiv(relevant_papers, display_date)
    else:
        logger.info("No new relevant papers to update in README.")


if __name__ == "__main__":
    run()

    # TODO-list
    """
    1. add an entrypoint for chatgpt and kimi chat
        1.1 https://chatgpt.com/?model=gpt-5&q=XXXXX
        1.2 http://kimi.com/_prefill_chat?prefill_prompt=XXXXX&send_immediately=true&force_search=false&enable_reasoning=false
    2. convert TLDR to chinese
    3. add email sending function
    4. implement a web UI for better visualization
    """
