"""
thumbnail_agent/tools.py
~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around the Tavily search API.
Raises a RuntimeError if the search fails so the caller can handle it.
"""

import logging

from tavily import TavilyClient

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Run a Tavily web search and return bullet-point results as a string.

    Args:
        query:       The search query.
        max_results: How many results to include (default 5).

    Returns:
        A newline-separated string of "- <title>: <snippet>" bullets.

    Raises:
        RuntimeError: If the Tavily API call fails.
    """
    logger.debug("Tavily search: %s", query)
    try:
        results = TavilyClient().search(query, max_results=max_results)["results"]
    except Exception as exc:
        raise RuntimeError(f"Tavily search failed: {exc}") from exc

    bullets = "\n".join(
        f"- {r['title']}: {r['content'][:200]}" for r in results
    )
    logger.debug("Search returned %d results", len(results))
    return bullets
