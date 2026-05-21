"""
thumbnail_agent/state.py
~~~~~~~~~~~~~~~~~~~~~~~~
Shared state schema for the LangGraph agent.

Key design decisions
--------------------
* ``total=False``     — Every key is optional at construction time.
                        We only populate fields that a given node writes,
                        so graph.invoke() only needs the seed inputs.

* ``Annotated[list, add]`` on ``history``
                      — LangGraph's default reducer *overwrites* state on
                        each node return.  We need history to *accumulate*
                        across iterations, so we use operator.add as the
                        reducer; each critic step appends ONE dict to the list.
"""

from operator import add
from typing import Annotated, TypedDict


class ThumbnailState(TypedDict, total=False):
    # ── Inputs — caller provides these in graph.invoke() ──────────────
    topic: str          # e.g. "Why Python is the best language for AI"
    target_rating: int  # loop stops when critic scores >= this (1-10)
    max_iterations: int # hard cap to avoid runaway cost

    # ── Written once by node_web_search ───────────────────────────────
    search_summary: str  # Tavily bullet-point research
    run_dir: str         # outputs/<timestamp>_<slug>/ — all files go here

    # ── Written once by node_strategy ────────────────────────────────
    strategy: dict       # structured visual strategy (subject, emotion, text…)

    # ── Overwritten every loop iteration ──────────────────────────────
    current_prompt: str  # DALL-E 3 image prompt (latest)
    image_path: str      # absolute path to the latest generated PNG
    rating: int          # latest critic score 1-10
    critique: str        # latest critic text (feeds next prompt rewrite)
    iteration: int       # how many images generated so far

    # ── Written once by node_saver ────────────────────────────────────
    final_report: str    # path to report.md

    # ── Append-only log — one dict per critic step ────────────────────
    # reducer = operator.add  →  lists are concatenated, never replaced
    history: Annotated[list, add]
