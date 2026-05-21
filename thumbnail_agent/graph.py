"""
thumbnail_agent/graph.py
~~~~~~~~~~~~~~~~~~~~~~~~
Wires up the 6 nodes and 1 conditional edge into a compiled LangGraph.

Graph layout:

    START
      │
      ▼
  web_search
      │
      ▼
   strategy          ← NEW: builds structured visual blueprint (runs once)
      │
      ▼
  prompt_writer  ◄──────────────────────┐
      │                                 │  (score too low)
      ▼                                 │
  generator                             │
      │                                 │
      ▼                                 │
   critic  ──── should_continue ────────┘
      │              │
      │         (score OK / cap hit)
      ▼
    saver
      │
      ▼
     END
"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    node_critic,
    node_generator,
    node_prompt_writer,
    node_saver,
    node_strategy,
    node_web_search,
    should_continue,
)
from .state import ThumbnailState


def build_graph():
    """
    Build and compile the Thumbnail Designer LangGraph agent.

    Returns:
        A compiled LangGraph runnable (call .invoke() or .stream()).
    """
    g = StateGraph(ThumbnailState)

    # ── Register nodes ────────────────────────────────────────────────
    g.add_node("web_search",    node_web_search)
    g.add_node("strategy",      node_strategy)      # NEW
    g.add_node("prompt_writer", node_prompt_writer)
    g.add_node("generator",     node_generator)
    g.add_node("critic",        node_critic)
    g.add_node("saver",         node_saver)

    # ── Linear edges ──────────────────────────────────────────────────
    g.add_edge(START,           "web_search")
    g.add_edge("web_search",    "strategy")          # NEW
    g.add_edge("strategy",      "prompt_writer")     # NEW
    g.add_edge("prompt_writer", "generator")
    g.add_edge("generator",     "critic")

    # ── Conditional edge — the reflexion loop ─────────────────────────
    g.add_conditional_edges(
        "critic",
        should_continue,
        {
            "prompt_writer": "prompt_writer",   # loop — score still low
            "saver":         "saver",           # done — score OK or cap hit
        },
    )

    g.add_edge("saver", END)

    return g.compile()
