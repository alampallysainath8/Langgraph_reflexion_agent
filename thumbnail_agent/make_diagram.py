"""
thumbnail_agent/make_diagram.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates graph.mmd (Mermaid source) and graph.png (rendered diagram).

Run from the project root:
    python -m thumbnail_agent.make_diagram

If PNG rendering fails (requires an internet call to mermaid.ink), the
Mermaid source is still written so you can paste it at https://mermaid.live
"""

import logging
from pathlib import Path

from rich.logging import RichHandler

from .graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False, show_path=False)],
)
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent  # saves graph.* next to the package


def main():
    graph = build_graph()
    compiled = graph  # already compiled

    # ── Mermaid source ─────────────────────────────────────────────────
    mmd_text = compiled.get_graph().draw_mermaid()
    mmd_path = OUT_DIR / "graph.mmd"
    mmd_path.write_text(mmd_text, encoding="utf-8")
    logger.info("graph.mmd  written → %s", mmd_path)

    # ── PNG render (requires network) ─────────────────────────────────
    try:
        png_bytes = compiled.get_graph().draw_mermaid_png(
            max_retries=5,
            retry_delay=2.0,
        )
        png_path = OUT_DIR / "graph.png"
        png_path.write_bytes(png_bytes)
        logger.info("graph.png  written → %s", png_path)
    except Exception as exc:
        logger.warning("PNG render failed: %s", exc)
        logger.info("Paste graph.mmd into https://mermaid.live to view the diagram.")


if __name__ == "__main__":
    main()
