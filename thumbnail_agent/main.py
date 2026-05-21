"""
thumbnail_agent/main.py
~~~~~~~~~~~~~~~~~~~~~~~
Entry point for the YouTube Thumbnail Designer agent.

Usage
-----
    # normal run
    python -m thumbnail_agent.main "Why Python is the best language for AI"

    # stream node-by-node updates to the terminal
    python -m thumbnail_agent.main "Why Python is the best language for AI" --stream

    # customise thresholds
    python -m thumbnail_agent.main "topic" --target 7 --max-iter 4
"""

import argparse
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from .graph import build_graph

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=True, show_path=False)],
)
logger  = logging.getLogger(__name__)
console = Console()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="YouTube Thumbnail Designer — Reflexion Agent"
    )
    parser.add_argument(
        "topic",
        help='Video topic, e.g. "Why Python is the best language for AI"',
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print live node-by-node updates instead of waiting for the final result",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=8,
        help="Target critic score to stop the loop (default: 8)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Maximum number of generate→critic iterations (default: 3)",
    )
    return parser.parse_args()


def main():
    """Run the thumbnail agent and print the final result."""
    args = _parse_args()

    graph = build_graph()

    # Seed state — only provide known-up-front values
    initial_state = {
        "topic":          args.topic,
        "target_rating":  args.target,
        "max_iterations": args.max_iter,
        "iteration":      0,
        "rating":         0,
        "critique":       "",
        "history":        [],
    }

    console.rule("[bold cyan]YouTube Thumbnail Designer — Reflexion Agent")
    logger.info("Topic      : %s", args.topic)
    logger.info("Target     : %d/10", args.target)
    logger.info("Max iters  : %d", args.max_iter)
    console.rule()

    try:
        if args.stream:
            # ── Streaming mode — print each node update as it arrives ──
            for chunk in graph.stream(initial_state, stream_mode="updates"):
                for node_name, update in chunk.items():
                    console.print(f"\n[bold green][{node_name}][/bold green]")
                    if "rating" in update:
                        console.print(f"  score   : [bold]{update['rating']}/10[/bold]")
                    if "critique" in update:
                        console.print(f"  critique: {update['critique'][:120]}…")
                    if "image_path" in update:
                        console.print(f"  image   : {update['image_path']}")
                    if "final_report" in update:
                        console.print(f"  report  : {update['final_report']}")
        else:
            # ── Batch mode — wait for full graph completion ────────────
            final = graph.invoke(initial_state)
            console.rule("[bold cyan]Done")
            console.print(f"  Best score : [bold]{final['rating']}/10[/bold]")
            console.print(f"  Iterations : {final['iteration']}")
            console.print(f"  Report     : {final['final_report']}")

    except Exception as exc:
        logger.error("Agent failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
