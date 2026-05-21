"""
thumbnail_agent/nodes.py
~~~~~~~~~~~~~~~~~~~~~~~~
All 5 LangGraph node functions + the conditional-edge decider.

Node execution order (happy path):
    web_search → prompt_writer → generator → critic → saver
                      ↑_________________________|  (loop back if score too low)
"""

import base64
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

from .prompts import (
    CRITIC_SYSTEM,
    PROMPT_WRITER_SYSTEM,
    PROMPT_WRITER_USER,
    REVISION_HINT,
    STRATEGY_SYSTEM,
    STRATEGY_USER,
)
from .state import ThumbnailState
from .tools import web_search

logger = logging.getLogger(__name__)

# ── Output directory (lives next to this package) ─────────────────────────
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── LLM clients ───────────────────────────────────────────────────────────
# gpt-4o-mini: fast + cheap for text generation
# gpt-4o:      vision-capable for the critic (reads the actual PNG)
_writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
_critic_llm = ChatOpenAI(model="gpt-4o",      temperature=0.2)

# Raw OpenAI client for DALL-E 3 image generation
_openai_client = OpenAI()


# ── Pydantic models ──────────────────────────────────────────────────────
class ThumbnailStrategy(BaseModel):
    """Structured visual strategy produced by node_strategy."""

    main_subject:      str = Field(description="Primary focal element (person, object)")
    secondary_subject: str = Field(description="Supporting element that adds context")
    emotion:           str = Field(description="Core emotion the viewer should feel")
    background:        str = Field(description="Specific background environment")
    style:             str = Field(description="Visual/lighting style, e.g. cinematic high-contrast")
    thumbnail_text:    str = Field(description="2-5 word bold text overlay")
    text_placement:    str = Field(description="Where the text sits: top/bottom/left/right/centre")
    attention_hook:    str = Field(description="What creates instant curiosity or shock")
    visual_hierarchy:  str = Field(description="Reading order: what the eye sees first, second, third")


class CritiqueOutput(BaseModel):
    """Structured rating + critique returned by the vision critic."""

    rating: int = Field(ge=1, le=10, description="Click-through score 1-10")
    critique: str = Field(description="3-5 sentence actionable critique")


# ─────────────────────────────────────────────────────────────────────────
# NODE 1 — web_search
# ─────────────────────────────────────────────────────────────────────────
def node_web_search(state: ThumbnailState) -> dict:
    """
    One-time Tavily search.

    Creates the run directory and stores:
    - ``search_summary``: bullet-point research for the prompt writer
    - ``run_dir``:        folder where all outputs for this run are saved
    - ``iteration``:      reset to 0
    """
    logger.info("[ web_search ] topic=%r", state["topic"])

    try:
        summary = web_search(f"YouTube thumbnail best practices for: {state['topic']}")
    except RuntimeError as exc:
        logger.warning("Web search failed (%s) — continuing with empty summary", exc)
        summary = "- No research available."

    # Build a filesystem-safe slug from the topic
    slug = "".join(c if c.isalnum() else "_" for c in state["topic"])[:40]
    run_dir = OUTPUTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[ web_search ] run_dir=%s", run_dir)
    return {
        "search_summary": summary,
        "run_dir":        str(run_dir),
        "iteration":      0,
    }


# ─────────────────────────────────────────────────────────────────────────
# NODE 2 — strategy
# ─────────────────────────────────────────────────────────────────────────
def node_strategy(state: ThumbnailState) -> dict:
    """
    Thumbnail Strategy Agent — runs once after web_search.

    Converts raw research into a structured visual blueprint:
    main_subject, secondary_subject, emotion, background, style,
    thumbnail_text, text_placement, attention_hook, visual_hierarchy.

    The prompt writer then uses this strategy instead of raw research
    bullets, giving every iteration a strong creative foundation.
    """
    logger.info("[ strategy ] building visual strategy for %r", state["topic"])

    user_msg = STRATEGY_USER.format(
        topic=state["topic"],
        search_summary=state["search_summary"],
    )

    try:
        result: ThumbnailStrategy = _writer_llm.with_structured_output(ThumbnailStrategy).invoke(
            [SystemMessage(content=STRATEGY_SYSTEM),
             HumanMessage(content=user_msg)]
        )
    except Exception as exc:
        raise RuntimeError(f"Strategy agent failed: {exc}") from exc

    strategy_dict = result.model_dump()
    logger.info("[ strategy ] text='%s'  emotion=%s",
                strategy_dict["thumbnail_text"], strategy_dict["emotion"])
    return {"strategy": strategy_dict}


# ─────────────────────────────────────────────────────────────────────────
# NODE 3 — prompt_writer
# ─────────────────────────────────────────────────────────────────────────
def node_prompt_writer(state: ThumbnailState) -> dict:
    """
    Write (or rewrite) the DALL-E 3 image prompt.

    On the first pass ``feedback`` is empty.
    On subsequent passes the previous critique is injected so the LLM
    addresses every weakness before we call the image API again.
    """
    logger.info("[ prompt_writer ] iteration=%d", state.get("iteration", 0))

    # Inject revision feedback only when we have a previous critique
    feedback = ""
    if state.get("critique"):
        feedback = REVISION_HINT.format(
            rating=state["rating"],
            critique=state["critique"],
        )

    # Format strategy dict as readable key: value lines for the LLM
    strategy_lines = "\n".join(
        f"  {k}: {v}" for k, v in state["strategy"].items()
    )

    user_msg = PROMPT_WRITER_USER.format(
        topic=state["topic"],
        strategy=strategy_lines,
        feedback=feedback,
    )

    try:
        response = _writer_llm.invoke(
            [SystemMessage(content=PROMPT_WRITER_SYSTEM),
             HumanMessage(content=user_msg)]
        )
        prompt = response.content.strip()
    except Exception as exc:
        raise RuntimeError(f"Prompt writer LLM failed: {exc}") from exc

    logger.debug("[ prompt_writer ] prompt=%s", prompt[:80])
    return {"current_prompt": prompt}


# ─────────────────────────────────────────────────────────────────────────
# NODE 3 — generator
# ─────────────────────────────────────────────────────────────────────────
def node_generator(state: ThumbnailState) -> dict:
    """
    Call gpt-image-1 and save the generated PNG to disk.

    Uses ``size="1536x1024"`` — the closest gpt-image-1 supports to the
    16:9 YouTube thumbnail aspect ratio (1280×720 recommended by YouTube).
    The model returns base64 JSON directly; no URL download is needed.

    Increments ``iteration`` by 1.
    """
    n = state["iteration"] + 1
    logger.info("[ generator ] iteration=%d — calling DALL-E 3", n)

    try:
        # gpt-image-1 is OpenAI's current image model (April 2025+).
        # It returns base64 JSON directly — no URL download required.
        # 1536x1024 is the closest supported size to a 16:9 YouTube thumbnail.
        response = _openai_client.images.generate(
            model="gpt-image-1",
            prompt=state["current_prompt"],
            size="1536x1024",
            quality="medium",  # gpt-image-1 values: low / medium / high / auto
            n=1,
        )
    except Exception as exc:
        raise RuntimeError(f"Image generation failed: {exc}") from exc

    # Decode base64 payload and write PNG to disk
    img_bytes = base64.b64decode(response.data[0].b64_json)
    out_path  = Path(state["run_dir"]) / f"iter_{n}.png"
    out_path.write_bytes(img_bytes)

    logger.info("[ generator ] saved %s", out_path)
    return {"image_path": str(out_path), "iteration": n}


# ─────────────────────────────────────────────────────────────────────────
# NODE 4 — critic
# ─────────────────────────────────────────────────────────────────────────
def node_critic(state: ThumbnailState) -> dict:
    """
    Vision LLM reads the generated PNG and returns a structured score.

    The PNG is passed as a base64 data URL so no public URL is needed.
    ``with_structured_output(CritiqueOutput)`` guarantees ``rating`` is
    a real integer, not a string.

    Appends one dict to ``history`` (the append reducer keeps all iterations).
    """
    logger.info("[ critic ] evaluating iter_%d", state["iteration"])

    # Encode the image to base64 for the vision API
    img_b64 = base64.b64encode(Path(state["image_path"]).read_bytes()).decode()
    data_url = f"data:image/png;base64,{img_b64}"

    try:
        result: CritiqueOutput = _critic_llm.with_structured_output(CritiqueOutput).invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM),
                HumanMessage(content=[
                    {"type": "text",
                     "text": f"Topic: {state['topic']}\n\nRate this thumbnail."},
                    {"type": "image_url",
                     "image_url": {"url": data_url}},
                ]),
            ]
        )
    except Exception as exc:
        raise RuntimeError(f"Critic LLM failed: {exc}") from exc

    logger.info("[ critic ] score=%d/10", result.rating)
    logger.info("[ critic ] %s", result.critique[:100])

    return {
        "rating":   result.rating,
        "critique": result.critique,
        # history reducer (operator.add) appends this list to existing history
        "history": [{
            "iteration":  state["iteration"],
            "prompt":     state["current_prompt"],
            "image_path": state["image_path"],
            "rating":     result.rating,
            "critique":   result.critique,
        }],
    }


# ─────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE — should_continue
# ─────────────────────────────────────────────────────────────────────────
def should_continue(state: ThumbnailState) -> str:
    """
    Decide whether to loop back to the prompt writer or finish.

    Returns
    -------
    "saver"         — rating reached target, or iteration cap hit → done
    "prompt_writer" — score still too low and iterations remain → loop
    """
    rating    = state["rating"]
    iteration = state["iteration"]
    target    = state.get("target_rating",   8)
    max_iter  = state.get("max_iterations",  3)

    if rating >= target:
        logger.info("[ should_continue ] rating %d >= target %d → saver", rating, target)
        return "saver"

    if iteration >= max_iter:
        logger.info("[ should_continue ] iteration %d >= max %d → saver", iteration, max_iter)
        return "saver"

    logger.info("[ should_continue ] rating %d < %d, iter %d/%d → loop",
                rating, target, iteration, max_iter)
    return "prompt_writer"


# ─────────────────────────────────────────────────────────────────────────
# NODE 5 — saver
# ─────────────────────────────────────────────────────────────────────────
def node_saver(state: ThumbnailState) -> dict:
    """
    Pick the highest-rated image, copy it as final.png, write report.md.

    Output layout inside ``run_dir``:
        iter_1.png, iter_2.png, …   — every generated thumbnail
        final.png                   — copy of the best-rated one
        report.md                   — full history in Markdown
    """
    logger.info("[ saver ] writing outputs for %d iteration(s)", state["iteration"])

    run_dir = Path(state["run_dir"])

    # Pick the best image from accumulated history
    best = max(state["history"], key=lambda h: h["rating"])

    # ── Build report.md ───────────────────────────────────────────────
    lines = [
        f"# YouTube Thumbnail — {state['topic']}",
        "",
        f"**Best rating**: {best['rating']}/10  |  "
        f"**Total iterations**: {state['iteration']}  |  "
        f"**Target**: {state.get('target_rating', 8)}/10",
        "",
        "---",
        "",
        "## Research Summary",
        "",
        state["search_summary"],
        "",
        "---",
        "",
        "## Iteration History",
        "",
    ]

    for h in state["history"]:
        lines += [
            f"### Iteration {h['iteration']}  —  Score: {h['rating']}/10",
            "",
            f"**Prompt used:**",
            f"> {h['prompt']}",
            "",
            f"**Critique:**  {h['critique']}",
            "",
            f"![iter_{h['iteration']}](./{Path(h['image_path']).name})",
            "",
        ]

    lines += [
        "---",
        "",
        "## Final (best iteration)",
        "",
        f"Score **{best['rating']}/10** — iteration {best['iteration']}",
        "",
        "![final](./final.png)",
    ]

    # Write report
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Copy best image as final.png
    final_path = run_dir / "final.png"
    final_path.write_bytes(Path(best["image_path"]).read_bytes())

    logger.info("[ saver ] final.png  → %s", final_path)
    logger.info("[ saver ] report.md  → %s", report_path)

    return {"final_report": str(report_path)}
