"""End-of-run tracker dump for multi-dataset experiment scripts (tracker.md)."""

from __future__ import annotations

from typing import Optional

# Same column order as exp/tracker.md → «Our Experiments — All Datasets»
TRACKER_MD_ORDER = [
    "PROTEINS",
    "MUTAG",
    "DD",
    "REDDIT-B",
    "REDDIT-M5",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "ZINC",
]


def _cell_md(d: Optional[dict]) -> str:
    if d is None:
        return "—"
    if d.get("regression"):
        return f"{d['mean']:.4f}±{d['std']:.4f}"
    return f"{d['mean']:.2f}±{d['std']:.2f}"


def _cell_log_line(name: str, d: Optional[dict]) -> str:
    if d is None:
        return f"  [TRACKER] {name}: —  (not run)"
    if d.get("regression"):
        return f"  [TRACKER] {name}: MAE={d['mean']:.4f}+/-{d['std']:.4f}"
    return f"  [TRACKER] {name}: {d['mean']:.2f}+/-{d['std']:.2f}%"


def _p(msg: str = "") -> None:
    """Print to stdout (job / Kaggle log) with flush so streaming logs show the block immediately."""
    print(msg, flush=True)


def print_exp_tracker_footer(
    exp_id: int,
    exp_title: str,
    results: dict,
    *,
    paper_note: bool = True,
) -> None:
    """
    results: dataset name -> return dict from core.trainer.run / run_k_fold, or None.
    Typically only keys for datasets that were executed are set; others are omitted (shown as —).
    All output goes to stdout (same as the rest of the run) — copy from the notebook / terminal log.
    """
    _p(f"\n{'=' * 70}")
    _p("  [LOG — copy block below into exp/tracker.md]")
    _p(f"  [TRACKER] EXP {exp_id:02d} — {exp_title}")
    _p("  → row «Our Experiments — All Datasets»")
    _p(f"{'=' * 70}")
    if paper_note:
        _p(
            "  vs paper: each filled cell compares to **Graph-JEPA (paper)** on that dataset "
            "(Table 1); a full row = full-suite comparison for datasets this script ran."
        )
    _p("")
    _p("  Per-dataset [TRACKER] lines (same numbers as FINAL RESULTS above):")
    for name in TRACKER_MD_ORDER:
        d = results.get(name)
        _p(_cell_log_line(name, d))
    _p("")
    _p("  One markdown row (PROTEINS … ZINC); paste into tracker table:")
    cells = [_cell_md(results.get(n)) for n in TRACKER_MD_ORDER]
    _p(f"  | {exp_id} | " + " | ".join(cells) + " |")
    _p(f"{'=' * 70}")
