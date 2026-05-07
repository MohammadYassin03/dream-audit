"""Generate assets/og-card.png. 1200x630, palette matched to the site.

Run once whenever the title or subtitle changes:
    python analysis/make_og_card.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.theme import PALETTE  # noqa: E402

OUT = ROOT / "assets" / "og-card.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _star_path(cx, cy, r_outer, r_inner=None):
    """5-point star centered at (cx, cy)."""
    if r_inner is None:
        r_inner = r_outer * 0.4
    pts = []
    for i in range(10):
        ang = (np.pi / 2) - i * (np.pi / 5)
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
    return np.array(pts)


def _add_star(ax, cx, cy, r, color, alpha=1.0):
    pts = _star_path(cx, cy, r)
    ax.add_patch(patches.Polygon(pts, closed=True,
                                 facecolor=color, edgecolor="none",
                                 alpha=alpha))


def main() -> None:
    W, H = 12, 6.3  # inches; at dpi=100, that's 1200x630
    fig, ax = plt.subplots(figsize=(W, H), dpi=100)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.3)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.patch.set_facecolor(PALETTE["cream"])
    ax.set_facecolor(PALETTE["cream"])

    # Subtle scattered stars in the margins (matching the site's textured background)
    rng = np.random.default_rng(7)
    for _ in range(34):
        x = rng.uniform(0.2, 11.8)
        y = rng.uniform(0.2, 6.1)
        # Avoid the centre block where text will sit (1.0 to 11.0 by 1.5 to 5.0)
        if 1.0 < x < 11.0 and 1.5 < y < 5.2:
            continue
        r = rng.uniform(0.06, 0.10)
        _add_star(ax, x, y, r, PALETTE["flag_navy"], alpha=0.10)

    # Title (centered)
    ax.text(6.0, 4.55, "THE AMERICAN DREAM,",
            family="serif", fontsize=46, fontweight="bold",
            color=PALETTE["ink"], ha="center", va="bottom")
    ax.text(6.0, 3.65, "Audited.",
            family="serif", fontsize=68, fontweight="bold", style="italic",
            color=PALETTE["flag_red"], ha="center", va="bottom")

    # Thin red rule beneath the title (mirrors the site's title rule)
    ax.add_patch(patches.Rectangle((4.5, 3.55), 3.0, 0.04,
                                   facecolor=PALETTE["flag_red"],
                                   edgecolor="none"))

    # Deck
    ax.text(6.0, 2.85,
            "Three lives, four demographics, six decades of US economic data.",
            family="serif", fontsize=22, style="italic",
            color=PALETTE["ink_soft"], ha="center", va="center")

    # Three centered ornament stars between deck and footer
    for i, off in enumerate([-0.55, 0.0, 0.55]):
        size = 0.13 if i == 1 else 0.10
        _add_star(ax, 6.0 + off, 1.85, size, PALETTE["gold"], alpha=0.95)

    # Footer kicker
    ax.text(6.0, 0.55,
            "DSAN  SCHOLARSHIP  PROJECT   ·   GEORGETOWN  UNIVERSITY   ·   SPRING  2026",
            family="sans-serif", fontsize=12, fontweight="bold",
            color=PALETTE["ink_soft"], ha="center", va="center")

    # Hairline rule above the kicker
    ax.add_patch(patches.Rectangle((1.5, 1.05), 9.0, 0.012,
                                   facecolor=PALETTE["rule"],
                                   edgecolor="none"))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(OUT, dpi=100, bbox_inches="tight", pad_inches=0,
                facecolor=fig.get_facecolor())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
