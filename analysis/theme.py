"""Matplotlib + plotly theme matching theme.scss "Tarnished Gold" palette.

Import this module before plotting anywhere in the project so figures match
the website's visual identity.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    # "Old Glory Under Glass": flag-coded, editorial register.
    "flag_navy":      "#1B3D6B",
    "flag_navy_mid":  "#3A5F8F",
    "flag_navy_pale": "#6E91B4",
    "flag_red":       "#9C2C3C",
    "flag_red_pale":  "#C46A6E",
    "cream":          "#ECEEF1",
    "cream_pale":     "#F4F5F7",
    "ink":            "#14233D",
    "ink_soft":       "#36456A",
    "gold":           "#B68B3E",
    "gold_deep":      "#8C6A2B",
    "gold_pale":      "#D8B978",
    "sage":           "#6B8770",
    "rule":           "#D8CFB8",
}

# Demographic colors, used consistently across every chart
DEMOGRAPHIC = {
    "white_men":      PALETTE["flag_navy"],
    "white_women":    PALETTE["flag_navy_pale"],
    "black_men":      PALETTE["flag_red"],
    "black_women":    "#D88575",
    "hispanic_men":   PALETTE["gold"],
    "hispanic_women": PALETTE["gold_pale"],
    "asian":          PALETTE["sage"],
    "all":            PALETTE["ink"],
}

PERSONA = {
    "starter":  PALETTE["flag_navy"],
    "builder":  PALETTE["gold_deep"],
    "finisher": PALETTE["flag_red"],
}


def apply_mpl_theme() -> None:
    """Apply the Tarnished Gold theme to matplotlib globally."""
    mpl.rcParams.update({
        # Figure
        "figure.facecolor":   PALETTE["cream_pale"],
        "axes.facecolor":     PALETTE["cream_pale"],
        "savefig.facecolor":  PALETTE["cream_pale"],
        "savefig.dpi":        160,
        "figure.dpi":         110,

        # Fonts
        "font.family":        "Inter",
        "font.size":          11,
        "axes.titlesize":     14,
        "axes.titleweight":   "semibold",
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,

        # Spines / grid
        "axes.edgecolor":     PALETTE["ink_soft"],
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.color":         PALETTE["rule"],
        "grid.linewidth":     0.5,
        "grid.linestyle":     "-",
        "axes.axisbelow":     True,

        # Ticks
        "xtick.color":        PALETTE["ink_soft"],
        "ytick.color":        PALETTE["ink_soft"],

        # Lines
        "lines.linewidth":    2.0,
        "lines.solid_capstyle": "round",

        # Color cycle (demographic order)
        "axes.prop_cycle": mpl.cycler(color=[
            PALETTE["flag_navy"], PALETTE["flag_red"],
            PALETTE["gold"], PALETTE["sage"],
            PALETTE["flag_navy_pale"], "#D88575",
        ]),
    })


def plotly_layout(**overrides) -> dict:
    """Plotly layout dict for the Old Glory Under Glass theme. Pass to fig.update_layout."""
    base = dict(
        font=dict(family="Inter, sans-serif", size=12, color=PALETTE["ink"]),
        title=dict(font=dict(family="Cormorant Garamond, serif", size=22, color=PALETTE["ink"])),
        paper_bgcolor=PALETTE["cream_pale"],
        plot_bgcolor=PALETTE["cream_pale"],
        colorway=[
            PALETTE["flag_navy"], PALETTE["flag_red"],
            PALETTE["gold"], PALETTE["sage"],
            PALETTE["flag_navy_pale"], "#D88575",
        ],
        xaxis=dict(
            gridcolor=PALETTE["rule"], linecolor=PALETTE["ink_soft"],
            zerolinecolor=PALETTE["rule"], tickfont=dict(color=PALETTE["ink_soft"]),
        ),
        yaxis=dict(
            gridcolor=PALETTE["rule"], linecolor=PALETTE["ink_soft"],
            zerolinecolor=PALETTE["rule"], tickfont=dict(color=PALETTE["ink_soft"]),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["ink_soft"])),
        margin=dict(l=60, r=30, t=60, b=50),
        hoverlabel=dict(
            bgcolor=PALETTE["ink"], font=dict(color=PALETTE["cream"], family="Inter"),
            bordercolor=PALETTE["gold"],
        ),
    )
    base.update(overrides)
    return base


__all__ = ["PALETTE", "DEMOGRAPHIC", "PERSONA", "apply_mpl_theme", "plotly_layout"]
