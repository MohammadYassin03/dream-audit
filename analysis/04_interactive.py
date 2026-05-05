"""Stage 4: build self-contained interactive HTML visualizations.

Each function writes one HTML file to interactive/, ready to embed in index.qmd
via <iframe>. Each chart is independently navigable, cite-able, and works
without the article context.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.theme import DEMOGRAPHIC, PALETTE, plotly_layout  # noqa: E402
PROC = ROOT / "data" / "processed"
OUT = ROOT / "interactive"
OUT.mkdir(parents=True, exist_ok=True)

DEM_LABELS = {
    "white_men":      "White men",
    "white_women":    "White women",
    "black_men":      "Black men",
    "black_women":    "Black women",
    "hispanic_men":   "Hispanic men",
    "hispanic_women": "Hispanic women",
}


# ---------------------------------------------------------------------------
# 1. Time Price Lattice — the signature chart
# ---------------------------------------------------------------------------
def time_price_lattice(item: str = "hours_for_home") -> None:
    """Hours of labor required to afford the median US home, by demographic and year."""
    tp = pd.read_parquet(PROC / "time_price.parquet")
    fig = go.Figure()
    for dem, label in DEM_LABELS.items():
        d = tp.query("demographic == @dem").sort_values("year")
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["year"], y=d[item],
            mode="lines",
            name=label,
            line=dict(color=DEMOGRAPHIC.get(dem, PALETTE["ink"]), width=2.4),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:,.0f} hours<extra></extra>",
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Hours of work to afford a median US home"),
        xaxis_title=None,
        yaxis_title="Hours of full-time labor",
        height=560,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    ))
    fig.write_html(
        OUT / "time_price_home.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# ---------------------------------------------------------------------------
# 2. Wage divergence — nominal hourly wage by demographic
# ---------------------------------------------------------------------------
def wage_divergence() -> None:
    wages = pd.read_parquet(PROC / "wages_demographic.parquet")
    wages["hourly_nominal"] = wages["weekly_earnings_nominal"] / 40
    fig = go.Figure()
    for dem, label in DEM_LABELS.items():
        d = wages.query("demographic == @dem").sort_values("year")
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["hourly_nominal"],
            mode="lines", name=label,
            line=dict(color=DEMOGRAPHIC.get(dem, PALETTE["ink"]), width=2.4),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: $%{y:,.2f}/hr (nominal)<extra></extra>",
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Nominal hourly wages, by race and sex"),
        xaxis_title=None,
        yaxis_title="USD per hour (nominal, full-time wage/salary workers)",
        height=520,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    ))
    fig.write_html(
        OUT / "wage_divergence.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# ---------------------------------------------------------------------------
# 3. Time Price Lattice — the signature heatmap viz
# ---------------------------------------------------------------------------
def time_price_lattice_heatmap() -> None:
    """Heatmap: 6 demographic rows × 6 decade columns. Cell = average hours
    of full-time labor required to afford the median US home in that decade.
    The story is meant to be read in two passes — across rows (the demographic
    spread within any decade) and down columns (each demographic's trajectory
    over time)."""
    tp = pd.read_parquet(PROC / "time_price.parquet")
    tp["decade"] = (tp["year"] // 10) * 10
    decade_avg = tp.groupby(["demographic", "decade"], as_index=False)["hours_for_home"].mean()

    # Demographic order for visual reading: most-burdened on top
    dem_order = ["hispanic_women", "black_women", "hispanic_men", "black_men", "white_women", "white_men"]
    decade_order = sorted(decade_avg["decade"].unique())
    decade_labels = {d: f"{d}s" for d in decade_order}

    z = []
    text = []
    for dem in dem_order:
        zrow = []
        trow = []
        for d in decade_order:
            v = decade_avg.query("demographic == @dem and decade == @d")["hours_for_home"]
            v = float(v.iloc[0]) if len(v) else float("nan")
            zrow.append(v)
            trow.append(f"{v:,.0f}" if v == v else "—")
        z.append(zrow)
        text.append(trow)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[decade_labels[d] for d in decade_order],
        y=[DEM_LABELS[d] for d in dem_order],
        text=text, texttemplate="%{text}",
        colorscale=[
            [0.0, PALETTE["cream_pale"]],
            [0.4, PALETTE["gold_pale"]],
            [0.7, PALETTE["flag_red_pale"]],
            [1.0, PALETTE["flag_red"]],
        ],
        colorbar=dict(
            title=dict(text="Hours", font=dict(size=11, color=PALETTE["ink_soft"])),
            tickfont=dict(color=PALETTE["ink_soft"], size=10),
            outlinewidth=0, thickness=12, len=0.6,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:,.0f} hours<extra></extra>",
        zmin=4000, zmax=22000,
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="The time price of a home, by decade and demographic"),
        height=440,
        xaxis=dict(side="top", tickfont=dict(size=12, color=PALETTE["ink_soft"]),
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickfont=dict(size=11, color=PALETTE["ink_soft"]),
                   gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        margin=dict(l=130, r=40, t=80, b=40),
    ))
    fig.write_html(
        OUT / "time_price_lattice.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# ---------------------------------------------------------------------------
# 4. Longevity Gap — life expectancy at birth, 1960-2018, with the
#    Black–White gap shaded between the lines
# ---------------------------------------------------------------------------
def longevity_gap() -> None:
    le = pd.read_parquet(PROC / "life_expectancy.parquet")
    le = le[le["year"] >= 1960]

    fig = go.Figure()
    # Shade the white/black gap as a band, by sex
    for sex, dash in [("women", None), ("men", "dot")]:
        white = le.query(f"demographic == 'white_{sex}'").sort_values("year")
        black = le.query(f"demographic == 'black_{sex}'").sort_values("year")
        if white.empty or black.empty:
            continue
        joined = white.merge(black, on="year", suffixes=("_w", "_b"))

        # Gap band (only for women; men band would clutter)
        if sex == "women":
            fig.add_trace(go.Scatter(
                x=joined["year"], y=joined["life_expectancy_w"],
                mode="lines", line=dict(width=0), showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=joined["year"], y=joined["life_expectancy_b"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(156, 44, 60, 0.10)",
                showlegend=False, hoverinfo="skip",
                name="Black-White gap (women)",
            ))

        fig.add_trace(go.Scatter(
            x=white["year"], y=white["life_expectancy"], mode="lines",
            name=f"White {sex}",
            line=dict(color=DEMOGRAPHIC[f"white_{sex}"], width=2.4, dash=dash),
            hovertemplate=f"<b>White {sex}</b><br>%{{x}}: %{{y:.1f}} years<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=black["year"], y=black["life_expectancy"], mode="lines",
            name=f"Black {sex}",
            line=dict(color=DEMOGRAPHIC[f"black_{sex}"], width=2.4, dash=dash),
            hovertemplate=f"<b>Black {sex}</b><br>%{{x}}: %{{y:.1f}} years<extra></extra>",
        ))

    fig.update_layout(**plotly_layout(
        title=dict(text="Life expectancy at birth, by race and sex (1960–2018)"),
        xaxis_title=None,
        yaxis_title="Years",
        height=520,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    ))
    fig.write_html(
        OUT / "longevity_gap.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# ---------------------------------------------------------------------------
# 5. Wealth Share — stacked area of net worth share by race, 1989–2025
# ---------------------------------------------------------------------------
def wealth_share_stack() -> None:
    dfa = pd.read_parquet(PROC / "dfa_race_shares.parquet")
    nw = dfa.query("metric == 'Net worth'").copy()
    # Order from largest base share to smallest for visual reading
    race_order = ["White", "Black", "Hispanic", "Other"]
    race_color = {
        "White":    PALETTE["flag_navy"],
        "Black":    PALETTE["flag_red"],
        "Hispanic": PALETTE["gold"],
        "Other":    PALETTE["sage"],
    }
    fig = go.Figure()
    for race in race_order:
        d = nw.query("race == @race").sort_values("date")
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["share"],
            name=race, mode="lines",
            line=dict(width=0.5, color=race_color[race]),
            stackgroup="one",
            fillcolor=race_color[race],
            hovertemplate=f"<b>{race}</b><br>%{{x|%Y Q%q}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Share of US household net worth, by race (1989–2025)"),
        xaxis_title=None,
        yaxis=dict(title="Share of total US net worth (%)",
                   ticksuffix="%", range=[0, 100]),
        height=520,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    ))
    fig.write_html(
        OUT / "wealth_share.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


def main() -> None:
    print("== Stage 4: interactive ==")
    time_price_lattice()
    print("  time_price_home.html")
    wage_divergence()
    print("  wage_divergence.html")
    time_price_lattice_heatmap()
    print("  time_price_lattice.html")
    longevity_gap()
    print("  longevity_gap.html")
    wealth_share_stack()
    print("  wealth_share.html")


if __name__ == "__main__":
    main()
