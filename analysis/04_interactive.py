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


# Time Price line chart (the original signature)
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


# Wage divergence: nominal hourly wage by demographic
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


# Time Price Lattice (signature heatmap viz)
def time_price_lattice_heatmap() -> None:
    """Heatmap: 6 demographic rows by 6 decade columns. Cell = average hours
    of full-time labor required to afford the median US home in that decade.
    The story is meant to be read in two passes: across rows (the demographic
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
            trow.append(f"{v:,.0f}" if v == v else "n/a")
        z.append(zrow)
        text.append(trow)

    zmin, zmax = 4000, 22000

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[decade_labels[d] for d in decade_order],
        y=[DEM_LABELS[d] for d in dem_order],
        # Top of the scale is capped at flag_red_pale (a dusty red) instead
        # of the deeper flag_red, so the entire range stays light enough for
        # a single dark-navy text color to read against any cell.
        colorscale=[
            [0.0, PALETTE["cream_pale"]],
            [0.4, PALETTE["gold_pale"]],
            [0.7, "#D89090"],
            [1.0, PALETTE["flag_red_pale"]],
        ],
        colorbar=dict(
            title=dict(text="Hours", font=dict(size=11, color=PALETTE["ink_soft"])),
            tickfont=dict(color=PALETTE["ink_soft"], size=10),
            outlinewidth=0, thickness=12, len=0.6,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:,.0f} hours<extra></extra>",
        zmin=zmin, zmax=zmax,
    ))

    # Single text color for every cell. The colorscale above stays light
    # enough across its full range that dark navy reads cleanly everywhere.
    # NaN cells (Hispanic in the 60s, before separate tracking) get a
    # muted "n/a" label so the empty space is read as missing-by-design.
    for ri, dem in enumerate(dem_order):
        for ci, dec in enumerate(decade_order):
            v = z[ri][ci]
            if v != v:  # NaN
                fig.add_annotation(
                    x=decade_labels[dec], y=DEM_LABELS[dem],
                    text="n/a",
                    showarrow=False,
                    font=dict(family="Inter, sans-serif", size=11,
                              color=PALETTE["ink_soft"], style="italic"),
                )
                continue
            fig.add_annotation(
                x=decade_labels[dec], y=DEM_LABELS[dem],
                text=text[ri][ci],
                showarrow=False,
                font=dict(family="Inter, sans-serif", size=12, color=PALETTE["ink"]),
            )

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


# Longevity Gap: life expectancy at birth, 1960 to 2018, with the
# Black/White gap shaded between the lines
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
                name="Black/White gap (women)",
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
        title=dict(text="Life expectancy at birth, by race and sex (1960 to 2018)"),
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


# Wealth Share: stacked area of net worth share by race, 1989 to 2025
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
        title=dict(text="Share of US household net worth, by race (1989 to 2025)"),
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


# Tuition Time Price: hours of work for one year of public 4-year
# tuition + required fees, by demographic, 1967 to 2021
def tuition_time_price() -> None:
    tuition = pd.read_parquet(PROC / "tuition.parquet").query(
        "institution_type == 'public' and cost_category == 'tuition_fees' and institution_level == '4yr'"
    )[["year", "amount_nominal"]]
    wages = pd.read_parquet(PROC / "wages_demographic_stitched.parquet")
    wages["hourly_nominal"] = wages["weekly_earnings_nominal"] / 40

    merged = wages.merge(tuition, on="year")
    merged["hours_for_tuition"] = merged["amount_nominal"] / merged["hourly_nominal"]
    merged = merged[merged["year"] <= 2021]  # tuition data ends 2021

    fig = go.Figure()
    for dem, label in DEM_LABELS.items():
        d = merged.query("demographic == @dem").sort_values("year")
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["hours_for_tuition"],
            mode="lines", name=label,
            line=dict(color=DEMOGRAPHIC.get(dem, PALETTE["ink"]), width=2.4),
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:,.0f}} hours<extra></extra>",
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Hours of work to afford one year of public four-year tuition and fees"),
        xaxis_title=None,
        yaxis_title="Hours of full-time labor",
        height=520,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    ))
    fig.write_html(
        OUT / "tuition_time_price.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# Medical CPI vs General CPI, both indexed to 1960=100. The "great
# decoupling" of healthcare from general inflation.
def medical_vs_general_cpi() -> None:
    long = pd.read_parquet(PROC / "series_long.parquet")
    cpi = long.query("series == 'CPIAUCSL'")[["date", "value"]].rename(columns={"value": "cpi"})
    med = long.query("series == 'CPIMEDSL'")[["date", "value"]].rename(columns={"value": "med"})
    merged = cpi.merge(med, on="date").sort_values("date")
    # Index both to Jan 1960 = 100
    base = merged[merged["date"].dt.year == 1960].iloc[0]
    merged["cpi_idx"] = merged["cpi"] / base["cpi"] * 100
    merged["med_idx"] = merged["med"] / base["med"] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["date"], y=merged["cpi_idx"], mode="lines", name="All consumer prices (CPI)",
        line=dict(color=PALETTE["flag_navy"], width=2.4),
        hovertemplate="<b>CPI</b><br>%{x|%Y}: %{y:,.0f} (1960=100)<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=merged["date"], y=merged["med_idx"], mode="lines", name="Medical care prices",
        line=dict(color=PALETTE["flag_red"], width=2.4),
        fill="tonexty", fillcolor="rgba(156, 44, 60, 0.10)",
        hovertemplate="<b>Medical CPI</b><br>%{x|%Y}: %{y:,.0f} (1960=100)<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Medical care prices vs. all consumer prices, 1960=100"),
        xaxis_title=None,
        yaxis_title="Index, 1960 = 100",
        height=480,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    ))
    fig.write_html(
        OUT / "medical_vs_cpi.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# Counterfactual: what would Black household net worth share look like under
# three alternative trajectories? Actual line + three counterfactuals from the
# 1989Q3 baseline. The chart asks the reader to weigh which counterfactual
# represents a fair benchmark and to read the actual line against it.
def counterfactual_wealth() -> None:
    dfa = pd.read_parquet(PROC / "dfa_race_shares.parquet")
    nw = dfa.query("metric == 'Net worth'").copy().sort_values("date")

    black = nw.query("race == 'Black'").reset_index(drop=True)
    other = nw.query("race == 'Other'").reset_index(drop=True)

    # Baselines (1989 Q3, the first quarter of DFA data)
    base_black = black["share"].iloc[0]
    base_other = other["share"].iloc[0]

    # Counterfactual 1: parity. Black population is ~13% of US population
    # (Census 2020). If wealth share matched population share, the line
    # would sit at 13.0%.
    cf_parity = pd.DataFrame({"date": black["date"], "share": 13.0})

    # Counterfactual 2: same growth as the "Other" category (predominantly
    # Asian American households). Apply the year-over-year ratio of Other's
    # share to a baseline equal to Black's 1989Q3 share.
    cf_other = pd.DataFrame({
        "date": black["date"],
        "share": base_black * (other["share"].values / base_other),
    })

    # Counterfactual 3: hold the gap constant at 1989 levels. White share
    # actually fell over time (90.3 -> 83.5). If Black share had moved in
    # lockstep with that decline, it would have stayed at 4.0% (no widening
    # of the gap).
    white = nw.query("race == 'White'").reset_index(drop=True)
    cf_constant = pd.DataFrame({
        "date": black["date"],
        "share": base_black + 0 * white["share"].values,  # flat at base
    })

    fig = go.Figure()

    # Counterfactual lines (light, dashed, sit behind)
    fig.add_trace(go.Scatter(
        x=cf_parity["date"], y=cf_parity["share"],
        mode="lines", name="If wealth share matched population share (~13%)",
        line=dict(color=PALETTE["sage"], width=1.6, dash="dot"),
        hovertemplate="<b>Population-parity counterfactual</b><br>%{x|%Y}: 13.0%%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cf_other["date"], y=cf_other["share"],
        mode="lines", name="If Black share grew like Asian American share",
        line=dict(color=PALETTE["gold"], width=1.6, dash="dash"),
        hovertemplate="<b>Asian-rate counterfactual</b><br>%{x|%Y}: %{y:.1f}%%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cf_constant["date"], y=cf_constant["share"],
        mode="lines", name="If 1989 gap had simply held (4.0% flat)",
        line=dict(color=PALETTE["flag_navy_pale"], width=1.6, dash="dashdot"),
        hovertemplate="<b>Hold-1989-gap counterfactual</b><br>%{x|%Y}: 4.0%%<extra></extra>",
    ))
    # Actual line (heavy, sits on top)
    fig.add_trace(go.Scatter(
        x=black["date"], y=black["share"],
        mode="lines", name="Actual",
        line=dict(color=PALETTE["flag_red"], width=3.0),
        hovertemplate="<b>Actual</b><br>%{x|%Y}: %{y:.1f}%%<extra></extra>",
    ))

    fig.update_layout(**plotly_layout(
        title=dict(text="Black household share of US net worth: actual vs three counterfactuals"),
        xaxis_title=None,
        yaxis=dict(title="Share of total US household net worth (%)",
                   ticksuffix="%", range=[0, 14]),
        height=560,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(size=10)),
    ))
    fig.write_html(
        OUT / "counterfactual.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# SOTU rhetoric chart: small multiples, one panel per topic. Each panel
# shows the smoothed share of paragraphs in each year's address that
# BERTopic assigned to that topic. 5-year centered rolling average
# dampens the spikiness of single-paragraph swings (each SOTU is short).
def sotu_rhetoric_river() -> None:
    from plotly.subplots import make_subplots

    fp = PROC / "sotu_topic_river.parquet"
    if not fp.exists():
        print("  (skip rhetoric river, run 05_nlp first)")
        return
    river = pd.read_parquet(fp)
    river["clean_label"] = river["label"].str.replace(r"^-?\d+_", "", regex=True).str.replace("_", ", ")

    # Plain-English short labels for each top-words signature
    label_overrides = {
        "tax, deficit, budget, spending":           "Budget &amp; taxes",
        "school, teachers, schools, high school":   "Education",
        "women, court, constitutional, equal":      "Women's rights",
        "crime, police, criminals, criminal":       "Crime &amp; policing",
        "agriculture, rural, grain, loan":          "Agriculture",
        "transportation, projects, construction":   "Infrastructure",
        "health, insurance, health care, care":     "Healthcare",
    }

    def short_for(words: str) -> str:
        for prefix, short in label_overrides.items():
            if words.startswith(prefix):
                return short
        return words[:24]

    # Smooth: 5-year centered rolling average per topic
    smoothed = []
    for tid, group in river.groupby("topic_id"):
        g = group.sort_values("year").copy()
        g["share_smoothed"] = g["share"].rolling(window=5, center=True, min_periods=1).mean()
        smoothed.append(g)
    river = pd.concat(smoothed, ignore_index=True)

    # Order topics by mean share (largest first)
    topic_order = (
        river.groupby("topic_id")["share"].mean().sort_values(ascending=False).index.tolist()
    )
    n = len(topic_order)
    cols = 4
    rows = (n + cols - 1) // cols

    palette = [
        PALETTE["flag_navy"], PALETTE["flag_red"], PALETTE["gold"],
        PALETTE["sage"], PALETTE["flag_navy_pale"], "#D88575", PALETTE["gold_pale"],
    ]

    short_labels = []
    for tid in topic_order:
        words = river.query("topic_id == @tid")["top_words"].iloc[0]
        short_labels.append(short_for(words))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=short_labels,
        horizontal_spacing=0.07,
        vertical_spacing=0.16,
        shared_xaxes=True,
    )

    for i, tid in enumerate(topic_order):
        r = i // cols + 1
        c = i % cols + 1
        d = river.query("topic_id == @tid").sort_values("year")
        words = d["top_words"].iloc[0]
        color = palette[i % len(palette)]
        # Faded raw points behind the smoothed line, for honesty about the data
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["share"] * 100,
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.25),
            showlegend=False, hoverinfo="skip",
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["share_smoothed"] * 100,
            mode="lines",
            line=dict(width=2.4, color=color),
            showlegend=False,
            customdata=[[words]] * len(d),
            hovertemplate=(f"<b>{short_labels[i]}</b><br>"
                           "Top words: %{customdata[0]}<br>"
                           "%{x}: %{y:.1f}% of paragraphs<extra></extra>"),
        ), row=r, col=c)

    layout = plotly_layout(
        title=dict(text="What presidents talked about, 1960 to 2024 (5-year rolling avg)"),
        height=560,
        showlegend=False,
        margin=dict(l=50, r=30, t=80, b=50),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor=PALETTE["rule"], linecolor=PALETTE["ink_soft"],
        tickfont=dict(size=10, color=PALETTE["ink_soft"]),
        dtick=20,
    )
    fig.update_yaxes(
        gridcolor=PALETTE["rule"], linecolor=PALETTE["ink_soft"],
        tickfont=dict(size=10, color=PALETTE["ink_soft"]),
        ticksuffix="%",
        rangemode="tozero",
    )
    # Smaller subplot title font
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=PALETTE["ink"], family="Cormorant Garamond, serif")

    fig.write_html(
        OUT / "sotu_rhetoric_river.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# Labor force participation: men's vs women's, 1960 to 2025. The rise of
# the dual-earner household.
def labor_force_participation() -> None:
    long = pd.read_parquet(PROC / "series_long.parquet")
    men = long.query("series == 'LNS11300001'").assign(group="Men")
    women = long.query("series == 'LNS11300002'").assign(group="Women")
    df = pd.concat([men, women])

    fig = go.Figure()
    for grp, color in [("Men", PALETTE["flag_navy"]), ("Women", PALETTE["flag_red"])]:
        d = df.query("group == @grp").sort_values("date")
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["value"], mode="lines", name=grp,
            line=dict(color=color, width=2.4),
            hovertemplate=f"<b>{grp}</b><br>%{{x|%Y-%m}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Civilian labor force participation rate, men vs. women"),
        xaxis_title=None,
        yaxis=dict(title="Percent of population 16+ in labor force", ticksuffix="%"),
        height=480,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    ))
    fig.write_html(
        OUT / "labor_force_participation.html",
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
    tuition_time_price()
    print("  tuition_time_price.html")
    medical_vs_general_cpi()
    print("  medical_vs_cpi.html")
    labor_force_participation()
    print("  labor_force_participation.html")
    counterfactual_wealth()
    print("  counterfactual.html")
    sotu_rhetoric_river()
    print("  sotu_rhetoric_river.html (if NLP stage has been run)")


if __name__ == "__main__":
    main()
