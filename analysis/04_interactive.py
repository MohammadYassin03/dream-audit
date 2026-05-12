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
        title=dict(text="Hours of Work to Afford a Median US Home"),
        xaxis_title="Year",
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


# Wage divergence: nominal hourly wage by demographic, with annotated
# policy timeline overlaid as vertical event markers at the top of the chart.
def wage_divergence() -> None:
    # Use stitched data so the chart spans 1967-2025, not just BLS 2000+
    wages = pd.read_parquet(PROC / "wages_demographic_stitched.parquet")
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

    # Policy events to overlay. (year, short label, longer hover description)
    events = [
        (1972, "Title IX",
         "Title IX (1972) banned sex discrimination in federally-funded education and "
         "is the year Hispanic origin was first separately tabulated in the CPS."),
        (1978, "401(k) created",
         "Revenue Act of 1978 created the 401(k) tax provision, which over the next "
         "two decades replaced defined-benefit pensions as the dominant retirement vehicle."),
        (1996, "Welfare reform",
         "Personal Responsibility and Work Opportunity Reconciliation Act (1996) ended "
         "the federal entitlement to cash assistance and pushed low-wage workers into "
         "the labor market."),
        (2008, "Great Recession",
         "The 2008 financial crisis triggered the longest period of nominal-wage "
         "stagnation since the 1970s. Wage recovery took roughly six years."),
        (2020, "COVID-19",
         "The COVID-19 pandemic compressed wage distributions dramatically. Many "
         "low-wage demographic cells saw their largest one-year nominal pay raise on record."),
    ]

    shapes = []
    annotations = []
    for year, label, desc in events:
        shapes.append(dict(
            type="line", xref="x", yref="paper",
            x0=year, x1=year, y0=0, y1=0.93,
            line=dict(color=PALETTE["ink_soft"], width=1, dash="dot"),
            opacity=0.55,
        ))
        annotations.append(dict(
            x=year, y=1.005, xref="x", yref="paper",
            text=label, showarrow=False,
            font=dict(size=10, color=PALETTE["ink_soft"], family="Inter, sans-serif"),
            xanchor="left", yanchor="bottom",
            textangle=-30,
            hovertext=desc,
        ))

    layout = plotly_layout(
        title=dict(text="Nominal Hourly Wages, by Race and Sex (1967 to 2025)"),
        xaxis_title="Year",
        yaxis_title="USD per hour (nominal, full-time wage/salary workers)",
        height=560,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(l=60, r=30, t=120, b=60),  # extra top room for event labels
    )
    layout["shapes"] = shapes
    layout["annotations"] = annotations
    fig.update_layout(**layout)

    fig.write_html(
        OUT / "wage_divergence.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# Time Price Explorer DATA. Same source as the calculator but expanded
# into a long-format table the dashboard can filter and reshape.
def explorer_data() -> None:
    import json
    wages = pd.read_parquet(PROC / "wages_demographic_stitched.parquet")
    wages["hourly_nominal"] = wages["weekly_earnings_nominal"] / 40

    long = pd.read_parquet(PROC / "series_long.parquet")
    home_q = long.query("series == 'MSPUS'").copy()
    home_q["year"] = home_q["date"].dt.year
    home_y = home_q.groupby("year", as_index=False)["value"].mean().rename(columns={"value": "home"})

    tuition = pd.read_parquet(PROC / "tuition.parquet").query(
        "institution_type == 'public' and cost_category == 'tuition_fees' and institution_level == '4yr'"
    )[["year", "amount_nominal"]].rename(columns={"amount_nominal": "tuition"})

    cpi_med = long.query("series == 'CPIMEDSL'").copy()
    cpi_med["year"] = cpi_med["date"].dt.year
    cpi_med_y = cpi_med.groupby("year", as_index=False)["value"].mean()
    base_med = cpi_med_y.loc[cpi_med_y["year"] == 2020, "value"].mean()
    # Anchor 2020 family premium at $21,342 (KFF 2020 EHB Survey)
    cpi_med_y["healthcare"] = 21342.0 * (cpi_med_y["value"] / base_med)

    # Combine costs by year
    cost_by_year = {}
    for y in range(1970, 2026):
        h = home_y.loc[home_y["year"] == y, "home"]
        t = tuition.loc[tuition["year"] == y, "tuition"]
        m = cpi_med_y.loc[cpi_med_y["year"] == y, "healthcare"]
        cost_by_year[y] = {
            "home":       float(h.iloc[0]) if len(h) else None,
            "tuition":    float(t.iloc[0]) if len(t) else None,
            "healthcare": float(m.iloc[0]) if len(m) else None,
        }

    # Long-format records: one row per (item, demographic, year)
    items = [
        {"key": "home", "label": "Median US home",
         "title": "Hours of Work to Afford the Median US Home",
         "y_axis": "Hours of full-time labor",
         "unit_short": "home", "unit_long": "the median US home"},
        {"key": "tuition", "label": "One year of public 4-yr tuition",
         "title": "Hours of Work for One Year of Public 4-Yr Tuition",
         "y_axis": "Hours of full-time labor",
         "unit_short": "year of school", "unit_long": "one year of public four-year tuition + fees"},
        {"key": "healthcare", "label": "Family health insurance premium",
         "title": "Hours of Work for One Year of Family Health Insurance",
         "y_axis": "Hours of full-time labor",
         "unit_short": "year of family premium",
         "unit_long": "one year of employer-sponsored family health insurance premium"},
    ]
    demographics = [
        {"key": "white_men",      "label": "White men",      "color": DEMOGRAPHIC["white_men"]},
        {"key": "white_women",    "label": "White women",    "color": DEMOGRAPHIC["white_women"]},
        {"key": "black_men",      "label": "Black men",      "color": DEMOGRAPHIC["black_men"]},
        {"key": "black_women",    "label": "Black women",    "color": DEMOGRAPHIC["black_women"]},
        {"key": "hispanic_men",   "label": "Hispanic men",   "color": DEMOGRAPHIC["hispanic_men"]},
        {"key": "hispanic_women", "label": "Hispanic women", "color": DEMOGRAPHIC["hispanic_women"]},
    ]

    records = []
    for item in items:
        for dem in demographics:
            d = wages.query("demographic == @dem['key']").sort_values("year")
            for r in d.itertuples():
                if not (1970 <= r.year <= 2025):
                    continue
                cost = cost_by_year.get(int(r.year), {}).get(item["key"])
                if cost is None or r.hourly_nominal is None:
                    continue
                records.append({
                    "item": item["key"],
                    "demographic": dem["key"],
                    "year": int(r.year),
                    "hours": cost / r.hourly_nominal,
                })

    # Add raw demographic variables: wages, life expectancy, wealth share
    le = pd.read_parquet(PROC / "life_expectancy.parquet") if (PROC / "life_expectancy.parquet").exists() else pd.DataFrame()
    dfa = pd.read_parquet(PROC / "dfa_race_shares.parquet") if (PROC / "dfa_race_shares.parquet").exists() else pd.DataFrame()

    extra_items = [
        {"key": "wage_hourly", "label": "Nominal hourly wage", "kind": "wage",
         "title": "Median Nominal Hourly Wage", "y_axis": "USD per hour (nominal)",
         "unit_short": "USD/hr", "unit_long": "the median hourly nominal wage"},
        {"key": "life_expectancy", "label": "Life expectancy at birth", "kind": "life_expectancy",
         "title": "Life Expectancy at Birth", "y_axis": "Years",
         "unit_short": "years", "unit_long": "life expectancy at birth"},
        {"key": "wealth_share", "label": "Share of US household net worth", "kind": "wealth_share",
         "title": "Share of US Household Net Worth (Federal Reserve DFA)",
         "y_axis": "Percent of total US household net worth",
         "unit_short": "% of net worth", "unit_long": "share of US household net worth"},
    ]

    extra_records = []
    for r in wages.itertuples():
        if 1970 <= r.year <= 2025:
            extra_records.append({
                "item": "wage_hourly",
                "demographic": r.demographic,
                "year": int(r.year),
                "hours": float(r.hourly_nominal),
            })
    if not le.empty:
        for _, r in le.iterrows():
            if r["demographic"] in [d["key"] for d in demographics] and 1970 <= r["year"] <= 2025 and pd.notna(r["life_expectancy"]):
                extra_records.append({
                    "item": "life_expectancy",
                    "demographic": r["demographic"],
                    "year": int(r["year"]),
                    "hours": float(r["life_expectancy"]),
                })
    if not dfa.empty:
        nw = dfa.query("metric == 'Net worth'").copy()
        nw["year"] = pd.to_datetime(nw["date"]).dt.year
        nw_y = nw.groupby(["year", "race"], as_index=False)["share"].mean()
        race_to_dem = {"White": "white_men", "Black": "black_men", "Hispanic": "hispanic_men"}
        for _, r in nw_y.iterrows():
            dem_key = race_to_dem.get(r["race"])
            if dem_key and 1970 <= r["year"] <= 2025:
                extra_records.append({
                    "item": "wealth_share",
                    "demographic": dem_key,
                    "year": int(r["year"]),
                    "hours": float(r["share"]),
                })

    payload = {
        "items": items + extra_items,
        "demographics": demographics,
        "year_min": 1970,
        "year_max": 2025,
        "records": records + extra_records,
    }
    (OUT / "explorer_data.json").write_text(json.dumps(payload))


# Calculator DATA: writes a single JSON with all the inputs the
# "What does it cost you" widget needs. For each (demographic, year)
# pair we ship the hourly wage; for each year we ship the cost of the
# three basket items (median home, public 4-yr tuition+fees, est. rent).
# The widget JS does the divisions client-side.
def calculator_data() -> None:
    import json
    wages = pd.read_parquet(PROC / "wages_demographic_stitched.parquet")
    wages["hourly_nominal"] = wages["weekly_earnings_nominal"] / 40

    long = pd.read_parquet(PROC / "series_long.parquet")
    home_q = long.query("series == 'MSPUS'").copy()
    home_q["year"] = home_q["date"].dt.year
    home_y = home_q.groupby("year", as_index=False)["value"].mean().rename(columns={"value": "home"})

    tuition = pd.read_parquet(PROC / "tuition.parquet").query(
        "institution_type == 'public' and cost_category == 'tuition_fees' and institution_level == '4yr'"
    )[["year", "amount_nominal"]].rename(columns={"amount_nominal": "tuition"})

    # Year range starts at 1970 (avoids a noisy 1967 first-data-year for both
    # home prices and Census P-38 wages) and ends at 2025.
    years = sorted(set(home_y["year"]) | set(tuition["year"]))
    years = [y for y in years if 1970 <= y <= 2025]

    def get(df, col, year):
        v = df.loc[df["year"] == year, col]
        return float(v.iloc[0]) if len(v) else None

    cost_by_year = {}
    for y in years:
        cost_by_year[y] = {
            "home":    get(home_y, "home", y),
            "tuition": get(tuition, "tuition", y),
        }

    # Hourly wage by (demographic, year), 1970+
    wage_by_dem_year = {}
    for dem in wages["demographic"].unique():
        d = wages.query("demographic == @dem")
        wage_by_dem_year[dem] = {
            int(r.year): float(r.hourly_nominal)
            for r in d.itertuples() if 1970 <= r.year <= 2025
        }

    payload = {
        "demographics": [
            {"key": "white_men",      "label": "White man"},
            {"key": "white_women",    "label": "White woman"},
            {"key": "black_men",      "label": "Black man"},
            {"key": "black_women",    "label": "Black woman"},
            {"key": "hispanic_men",   "label": "Hispanic man"},
            {"key": "hispanic_women", "label": "Hispanic woman"},
        ],
        "items": [
            {"key": "home",    "label": "the median US home",            "unit": "home"},
            {"key": "tuition", "label": "one year of public 4-yr tuition","unit": "year of school"},
        ],
        "years": years,
        "wage": wage_by_dem_year,
        "cost": cost_by_year,
    }
    (OUT / "calculator_data.json").write_text(json.dumps(payload))


# Time Price Lattice DATA for inline scrollytelling render. Exports the
# computed z-matrix and labels as JSON so the article page can render the
# same lattice inline (not in an iframe) and scrollama can highlight
# specific cells in response to scroll events.
def time_price_lattice_scrolly_data() -> None:
    import json
    tp = pd.read_parquet(PROC / "time_price.parquet")
    tp["decade"] = (tp["year"] // 10) * 10
    decade_avg = tp.groupby(["demographic", "decade"], as_index=False)["hours_for_home"].mean()

    dem_order = ["hispanic_women", "black_women", "hispanic_men",
                 "black_men", "white_women", "white_men"]
    decade_order = sorted(decade_avg["decade"].unique().tolist())
    decade_labels = [f"{d}s" for d in decade_order]
    dem_labels = [DEM_LABELS[d] for d in dem_order]

    z = []
    text = []
    for dem in dem_order:
        zrow, trow = [], []
        for d in decade_order:
            v = decade_avg.query("demographic == @dem and decade == @d")["hours_for_home"]
            v = float(v.iloc[0]) if len(v) else None
            zrow.append(v)
            trow.append(f"{v:,.0f}" if v is not None else "n/a")
        z.append(zrow)
        text.append(trow)

    payload = {
        "x": decade_labels,
        "y": dem_labels,
        "z": z,
        "text": text,
        "zmin": 4000,
        "zmax": 22000,
        "dem_order": dem_order,
        "decade_order": decade_order,
        "palette": {
            "ink": PALETTE["ink"],
            "ink_soft": PALETTE["ink_soft"],
            "cream": PALETTE["cream"],
            "cream_pale": PALETTE["cream_pale"],
            "gold": PALETTE["gold"],
            "gold_pale": PALETTE["gold_pale"],
            "flag_red": PALETTE["flag_red"],
            "flag_red_pale": PALETTE["flag_red_pale"],
            "rule": PALETTE["rule"],
        },
    }
    (OUT / "lattice_data.json").write_text(json.dumps(payload, indent=2))


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
        title=dict(text="The Time Price of a Home, by Decade and Demographic"),
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
        title=dict(text="Life Expectancy at Birth, by Race and Sex (1960 to 2018)"),
        xaxis_title="Year",
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
        title=dict(text="Share of US Household Net Worth, by Race (1989 to 2025)"),
        xaxis_title="Year",
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
    merged = merged[merged["year"] <= 2022]  # tuition data ends 2022 (NCES d23)

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
        title=dict(text="Hours of Work to Afford One Year of Public Four-Year Tuition and Fees"),
        xaxis_title="Year",
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
        title=dict(text="Medical Care Prices vs. All Consumer Prices, 1960=100"),
        xaxis_title="Year",
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


# Dream Scoreboard: a radar chart that compares the 1965 era against the
# 2025 era across six dimensions of "the American Dream." Each axis is
# normalized so that 1.0 means the Dream's strongest reading on that axis
# in the audit's window, and 0 means the weakest. Two filled rings.
def dream_scoreboard() -> None:
    from plotly.subplots import make_subplots  # noqa: F401

    # Each axis: (label, era1_score, era2_score, era1_tooltip, era2_tooltip).
    # Scores are normalized 0 to 1 within the audit window. Tooltips spell
    # out the year, the raw reading, and what the score represents so the
    # radar polygon hover does not strand the user on a label-only fragment.
    axes = [
        ("Home affordability",     1.00, 0.48,
         "Year: 1970<br>Reading: 5,900 hours of labor at the<br>white-man median wage to afford<br>the median US home.<br>Score 1.00 sets the audit baseline.",
         "Year: 2025<br>Reading: 12,275 hours at the<br>white-man median wage for the<br>same median US home.<br>Lower score = less affordable."),
        ("College affordability",  1.00, 0.30,
         "Year: 1970<br>Reading: 99 hours for one year of<br>public four-year tuition and fees<br>at the white-man wage.<br>Score 1.00 sets the baseline.",
         "Year: 2022<br>Reading: 332 hours for the same<br>year of public four-year tuition.<br>Score 0.30."),
        ("Healthcare-to-CPI parity", 1.00, 0.41,
         "Year: 1965<br>Reading: medical CPI moving in<br>lockstep with overall CPI.<br>Score 1.00 sets the parity baseline.",
         "Year: 2025<br>Reading: medical CPI has grown<br>roughly 2.4x as fast as overall CPI<br>since 1965.<br>Score 0.41."),
        ("Employer-funded retirement", 1.00, 0.32,
         "Year: 1979<br>Reading: 38% of private-sector<br>workers had a defined-benefit<br>pension.<br>Score 1.00 sets the baseline.",
         "Year: 2018<br>Reading: 12% of private-sector<br>workers had a defined-benefit<br>pension.<br>Score 0.32."),
        ("Longevity equity",       0.03, 0.66,
         "Year: 1960<br>Reading: the gap in life expectancy<br>between Black and white women<br>was 7.8 years.<br>Score 0.03 (wide gap).",
         "Year: 2017<br>Reading: the same gap had<br>narrowed to 2.7 years.<br>Score 0.66 (closing, not closed)."),
        ("Real household income",  0.60, 1.00,
         "Year: 1965<br>Reading: $48k median real<br>household income, in 2024 dollars.<br>Score 0.60.",
         "Year: 2024<br>Reading: $80k median real<br>household income, in 2024 dollars.<br>Score 1.00, the highest in window."),
    ]

    labels = [a[0] for a in axes]
    era1 = [a[1] for a in axes]
    era2 = [a[2] for a in axes]
    tooltip1 = [a[3] for a in axes]
    tooltip2 = [a[4] for a in axes]

    # Close the polygon
    labels_c = labels + [labels[0]]
    era1_c = era1 + [era1[0]]
    era2_c = era2 + [era2[0]]
    tooltip1_c = tooltip1 + [tooltip1[0]]
    tooltip2_c = tooltip2 + [tooltip2[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=era1_c, theta=labels_c,
        fill="toself", name="1965 era",
        mode="lines+markers",
        line=dict(color=PALETTE["flag_navy"], width=2.4),
        marker=dict(size=8, color=PALETTE["flag_navy"], line=dict(color="white", width=1.5)),
        fillcolor="rgba(27, 61, 107, 0.18)",
        customdata=[[t] for t in tooltip1_c],
        hovertemplate="<b>%{theta}</b><br><i>1965 era</i><br>%{customdata[0]}<extra></extra>",
        hoveron="points",
    ))
    fig.add_trace(go.Scatterpolar(
        r=era2_c, theta=labels_c,
        fill="toself", name="2025 era",
        mode="lines+markers",
        line=dict(color=PALETTE["flag_red"], width=2.4),
        marker=dict(size=8, color=PALETTE["flag_red"], line=dict(color="white", width=1.5)),
        fillcolor="rgba(156, 44, 60, 0.18)",
        customdata=[[t] for t in tooltip2_c],
        hovertemplate="<b>%{theta}</b><br><i>2025 era</i><br>%{customdata[0]}<extra></extra>",
        hoveron="points",
    ))

    fig.update_layout(**plotly_layout(
        title=dict(text="The Dream Scoreboard, 1965 vs 2025"),
        height=560,
        polar=dict(
            bgcolor=PALETTE["cream_pale"],
            radialaxis=dict(
                visible=True, range=[0, 1.05],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(size=9, color=PALETTE["ink_soft"]),
                gridcolor=PALETTE["rule"],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=PALETTE["ink"]),
                gridcolor=PALETTE["rule"],
                linecolor=PALETTE["ink_soft"],
            ),
        ),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        margin=dict(l=80, r=80, t=80, b=60),
        hoverlabel=dict(
            align="left",
            bgcolor="rgba(20, 35, 61, 0.94)",
            bordercolor="rgba(255,255,255,0.4)",
            font=dict(color="white", size=12, family="Inter, system-ui, sans-serif"),
            namelength=-1,
        ),
    ))
    fig.write_html(
        OUT / "dream_scoreboard.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


# Household budget composition: two stacked bars side by side, 1960 vs 2020.
# Categories use the BLS Consumer Expenditure Survey's 1960-61 and 2020-21
# household spending shares so the comparison is apples-to-apples.
def budget_composition() -> None:
    # Categories x [1960, 2020] share of household expenditures (%).
    # Sources: BLS CES 1960-61 historical tables (1960 column) and
    # BLS CES 2020 ("Consumer Expenditures in 2020", BLS Report 1098).
    # Categories collapsed where useful for comparability.
    categories = [
        ("Housing",            27.0, 33.0, PALETTE["flag_navy"]),
        ("Food",               24.0, 12.0, PALETTE["gold_deep"]),
        ("Transportation",     14.0, 16.0, PALETTE["flag_navy_pale"]),
        ("Healthcare",          5.0,  8.0, PALETTE["flag_red"]),
        ("Apparel",            10.0,  2.5, PALETTE["sage"]),
        ("Insurance & pensions", 5.0, 11.5, PALETTE["gold"]),
        ("Education",           1.0,  2.5, "#D8896E"),
        ("Other",              14.0, 14.5, PALETTE["sage_pale"] if "sage_pale" in PALETTE else "#B5BFA5"),
    ]

    fig = go.Figure()
    for cat, v1960, v2020, color in categories:
        fig.add_trace(go.Bar(
            x=["1960-61", "2020-21"],
            y=[v1960, v2020],
            name=cat,
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{v1960:.1f}%", f"{v2020:.1f}%"],
            textposition="inside",
            textfont=dict(color=PALETTE["cream"], size=10),
            hovertemplate=f"<b>{cat}</b><br>%{{x}}: %{{y:.1f}}%%<extra></extra>",
        ))

    fig.update_layout(**plotly_layout(
        title=dict(text="What Households Spent Their Money On, 1960 vs 2020"),
        barmode="stack",
        bargap=0.55,
        height=560,
        xaxis=dict(
            title="Survey period",
            tickfont=dict(size=14, color=PALETTE["ink"]),
        ),
        yaxis=dict(
            title="Share of total household expenditures",
            ticksuffix="%",
            range=[0, 100],
        ),
        legend=dict(
            orientation="v", y=0.5, x=1.02, xanchor="left",
            font=dict(size=10),
        ),
        margin=dict(l=80, r=200, t=80, b=60),
    ))
    fig.write_html(
        OUT / "budget_composition.html",
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
        title=dict(text="Black Household Share of US Net Worth: Actual vs Three Counterfactuals"),
        xaxis_title="Year",
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
        horizontal_spacing=0.09,
        vertical_spacing=0.18,
        shared_xaxes=False,
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
        title=dict(text="What Presidents Talked About, 1960 to 2024 (5-Year Rolling Average)"),
        height=820,
        showlegend=False,
        margin=dict(l=130, r=40, t=90, b=110),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor=PALETTE["rule"], linecolor=PALETTE["ink_soft"],
        tickfont=dict(size=10, color=PALETTE["ink_soft"]),
        dtick=20,
        showticklabels=True,
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

    # Figure-level axis labels. Y label sits in the wider left margin so it
    # never overlaps the leftmost subplot's tick numbers.
    fig.add_annotation(
        text="Year",
        xref="paper", yref="paper",
        x=0.5, y=-0.10,
        showarrow=False,
        font=dict(size=13, color=PALETTE["ink_soft"], family="Inter, system-ui, sans-serif"),
    )
    fig.add_annotation(
        text="Percent of SOTU paragraphs covering this topic",
        xref="paper", yref="paper",
        x=-0.11, y=0.5,
        textangle=-90,
        showarrow=False,
        font=dict(size=13, color=PALETTE["ink_soft"], family="Inter, system-ui, sans-serif"),
    )

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
        title=dict(text="Civilian Labor Force Participation Rate, Men vs. Women"),
        xaxis_title="Year",
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


# Choropleth: state-level life expectancy at birth (CDC NCHS 2018).
# A regional counterpoint to the national life-expectancy line chart.
def state_life_expectancy_map() -> None:
    csv = ROOT / "data" / "raw" / "cdc_state_life_expectancy_2018.csv"
    if not csv.exists():
        print("  (skip state map, raw CSV missing)")
        return
    df = pd.read_csv(csv)
    df = df.query("Sex == 'Total' and State != 'United States'").copy()

    state_to_code = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
        "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
        "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
        "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
        "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
        "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
        "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
        "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
        "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
        "Wisconsin": "WI", "Wyoming": "WY",
    }
    df["code"] = df["State"].map(state_to_code)
    df = df.dropna(subset=["code"]).sort_values("LEB")

    # Use a navy-to-red diverging-style scale anchored on the audit palette
    colorscale = [
        [0.00, PALETTE["flag_red"]],
        [0.50, PALETTE["cream_pale"]],
        [1.00, PALETTE["flag_navy"]],
    ]

    fig = go.Figure(go.Choropleth(
        locations=df["code"],
        z=df["LEB"],
        locationmode="USA-states",
        colorscale=colorscale,
        zmin=df["LEB"].min(),
        zmax=df["LEB"].max(),
        marker_line_color="white",
        marker_line_width=0.6,
        colorbar=dict(
            title=dict(text="Years", font=dict(size=11, color=PALETTE["ink_soft"])),
            tickfont=dict(size=10, color=PALETTE["ink_soft"]),
            thickness=14,
            len=0.7,
            ticksuffix=" yr",
        ),
        customdata=df[["State"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>Life expectancy at birth: %{z:.1f} years<extra></extra>",
    ))

    # Range annotation: best and worst
    best = df.iloc[-1]
    worst = df.iloc[0]

    fig.update_layout(**plotly_layout(
        title=dict(text="Life Expectancy at Birth, by State (CDC NCHS, 2018)"),
        height=560,
        geo=dict(
            scope="usa",
            projection=dict(type="albers usa"),
            bgcolor="rgba(0,0,0,0)",
            lakecolor=PALETTE["cream_pale"],
            showlakes=True,
        ),
        margin=dict(l=20, r=20, t=80, b=70),
        annotations=[
            dict(
                text=(f"Range across states: <b>{worst['State']}</b> at {worst['LEB']:.1f} years "
                      f"to <b>{best['State']}</b> at {best['LEB']:.1f} years."),
                xref="paper", yref="paper",
                x=0.5, y=-0.08, showarrow=False,
                font=dict(size=12, color=PALETTE["ink_soft"], family="Inter, system-ui, sans-serif"),
            ),
        ],
    ))
    fig.write_html(
        OUT / "state_life_expectancy_map.html",
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
    time_price_lattice_scrolly_data()
    print("  lattice_data.json (for inline scrollytelling)")
    calculator_data()
    print("  calculator_data.json (for the calculator widget)")
    explorer_data()
    print("  explorer_data.json (for the time-price explorer)")
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
    dream_scoreboard()
    print("  dream_scoreboard.html")
    budget_composition()
    print("  budget_composition.html")
    counterfactual_wealth()
    print("  counterfactual.html")
    sotu_rhetoric_river()
    print("  sotu_rhetoric_river.html (if NLP stage has been run)")
    state_life_expectancy_map()
    print("  state_life_expectancy_map.html")


if __name__ == "__main__":
    main()
