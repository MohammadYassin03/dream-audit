"""Stage 3: analytical transforms — the time-price calculator and persona aggregations.

The "time price" of a good = nominal price ÷ nominal hourly wage. It is
unit-agnostic to inflation and gives a directly comparable measure of how many
hours of labor a good costs across decades and demographics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

HOURS_PER_WEEK_FT = 40  # full-time convention for converting weekly to hourly
MAX_YEAR = 2025         # clip partial-year (Q1 2026) data for clean trailing edge


def load_long() -> pd.DataFrame:
    return pd.read_parquet(PROC / "series_long.parquet")


def load_wages() -> pd.DataFrame:
    """Prefer the Census-BLS stitched series (1967+) when available; fall back
    to BLS-only (2000+) if stitching hasn't been run yet."""
    stitched = PROC / "wages_demographic_stitched.parquet"
    if stitched.exists():
        df = pd.read_parquet(stitched)
        return df[["year", "demographic", "weekly_earnings_nominal", "source"]]
    return pd.read_parquet(PROC / "wages_demographic.parquet").assign(source="bls")


def hourly_wages_by_demographic() -> pd.DataFrame:
    """Annual nominal hourly wage by demographic, derived from stitched
    Census P-38 (pre-2000) + BLS LEU (2000+) weekly earnings."""
    wages = load_wages()
    wages["hourly_nominal"] = wages["weekly_earnings_nominal"] / HOURS_PER_WEEK_FT
    return wages[["year", "demographic", "hourly_nominal", "source"]]


def annual_series(long: pd.DataFrame, series_id: str) -> pd.DataFrame:
    s = long.query(f"series == '{series_id}'").copy()
    s["year"] = s["date"].dt.year
    return s.groupby("year", as_index=False)["value"].mean().rename(columns={"value": series_id})


def _clip_to_year(df: pd.DataFrame, year_col: str = "year", max_year: int = MAX_YEAR) -> pd.DataFrame:
    return df[df[year_col] <= max_year]


def time_price_table(long: pd.DataFrame) -> pd.DataFrame:
    """Hours of labor (per demographic) required to afford a basket of essentials.

    Basket items:
        home   — median sales price (one-time, hours per home)
        rent   — annual rent for median renter (rough; uses CPI rent index x base)
        college — annual public 4-yr tuition+fees (NCES; deferred to next pass)
        car    — placeholder, deferred
    """
    wages = hourly_wages_by_demographic()

    home = annual_series(long, "MSPUS")
    rent_idx = annual_series(long, "CUUR0000SEHA")
    cpi = annual_series(long, "CPIAUCSL")

    base = wages.merge(home, on="year", how="left").merge(rent_idx, on="year", how="left").merge(cpi, on="year", how="left")
    base = _clip_to_year(base)
    base["hours_for_home"] = base["MSPUS"] / base["hourly_nominal"]

    # Rent: convert CPI-rent index to a notional dollar burden by anchoring
    # to a known $/month median in 2020 ($1,200 nationally per ACS) then
    # scaling by the index ratio. This is a first-pass approximation that
    # 02_clean refines once we wire in HUD Fair Market Rents.
    anchor_year = 2020
    anchor_rent_monthly = 1200.0
    anchor_idx = base.loc[base["year"] == anchor_year, "CUUR0000SEHA"].mean()
    base["est_rent_monthly"] = anchor_rent_monthly * (base["CUUR0000SEHA"] / anchor_idx)
    base["hours_for_rent_year"] = (base["est_rent_monthly"] * 12) / base["hourly_nominal"]

    return base


def persona_starter(long: pd.DataFrame, time_price: pd.DataFrame) -> pd.DataFrame:
    """The Starter (~22-25): how long to afford a year of college + a year of rent."""
    out = time_price.assign(
        hours_year_school_rent=lambda d: d["hours_for_rent_year"]  # college added in next pass
    )
    return out[["year", "demographic", "hourly_nominal", "hours_for_rent_year"]]


def persona_builder(long: pd.DataFrame, time_price: pd.DataFrame) -> pd.DataFrame:
    """The Builder (~35-45): home + healthcare burden."""
    return time_price[["year", "demographic", "hours_for_home", "hours_for_rent_year"]]


def persona_finisher(long: pd.DataFrame, time_price: pd.DataFrame) -> pd.DataFrame:
    """The Finisher (~65+): retirement runway. Healthcare cost share placeholder."""
    return time_price[["year", "demographic", "hours_for_home"]]


def headline_stats(time_price: pd.DataFrame) -> dict:
    """Numbers for the article kickers — write to data/processed/headline_stats.csv."""
    out = {}
    for dem in ("white_men", "white_women", "black_men", "black_women",
                "hispanic_men", "hispanic_women"):
        d = time_price.query("demographic == @dem")
        if d.empty:
            continue
        for yr in (1970, 1980, 2000, 2020):
            v = d.query("year == @yr")["hours_for_home"].mean()
            if not np.isnan(v):
                out[f"{dem}_hours_home_{yr}"] = round(v, 0)
    # Headline gap: Black women 2020 vs White men 2020
    bw = out.get("black_women_hours_home_2020")
    wm = out.get("white_men_hours_home_2020")
    if bw and wm:
        out["bw_wm_gap_hours_2020"] = round(bw - wm, 0)
        out["bw_wm_ratio_2020"] = round(bw / wm, 2)
    return out


def main() -> None:
    print("== Stage 3: analyze ==")
    long = load_long()

    tp = time_price_table(long)
    tp.to_parquet(PROC / "time_price.parquet", index=False)
    print(f"  time_price.parquet: {len(tp):,} rows")

    persona_starter(long, tp).to_parquet(PROC / "persona_starter.parquet", index=False)
    persona_builder(long, tp).to_parquet(PROC / "persona_builder.parquet", index=False)
    persona_finisher(long, tp).to_parquet(PROC / "persona_finisher.parquet", index=False)
    print("  persona_*.parquet (3 files)")

    stats = headline_stats(tp)
    pd.DataFrame(list(stats.items()), columns=["metric", "value"]).to_csv(
        PROC / "headline_stats.csv", index=False
    )
    (PROC / "headline_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  headline_stats: {stats}")


if __name__ == "__main__":
    main()
