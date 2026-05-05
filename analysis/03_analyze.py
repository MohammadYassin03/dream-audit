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


def load_long() -> pd.DataFrame:
    return pd.read_parquet(PROC / "series_long.parquet")


def load_wages() -> pd.DataFrame:
    return pd.read_parquet(PROC / "wages_demographic.parquet")


def hourly_wages_by_demographic() -> pd.DataFrame:
    """Annual nominal hourly wage by demographic, derived from BLS weekly earnings."""
    wages = load_wages()
    wages["hourly_nominal"] = wages["weekly_earnings_nominal"] / HOURS_PER_WEEK_FT
    return wages[["year", "demographic", "hourly_nominal"]]


def annual_series(long: pd.DataFrame, series_id: str) -> pd.DataFrame:
    s = long.query(f"series == '{series_id}'").copy()
    s["year"] = s["date"].dt.year
    return s.groupby("year", as_index=False)["value"].mean().rename(columns={"value": series_id})


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
    # Hours to afford median home, white men, 1960 vs 2020
    wm = time_price.query("demographic == 'white_men'")
    if not wm.empty:
        v_60 = wm.query("year == 1979")["hours_for_home"].mean()  # earliest dem cut
        v_20 = wm.query("year == 2020")["hours_for_home"].mean()
        out["wm_hours_home_1979"] = round(v_60, 0)
        out["wm_hours_home_2020"] = round(v_20, 0)
    # Black women 2020
    bw_2020 = time_price.query("demographic == 'black_women' and year == 2020")["hours_for_home"].mean()
    out["bw_hours_home_2020"] = round(bw_2020, 0) if not np.isnan(bw_2020) else None
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
