"""Stage 2: clean raw data into a tidy long-format master table.

Outputs:
    data/processed/series_long.parquet  — tidy (date, series, value) for all numeric series
    data/processed/series_meta.csv      — series metadata (units, source, demographic)
    data/processed/cpi.csv              — monthly CPI for deflation
    data/processed/wages_demographic.parquet  — wages by race × sex, annualized
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def load_fred(series_id: str) -> pd.DataFrame:
    fp = RAW / "fred" / f"{series_id}.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, parse_dates=["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["series"] = series_id
    return df[["date", "series", "value"]]


SERIES_META = pd.DataFrame([
    # series_id, label, source, frequency, demographic, units
    ("CPIAUCSL",       "CPI (all urban)",                   "FRED/BLS",    "M", "all",            "index 1982-84=100"),
    ("PCEPI",          "PCE Price Index",                   "FRED/BEA",    "M", "all",            "index 2017=100"),
    ("CPIMEDSL",       "CPI Medical Care",                  "FRED/BLS",    "M", "all",            "index 1982-84=100"),
    ("CUUR0000SEHA",   "CPI Rent of Primary Residence",     "FRED/BLS",    "M", "all",            "index 1982-84=100"),
    ("MEHOINUSA672N",  "Real Median HH Income",             "FRED/Census", "A", "all",            "2022 USD"),
    ("LES1252881600Q", "Real Median Weekly Earnings",       "FRED/BLS",    "Q", "all_real",       "1982-84 USD"),
    ("LEU0252881500Q", "Nominal Weekly Earnings, all",      "FRED/BLS",    "Q", "all",            "USD"),
    ("LEU0252883900Q", "Nominal Weekly Earnings",           "FRED/BLS",    "Q", "white_men",      "USD"),
    ("LEU0252884200Q", "Nominal Weekly Earnings",           "FRED/BLS",    "Q", "white_women",    "USD"),
    ("LEU0252884500Q", "Nominal Weekly Earnings",           "FRED/BLS",    "Q", "black_all",      "USD"),
    ("LEU0252884800Q", "Nominal Weekly Earnings",           "FRED/BLS",    "Q", "black_men",      "USD"),
    ("LEU0252885100Q", "Nominal Weekly Earnings",           "FRED/BLS",    "Q", "black_women",    "USD"),
    ("LEU0252885400Q", "Nominal Weekly Earnings",           "FRED/BLS",    "Q", "hispanic_all",   "USD"),
    ("LEU0252885700Q", "Nominal Weekly Earnings",           "FRED/BLS",    "Q", "hispanic_men",   "USD"),
    ("LEU0252886000Q", "Nominal Weekly Earnings",           "FRED/BLS",    "Q", "hispanic_women", "USD"),
    ("MSPUS",          "Median Home Sales Price",           "FRED/Census", "Q", "all",            "USD"),
    ("RHORUSQ156N",    "Homeownership Rate",                "FRED/Census", "Q", "all",            "%"),
    ("MORTGAGE30US",   "30-Yr Fixed Mortgage",              "FRED/Freddie","W", "all",            "%"),
    ("AWHNONAG",       "Avg Weekly Hours, Private",         "FRED/BLS",    "M", "all",            "hours"),
    ("LNS11300002",    "Labor Force Participation, Women",  "FRED/BLS",    "M", "women",          "%"),
    ("LNS11300001",    "Labor Force Participation, Men",    "FRED/BLS",    "M", "men",            "%"),
    ("SLOAS",          "Student Loans Outstanding",         "FRED/Fed",    "Q", "all",            "billions USD"),
    ("WFRBST01134",    "Net Worth Share, Top 1%",           "FRED/Fed DFA","Q", "top_1pct",       "%"),
    ("WFRBSB50215",    "Net Worth Share, Bottom 50%",       "FRED/Fed DFA","Q", "bottom_50pct",   "%"),
], columns=["series", "label", "source", "frequency", "demographic", "units"])


def main() -> None:
    print("== Stage 2: clean ==")
    fred_dir = RAW / "fred"
    if not fred_dir.exists():
        raise SystemExit("Run 01_acquire.py first")

    frames = [load_fred(sid) for sid in SERIES_META["series"]]
    long = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    long.to_parquet(PROC / "series_long.parquet", index=False)
    print(f"  series_long.parquet: {len(long):,} rows, {long['series'].nunique()} series")

    SERIES_META.to_csv(PROC / "series_meta.csv", index=False)
    print(f"  series_meta.csv:     {len(SERIES_META)} series")

    # CPI helper — used by every downstream deflation step
    cpi = long.query("series == 'CPIAUCSL'")[["date", "value"]].rename(columns={"value": "cpi"})
    cpi.to_csv(PROC / "cpi.csv", index=False)
    print(f"  cpi.csv:             {len(cpi):,} months")

    # Wages by demographic, annualized — central to time-price calc.
    # Six primary cells (race × sex). The "all" / "*_all" series are kept in
    # series_long.parquet for cross-checks but excluded from the demographic table.
    dem_series = SERIES_META[SERIES_META["demographic"].isin(
        ["white_men", "white_women", "black_men", "black_women", "hispanic_men", "hispanic_women"]
    )]
    wages = long.merge(dem_series[["series", "demographic"]], on="series")
    wages = wages.rename(columns={"value": "weekly_earnings_nominal"})
    wages["year"] = wages["date"].dt.year
    annual = (wages.groupby(["year", "demographic"], as_index=False)["weekly_earnings_nominal"].mean())
    annual.to_parquet(PROC / "wages_demographic.parquet", index=False)
    print(f"  wages_demographic.parquet: {len(annual):,} rows ({annual['demographic'].nunique()} groups)")

    print("Done.")


if __name__ == "__main__":
    main()
