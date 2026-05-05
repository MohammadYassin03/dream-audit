"""Stage 1: acquire raw data from FRED, BLS, Census, NCES, NHE, SCF, and SOTU corpus.

All downloads cache to data/raw/. Re-running is idempotent — existing files are
skipped unless --force is passed.

Usage:
    export FRED_API_KEY=...
    python -m analysis.01_acquire [--force]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CORPUS = ROOT / "data" / "corpus"
RAW.mkdir(parents=True, exist_ok=True)
CORPUS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FRED series — the macroeconomic backbone
# ---------------------------------------------------------------------------
FRED_SERIES: dict[str, str] = {
    # Inflation deflators
    "CPIAUCSL":         "CPI All Urban Consumers (monthly, 1947+)",
    "PCEPI":            "PCE Price Index (monthly, 1959+)",
    "CPIMEDSL":         "CPI Medical Care",
    "CUUR0000SEHA":     "CPI Rent of Primary Residence",

    # Wages / income
    "MEHOINUSA672N":    "Real Median Household Income (annual)",
    "LES1252881600Q":   "Median Weekly Earnings, Full-Time Wage/Salary (quarterly)",

    # Wage demographic cuts (BLS LEU series via FRED — nominal weekly earnings,
    # full-time wage/salary, 16+, quarterly, 1979+)
    "LEU0252881500Q":   "Median Weekly Earnings, All 16+ (overall)",
    "LEU0252883900Q":   "Median Weekly Earnings, White Men 16+",
    "LEU0252884200Q":   "Median Weekly Earnings, White Women 16+",
    "LEU0252884500Q":   "Median Weekly Earnings, Black or African American 16+ (overall)",
    "LEU0252884800Q":   "Median Weekly Earnings, Black Men 16+",
    "LEU0252885100Q":   "Median Weekly Earnings, Black Women 16+",
    "LEU0252885400Q":   "Median Weekly Earnings, Hispanic or Latino 16+ (overall)",
    "LEU0252885700Q":   "Median Weekly Earnings, Hispanic Men 16+",
    "LEU0252886000Q":   "Median Weekly Earnings, Hispanic Women 16+",

    # Housing
    "MSPUS":            "Median Sales Price of Houses Sold (quarterly)",
    "RHORUSQ156N":      "Homeownership Rate (quarterly)",
    "MORTGAGE30US":     "30-Year Fixed Mortgage Rate (weekly)",

    # Hours / labor
    "AWHNONAG":         "Average Weekly Hours, Total Private (monthly)",
    "LNS11300002":      "Civilian Labor Force Participation, Women 16+",
    "LNS11300001":      "Civilian Labor Force Participation, Men 16+",

    # Education
    "SLOAS":            "Student Loans Owned and Securitized (quarterly, 2006+)",

    # Wealth shares (DFA — overall by wealth quintile, not race; race cuts
    # are pulled directly from Fed DFA bulk download in download_dfa_race())
    "WFRBST01134":      "Share of Total Net Worth, Top 1%",
    "WFRBSB50215":      "Share of Total Net Worth, Bottom 50%",
    "WFRBLN09053":      "Net Worth Held by Top 0.1%",

    # Retirement / safety net
    "FYONGDA188S":      "Federal Outlays as % of GDP (context)",
}


def fred_url(series_id: str, api_key: str) -> str:
    return (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
        "&observation_start=1960-01-01"
    )


def download_fred(series_ids: Iterable[str], force: bool = False) -> None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit("Set FRED_API_KEY env var (https://fred.stlouisfed.org/docs/api/api_key.html)")
    out_dir = RAW / "fred"
    out_dir.mkdir(exist_ok=True)
    for sid in series_ids:
        out = out_dir / f"{sid}.csv"
        if out.exists() and not force:
            print(f"  [skip] {sid}")
            continue
        print(f"  [fred] {sid}")
        r = requests.get(fred_url(sid, api_key), timeout=30)
        if r.status_code != 200:
            print(f"    !! status {r.status_code} for {sid} — skipping")
            continue
        obs = r.json().get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty:
            print(f"    !! empty for {sid}")
            continue
        df.to_csv(out, index=False)
        time.sleep(0.25)


# ---------------------------------------------------------------------------
# NHE — National Health Expenditures (CMS)
# ---------------------------------------------------------------------------
def download_nhe() -> None:
    """CMS National Health Expenditure historical tables, 1960+. Manual mirror."""
    # CMS hosts these as Excel; the canonical "NHE Summary" table includes
    # per-capita NHE and out-of-pocket per capita back to 1960.
    url = "https://www.cms.gov/files/zip/nhe-tables.zip"
    out = RAW / "nhe" / "nhe-tables.zip"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        print("  [skip] NHE tables")
        return
    print("  [nhe] downloading CMS tables")
    r = requests.get(url, timeout=60)
    out.write_bytes(r.content)


# ---------------------------------------------------------------------------
# NCES — College tuition since 1963 (Digest Table 330.10)
# ---------------------------------------------------------------------------
def download_nces_tuition() -> None:
    """NCES Digest Table 330.10 — average tuition + fees + room/board, by institution
    type, current and constant dollars, since 1963."""
    url = "https://nces.ed.gov/programs/digest/d22/tables/xls/tabn330.10.xlsx"
    out = RAW / "nces" / "tabn330_10.xlsx"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        print("  [skip] NCES tuition")
        return
    print("  [nces] tuition Table 330.10")
    r = requests.get(url, timeout=60)
    if r.status_code == 200:
        out.write_bytes(r.content)
    else:
        print(f"    !! status {r.status_code}")


def download_dfa_race() -> None:
    """Federal Reserve Distributional Financial Accounts — wealth holdings by
    race × quartile, quarterly 1989-present. Bulk CSV.
    """
    url = "https://www.federalreserve.gov/releases/z1/dataviz/dfa/distribute/chart/data.csv"
    out = RAW / "dfa" / "dfa_by_race.csv"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        print("  [skip] DFA by race")
        return
    print("  [dfa] Federal Reserve DFA wealth-by-race")
    r = requests.get(url, timeout=60)
    if r.status_code == 200:
        out.write_bytes(r.content)
    else:
        print(f"    !! status {r.status_code}")


# ---------------------------------------------------------------------------
# State of the Union corpus — UCSB American Presidency Project mirror
# ---------------------------------------------------------------------------
SOTU_YEARS = list(range(1960, 2026))


def download_sotu_corpus() -> None:
    """Fetch SOTU addresses 1960–2025 from a stable public source.

    First try the Miller Center / GovInfo combo. Fallback: scrape American
    Presidency Project (UCSB). The clean text is what matters; per-paragraph
    structure is preserved.
    """
    out_dir = CORPUS / "sotu"
    out_dir.mkdir(exist_ok=True)
    # Implementation: deferred to acquire stage; uses requests + BeautifulSoup.
    # Stub: this function is intentionally left for the next pass once the
    # site URL pattern is verified (UCSB's URL scheme uses node IDs, not years,
    # so we'll need to map year -> node first).
    print("  [sotu] (next pass — corpus fetch implemented after API discovery)")


# ---------------------------------------------------------------------------
# CDC NCHS — Life expectancy by race × sex, 1960+
# ---------------------------------------------------------------------------
def download_life_expectancy() -> None:
    # CDC NCHS National Vital Statistics System — life tables back to 1900.
    # Series: "United States Life Tables" annual reports.
    out = RAW / "cdc" / "life_expectancy_race_sex.csv"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        print("  [skip] life expectancy")
        return
    # Canonical: NCHS "Health, United States" Trend Tables. This is a manual
    # consolidation step — for the first pass we rely on a curated CSV that
    # 02_clean.py expects to find.
    print("  [cdc] (life expectancy table to be sourced from Health, US Trend Tables)")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = p.parse_args()

    print("== Stage 1: acquire ==")
    print("\n[FRED]")
    download_fred(FRED_SERIES.keys(), force=args.force)

    print("\n[NHE — CMS]")
    download_nhe()

    print("\n[NCES tuition]")
    download_nces_tuition()

    print("\n[Fed DFA by race]")
    download_dfa_race()

    print("\n[SOTU corpus]")
    download_sotu_corpus()

    print("\n[CDC life expectancy]")
    download_life_expectancy()

    print("\nDone.")


if __name__ == "__main__":
    main()
