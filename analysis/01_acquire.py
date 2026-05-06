"""Stage 1: acquire raw data from FRED, BLS, Census, NCES, NHE, DFA, CDC,
and SOTU corpus.

All downloads cache to data/raw/. Re-running is idempotent. Existing files are
skipped unless --force is passed.

Usage:
    export FRED_API_KEY=...
    python analysis/01_acquire.py [--force]
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


# FRED series: the macroeconomic backbone
FRED_SERIES: dict[str, str] = {
    # Inflation deflators
    "CPIAUCSL":         "CPI All Urban Consumers (monthly, 1947+)",
    "PCEPI":            "PCE Price Index (monthly, 1959+)",
    "CPIMEDSL":         "CPI Medical Care",
    "CUUR0000SEHA":     "CPI Rent of Primary Residence",

    # Wages / income
    "MEHOINUSA672N":    "Real Median Household Income (annual)",
    "LES1252881600Q":   "Median Weekly Earnings, Full-Time Wage/Salary (quarterly)",

    # Wage demographic cuts. BLS LEU series via FRED. Nominal weekly earnings,
    # full-time wage and salary workers, 16+, quarterly, 2000+.
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

    # Wealth shares (DFA, overall by wealth quintile, not race; race cuts
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
            print(f"    !! status {r.status_code} for {sid}, skipping")
            continue
        obs = r.json().get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty:
            print(f"    !! empty for {sid}")
            continue
        df.to_csv(out, index=False)
        time.sleep(0.25)


def download_nhe() -> None:
    """CMS National Health Expenditure historical tables, 1960+."""
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


def download_nces_tuition() -> None:
    """NCES Digest Table 330.10. Average tuition + fees + room/board, by
    institution type, current and constant dollars, since 1963."""
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


# Census P-38: median earnings by race, sex, year. Full-Time, Year-Round All
# Workers. Annual back to 1955 (white), 1967 (black), 1972 (hispanic). Used to
# backfill BLS demographic wage series pre-2000.
CENSUS_P38_RACES: dict[str, str] = {
    "white":         "p38w.xlsx",
    "black":         "p38b.xlsx",
    "hispanic":      "p38h.xlsx",
    "asian":         "p38a.xlsx",
    "white_nonhisp": "p38wnh.xlsx",
    "all_races":     "p38ar.xlsx",
}


def download_census_p38() -> None:
    """Census P-38: Full-Time Year-Round Workers, median annual earnings by sex,
    one Excel per race/ethnicity."""
    base = "https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-income-people"
    out_dir = RAW / "census"
    out_dir.mkdir(exist_ok=True)
    for race, fname in CENSUS_P38_RACES.items():
        out = out_dir / fname
        if out.exists():
            print(f"  [skip] census P-38 {race}")
            continue
        print(f"  [census] P-38 {race}")
        r = requests.get(f"{base}/{fname}", timeout=60)
        if r.status_code == 200:
            out.write_bytes(r.content)
        else:
            print(f"    !! status {r.status_code}")


def download_dfa_race() -> None:
    """Federal Reserve Distributional Financial Accounts, full bulk download.
    Includes wealth/income/networth shares and levels by race, age, generation,
    education, and income decile, quarterly 1989Q3 to present.
    """
    import zipfile
    out_dir = RAW / "dfa"
    out_dir.mkdir(exist_ok=True)
    zip_path = out_dir / "dfa.zip"
    if (out_dir / "dfa-race-levels.csv").exists():
        print("  [skip] DFA bulk")
        return
    url = "https://www.federalreserve.gov/releases/z1/dataviz/download/zips/dfa.zip"
    print("  [dfa] Federal Reserve DFA bulk")
    r = requests.get(url, timeout=90)
    if r.status_code != 200:
        print(f"    !! status {r.status_code}")
        return
    zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path) as z:
        # Extract just the by-race files; the full archive also has age/edu/etc.
        # which we may want later but don't need now.
        wanted = ("dfa-race-levels.csv", "dfa-race-shares.csv",
                  "dfa-data-definitions.txt")
        for name in wanted:
            try:
                z.extract(name, out_dir)
            except KeyError:
                print(f"    (missing in zip: {name})")


def download_cdc_life_expectancy() -> None:
    """CDC NCHS death rates and life expectancy at birth, 1900 to present,
    by race and sex. CDC open-data Socrata endpoint."""
    url = "https://data.cdc.gov/api/views/w9j2-ggv5/rows.csv?accessType=DOWNLOAD"
    out = RAW / "cdc" / "life_expectancy.csv"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        print("  [skip] CDC life expectancy")
        return
    print("  [cdc] life expectancy by race x sex")
    r = requests.get(url, timeout=60)
    if r.status_code == 200:
        out.write_bytes(r.content)
    else:
        print(f"    !! status {r.status_code}")


# State of the Union corpus, UCSB American Presidency Project mirror
SOTU_YEARS = list(range(1960, 2026))


def download_sotu_corpus() -> None:
    """Fetch SOTU addresses 1960 to 2025.

    Primary source: stdlib-js/datasets-sotu (covers 1790 to 2021). For each
    year in 1960 to 2021 the repo has files like 1965_lyndon_b_johnson_d.txt.
    The repo's directory listing is fetched via the GitHub Contents API,
    then each file's raw URL is downloaded.

    Post-2021 addresses (2022-2025) are scraped from the Miller Center, which
    has stable per-speech URLs.
    """
    out_dir = CORPUS / "sotu"
    out_dir.mkdir(exist_ok=True)

    # Primary: stdlib-js/datasets-sotu
    api = "https://api.github.com/repos/stdlib-js/datasets-sotu/contents/data"
    try:
        r = requests.get(api, timeout=30)
        listing = r.json() if r.status_code == 200 else []
    except Exception as e:
        print(f"    !! github listing failed: {e}")
        listing = []

    txt_files = [x for x in listing if isinstance(x, dict) and x.get("name", "").endswith(".txt")]
    fetched = 0
    for entry in txt_files:
        name = entry["name"]
        # Filter to 1960-2025 by parsing the year prefix
        try:
            year = int(name[:4])
        except ValueError:
            continue
        if year < 1960 or year > 2025:
            continue
        out = out_dir / name
        if out.exists():
            continue
        raw_url = entry.get("download_url")
        if not raw_url:
            continue
        try:
            rr = requests.get(raw_url, timeout=30)
            if rr.status_code == 200:
                out.write_bytes(rr.content)
                fetched += 1
                time.sleep(0.05)
        except Exception:
            pass
    print(f"  [sotu] stdlib-js corpus: {fetched} files fetched, "
          f"{len(list(out_dir.glob('*.txt')))} total in 1960-2025 range")

    # Recent SOTUs (2022-2025) from Miller Center. URL slugs are date-based:
    # /the-presidency/presidential-speeches/<month>-<day>-<year>-state-union
    miller_recent = [
        ("2022", "march-1-2022-state-union-address"),
        ("2023", "february-7-2023-state-union-address"),
        ("2024", "march-7-2024-state-union-address"),
        ("2025", "march-4-2025-joint-address-congress"),  # Trump 2025 was a joint address, not SOTU
    ]
    for year, slug in miller_recent:
        out = out_dir / f"{year}_recent.txt"
        if out.exists():
            continue
        url = f"https://millercenter.org/the-presidency/presidential-speeches/{slug}"
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        except Exception:
            continue
        if r.status_code == 200:
            # Crude extraction: between <div class="transcript-inner"> tags or
            # similar. We'll defer the real parsing to 02_clean and just save
            # the raw HTML for now.
            out.write_text(r.text, encoding="utf-8")
            print(f"  [sotu] {year} recent: saved")
        else:
            print(f"  [sotu] {year} recent: status {r.status_code}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = p.parse_args()

    print("== Stage 1: acquire ==")
    print("\n[FRED]")
    download_fred(FRED_SERIES.keys(), force=args.force)

    print("\n[NHE / CMS]")
    download_nhe()

    print("\n[NCES tuition]")
    download_nces_tuition()

    print("\n[Census P-38 historical income by race x sex]")
    download_census_p38()

    print("\n[Fed DFA bulk]")
    download_dfa_race()

    print("\n[CDC life expectancy]")
    download_cdc_life_expectancy()

    print("\n[SOTU corpus]")
    download_sotu_corpus()

    print("\nDone.")


if __name__ == "__main__":
    main()
