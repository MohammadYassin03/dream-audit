"""Stage 2: clean raw data into a tidy long-format master table.

Outputs:
    data/processed/series_long.parquet, tidy (date, series, value) for all numeric series
    data/processed/series_meta.csv, series metadata (units, source, demographic)
    data/processed/cpi.csv, monthly CPI for deflation
    data/processed/wages_demographic.parquet, wages by race x sex, annualized
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


# Census P-38 backfill, pre-2000 demographic wages
P38_FILE_BY_RACE = {
    "white":    "p38w.xlsx",
    "black":    "p38b.xlsx",
    "hispanic": "p38h.xlsx",
}

# Map race x sex pair to the BLS demographic cell name used in series_meta.
RACE_SEX_TO_DEMOGRAPHIC = {
    ("white", "male"):      "white_men",
    ("white", "female"):    "white_women",
    ("black", "male"):      "black_men",
    ("black", "female"):    "black_women",
    ("hispanic", "male"):   "hispanic_men",
    ("hispanic", "female"): "hispanic_women",
}


def _parse_p38(race: str) -> pd.DataFrame:
    """Parse one Census P-38 Excel file into long format.

    Columns produced: year, sex, annual_earnings_nominal.
    Footnote markers like "(12)" are stripped from the year column. Where a
    methodology revision causes a duplicate year (Census reports both the
    old and new basis for that year), keep the first occurrence (the
    post-revision figure).
    """
    fp = RAW / "census" / P38_FILE_BY_RACE[race]
    raw = pd.read_excel(fp, header=None)
    rows = raw[raw[0].astype(str).str.match(r"^\s*[12][0-9]{3}", na=False)].copy()
    rows[0] = rows[0].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    rows = rows.drop_duplicates(subset=[0], keep="first")

    male = pd.DataFrame({
        "year": rows[0],
        "sex": "male",
        "annual_earnings_nominal": pd.to_numeric(rows[2], errors="coerce"),
    })
    female = pd.DataFrame({
        "year": rows[0],
        "sex": "female",
        "annual_earnings_nominal": pd.to_numeric(rows[5], errors="coerce"),
    })
    out = pd.concat([male, female], ignore_index=True).dropna()
    out["race"] = race
    return out


def _build_stitched_wages(annual_bls: pd.DataFrame) -> pd.DataFrame:
    """Stitch Census P-38 (1967+, annual / 52 -> implied weekly) to BLS LEU
    (2000+, observed weekly) at the 2000 break, with a multiplicative scale
    factor that anchors Census levels to BLS at the boundary.

    Produces (year, demographic, weekly_earnings_nominal, source) where
    source is one of {'census_p38_stitched', 'bls'}.
    """
    pieces = []
    for race in P38_FILE_BY_RACE:
        pieces.append(_parse_p38(race))
    census_long = pd.concat(pieces, ignore_index=True)

    # Map race x sex to demographic
    census_long["demographic"] = census_long.apply(
        lambda r: RACE_SEX_TO_DEMOGRAPHIC.get((r["race"], r["sex"])), axis=1
    )
    census_long = census_long.dropna(subset=["demographic"])
    census_long["weekly_implied"] = census_long["annual_earnings_nominal"] / 52.0

    # BLS annual averages from raw quarterly observations
    bls_annual = (
        annual_bls.groupby(["year", "demographic"], as_index=False)["weekly_earnings_nominal"]
        .mean()
    )

    out_pieces = []
    for dem in RACE_SEX_TO_DEMOGRAPHIC.values():
        c = census_long.query("demographic == @dem")[["year", "weekly_implied"]].sort_values("year")
        b = bls_annual.query("demographic == @dem")[["year", "weekly_earnings_nominal"]].sort_values("year")
        if c.empty or b.empty:
            continue

        # Scale factor at the BLS/Census overlap boundary (first BLS year that
        # also exists in Census). Average across a 3-yr window for stability.
        boundary = b["year"].min()
        win = list(range(boundary, boundary + 3))
        c_win = c[c["year"].isin(win)]["weekly_implied"].mean()
        b_win = b[b["year"].isin(win)]["weekly_earnings_nominal"].mean()
        if c_win > 0:
            scale = b_win / c_win
        else:
            scale = 1.0

        # Pre-boundary: scaled Census. Boundary onward: BLS observed.
        pre = c[c["year"] < boundary].copy()
        pre["weekly_earnings_nominal"] = pre["weekly_implied"] * scale
        pre = pre[["year", "weekly_earnings_nominal"]]
        pre["source"] = "census_p38_stitched"

        post = b.copy()
        post["source"] = "bls"

        merged = pd.concat([pre, post], ignore_index=True)
        merged["demographic"] = dem
        merged["scale_factor_to_bls"] = scale
        out_pieces.append(merged)

    return pd.concat(out_pieces, ignore_index=True)


def clean_life_expectancy() -> pd.DataFrame:
    """CDC NCHS life expectancy + age-adjusted death rate, 1900 to 2018,
    by race x sex. Filters out 'Both Sexes' and 'All Races' aggregates so the
    output matches the audit's race x sex demographic convention.
    """
    fp = RAW / "cdc" / "life_expectancy.csv"
    df = pd.read_csv(fp).rename(columns={
        "Year": "year",
        "Race": "race",
        "Sex": "sex",
        "Average Life Expectancy (Years)": "life_expectancy",
        "Age-adjusted Death Rate": "death_rate",
    })
    out = df.copy()
    sex_map = {"Male": "men", "Female": "women", "Both Sexes": "all"}
    race_map = {"White": "white", "Black": "black", "All Races": "all"}
    out["sex_norm"] = out["sex"].map(sex_map)
    out["race_norm"] = out["race"].map(race_map)
    out["demographic"] = out.apply(
        lambda r: f"{r['race_norm']}_{r['sex_norm']}"
        if r["race_norm"] != "all" and r["sex_norm"] != "all"
        else f"{r['race_norm']}_total" if r["sex_norm"] == "all"
        else f"all_{r['sex_norm']}",
        axis=1
    )
    return out[["year", "demographic", "life_expectancy", "death_rate", "race", "sex"]]


def clean_nces_tuition() -> pd.DataFrame:
    """NCES Digest Table 330.10. Average tuition, fees, room and board.

    Returns long format with columns:
        year (int)            academic year start (1963 from "1963-64")
        institution_type      'all', 'public', 'private'
        cost_category         'total', 'tuition_fees'
        institution_level     'all', '4yr', '2yr'
        amount_nominal        current-dollar US$
    """
    fp = RAW / "nces" / "tabn330_10.xlsx"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_excel(fp, header=None)

    # Section header rows discovered by inspection
    sections = [
        (5,  60, "all"),       # "All institutions"
        (61, 116, "public"),   # "Public institutions"
        (117, 172, "private"), # "Private (nonprofit + for-profit)"
    ]
    pieces = []
    for start, end, label in sections:
        block = df.iloc[start + 1:end + 1].copy()
        # Year column: e.g. "1963-64 ". Strip and take leading 4 digits.
        block["year"] = block[0].astype(str).str.extract(r"(\d{4})")[0]
        block = block.dropna(subset=["year"])
        block["year"] = block["year"].astype(int)

        # Current-dollar columns (cols 13 to 21):
        #   13-15: total tuition+fees+room+board, All / 4yr / 2yr
        #   16-18: tuition+required-fees only, All / 4yr / 2yr
        col_map = {
            ("total", "all"):           13,
            ("total", "4yr"):           14,
            ("total", "2yr"):           15,
            ("tuition_fees", "all"):    16,
            ("tuition_fees", "4yr"):    17,
            ("tuition_fees", "2yr"):    18,
        }
        for (cat, lvl), col in col_map.items():
            piece = block[["year", col]].rename(columns={col: "amount_nominal"})
            piece["amount_nominal"] = pd.to_numeric(piece["amount_nominal"], errors="coerce")
            piece = piece.dropna(subset=["amount_nominal"])
            piece["institution_type"] = label
            piece["cost_category"] = cat
            piece["institution_level"] = lvl
            pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def clean_dfa_race() -> pd.DataFrame:
    """Federal Reserve DFA race shares. Quarterly 1989Q3+, by race x asset class.

    Output: long format (date, year, race, metric, share).
    """
    fp = RAW / "dfa" / "dfa-race-shares.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["date"] = pd.PeriodIndex(df["Date"].str.replace(":", ""), freq="Q").to_timestamp()
    df["year"] = df["date"].dt.year
    df = df.rename(columns={"Category": "race"})
    metric_cols = [c for c in df.columns if c not in ("Date", "date", "year", "race")]
    long = df.melt(id_vars=["date", "year", "race"],
                   value_vars=metric_cols, var_name="metric", value_name="share")
    long["share"] = pd.to_numeric(long["share"], errors="coerce")
    return long.dropna()


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

    # CPI helper, used by every downstream deflation step
    cpi = long.query("series == 'CPIAUCSL'")[["date", "value"]].rename(columns={"value": "cpi"})
    cpi.to_csv(PROC / "cpi.csv", index=False)
    print(f"  cpi.csv:             {len(cpi):,} months")

    # Wages by demographic, annualized. Central to the time-price calc.
    # Six primary cells (race x sex). The "all" / "*_all" series stay in
    # series_long.parquet for cross-checks but are excluded from the
    # demographic table.
    dem_series = SERIES_META[SERIES_META["demographic"].isin(
        ["white_men", "white_women", "black_men", "black_women", "hispanic_men", "hispanic_women"]
    )]
    wages = long.merge(dem_series[["series", "demographic"]], on="series")
    wages = wages.rename(columns={"value": "weekly_earnings_nominal"})
    wages["year"] = wages["date"].dt.year
    annual = (wages.groupby(["year", "demographic"], as_index=False)["weekly_earnings_nominal"].mean())
    annual.to_parquet(PROC / "wages_demographic.parquet", index=False)
    print(f"  wages_demographic.parquet: {len(annual):,} rows (BLS only, 2000+)")

    # NCES tuition. Public/Private/All, by 4yr/2yr, by Total or Tuition+Fees, 1963-2022.
    tuition = clean_nces_tuition()
    if not tuition.empty:
        tuition.to_parquet(PROC / "tuition.parquet", index=False)
        print(f"  tuition.parquet: {len(tuition):,} rows ({tuition['year'].min()} to {tuition['year'].max()})")
    else:
        print("  (skip tuition, NCES file missing)")

    # CDC life expectancy. Long arc 1900-2018, race x sex.
    if (RAW / "cdc" / "life_expectancy.csv").exists():
        le = clean_life_expectancy()
        le.to_parquet(PROC / "life_expectancy.parquet", index=False)
        race_sex = le[le["demographic"].str.match(r"^(white|black)_(men|women)$")]
        coverage = race_sex.groupby("demographic")["year"].agg(["min", "max", "count"])
        print("  life_expectancy.parquet:")
        print(coverage.to_string().replace("\n", "\n    "))
    else:
        print("  (skip life expectancy, CDC file missing)")

    # Fed DFA wealth shares. Race by asset class, quarterly 1989Q3+.
    dfa = clean_dfa_race()
    if not dfa.empty:
        dfa.to_parquet(PROC / "dfa_race_shares.parquet", index=False)
        print(f"  dfa_race_shares.parquet: {len(dfa):,} rows ({dfa['race'].nunique()} races, "
              f"{dfa['metric'].nunique()} metrics)")
    else:
        print("  (skip DFA, bulk file missing)")

    # Stitched series. Census P-38 backfill (1967+) joined to BLS at 2000.
    # This is what downstream time-price calculations should use.
    if (RAW / "census" / "p38w.xlsx").exists():
        stitched = _build_stitched_wages(annual)
        stitched.to_parquet(PROC / "wages_demographic_stitched.parquet", index=False)
        coverage = stitched.groupby("demographic")["year"].agg(["min", "max", "count"])
        print("  wages_demographic_stitched.parquet:")
        print(coverage.to_string().replace("\n", "\n    "))
    else:
        print("  (skip stitched, Census P-38 files not yet downloaded)")

    print("Done.")


if __name__ == "__main__":
    main()
