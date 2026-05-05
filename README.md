# The American Dream, Audited

A data-driven audit of how the US economic promise played out across three life stages and four demographic groups, 1960 to today.

**Submission for:** DSAN Spring 2026 scholarship, Georgetown University, Department of Data Science and Analytics
**Live site:** https://dream.my677.georgetown.domains/

> Submission is reviewed anonymously. The author is identified only by NetID (`my677`).

## Build

```bash
# Python 3.11+ with pandas, numpy, plotly, requests, pyarrow, openpyxl,
# scikit-learn, sentence-transformers, bertopic, transformers, torch

# 1. Set FRED API key (free, register at https://fred.stlouisfed.org/docs/api/api_key.html)
export FRED_API_KEY=...

# 2. Run pipeline (idempotent; caches to data/raw)
python analysis/01_acquire.py
python analysis/02_clean.py
python analysis/03_analyze.py
python analysis/04_interactive.py
python analysis/05_nlp.py

# 3. Render site
quarto render
quarto preview
```

## Repository structure

```
analysis/      modular Python pipeline (01_acquire to 05_nlp)
data/          raw, processed, corpus
figures/       static plots
interactive/   self-contained HTML viz embedded in the article
assets/        images, fonts, og-card
```

## Methodology
See `appendix.qmd` for full methodology, data sources, and caveats.
