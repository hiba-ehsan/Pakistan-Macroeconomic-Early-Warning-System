# Pakistan Macroeconomic Early Warning System

**A Multi-GARCH & Markov Regime-Switching Framework for Detecting Institutional Fragility**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Quarto](https://img.shields.io/badge/Report-Quarto-blue?logo=quarto)](https://quarto.org)

---

## Overview

This project builds a **reproducible, Python-based early-warning framework** to identify which of Pakistan's core macroeconomic indicators enters a high-volatility regime first before a major economic crisis materializes.

Using 20 years of monthly data (2005–2026) sourced from the **State Bank of Pakistan (SBP)**, the framework applies:

- **GARCH(1,1) & EGARCH(1,1)** models for conditional volatility estimation
- **Markov Regime-Switching** models (2-state) for regime probability extraction
- **Lead-time analysis** measuring how many months before each crisis onset an indicator crossed the high-volatility threshold
- **Robustness sweeps** across multiple thresholds, lookback windows, and smoothed vs. filtered probabilities

### Key Finding

> **The PKR/USD exchange rate is the most consistent leading indicator** : it entered the high-volatility regime ahead of **3 out of 4** major crises and survived sensitivity checks across all parameter combinations tested.

---

## Crises Examined

| Crisis | Period | Trigger |
|--------|--------|---------|
| **2008 Crisis** | Jan 2008 – Jun 2009 | Global financial crisis → $7.6B IMF bailout |
| **2013 Crisis** | Jan 2013 – Jun 2014 | Energy crisis + FX reserve depletion |
| **2018–19 Crisis** | Jan 2018 – Dec 2019 | Balance-of-payments gap → 30% rupee devaluation |
| **2022–23 Crisis** | Jan 2022 – Dec 2023 | Commodity shocks, inflation peaked 38%, near-default |

## Indicators Analyzed

| Indicator | Frequency | Transformation |
|-----------|-----------|----------------|
| CPI Inflation (YoY) | Monthly | First difference |
| PKR/USD Exchange Rate | Monthly | Log-difference (%) |
| Current Account Balance | Quarterly → Monthly (forward-fill) | First difference |
| GDP Growth Rate | Annual → Monthly (broadcast) | First difference |

---

## Project Structure

```
Multi-GARCH/
├── data/                          # Raw datasets (SBP sources)
│   ├── pakistan_cpi_monthly_2001_2026.csv
│   ├── PKR_USD_Monthly_2005_2026.csv
│   ├── _Current_Account_Balance__Summary_of_Balance_of_Payments_as_per_BPM6.csv
│   └── Gross_Domestic_Product_of_Pakistan_at_constant_basic_prices_of_2015-16.csv
│
├── src/                           # Modular Python source code
│   ├── loading.py                 # Data loading utilities per indicator
│   ├── preprocessing.py           # Monthly harmonization & stationarity transforms
│   ├── diagnostics.py             # ADF, KPSS, ARCH-LM, Ljung-Box tests + ACF/PACF plots
│   ├── garch.py                   # GARCH/EGARCH fitting, model comparison, conditional volatility
│   └── markov.py                  # Markov regime-switching models, regime probabilities, lead times
│
├── report/                        # Quarto report (full analysis + narrative)
│   ├── pakistan_macro_report_final.qmd    # Reproducible Quarto document
│   ├── pakistan_macro_report_final.pdf    # Rendered PDF report
│   └── pakistan_macro_report_final.tex    # LaTeX intermediate
│
├── _quarto.yml                    # Quarto project configuration
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Quarto** (optional, only needed to re-render the report) — [Install Quarto](https://quarto.org/docs/get-started/)

### Installation

```bash
# Clone the repository
git clone https://github.com/<hiba-ehsan>/Pakistan-Macroeconomic-Early-Warning-System.git
cd Pakistan-Macroeconomic-Early-Warning-System

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Render the Report

```bash
quarto render report/pakistan_macro_report_final.qmd
```

This regenerates the full analysis, all the figures, tables, diagnostics, and conclusions, from source data.

### Use the Source Modules Directly

```python
from src.loading import cpi, fx, cabalance, gdp
from src.preprocessing import monthlyframe, transformations, SERIES_LABELS
from src.garch import allgarch, summary, conditionalvol
from src.markov import allmarkov, regimeprobs, leadtimes

# Load & harmonize data
macro_m = monthlyframe(cpi(), fx(), cabalance(), gdp())
diff_df = transformations(macro_m)

# Fit GARCH models
garch_results = allgarch(diff_df, SERIES_LABELS)
print(summary(garch_results))

# Fit Markov models & compute lead times
markov_results = allmarkov(diff_df, SERIES_LABELS)
prob_df = regimeprobs(markov_results, diff_df)
lead_df = leadtimes(prob_df, SERIES_LABELS)
print(lead_df)
```

---

## Methodology

```
Raw Data (SBP)
    │
    ▼
Harmonize to Monthly Timeline (2005–2026)
    │
    ▼
Stationarity Transforms (diff / log-diff)
    │
    ▼
Diagnostic Tests (ADF, KPSS, ARCH-LM)
    │
    ▼
┌───────────────────┬────────────────────────┐
│  GARCH / EGARCH   │  Markov Regime-Switch  │
│  → Cond. Vol.     │  → P(High Volatility)  │
└───────────────────┴────────────────────────┘
    │
    ▼
Lead-Time Analysis: months before crisis onset
    │
    ▼
Robustness Sweeps (threshold × lookback × smoothed vs. filtered)
    │
    ▼
Conclusion: FX Rate = most consistent early-warning signal
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| [`arch`](https://arch.readthedocs.io/) | GARCH / EGARCH volatility models |
| [`statsmodels`](https://www.statsmodels.org/) | Markov regime-switching, ADF, KPSS, ARCH-LM, Ljung-Box |
| [`pandas`](https://pandas.pydata.org/) | Data wrangling & time series |
| [`numpy`](https://numpy.org/) | Numerical operations |
| [`matplotlib`](https://matplotlib.org/) | Static plots |
| [`seaborn`](https://seaborn.pydata.org/) | Correlation heatmaps |
| [`scipy`](https://scipy.org/) | Statistical functions |

---

## Limitations

- **GDP** is broadcast from annual to monthly, so regime timing for GDP is a coarse benchmark, not a genuine monthly signal.
- **Current Account** GARCH model fails the post-fit Ljung-Box diagnostic, making it the weakest indicator in this framework.
- Main lead-time results use **smoothed probabilities** (retrospective). Section 8 of the report compares against filtered (real-time) probabilities.
- This is a **descriptive, retrospective** analysis and not an operational real-time early-warning system. Out-of-sample validation with expanding-window estimation is a natural next step.

---

## References

1. Engle, R.F. (1982). *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation.* Econometrica, 50(4).
2. Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica, 57(2).
3. Abiad, A. (2003). *Early Warning Systems: A Survey and a Regime-Switching Approach.* IMF Working Paper WP/03/32.
4. Ahmed, A.M. & Qayyum, A. (2007). *Do Fiscal and Monetary Policies Lead to a Crisis? Evidence from Pakistan.* The Pakistan Development Review, 46(4).
5. 5. Mahmood, M., & Chaudhry, S. (2020). Pakistan's Balance-of-Payments Crisis and Some Policy Options. *The Lahore Journal of Economics*, 25(2), 55–92.
6. Gohar, M.M., & Kanwal, L. (2024). Assessing The Governance Of Coalition Government: The Case Of Pakistan (2008-2013). *Migration Letters*, 21(S14), 1008–1017.


---

## Author

**Hiba Ehsan** 
Linkedin: https://www.linkedin.com/in/hiba-ehsan/
---

## License

This project is licensed under the [MIT License](LICENSE).
