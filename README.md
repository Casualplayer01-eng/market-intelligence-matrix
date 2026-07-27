# Global Market Entry Simulation Engine

![Market Matrix](matrix.png)

A decision-support tool that automates the evaluation of global market trends using live World Bank economic data and a weighted scoring algorithm based on the GE-McKinsey Nine-Box Matrix.

[View Live Dashboard](https://market-entry-simulator.streamlit.app/) | [Download Market Entry Simulation Case Study (PDF)](case-study.pdf)

---

## How it works

The `RISK_ADJ_SCORE` for each market is derived from four weighted factors:

- **GDP growth (40%)** calculated from live World Bank data using a weighted historical average.
- **Labor capacity (30%)** based on labour force participation estimates.
- **Infrastructure (20%)** using structural market readiness scores.
- **Inflation risk (10%)** combining historical inflation and its volatility.

Economic projections (2035–2050) are generated using Monte Carlo simulation with Gaussian stochastic shocks (σ = 0.005), providing more robust long-term forecasts while accounting for macroeconomic uncertainty.
 
---

## Architecture

- **ETL pipeline:** Python-based automation running via GitHub Actions to fetch live World Bank data and persist the results to  `market_engine_cache.csv`.
- **Dashboard:** Streamlit/Plotly interface that visualizes data from the cached CSV, ensuring low-latency performance and decoupling the UI from external API constraints.
- **Result:** Markets are separated into "Target," "Watch," or "Avoid" categories, providing clear actionable business insights.
  
---

## Project Structure

```
├── .github/workflows/      # CI/CD pipeline for automated data updates
├── app.py                  # Streamlit dashboard
├── main.py                 # ETL engine and scoring logic
├── market_engine_cache.csv # Processed market data (auto-updated)
└── requirements.txt        # Dependencies
```

---

## How to set it up locally

```bash
git clone [your-repo-url]
pip install -r requirements.txt
python main.py        # updates the cache
streamlit run app.py  # launch the dashboard
```

---
