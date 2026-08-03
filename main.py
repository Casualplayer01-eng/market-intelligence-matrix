"""
World Bank Macroeconomic ETL & Portfolio Market Engine
======================================================
Production-ready ETL pipeline fetching live World Bank indicator data,
computing macro metrics, running Gaussian shock projections, and generating
investment recommendations.

Includes persistent HTTP session retry handling, data validation, and automated
cache fallback to ensure pipeline resilience in CI/CD environments.
"""

import datetime
import logging
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("WB_ETL_Engine")

# --- CONFIGURATION & CONSTANTS ---
CACHE_FILE = "market_engine_cache.csv"
COUNTRIES = [
    "USA", "JPN", "CHN", "IND", "CHE", "KOR", "NLD", "SAU", "ARE", 
    "SGP", "DEU", "PHL", "MYS", "QAT", "BHR", "CAN", "FRA", "GBR"
]

# Official World Bank API Indicators
INDICATOR_GDP_GROWTH = "NY.GDP.MKTP.KD.ZG"  # GDP growth (annual %)
INDICATOR_INFLATION  = "FP.CPI.TOTL.ZG"      # Inflation, consumer prices (annual %)

# Preserved Architectural Dictionaries
EODB_SCORES = {
    "USA": 0.90, "JPN": 0.85, "CHN": 0.70, "IND": 0.60, "CHE": 0.95, 
    "KOR": 0.80, "NLD": 0.90, "SAU": 0.65, "ARE": 0.85, "SGP": 0.99, 
    "DEU": 0.80, "PHL": 0.50, "MYS": 0.75, "QAT": 0.70, "BHR": 0.70, 
    "CAN": 0.90, "FRA": 0.80, "GBR": 0.90
}

LABOR_ESTIMATES = {
    "USA": 0.63, "JPN": 0.62, "CHN": 0.67, "IND": 0.52, "CHE": 0.67, 
    "KOR": 0.64, "NLD": 0.67, "SAU": 0.61, "ARE": 0.72, "SGP": 0.68, 
    "DEU": 0.61, "PHL": 0.64, "MYS": 0.65, "QAT": 0.85, "BHR": 0.70, 
    "CAN": 0.66, "FRA": 0.56, "GBR": 0.63
}


def create_robust_session() -> requests.Session:
    """
    Creates a persistent requests.Session configured with exponential backoff retries
    and standard browser User-Agent headers to handle transient World Bank API failures.
    """
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,  # Delays: 1s, 2s, 4s, 8s, 16s
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        "User-Agent": "WorldBankETLEngine/1.0",
        "Accept": "application/json"
    })
    return session


def fetch_wb_indicator(
    session: requests.Session, 
    country: str, 
    indicator: str, 
    start_year: int = 2014, 
    end_year: int = 2025
) -> List[float]:
    """
    Fetches raw indicator series from World Bank API with timeout and explicit validation.
    Raises RuntimeError on failed HTTP requests or invalid response payloads.
    """
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 50
    }
    
    try:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP request failed for country {country}, indicator {indicator}: {exc}") from exc

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"Invalid API payload structure for country '{country}', indicator '{indicator}'.")
    
    records = payload[1]
    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError(f"No indicator records found for country '{country}', indicator '{indicator}'.")

    valid_values: List[float] = []
    for entry in records:
        val = entry.get("value")
        if val is not None:
            try:
                valid_values.append(float(val))
            except (ValueError, TypeError):
                continue

    if not valid_values:
        raise RuntimeError(f"Zero valid numerical observations returned for country '{country}', indicator '{indicator}'.")

    return valid_values


def compute_macro_metrics(gdp_series: List[float], inf_series: List[float]) -> Tuple[float, float, float]:
    """
    Computes simple arithmetic mean (np.mean) for GDP growth and inflation,
    and inflation volatility (sample std dev). Converts percentage to decimal.
    """
    gdp_growth = float(np.mean(gdp_series)) / 100.0
    inf_avg = float(np.mean(inf_series)) / 100.0
    inf_vol = float(np.std(inf_series, ddof=1)) / 100.0 if len(inf_series) > 1 else 0.0
    return gdp_growth, inf_avg, inf_vol


def run_gaussian_shock_projections(
    gdp_growth: float,
    start_index: float = 100.0,
    target_years: List[int] = [2035, 2040, 2045, 2050],
    stoch_std: float = 0.005
) -> Dict[str, float]:
    """
    Generates GDP growth projections using Gaussian stochastic shocks across target horizon years.
    """
    projections = {}
    base_year = 2025
    for year in target_years:
        years_span = year - base_year
        shock = np.random.normal(0, stoch_std)
        effective_rate = gdp_growth + shock
        proj_val = start_index * ((1.0 + effective_rate) ** years_span)
        projections[f"Proj_{year}"] = float(round(proj_val, 2))
    return projections


def load_cached_data() -> Dict[str, Dict[str, Any]]:
    """Loads existing cache file if present for fallback operations."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        cache_df = pd.read_csv(CACHE_FILE)
        return cache_df.set_index("country").to_dict(orient="index")
    except Exception as exc:
        logger.warning(f"Could not read existing cache file {CACHE_FILE}: {exc}")
        return {}


def build_engine() -> pd.DataFrame:
    """
    Main ETL execution pipeline:
      1. Sequential live fetch with persistent Session, retry backoff, and timeouts.
      2. Automatic fallback to market_engine_cache.csv on transient API failure.
      3. Original weighted scoring formula.
      4. Gaussian shock projections.
      5. Quantile recommendation grouping via pd.qcut().
    """
    print("--- STARTING WORLD BANK ETL ENGINE ---")
    session = create_robust_session()
    cached_records = load_cached_data()

    country_data: List[Dict[str, Any]] = []
    
    live_countries: List[str] = []
    cached_countries: List[str] = []
    failed_countries: List[str] = []

    for code in COUNTRIES:
        logger.info(f"Fetching live World Bank data for: {code}")
        try:
            gdp_series = fetch_wb_indicator(session, code, INDICATOR_GDP_GROWTH)
            inf_series = fetch_wb_indicator(session, code, INDICATOR_INFLATION)
            
            gdp_growth, inf_avg, inf_vol = compute_macro_metrics(gdp_series, inf_series)
            labor = LABOR_ESTIMATES.get(code, 0.60)
            infra = EODB_SCORES.get(code, 0.50)
            inf_risk = inf_avg + (0.5 * inf_vol)

            country_data.append({
                "country": code,
                "GDP_Growth": round(gdp_growth, 4),
                "Labor_Participation": labor,
                "Infrastructure": infra,
                "Inflation_Avg": inf_avg,
                "Inflation_Vol": inf_vol,
                "Inflation_Risk": inf_risk
            })
            live_countries.append(code)

        except Exception as exc:
            logger.warning(f"Failed to fetch live data for {code}: {exc}")
            if code in cached_records:
                logger.info(f"LOG: Using cached values from {CACHE_FILE} for country: {code}")
                c_row = cached_records[code]
                country_data.append({
                    "country": code,
                    "GDP_Growth": c_row.get("GDP_Growth", 0.02),
                    "Labor_Participation": c_row.get("Labor_Participation", LABOR_ESTIMATES.get(code, 0.60)),
                    "Infrastructure": c_row.get("Infrastructure", EODB_SCORES.get(code, 0.50)),
                    "Inflation_Risk": c_row.get("Inflation_Risk", 0.02)
                })
                cached_countries.append(code)
            else:
                logger.error(f"No cache entry found for {code}. Skipping.")
                failed_countries.append(code)

    if not country_data:
        raise RuntimeError("Pipeline failed: Unable to obtain data from either live API or cache.")

    df = pd.DataFrame(country_data)

    # Ensure Inflation_Risk is calculated if missing from fallback
    if "Inflation_Risk" not in df.columns:
        df["Inflation_Risk"] = df["Inflation_Avg"] + (0.5 * df["Inflation_Vol"])

    # Original Weighted Scoring Formula
    df["RISK_ADJ_SCORE"] = (
        (df["GDP_Growth"] * 0.4) + 
        (df["Labor_Participation"] * 0.3) + 
        (df["Infrastructure"] * 0.2) - 
        (df["Inflation_Risk"] * 0.1)
    ).round(4)

    # Gaussian Shock Projections
    proj_rows = []
    for _, row in df.iterrows():
        projs = run_gaussian_shock_projections(gdp_growth=row["GDP_Growth"])
        proj_rows.append(projs)

    proj_df = pd.DataFrame(proj_rows)
    df = pd.concat([df, proj_df], axis=1)

    # Recommendations using pd.qcut()
    df["Recommendation"] = pd.qcut(
        df["RISK_ADJ_SCORE"], 
        q=3, 
        labels=["Avoid", "Watch", "Target"]
    )

    df["Last_Updated"] = datetime.datetime.now().strftime("%Y-%m-%d")

    # Final Schema Match
    schema_cols = [
        "country", "RISK_ADJ_SCORE", "GDP_Growth", "Labor_Participation", 
        "Infrastructure", "Proj_2035", "Proj_2040", "Proj_2045", "Proj_2050", 
        "Recommendation", "Last_Updated"
    ]
    
    final_df = df[schema_cols].copy()
    final_df.to_csv(CACHE_FILE, index=False)

    # Execution Summary
    print("\n================ EXECUTION SUMMARY ================")
    print(f"Live Data Fetch Succeeded ({len(live_countries)}): {', '.join(live_countries) if live_countries else 'None'}")
    print(f"Fallback to Cache Used     ({len(cached_countries)}): {', '.join(cached_countries) if cached_countries else 'None'}")
    print(f"Failed / Excluded          ({len(failed_countries)}): {', '.join(failed_countries) if failed_countries else 'None'}")
    print("===================================================\n")

    print(f"--- PIPELINE COMPLETE: Cache saved to {CACHE_FILE} ---")
    print(final_df.to_string(index=False))
    
    return final_df


if __name__ == "__main__":
    build_engine()
