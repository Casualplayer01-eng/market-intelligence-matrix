"""
World Bank Macroeconomic ETL & Portfolio Market Engine
======================================================
Production-ready ETL pipeline fetching live World Bank indicator data,
computing weighted growth and risk metrics, running Monte Carlo projections,
and generating investment recommendations.

Fully compatible with Streamlit, GitHub Actions, and CSV caching.
"""

import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    Creates a requests.Session configured with exponential backoff retries
    and standard browser User-Agent headers to prevent CDN throttling.
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WorldBankETLEngine/2.0",
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
    Fetches raw indicator series from World Bank API with explicit validation.
    Returns chronologically sorted list of float values (oldest to newest).
    Raises RuntimeError on invalid API responses to prevent silent default values.
    """
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 50
    }
    
    try:
        response = session.get(url, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP request failure for country {country}, indicator {indicator}: {exc}") from exc

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"Invalid API response structure for country '{country}', indicator '{indicator}'. Payload: {payload}")
    
    records = payload[1]
    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError(f"No indicator records found for country '{country}', indicator '{indicator}'.")

    valid_points: List[Tuple[int, float]] = []
    for entry in records:
        val = entry.get("value")
        year_str = entry.get("date")
        if val is not None and year_str is not None:
            try:
                valid_points.append((int(year_str), float(val)))
            except (ValueError, TypeError):
                continue

    if not valid_points:
        raise RuntimeError(f"Zero valid numerical observations returned for country '{country}', indicator '{indicator}'.")

    # Sort chronologically (oldest -> newest)
    valid_points.sort(key=lambda x: x[0])
    return [pt[1] for pt in valid_points]


def compute_weighted_growth(growth_rates_pct: List[float]) -> float:
    """
    Computes linearly weighted average growth rate (decimal).
    Recent years receive higher weight (w_1=1, w_2=2, ..., w_N=N).
    Converts raw percentage points (e.g. 3.5%) to decimal (0.035).
    """
    n = len(growth_rates_pct)
    weights = np.arange(1, n + 1, dtype=float)
    weighted_avg_pct = np.sum(np.array(growth_rates_pct) * weights) / np.sum(weights)
    return float(weighted_avg_pct / 100.0)


def compute_inflation_metrics(inflation_rates_pct: List[float]) -> Tuple[float, float]:
    """
    Computes linearly weighted mean inflation rate and inflation volatility (sample std dev).
    Returns (weighted_mean_decimal, volatility_std_decimal).
    """
    n = len(inflation_rates_pct)
    weights = np.arange(1, n + 1, dtype=float)
    arr = np.array(inflation_rates_pct)
    
    weighted_mean_pct = np.sum(arr * weights) / np.sum(weights)
    volatility_pct = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    return float(weighted_mean_pct / 100.0), float(volatility_pct / 100.0)


def run_monte_carlo_projections(
    base_velocity: float,
    start_index: float = 100.0,
    target_years: List[int] = [2035, 2040, 2045, 2050],
    n_simulations: int = 1000,
    stoch_std: float = 0.005,
    decay_rate: float = 0.98,
    steady_state_growth: float = 0.02
) -> Dict[str, float]:
    """
    Runs a 1,000-path Monte Carlo simulation with growth rate mean-reversion.
    Growth velocity decays toward steady-state equilibrium (2.0% p.a.).
    Returns expected mean projected index values for target years.
    """
    n_years_span = max(target_years) - 2025
    simulations = np.zeros((n_simulations, len(target_years)))
    
    for sim in range(n_simulations):
        val = start_index
        vel = base_velocity
        current_year = 2025
        target_idx = 0
        
        for _ in range(1, (n_years_span // 5) + 1):
            shock = np.random.normal(0, stoch_std)
            block_vel = float(np.clip(vel + shock, -0.02, 0.10))
            
            # Compound index across 5-year block
            val *= ((1.0 + block_vel) ** 5)
            
            # Mean revert growth velocity toward steady-state equilibrium
            vel = vel * decay_rate + steady_state_growth * (1.0 - decay_rate)
            
            current_year += 5
            if current_year in target_years:
                simulations[sim, target_idx] = val
                target_idx += 1

    expected_vals = np.mean(simulations, axis=0)
    return {f"Proj_{y}": float(round(expected_vals[i], 2)) for i, y in enumerate(target_years)}


def fetch_country_macro_data(session: requests.Session, code: str) -> Dict[str, Any]:
    """
    Fetches GDP growth and Inflation data for a single country and computes raw metrics.
    """
    logger.info(f"Fetching live World Bank data for: {code}")
    
    gdp_series = fetch_wb_indicator(session, code, INDICATOR_GDP_GROWTH)
    inf_series = fetch_wb_indicator(session, code, INDICATOR_INFLATION)
    
    gdp_velocity = compute_weighted_growth(gdp_series)
    gdp_velocity = float(np.clip(gdp_velocity, -0.02, 0.10))
    
    inf_avg, inf_vol = compute_inflation_metrics(inf_series)
    
    labor = LABOR_ESTIMATES.get(code, 0.60)
    infra = EODB_SCORES.get(code, 0.50)
    
    # Risk Metric combining average inflation and volatility
    inf_risk = inf_avg + (0.5 * inf_vol)
    
    return {
        "country": code,
        "GDP_Growth": gdp_velocity,
        "Labor_Participation": labor,
        "Infrastructure": infra,
        "Inflation_Avg": inf_avg,
        "Inflation_Vol": inf_vol,
        "Inflation_Risk": inf_risk
    }


def build_engine() -> pd.DataFrame:
    """
    Main ETL execution pipeline:
      1. Parallel live API extraction with Session pooling & retry backoff.
      2. Feature scaling & consultant risk-adjusted scoring.
      3. Monte Carlo projections with growth rate mean-reversion.
      4. Recommendation categorization & CSV Export.
    """
    print("--- STARTING WORLD BANK ETL ENGINE ---")
    session = create_robust_session()
    raw_country_data: List[Dict[str, Any]] = []
    
    # Parallel API execution via ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_country = {
            executor.submit(fetch_country_macro_data, session, code): code 
            for code in COUNTRIES
        }
        
        for future in as_completed(future_to_country):
            code = future_to_country[future]
            try:
                data = future.result()
                raw_country_data.append(data)
            except Exception as exc:
                logger.error(f"CRITICAL: Data pipeline halted for '{code}': {exc}")
                raise RuntimeError(f"Pipeline failed due to API/data extraction error on {code}: {exc}") from exc

    df = pd.DataFrame(raw_country_data)

    # Min-Max Normalization across country batch
    def min_max_scale(series: pd.Series) -> pd.Series:
        s_min, s_max = series.min(), series.max()
        return pd.Series(0.5, index=series.index) if s_max == s_min else (series - s_min) / (s_max - s_min)

    gdp_norm = min_max_scale(df["GDP_Growth"])
    labor_norm = min_max_scale(df["Labor_Participation"])
    infra_norm = min_max_scale(df["Infrastructure"])
    inf_risk_norm = min_max_scale(df["Inflation_Risk"])

    # Risk-Adjusted Score Calculation
    df["RISK_ADJ_SCORE"] = (
        (gdp_norm * 0.4) + 
        (labor_norm * 0.3) + 
        (infra_norm * 0.2) - 
        (inf_risk_norm * 0.1)
    ).round(4)

    df["GDP_Growth"] = df["GDP_Growth"].round(4)

    # Monte Carlo Projections
    proj_rows = []
    for _, row in df.iterrows():
        projs = run_monte_carlo_projections(
            base_velocity=row["GDP_Growth"],
            start_index=100.0,
            target_years=[2035, 2040, 2045, 2050],
            n_simulations=1000
        )
        proj_rows.append(projs)

    proj_df = pd.DataFrame(proj_rows)
    df = pd.concat([df, proj_df], axis=1)

    # Absolute Threshold Recommendations
    def classify_recommendation(score: float) -> str:
        if score >= 0.45:
            return "Target"
        elif score >= 0.30:
            return "Watch"
        else:
            return "Avoid"

    df["Recommendation"] = df["RISK_ADJ_SCORE"].apply(classify_recommendation)
    df["Last_Updated"] = datetime.datetime.now().strftime("%Y-%m-%d")

    # Final Schema Match
    schema_cols = [
        "country", "RISK_ADJ_SCORE", "GDP_Growth", "Labor_Participation", 
        "Infrastructure", "Proj_2035", "Proj_2040", "Proj_2045", "Proj_2050", 
        "Recommendation", "Last_Updated"
    ]
    
    final_df = df[schema_cols].copy()
    final_df.to_csv(CACHE_FILE, index=False)
    
    print(f"\n--- PIPELINE COMPLETE: Cache saved to {CACHE_FILE} ---")
    print(final_df.to_string(index=False))
    
    return final_df


if __name__ == "__main__":
    build_engine()
