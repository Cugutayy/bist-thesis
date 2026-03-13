"""
Utility functions for BIST Thesis Project
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


# ─── Caching ─────────────────────────────────────────────
def get_cache_path(key: str, extension: str = "csv") -> Path:
    """Generate a cache file path from a key string."""
    hashed = hashlib.md5(key.encode()).hexdigest()[:12]
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:50]
    return config.CACHE_DIR / f"{safe_key}_{hashed}.{extension}"


def load_from_cache(key: str, max_age_hours: int = 24) -> pd.DataFrame | None:
    """Load a DataFrame from cache if it exists and is fresh."""
    cache_path = get_cache_path(key)
    if not cache_path.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    if age > timedelta(hours=max_age_hours):
        return None
    try:
        return pd.read_csv(cache_path, parse_dates=True)
    except Exception as e:
        logger.warning(f"Cache read failed for {key}: {e}")
        return None


def save_to_cache(df: pd.DataFrame, key: str) -> None:
    """Save a DataFrame to cache."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(key)
    df.to_csv(cache_path, index=True)
    logger.debug(f"Cached {key} → {cache_path}")


# ─── Date Utilities ──────────────────────────────────────
def trading_days_between(start: str, end: str) -> int:
    """Estimate number of trading days between two dates (BIST ~250/year)."""
    d1 = pd.Timestamp(start)
    d2 = pd.Timestamp(end)
    return int(np.busday_count(d1.date(), d2.date()))


def nearest_trading_day(date: str, direction: str = "forward") -> pd.Timestamp:
    """Find nearest trading day (simple weekday approximation)."""
    dt = pd.Timestamp(date)
    if direction == "forward":
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
    else:
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)
    return dt


# ─── Return Calculations ────────────────────────────────
def simple_return(p_start: float, p_end: float) -> float:
    """Simple return: (P_end - P_start) / P_start."""
    if p_start == 0 or pd.isna(p_start) or pd.isna(p_end):
        return np.nan
    return (p_end - p_start) / p_start


def log_return(p_start: float, p_end: float) -> float:
    """Log return: ln(P_end / P_start)."""
    if p_start <= 0 or p_end <= 0 or pd.isna(p_start) or pd.isna(p_end):
        return np.nan
    return np.log(p_end / p_start)


def annualize_return(total_return: float, days: int) -> float:
    """Annualize a return given number of trading days."""
    if days <= 0 or pd.isna(total_return):
        return np.nan
    return (1 + total_return) ** (252 / days) - 1


def deflate_return(nominal_return: float, inflation_rate: float) -> float:
    """Convert nominal return to real return using Fisher equation."""
    return (1 + nominal_return) / (1 + inflation_rate) - 1


# ─── Statistical Helpers ─────────────────────────────────
def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a series at given quantiles."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def newey_west_se(residuals: np.ndarray, X: np.ndarray, lags: int = None) -> np.ndarray:
    """Calculate Newey-West heteroscedasticity and autocorrelation robust standard errors."""
    n, k = X.shape
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    # Meat of the sandwich
    u = residuals.reshape(-1, 1)
    S = (X * u).T @ (X * u)  # HAC=0 part

    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)  # Bartlett kernel
        Gamma_l = (X[l:] * u[l:]).T @ (X[:-l] * u[:-l])
        S += w * (Gamma_l + Gamma_l.T)

    # Bread
    XtX_inv = np.linalg.inv(X.T @ X)
    V = n * XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(V) / n)


# ─── Formatting ──────────────────────────────────────────
def format_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, currency: str = "TL") -> str:
    """Format a number as currency."""
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1e9:
        return f"{value/1e9:.1f}B {currency}"
    if abs(value) >= 1e6:
        return f"{value/1e6:.1f}M {currency}"
    if abs(value) >= 1e3:
        return f"{value/1e3:.1f}K {currency}"
    return f"{value:.0f} {currency}"


def significance_stars(p_value: float) -> str:
    """Return significance stars for p-value."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "†"
    return ""


# ─── DataFrame Helpers ───────────────────────────────────
def ensure_datetime_index(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """Ensure DataFrame has a datetime index."""
    if col and col in df.columns:
        df = df.set_index(col)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def align_series(*series_list: pd.Series) -> list[pd.Series]:
    """Align multiple time series to common dates."""
    common_idx = series_list[0].dropna().index
    for s in series_list[1:]:
        common_idx = common_idx.intersection(s.dropna().index)
    return [s.loc[common_idx] for s in series_list]


def save_results(df: pd.DataFrame, name: str, formats: list[str] = None) -> list[Path]:
    """Save results to processed directory in multiple formats."""
    if formats is None:
        formats = ["csv"]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    for fmt in formats:
        path = config.PROCESSED_DIR / f"{name}.{fmt}"
        if fmt == "csv":
            df.to_csv(path)
        elif fmt == "xlsx":
            df.to_excel(path, engine="openpyxl")
        elif fmt == "json":
            df.to_json(path, orient="records", date_format="iso")
        paths.append(path)
        logger.info(f"Saved {name}.{fmt}")

    return paths
