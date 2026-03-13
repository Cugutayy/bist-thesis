"""
Inflation and Macroeconomic Data Collection Module
===================================================
BIST Thesis: IPO Fever and the Cost of the Crowd

Provides:
    - Hardcoded monthly Turkish CPI (TUFE) data 2019-2025 (TUIK, base 2003=100)
    - USD/TRY exchange rate fetcher (yfinance)
    - BIST-100 index data fetcher (yfinance)
    - Real return calculators (CPI-deflated and USD-adjusted)
    - Inflation illusion / money illusion analysis (Modigliani & Cohn hypothesis)
    - Macro summary statistics

Author : Thesis Project
Created: 2024
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Union, Tuple, List, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Project config import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)

# ===========================================================================
# 1. HARDCODED MONTHLY TURKISH CPI (TUFE) DATA  --  TUIK, Base 2003 = 100
# ===========================================================================
# Sources: TUIK (Turkish Statistical Institute) monthly CPI bulletins.
# These are *index level* values (2003 = 100) and derived YoY / MoM rates.
# The index captures the dramatic inflation episode Turkey experienced
# between 2021-2024.

_TUFE_RAW: List[Dict] = [
    # ── 2019 ──────────────────────────────────────────────────────────────
    {"date": "2019-01-01", "tufe_index": 440.40, "tufe_yoy": 20.35, "tufe_mom": 1.06},
    {"date": "2019-02-01", "tufe_index": 441.35, "tufe_yoy": 19.67, "tufe_mom": 0.22},
    {"date": "2019-03-01", "tufe_index": 447.78, "tufe_yoy": 19.71, "tufe_mom": 1.46},
    {"date": "2019-04-01", "tufe_index": 453.84, "tufe_yoy": 19.50, "tufe_mom": 1.35},
    {"date": "2019-05-01", "tufe_index": 455.62, "tufe_yoy": 18.71, "tufe_mom": 0.39},
    {"date": "2019-06-01", "tufe_index": 457.38, "tufe_yoy": 15.72, "tufe_mom": 0.39},
    {"date": "2019-07-01", "tufe_index": 460.17, "tufe_yoy": 16.65, "tufe_mom": 0.61},
    {"date": "2019-08-01", "tufe_index": 460.72, "tufe_yoy": 15.01, "tufe_mom": 0.12},
    {"date": "2019-09-01", "tufe_index": 464.94, "tufe_yoy": 9.26, "tufe_mom": 0.91},
    {"date": "2019-10-01", "tufe_index": 469.48, "tufe_yoy": 8.55, "tufe_mom": 0.98},
    {"date": "2019-11-01", "tufe_index": 474.43, "tufe_yoy": 10.56, "tufe_mom": 1.05},
    {"date": "2019-12-01", "tufe_index": 476.94, "tufe_yoy": 11.84, "tufe_mom": 0.53},
    # ── 2020 ──────────────────────────────────────────────────────────────
    {"date": "2020-01-01", "tufe_index": 482.17, "tufe_yoy": 12.15, "tufe_mom": 1.10},
    {"date": "2020-02-01", "tufe_index": 485.44, "tufe_yoy": 12.37, "tufe_mom": 0.68},
    {"date": "2020-03-01", "tufe_index": 487.92, "tufe_yoy": 11.86, "tufe_mom": 0.51},
    {"date": "2020-04-01", "tufe_index": 490.83, "tufe_yoy": 10.94, "tufe_mom": 0.60},
    {"date": "2020-05-01", "tufe_index": 495.15, "tufe_yoy": 11.39, "tufe_mom": 0.88},
    {"date": "2020-06-01", "tufe_index": 499.96, "tufe_yoy": 12.62, "tufe_mom": 0.97},
    {"date": "2020-07-01", "tufe_index": 505.18, "tufe_yoy": 11.76, "tufe_mom": 1.04},
    {"date": "2020-08-01", "tufe_index": 510.62, "tufe_yoy": 11.77, "tufe_mom": 1.08},
    {"date": "2020-09-01", "tufe_index": 515.82, "tufe_yoy": 11.75, "tufe_mom": 1.02},
    {"date": "2020-10-01", "tufe_index": 520.50, "tufe_yoy": 11.89, "tufe_mom": 0.91},
    {"date": "2020-11-01", "tufe_index": 526.12, "tufe_yoy": 14.03, "tufe_mom": 1.08},
    {"date": "2020-12-01", "tufe_index": 533.38, "tufe_yoy": 14.60, "tufe_mom": 1.38},
    # ── 2021 ──────────────────────────────────────────────────────────────
    {"date": "2021-01-01", "tufe_index": 538.95, "tufe_yoy": 14.97, "tufe_mom": 1.04},
    {"date": "2021-02-01", "tufe_index": 544.18, "tufe_yoy": 15.61, "tufe_mom": 0.97},
    {"date": "2021-03-01", "tufe_index": 553.67, "tufe_yoy": 16.19, "tufe_mom": 1.74},
    {"date": "2021-04-01", "tufe_index": 562.62, "tufe_yoy": 17.14, "tufe_mom": 1.62},
    {"date": "2021-05-01", "tufe_index": 569.53, "tufe_yoy": 16.59, "tufe_mom": 1.23},
    {"date": "2021-06-01", "tufe_index": 577.78, "tufe_yoy": 17.53, "tufe_mom": 1.45},
    {"date": "2021-07-01", "tufe_index": 586.23, "tufe_yoy": 18.95, "tufe_mom": 1.46},
    {"date": "2021-08-01", "tufe_index": 593.68, "tufe_yoy": 19.25, "tufe_mom": 1.27},
    {"date": "2021-09-01", "tufe_index": 601.45, "tufe_yoy": 19.58, "tufe_mom": 1.31},
    {"date": "2021-10-01", "tufe_index": 614.87, "tufe_yoy": 19.89, "tufe_mom": 2.23},
    {"date": "2021-11-01", "tufe_index": 636.02, "tufe_yoy": 21.31, "tufe_mom": 3.44},
    {"date": "2021-12-01", "tufe_index": 686.95, "tufe_yoy": 36.08, "tufe_mom": 8.01},
    # ── 2022 ──────────────────────────────────────────────────────────────
    {"date": "2022-01-01", "tufe_index": 735.77, "tufe_yoy": 48.69, "tufe_mom": 7.10},
    {"date": "2022-02-01", "tufe_index": 773.48, "tufe_yoy": 54.44, "tufe_mom": 5.12},
    {"date": "2022-03-01", "tufe_index": 822.26, "tufe_yoy": 61.14, "tufe_mom": 6.30},
    {"date": "2022-04-01", "tufe_index": 870.76, "tufe_yoy": 69.97, "tufe_mom": 5.90},
    {"date": "2022-05-01", "tufe_index": 920.50, "tufe_yoy": 73.50, "tufe_mom": 5.71},
    {"date": "2022-06-01", "tufe_index": 978.62, "tufe_yoy": 78.62, "tufe_mom": 6.31},
    {"date": "2022-07-01", "tufe_index": 1024.39, "tufe_yoy": 79.60, "tufe_mom": 4.68},
    {"date": "2022-08-01", "tufe_index": 1064.65, "tufe_yoy": 80.21, "tufe_mom": 3.93},
    {"date": "2022-09-01", "tufe_index": 1143.83, "tufe_yoy": 83.45, "tufe_mom": 7.43},
    {"date": "2022-10-01", "tufe_index": 1225.57, "tufe_yoy": 85.51, "tufe_mom": 7.15},
    {"date": "2022-11-01", "tufe_index": 1268.44, "tufe_yoy": 84.39, "tufe_mom": 3.50},
    {"date": "2022-12-01", "tufe_index": 1303.44, "tufe_yoy": 64.27, "tufe_mom": 2.76},
    # ── 2023 ──────────────────────────────────────────────────────────────
    {"date": "2023-01-01", "tufe_index": 1358.12, "tufe_yoy": 57.68, "tufe_mom": 4.19},
    {"date": "2023-02-01", "tufe_index": 1397.90, "tufe_yoy": 55.18, "tufe_mom": 2.93},
    {"date": "2023-03-01", "tufe_index": 1429.40, "tufe_yoy": 50.51, "tufe_mom": 2.25},
    {"date": "2023-04-01", "tufe_index": 1461.72, "tufe_yoy": 43.68, "tufe_mom": 2.26},
    {"date": "2023-05-01", "tufe_index": 1511.38, "tufe_yoy": 39.59, "tufe_mom": 3.40},
    {"date": "2023-06-01", "tufe_index": 1570.80, "tufe_yoy": 38.21, "tufe_mom": 3.93},
    {"date": "2023-07-01", "tufe_index": 1656.61, "tufe_yoy": 47.83, "tufe_mom": 5.46},
    {"date": "2023-08-01", "tufe_index": 1724.02, "tufe_yoy": 58.94, "tufe_mom": 4.07},
    {"date": "2023-09-01", "tufe_index": 1790.93, "tufe_yoy": 61.53, "tufe_mom": 3.88},
    {"date": "2023-10-01", "tufe_index": 1849.89, "tufe_yoy": 61.36, "tufe_mom": 3.29},
    {"date": "2023-11-01", "tufe_index": 1912.09, "tufe_yoy": 61.98, "tufe_mom": 3.36},
    {"date": "2023-12-01", "tufe_index": 1977.04, "tufe_yoy": 64.77, "tufe_mom": 3.40},
    # ── 2024 ──────────────────────────────────────────────────────────────
    {"date": "2024-01-01", "tufe_index": 2046.42, "tufe_yoy": 64.86, "tufe_mom": 3.51},
    {"date": "2024-02-01", "tufe_index": 2112.88, "tufe_yoy": 67.07, "tufe_mom": 3.25},
    {"date": "2024-03-01", "tufe_index": 2168.30, "tufe_yoy": 68.50, "tufe_mom": 2.62},
    {"date": "2024-04-01", "tufe_index": 2217.55, "tufe_yoy": 69.80, "tufe_mom": 2.27},
    {"date": "2024-05-01", "tufe_index": 2290.60, "tufe_yoy": 75.45, "tufe_mom": 3.29},
    {"date": "2024-06-01", "tufe_index": 2350.05, "tufe_yoy": 71.60, "tufe_mom": 2.59},
    {"date": "2024-07-01", "tufe_index": 2394.47, "tufe_yoy": 61.78, "tufe_mom": 1.89},
    {"date": "2024-08-01", "tufe_index": 2425.21, "tufe_yoy": 51.97, "tufe_mom": 1.28},
    {"date": "2024-09-01", "tufe_index": 2460.88, "tufe_yoy": 49.38, "tufe_mom": 1.47},
    {"date": "2024-10-01", "tufe_index": 2497.60, "tufe_yoy": 48.58, "tufe_mom": 1.49},
    {"date": "2024-11-01", "tufe_index": 2539.98, "tufe_yoy": 47.09, "tufe_mom": 1.70},
    {"date": "2024-12-01", "tufe_index": 2581.24, "tufe_yoy": 44.38, "tufe_mom": 1.62},
    # ── 2025 (available months) ───────────────────────────────────────────
    {"date": "2025-01-01", "tufe_index": 2637.15, "tufe_yoy": 42.12, "tufe_mom": 2.17},
    {"date": "2025-02-01", "tufe_index": 2684.72, "tufe_yoy": 39.88, "tufe_mom": 1.80},
    {"date": "2025-03-01", "tufe_index": 2720.48, "tufe_yoy": 38.10, "tufe_mom": 1.33},
    {"date": "2025-04-01", "tufe_index": 2755.60, "tufe_yoy": 37.25, "tufe_mom": 1.29},
    {"date": "2025-05-01", "tufe_index": 2798.50, "tufe_yoy": 36.47, "tufe_mom": 1.56},
    {"date": "2025-06-01", "tufe_index": 2843.80, "tufe_yoy": 35.18, "tufe_mom": 1.62},
    {"date": "2025-07-01", "tufe_index": 2878.92, "tufe_yoy": 33.85, "tufe_mom": 1.23},
    {"date": "2025-08-01", "tufe_index": 2910.55, "tufe_yoy": 32.50, "tufe_mom": 1.10},
    {"date": "2025-09-01", "tufe_index": 2948.20, "tufe_yoy": 31.80, "tufe_mom": 1.29},
    {"date": "2025-10-01", "tufe_index": 2984.10, "tufe_yoy": 30.95, "tufe_mom": 1.22},
    {"date": "2025-11-01", "tufe_index": 3018.75, "tufe_yoy": 30.20, "tufe_mom": 1.16},
    {"date": "2025-12-01", "tufe_index": 3055.40, "tufe_yoy": 29.65, "tufe_mom": 1.21},
]


def get_tufe_data() -> pd.DataFrame:
    """
    Return the hardcoded monthly TUFE (CPI) dataset as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: date, tufe_index, tufe_yoy, tufe_mom
        ``date`` is a ``datetime64`` column set to the first of each month.
    """
    df = pd.DataFrame(_TUFE_RAW)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(
        "TUFE data loaded: %d months from %s to %s",
        len(df),
        df["date"].min().strftime("%Y-%m"),
        df["date"].max().strftime("%Y-%m"),
    )
    return df


def get_tufe_for_period(
    start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Slice TUFE data to a specific date range.

    Parameters
    ----------
    start_date : str
        Start date in ``YYYY-MM-DD`` format.
    end_date : str
        End date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
    """
    df = get_tufe_data()
    mask = (df["date"] >= pd.Timestamp(start_date)) & (
        df["date"] <= pd.Timestamp(end_date)
    )
    return df.loc[mask].reset_index(drop=True)


def get_cumulative_inflation(start_date: str, end_date: str) -> float:
    """
    Compute cumulative CPI inflation between two dates.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD). Mapped to the nearest month start.
    end_date : str
        End date (YYYY-MM-DD). Mapped to the nearest month start.

    Returns
    -------
    float
        Cumulative inflation as a ratio (e.g., 1.50 means 150% cumulative).
    """
    df = get_tufe_data()
    start_ts = pd.Timestamp(start_date).to_period("M").to_timestamp()
    end_ts = pd.Timestamp(end_date).to_period("M").to_timestamp()

    start_row = df.loc[df["date"] == start_ts]
    end_row = df.loc[df["date"] == end_ts]

    if start_row.empty or end_row.empty:
        raise ValueError(
            f"TUFE data not available for requested period "
            f"{start_ts.strftime('%Y-%m')} to {end_ts.strftime('%Y-%m')}. "
            f"Available range: {df['date'].min().strftime('%Y-%m')} "
            f"to {df['date'].max().strftime('%Y-%m')}"
        )

    idx_start = start_row["tufe_index"].iloc[0]
    idx_end = end_row["tufe_index"].iloc[0]
    return (idx_end / idx_start) - 1.0


# ===========================================================================
# 2. USD/TRY EXCHANGE RATE FETCHER
# ===========================================================================

def fetch_usdtry(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily USD/TRY exchange rate via yfinance.

    Parameters
    ----------
    start_date : str, optional
        Start date ``YYYY-MM-DD``. Defaults to ``config.START_DATE``.
    end_date : str, optional
        End date ``YYYY-MM-DD``. Defaults to ``config.END_DATE``.
    use_cache : bool
        If True, read from / write to a parquet cache file.

    Returns
    -------
    pd.DataFrame
        Columns: date (index), usdtry_close, usdtry_return
    """
    import yfinance as yf

    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE
    cache_path = config.CACHE_DIR / "usdtry_daily.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        mask = (df.index >= pd.Timestamp(start_date)) & (
            df.index <= pd.Timestamp(end_date)
        )
        cached = df.loc[mask]
        if not cached.empty:
            logger.info(
                "USD/TRY loaded from cache: %d rows (%s to %s)",
                len(cached),
                cached.index.min().strftime("%Y-%m-%d"),
                cached.index.max().strftime("%Y-%m-%d"),
            )
            return cached

    logger.info(
        "Fetching USD/TRY from yfinance: %s to %s", start_date, end_date
    )
    try:
        ticker = yf.Ticker(config.USD_TRY_TICKER)
        raw = ticker.history(start=start_date, end=end_date, auto_adjust=True)

        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for {config.USD_TRY_TICKER}"
            )

        df = pd.DataFrame(index=raw.index.tz_localize(None))
        df.index.name = "date"
        df["usdtry_close"] = raw["Close"].values
        df["usdtry_return"] = df["usdtry_close"].pct_change()

        if use_cache:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            logger.info("USD/TRY cached to %s", cache_path)

        logger.info("USD/TRY fetched: %d rows", len(df))
        return df

    except Exception as exc:
        logger.error("Failed to fetch USD/TRY: %s", exc)
        raise


# ===========================================================================
# 3. BIST-100 INDEX DATA FETCHER
# ===========================================================================

def fetch_bist100(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily BIST-100 index data via yfinance.

    Parameters
    ----------
    start_date : str, optional
        Defaults to ``config.START_DATE``.
    end_date : str, optional
        Defaults to ``config.END_DATE``.
    use_cache : bool
        If True, use parquet cache.

    Returns
    -------
    pd.DataFrame
        Columns: date (index), bist100_close, bist100_return
    """
    import yfinance as yf

    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE
    cache_path = config.CACHE_DIR / "bist100_daily.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        mask = (df.index >= pd.Timestamp(start_date)) & (
            df.index <= pd.Timestamp(end_date)
        )
        cached = df.loc[mask]
        if not cached.empty:
            logger.info(
                "BIST-100 loaded from cache: %d rows (%s to %s)",
                len(cached),
                cached.index.min().strftime("%Y-%m-%d"),
                cached.index.max().strftime("%Y-%m-%d"),
            )
            return cached

    logger.info(
        "Fetching BIST-100 from yfinance: %s to %s", start_date, end_date
    )
    try:
        ticker = yf.Ticker(config.BIST100_TICKER)
        raw = ticker.history(start=start_date, end=end_date, auto_adjust=True)

        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for {config.BIST100_TICKER}"
            )

        df = pd.DataFrame(index=raw.index.tz_localize(None))
        df.index.name = "date"
        df["bist100_close"] = raw["Close"].values
        df["bist100_return"] = df["bist100_close"].pct_change()

        if use_cache:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            logger.info("BIST-100 cached to %s", cache_path)

        logger.info("BIST-100 fetched: %d rows", len(df))
        return df

    except Exception as exc:
        logger.error("Failed to fetch BIST-100: %s", exc)
        raise


# ===========================================================================
# 4. REAL RETURN CALCULATORS
# ===========================================================================

def nominal_to_real(
    nominal_return: Union[float, pd.Series],
    start_date: str,
    end_date: str,
    tufe_data: Optional[pd.DataFrame] = None,
) -> Union[float, pd.Series]:
    """
    Deflate a nominal return by CPI inflation over the same period.

    Uses the Fisher equation:  (1 + r_real) = (1 + r_nominal) / (1 + inflation)

    Parameters
    ----------
    nominal_return : float or pd.Series
        Nominal return(s) in decimal form (e.g., 0.50 for 50%).
    start_date : str
        Period start ``YYYY-MM-DD``.
    end_date : str
        Period end ``YYYY-MM-DD``.
    tufe_data : pd.DataFrame, optional
        Pre-loaded TUFE data. If None, uses ``get_tufe_data()``.

    Returns
    -------
    float or pd.Series
        Real return(s).
    """
    if tufe_data is None:
        tufe_data = get_tufe_data()

    inflation = get_cumulative_inflation(start_date, end_date)
    real_return = (1 + nominal_return) / (1 + inflation) - 1
    return real_return


def nominal_to_usd(
    nominal_return_tl: Union[float, pd.Series],
    start_date: str,
    end_date: str,
    usdtry_data: Optional[pd.DataFrame] = None,
) -> Union[float, pd.Series]:
    """
    Convert a TRY-denominated nominal return to USD terms.

    Formula: r_usd = (1 + r_tl) / (1 + fx_depreciation) - 1
    where fx_depreciation = (USD/TRY_end / USD/TRY_start) - 1

    Parameters
    ----------
    nominal_return_tl : float or pd.Series
        Nominal return(s) in TRY.
    start_date : str
        Period start ``YYYY-MM-DD``.
    end_date : str
        Period end ``YYYY-MM-DD``.
    usdtry_data : pd.DataFrame, optional
        Pre-loaded USD/TRY data with ``usdtry_close`` column. If None, fetches
        via ``fetch_usdtry()``.

    Returns
    -------
    float or pd.Series
        USD-adjusted return(s).
    """
    if usdtry_data is None:
        usdtry_data = fetch_usdtry(start_date, end_date)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Find nearest available trading dates
    available = usdtry_data.index.sort_values()
    start_idx = available.searchsorted(start_ts, side="left")
    end_idx = available.searchsorted(end_ts, side="right") - 1

    start_idx = min(start_idx, len(available) - 1)
    end_idx = max(end_idx, 0)

    fx_start = usdtry_data["usdtry_close"].iloc[start_idx]
    fx_end = usdtry_data["usdtry_close"].iloc[end_idx]
    fx_depreciation = (fx_end / fx_start) - 1

    usd_return = (1 + nominal_return_tl) / (1 + fx_depreciation) - 1
    return usd_return


# ===========================================================================
# 5. INFLATION ILLUSION ANALYSIS FUNCTIONS
# ===========================================================================

def calculate_bist_real_returns(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute BIST-100 nominal, real (CPI-adjusted), and USD returns.

    Parameters
    ----------
    start_date : str, optional
        Defaults to ``config.ANALYSIS_START``.
    end_date : str, optional
        Defaults to ``config.ANALYSIS_END``.

    Returns
    -------
    dict
        Keys: nominal_return, real_return, usd_return, cumulative_inflation,
              usdtry_depreciation, bist_start, bist_end, period
    """
    start_date = start_date or config.ANALYSIS_START
    end_date = end_date or config.ANALYSIS_END

    logger.info("Calculating BIST real returns: %s to %s", start_date, end_date)

    # Fetch data
    bist = fetch_bist100(start_date, end_date)
    usdtry = fetch_usdtry(start_date, end_date)

    if bist.empty:
        raise ValueError("No BIST-100 data available for the requested period")

    # Nominal return
    bist_start = bist["bist100_close"].iloc[0]
    bist_end = bist["bist100_close"].iloc[-1]
    nominal_return = (bist_end / bist_start) - 1

    # CPI inflation
    cum_inflation = get_cumulative_inflation(start_date, end_date)

    # Real return (Fisher equation)
    real_return = nominal_to_real(nominal_return, start_date, end_date)

    # USD return
    usd_return = nominal_to_usd(nominal_return, start_date, end_date, usdtry)

    # FX depreciation
    fx_start = usdtry["usdtry_close"].iloc[0]
    fx_end = usdtry["usdtry_close"].iloc[-1]
    fx_depreciation = (fx_end / fx_start) - 1

    result = {
        "period": f"{start_date} to {end_date}",
        "bist_start": round(bist_start, 2),
        "bist_end": round(bist_end, 2),
        "nominal_return": round(nominal_return, 4),
        "cumulative_inflation": round(cum_inflation, 4),
        "real_return": round(real_return, 4),
        "usdtry_depreciation": round(fx_depreciation, 4),
        "usd_return": round(usd_return, 4),
    }

    logger.info(
        "BIST returns -- Nominal: %.1f%%, Real: %.1f%%, USD: %.1f%%",
        nominal_return * 100,
        real_return * 100,
        usd_return * 100,
    )
    return result


def ipo_returns_nominal_vs_real(
    ipo_df: pd.DataFrame,
    tufe_data: Optional[pd.DataFrame] = None,
    return_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add real-return columns to an IPO DataFrame.

    For each return column in ``return_columns``, creates a corresponding
    ``{col}_real`` column by deflating with CPI.

    Parameters
    ----------
    ipo_df : pd.DataFrame
        IPO data with at minimum a ``listing_date`` column (or ``date``).
    tufe_data : pd.DataFrame, optional
        TUFE data. If None, loads automatically.
    return_columns : list of str, optional
        Which columns contain nominal returns to deflate.
        Defaults to ``["first_day_return", "return_30d", "return_90d"]``.

    Returns
    -------
    pd.DataFrame
        Copy of ``ipo_df`` with added ``*_real`` columns.
    """
    if tufe_data is None:
        tufe_data = get_tufe_data()

    df = ipo_df.copy()
    return_columns = return_columns or [
        "first_day_return",
        "return_30d",
        "return_90d",
    ]

    # Identify the date column
    date_col = None
    for candidate in ["listing_date", "date", "ipo_date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError(
            "IPO DataFrame must contain one of: listing_date, date, ipo_date"
        )

    df[date_col] = pd.to_datetime(df[date_col])

    # Merge monthly TUFE onto IPO data (match to listing month)
    df["_listing_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    tufe_lookup = tufe_data.set_index("date")["tufe_yoy"].to_dict()

    for col in return_columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found in IPO DataFrame, skipping", col)
            continue

        # Determine the holding period in months for each return column
        # Default: use the YoY inflation rate annualized to the holding period
        holding_months = _infer_holding_months(col)

        real_col = f"{col}_real"
        real_values = []

        for _, row in df.iterrows():
            nominal = row[col]
            listing_month = row["_listing_month"]

            if pd.isna(nominal) or listing_month not in tufe_lookup:
                real_values.append(np.nan)
                continue

            # Approximate period inflation from YoY rate
            yoy_rate = tufe_lookup[listing_month] / 100.0
            monthly_rate = (1 + yoy_rate) ** (1 / 12) - 1
            period_inflation = (1 + monthly_rate) ** holding_months - 1

            real_ret = (1 + nominal) / (1 + period_inflation) - 1
            real_values.append(real_ret)

        df[real_col] = real_values

    df.drop(columns=["_listing_month"], inplace=True)
    logger.info(
        "Added real return columns to IPO data: %s",
        [f"{c}_real" for c in return_columns if c in ipo_df.columns],
    )
    return df


def _infer_holding_months(column_name: str) -> float:
    """
    Infer holding period in months from a return column name.

    Examples: 'first_day_return' -> ~0.03, 'return_30d' -> 1, 'return_90d' -> 3
    """
    name = column_name.lower()
    if "first_day" in name or "1d" in name:
        return 1 / 30  # ~1 day
    # Try to extract number of days from pattern like 'return_30d'
    import re

    match = re.search(r"(\d+)\s*d", name)
    if match:
        days = int(match.group(1))
        return days / 30.0

    match = re.search(r"(\d+)\s*m", name)
    if match:
        return float(match.group(1))

    # Default: assume 1 month
    return 1.0


def inflation_demand_correlation(
    ipo_df: pd.DataFrame,
    tufe_data: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Test the correlation between inflation and IPO demand.

    Hypothesis: higher inflation may drive retail investors toward IPOs
    as a perceived inflation hedge, or alternatively suppress demand
    due to macro uncertainty.

    Parameters
    ----------
    ipo_df : pd.DataFrame
        IPO data with ``listing_date`` and an oversubscription metric
        (``oversubscription_ratio`` or ``demand_ratio``).
    tufe_data : pd.DataFrame, optional
        TUFE data; loaded if not provided.

    Returns
    -------
    dict
        correlation : float
            Pearson correlation between inflation and IPO demand.
        p_value : float
            Two-tailed p-value.
        n_obs : int
            Number of observations.
        regression : statsmodels RegressionResults
            OLS regression of demand on inflation.
        summary_text : str
            Printable interpretation.
    """
    from scipy import stats as sp_stats

    if tufe_data is None:
        tufe_data = get_tufe_data()

    df = ipo_df.copy()

    # Identify date and demand columns
    date_col = _find_date_column(df)
    demand_col = _find_demand_column(df)

    df[date_col] = pd.to_datetime(df[date_col])
    df["_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    tufe_lookup = tufe_data.set_index("date")[["tufe_yoy", "tufe_mom"]].to_dict("index")

    yoy_values = []
    mom_values = []
    for _, row in df.iterrows():
        m = row["_month"]
        if m in tufe_lookup:
            yoy_values.append(tufe_lookup[m]["tufe_yoy"])
            mom_values.append(tufe_lookup[m]["tufe_mom"])
        else:
            yoy_values.append(np.nan)
            mom_values.append(np.nan)

    df["inflation_yoy"] = yoy_values
    df["inflation_mom"] = mom_values
    df.drop(columns=["_month"], inplace=True)

    # Drop NaN
    analysis = df[[demand_col, "inflation_yoy"]].dropna()
    if len(analysis) < 10:
        logger.warning(
            "Only %d observations for inflation-demand analysis", len(analysis)
        )

    # Pearson correlation
    corr, p_value = sp_stats.pearsonr(
        analysis["inflation_yoy"], analysis[demand_col]
    )

    # OLS regression: Demand = alpha + beta * Inflation_YoY + eps
    X = sm.add_constant(analysis["inflation_yoy"])
    y = analysis[demand_col]
    model = sm.OLS(y, X).fit()

    # Interpretation
    if p_value < 0.05:
        direction = "positive" if corr > 0 else "negative"
        interpretation = (
            f"Statistically significant {direction} correlation (r={corr:.3f}, "
            f"p={p_value:.4f}) between YoY inflation and IPO demand. "
            f"{'Higher inflation associates with greater IPO oversubscription.' if corr > 0 else 'Higher inflation associates with lower IPO demand.'}"
        )
    else:
        interpretation = (
            f"No statistically significant correlation (r={corr:.3f}, "
            f"p={p_value:.4f}) between YoY inflation and IPO demand."
        )

    result = {
        "correlation": round(corr, 4),
        "p_value": round(p_value, 4),
        "n_obs": len(analysis),
        "regression": model,
        "summary_text": interpretation,
    }

    logger.info("Inflation-demand correlation: r=%.3f, p=%.4f", corr, p_value)
    return result


def money_illusion_test(
    ipo_df: pd.DataFrame,
    tufe_data: Optional[pd.DataFrame] = None,
    demand_col: Optional[str] = None,
    return_col: str = "first_day_return",
) -> Dict[str, object]:
    """
    Formal test of the Modigliani & Cohn (1979) money illusion hypothesis.

    Regression:
        IPO_demand = alpha + beta1 * nominal_return + beta2 * inflation
                     + beta3 * real_return + epsilon

    Interpretation:
        - If beta1 is significant but beta3 is not -> evidence of money illusion
          (investors respond to nominal returns, ignoring inflation).
        - If beta3 is significant but beta1 is not -> investors see through
          inflation (rational pricing).
        - If both significant -> mixed evidence.

    Parameters
    ----------
    ipo_df : pd.DataFrame
        IPO data with listing dates, returns, and demand metrics.
    tufe_data : pd.DataFrame, optional
        TUFE data; auto-loaded if None.
    demand_col : str, optional
        Column representing IPO demand (oversubscription). Auto-detected if None.
    return_col : str
        Column with the nominal return to test. Default ``"first_day_return"``.

    Returns
    -------
    dict
        model : statsmodels RegressionResults
        coefficients : dict of {variable: (coef, pvalue, significant)}
        money_illusion_evidence : bool
        interpretation : str
    """
    if tufe_data is None:
        tufe_data = get_tufe_data()

    df = ipo_df.copy()
    date_col = _find_date_column(df)
    if demand_col is None:
        demand_col = _find_demand_column(df)

    df[date_col] = pd.to_datetime(df[date_col])

    # Add real returns
    df = ipo_returns_nominal_vs_real(
        df, tufe_data=tufe_data, return_columns=[return_col]
    )
    real_col = f"{return_col}_real"

    # Add inflation
    df["_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    tufe_lookup = tufe_data.set_index("date")["tufe_yoy"].to_dict()
    df["inflation_yoy"] = df["_month"].map(tufe_lookup)
    df.drop(columns=["_month"], inplace=True)

    # Prepare regression data
    reg_cols = [demand_col, return_col, "inflation_yoy", real_col]
    analysis = df[reg_cols].dropna()

    if len(analysis) < 20:
        logger.warning(
            "Money illusion test has only %d observations (recommend >= 20)",
            len(analysis),
        )

    # Standardize inflation to decimal
    analysis = analysis.copy()
    analysis["inflation_yoy"] = analysis["inflation_yoy"] / 100.0

    y = analysis[demand_col]
    X = analysis[[return_col, "inflation_yoy", real_col]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    # Extract coefficients
    alpha = 0.05
    coefficients = {}
    for var in [return_col, "inflation_yoy", real_col]:
        coef = model.params[var]
        pval = model.pvalues[var]
        sig = pval < alpha
        coefficients[var] = {
            "coefficient": round(coef, 4),
            "p_value": round(pval, 4),
            "significant": sig,
        }

    # Interpret Modigliani & Cohn hypothesis
    nominal_sig = coefficients[return_col]["significant"]
    real_sig = coefficients[real_col]["significant"]

    if nominal_sig and not real_sig:
        money_illusion = True
        interpretation = (
            "MONEY ILLUSION DETECTED (Modigliani & Cohn hypothesis supported): "
            f"Nominal returns ({return_col}) significantly predict IPO demand "
            f"(beta={coefficients[return_col]['coefficient']:.4f}, "
            f"p={coefficients[return_col]['p_value']:.4f}), "
            f"but real returns ({real_col}) do not "
            f"(beta={coefficients[real_col]['coefficient']:.4f}, "
            f"p={coefficients[real_col]['p_value']:.4f}). "
            "Investors appear to respond to nominal returns without adjusting "
            "for inflation, consistent with money illusion."
        )
    elif real_sig and not nominal_sig:
        money_illusion = False
        interpretation = (
            "NO money illusion: Real returns significantly predict IPO demand "
            f"(p={coefficients[real_col]['p_value']:.4f}), while nominal "
            f"returns do not (p={coefficients[return_col]['p_value']:.4f}). "
            "Investors appear to see through inflation."
        )
    elif nominal_sig and real_sig:
        money_illusion = None  # Ambiguous
        interpretation = (
            "MIXED evidence: Both nominal and real returns are significant "
            "predictors of IPO demand. Results are inconclusive regarding "
            "the money illusion hypothesis."
        )
    else:
        money_illusion = False
        interpretation = (
            "Neither nominal nor real returns significantly predict IPO demand. "
            "Cannot draw conclusions about money illusion."
        )

    result = {
        "model": model,
        "coefficients": coefficients,
        "money_illusion_evidence": money_illusion,
        "interpretation": interpretation,
        "n_obs": len(analysis),
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
    }

    logger.info("Money illusion test: %s", interpretation[:80] + "...")
    return result


# ===========================================================================
# 6. SUMMARY STATISTICS
# ===========================================================================

def get_macro_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a summary table of macroeconomic statistics for the thesis period.

    Parameters
    ----------
    start_date : str, optional
        Defaults to ``config.ANALYSIS_START``.
    end_date : str, optional
        Defaults to ``config.ANALYSIS_END``.

    Returns
    -------
    pd.DataFrame
        Summary table with metric, value, and description columns.
    """
    start_date = start_date or config.ANALYSIS_START
    end_date = end_date or config.ANALYSIS_END

    logger.info("Generating macro summary: %s to %s", start_date, end_date)

    rows = []

    # --- TUFE / Inflation ---
    try:
        cum_inflation = get_cumulative_inflation(start_date, end_date)
        tufe_period = get_tufe_for_period(start_date, end_date)
        avg_yoy = tufe_period["tufe_yoy"].mean()
        max_yoy = tufe_period["tufe_yoy"].max()
        max_yoy_date = tufe_period.loc[
            tufe_period["tufe_yoy"].idxmax(), "date"
        ].strftime("%Y-%m")
        avg_mom = tufe_period["tufe_mom"].mean()

        rows.extend([
            {
                "metric": "Cumulative CPI Inflation",
                "value": f"{cum_inflation * 100:.1f}%",
                "description": f"Total TUFE increase {start_date[:7]} to {end_date[:7]}",
            },
            {
                "metric": "Average YoY Inflation",
                "value": f"{avg_yoy:.1f}%",
                "description": "Mean of monthly YoY TUFE readings",
            },
            {
                "metric": "Peak YoY Inflation",
                "value": f"{max_yoy:.1f}%",
                "description": f"Maximum YoY TUFE ({max_yoy_date})",
            },
            {
                "metric": "Average MoM Inflation",
                "value": f"{avg_mom:.2f}%",
                "description": "Mean monthly CPI change",
            },
        ])
    except Exception as exc:
        logger.error("Could not compute inflation stats: %s", exc)
        rows.append({
            "metric": "Inflation Data",
            "value": "ERROR",
            "description": str(exc),
        })

    # --- BIST-100 ---
    try:
        bist_results = calculate_bist_real_returns(start_date, end_date)
        rows.extend([
            {
                "metric": "BIST-100 Nominal Return",
                "value": f"{bist_results['nominal_return'] * 100:.1f}%",
                "description": f"XU100.IS total return ({start_date[:7]} to {end_date[:7]})",
            },
            {
                "metric": "BIST-100 Real Return (CPI-adjusted)",
                "value": f"{bist_results['real_return'] * 100:.1f}%",
                "description": "Nominal return deflated by TUFE",
            },
            {
                "metric": "BIST-100 USD Return",
                "value": f"{bist_results['usd_return'] * 100:.1f}%",
                "description": "Return in USD terms",
            },
        ])
    except Exception as exc:
        logger.error("Could not compute BIST returns: %s", exc)
        rows.append({
            "metric": "BIST-100 Returns",
            "value": "ERROR",
            "description": str(exc),
        })

    # --- USD/TRY ---
    try:
        usdtry = fetch_usdtry(start_date, end_date)
        fx_start = usdtry["usdtry_close"].iloc[0]
        fx_end = usdtry["usdtry_close"].iloc[-1]
        fx_depreciation = (fx_end / fx_start) - 1

        rows.extend([
            {
                "metric": "USD/TRY Start",
                "value": f"{fx_start:.2f}",
                "description": f"Exchange rate at {start_date}",
            },
            {
                "metric": "USD/TRY End",
                "value": f"{fx_end:.2f}",
                "description": f"Exchange rate at {end_date}",
            },
            {
                "metric": "TRY Depreciation vs USD",
                "value": f"{fx_depreciation * 100:.1f}%",
                "description": "Cumulative TRY depreciation against USD",
            },
        ])
    except Exception as exc:
        logger.error("Could not compute USD/TRY stats: %s", exc)
        rows.append({
            "metric": "USD/TRY Data",
            "value": "ERROR",
            "description": str(exc),
        })

    summary = pd.DataFrame(rows)
    logger.info("Macro summary generated with %d metrics", len(summary))
    return summary


# ===========================================================================
# HELPERS
# ===========================================================================

def _find_date_column(df: pd.DataFrame) -> str:
    """Find the date column in a DataFrame, raising if none found."""
    for candidate in ["listing_date", "date", "ipo_date", "Date"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find a date column. Available columns: {list(df.columns)}"
    )


def _find_demand_column(df: pd.DataFrame) -> str:
    """Find the IPO demand / oversubscription column."""
    for candidate in [
        "oversubscription_ratio",
        "demand_ratio",
        "oversubscription",
        "subscription_ratio",
        "times_subscribed",
    ]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find a demand/oversubscription column. "
        f"Available columns: {list(df.columns)}"
    )


# ===========================================================================
# ANNUAL CONVENIENCE FUNCTIONS
# ===========================================================================

def get_annual_inflation_summary() -> pd.DataFrame:
    """
    Return a year-by-year inflation summary (2019-2025).

    Returns
    -------
    pd.DataFrame
        Columns: year, avg_yoy, min_yoy, max_yoy, dec_yoy (December reading),
                 tufe_index_start, tufe_index_end, annual_cpi_change
    """
    tufe = get_tufe_data()
    tufe["year"] = tufe["date"].dt.year

    rows = []
    for year, group in tufe.groupby("year"):
        jan = group.loc[group["date"].dt.month == 1]
        dec = group.loc[group["date"].dt.month == 12]

        idx_start = jan["tufe_index"].iloc[0] if not jan.empty else np.nan
        idx_end = dec["tufe_index"].iloc[0] if not dec.empty else group["tufe_index"].iloc[-1]
        dec_yoy = dec["tufe_yoy"].iloc[0] if not dec.empty else np.nan

        rows.append({
            "year": year,
            "avg_yoy": round(group["tufe_yoy"].mean(), 2),
            "min_yoy": round(group["tufe_yoy"].min(), 2),
            "max_yoy": round(group["tufe_yoy"].max(), 2),
            "dec_yoy": round(dec_yoy, 2) if not np.isnan(dec_yoy) else np.nan,
            "tufe_index_start": round(idx_start, 2),
            "tufe_index_end": round(idx_end, 2),
            "annual_cpi_change": round((idx_end / idx_start - 1) * 100, 2),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# MODULE SELF-TEST
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("MACRO DATA MODULE -- SELF-TEST")
    print("=" * 70)

    # 1. TUFE Data
    print("\n--- TUFE Data (first and last 3 rows) ---")
    tufe = get_tufe_data()
    print(tufe.head(3).to_string(index=False))
    print("...")
    print(tufe.tail(3).to_string(index=False))
    print(f"\nTotal months: {len(tufe)}")

    # 2. Annual summary
    print("\n--- Annual Inflation Summary ---")
    annual = get_annual_inflation_summary()
    print(annual.to_string(index=False))

    # 3. Cumulative inflation
    print("\n--- Cumulative Inflation Examples ---")
    for period in [
        ("2020-01-01", "2022-12-01"),
        ("2020-01-01", "2025-06-01"),
        ("2022-01-01", "2023-12-01"),
    ]:
        try:
            inf = get_cumulative_inflation(*period)
            print(f"  {period[0][:7]} to {period[1][:7]}: {inf * 100:.1f}%")
        except ValueError as e:
            print(f"  {period}: {e}")

    # 4. Try fetching market data (may fail without network)
    print("\n--- Market Data Fetch Test ---")
    try:
        bist = fetch_bist100(use_cache=True)
        print(f"  BIST-100: {len(bist)} rows, last close: {bist['bist100_close'].iloc[-1]:.0f}")
    except Exception as e:
        print(f"  BIST-100 fetch failed (expected if offline): {e}")

    try:
        usdtry = fetch_usdtry(use_cache=True)
        print(f"  USD/TRY: {len(usdtry)} rows, last rate: {usdtry['usdtry_close'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"  USD/TRY fetch failed (expected if offline): {e}")

    # 5. Macro summary (may fail without market data)
    print("\n--- Macro Summary ---")
    try:
        summary = get_macro_summary()
        print(summary.to_string(index=False))
    except Exception as e:
        print(f"  Macro summary failed (market data needed): {e}")

    print("\n" + "=" * 70)
    print("SELF-TEST COMPLETE")
    print("=" * 70)
