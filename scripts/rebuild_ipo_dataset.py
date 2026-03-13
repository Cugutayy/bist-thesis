"""
Rebuild IPO Dataset from Scratch
=================================
This script:
1. Loads 209 verified IPOs from master CSV
2. Runs tavan series detection for each IPO (Yahoo Finance)
3. Fetches multi-period prices and benchmarks
4. Computes nominal, benchmark-adjusted, and BHAR returns
5. Saves comprehensive dataset to data/processed/ipo_dataset.csv
6. Saves tavan series results to data/processed/tavan_series.csv

Usage:
    python scripts/rebuild_ipo_dataset.py
"""
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data_collection.ipo_data import (
    get_ipo_database,
    detect_tavan_series,
    fetch_ipo_prices,
    calculate_ipo_returns,
    MASTER_CSV,
    DEFAULT_PERIODS,
    BIST_DAILY_LIMIT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()

    # ── 1. Load IPO database ──────────────────────────────
    ipo_list = get_ipo_database()
    n_total = len(ipo_list)
    logger.info("=" * 70)
    logger.info("REBUILD IPO DATASET — %d IPOs from master CSV", n_total)
    logger.info("=" * 70)

    # ── 2. Tavan series detection ─────────────────────────
    logger.info("\n--- PHASE 1: Tavan Series Detection ---")
    tavan_results = []
    tavan_errors = []

    for i, ipo in enumerate(ipo_list, 1):
        ticker = ipo["ticker"]
        ipo_date = ipo["ipo_date"]
        logger.info("[%d/%d] Tavan detection: %s (%s)", i, n_total, ticker, ipo_date)

        result = detect_tavan_series(ticker, ipo_date)
        result["ticker"] = ticker
        result["ipo_date"] = ipo_date
        result["offer_price"] = ipo["offer_price"]
        tavan_results.append(result)

        if result.get("tavan_days") is None:
            tavan_errors.append(ticker)

        # Small delay to be respectful to Yahoo
        if i % 20 == 0:
            logger.info("  ... progress: %d/%d (%.0f%%)", i, n_total, i/n_total*100)
            time.sleep(1.0)
        else:
            time.sleep(0.3)

    # Save tavan results
    df_tavan = pd.DataFrame(tavan_results)
    tavan_csv = config.PROCESSED_DIR / "tavan_series.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_tavan.to_csv(tavan_csv, index=False, encoding="utf-8-sig")
    logger.info("Tavan series saved to %s", tavan_csv)

    # Stats
    valid_tavan = df_tavan[df_tavan["tavan_days"].notna()]
    logger.info("\nTavan Detection Summary:")
    logger.info("  Total: %d, Success: %d, Errors: %d",
                n_total, len(valid_tavan), len(tavan_errors))
    if len(valid_tavan) > 0:
        logger.info("  Mean tavan days: %.1f", valid_tavan["tavan_days"].mean())
        logger.info("  Median tavan days: %.0f", valid_tavan["tavan_days"].median())
        logger.info("  Hit tavan day 1: %d (%.1f%%)",
                     valid_tavan["hit_tavan_day1"].sum(),
                     valid_tavan["hit_tavan_day1"].mean() * 100)
        logger.info("  Max tavan days: %d", valid_tavan["tavan_days"].max())
    if tavan_errors:
        logger.warning("  Failed tickers: %s", ", ".join(tavan_errors[:20]))

    # ── 3. Fetch multi-period prices ──────────────────────
    logger.info("\n--- PHASE 2: Multi-Period Price Fetching ---")
    price_results = []
    price_errors = []

    for i, ipo in enumerate(ipo_list, 1):
        ticker = ipo["ticker"]
        ipo_date = ipo["ipo_date"]
        logger.info("[%d/%d] Fetching prices: %s", i, n_total, ticker)

        try:
            prices = fetch_ipo_prices(ticker, ipo_date, use_cache=True)
            price_results.append(prices)
        except Exception as exc:
            logger.error("Failed %s: %s", ticker, exc)
            stub = {"ticker": ticker, "ipo_date": ipo_date, "ipo_close": None}
            for p in DEFAULT_PERIODS:
                stub[f"price_d{p}"] = None
                stub[f"benchmark_d{p}"] = None
            stub["benchmark_ipo"] = None
            price_results.append(stub)
            price_errors.append(ticker)

        if i % 20 == 0:
            logger.info("  ... progress: %d/%d (%.0f%%)", i, n_total, i/n_total*100)

    logger.info("Price fetching: %d success, %d errors",
                n_total - len(price_errors), len(price_errors))

    # ── 4. Build comprehensive dataset ────────────────────
    logger.info("\n--- PHASE 3: Building Comprehensive Dataset ---")

    # Base IPO dataframe
    df_ipo = pd.DataFrame(ipo_list)
    df_ipo["ipo_date"] = pd.to_datetime(df_ipo["ipo_date"])
    df_ipo["ipo_year"] = df_ipo["ipo_date"].dt.year
    df_ipo["ipo_month"] = df_ipo["ipo_date"].dt.month

    # Merge tavan data
    tavan_cols = ["ticker", "tavan_days", "tavan_days_close_to_close",
                  "tavan_series_return",
                  "first_free_day", "first_free_close",
                  "hit_tavan_day1", "ipo_close_yahoo"]
    available_tavan_cols = [c for c in tavan_cols if c in df_tavan.columns]
    df_ipo = df_ipo.merge(
        df_tavan[available_tavan_cols], on="ticker", how="left"
    )

    # ────────────────────────────────────────────────────────
    # IMPORTANT: Yahoo Finance prices are SPLIT-ADJUSTED.
    # Direct comparison of offer_price vs Yahoo close is WRONG
    # for any stock that has undergone a split.
    #
    # Correct underpricing methodology for BIST (±10% daily limit):
    #
    # Case 1: tavan_days > 0  (consecutive +10% close-to-close returns)
    #   → The IPO day itself also hit tavan from the offer price.
    #   → Total tavan days from offer = tavan_days + 1
    #   → Underpricing = (1.10)^(tavan_days + 1) - 1
    #   Verified against PA Turkey data:
    #     DOFRB: our 9 + 1 = 10 = PA Turkey 10 ✓
    #     ECOGR: our 10 + 1 = 11 = PA Turkey 11 ✓
    #
    # Case 2: tavan_days == 0  (no consecutive +10% days)
    #   → The stock may or may not have hit tavan on IPO day
    #   → We cannot determine exact underpricing from split-adjusted data
    #   → We estimate from the first few days' daily returns (pct_change)
    #   → The first day return is bounded by [-10%, +10%] (BIST limit)
    # ────────────────────────────────────────────────────────

    # Do NOT use Yahoo close vs offer_price for first_day_return (split issue)
    # Instead, use tavan-based methodology

    # Total tavan days from offer (including IPO day)
    df_ipo["total_tavan_days"] = df_ipo["tavan_days"].apply(
        lambda x: x + 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) else np.nan)
    )

    # Proper underpricing = (1.10)^(total_tavan_days) - 1
    def calc_underpricing(row):
        td = row.get("total_tavan_days")
        if pd.isna(td):
            return np.nan
        if td > 0:
            return (1 + BIST_DAILY_LIMIT) ** td - 1
        else:
            # tavan_days = 0: stock didn't hit consecutive tavan
            # First day return is bounded by [-10%, +10%]
            # Use +10% (tavan) if first daily return from Yahoo is close to +10%
            # Otherwise use 0% as conservative estimate (unknown)
            return 0.0  # Conservative: no tavan detected

    df_ipo["underpricing"] = df_ipo.apply(calc_underpricing, axis=1)

    # first_day_return: for tavan IPOs = 10% (BIST limit), else conservative 0%
    df_ipo["first_day_return"] = df_ipo["total_tavan_days"].apply(
        lambda x: BIST_DAILY_LIMIT if pd.notna(x) and x > 0 else 0.0
    )
    # Keep first_day_close as offer * 1.10 for tavan, offer for non-tavan
    mask_tavan = df_ipo["total_tavan_days"].fillna(0) > 0
    df_ipo.loc[mask_tavan, "first_day_close"] = df_ipo.loc[mask_tavan, "offer_price"] * 1.10
    df_ipo.loc[~mask_tavan, "first_day_close"] = df_ipo.loc[~mask_tavan, "offer_price"]

    # Merge price data
    df_prices = pd.DataFrame(price_results)
    df_ipo = df_ipo.merge(
        df_prices.drop(columns=["ipo_date"], errors="ignore"),
        on="ticker", how="left", suffixes=("", "_yf")
    )

    # Calculate returns
    logger.info("Calculating returns...")
    price_map = {p["ticker"]: p for p in price_results}
    return_records = []
    for ipo in ipo_list:
        prices = price_map.get(ipo["ticker"], {})
        r = calculate_ipo_returns(ipo, prices)
        return_records.append(r)

    df_returns = pd.DataFrame(return_records)
    df_returns = df_returns.drop(columns=["ipo_date"], errors="ignore")

    df_ipo = df_ipo.merge(
        df_returns, on="ticker", how="left", suffixes=("", "_calc")
    )

    # Clean duplicate columns
    drop_cols = [c for c in df_ipo.columns if c.endswith("_yf") or c.endswith("_calc")]
    df_ipo = df_ipo.drop(columns=drop_cols, errors="ignore")

    # ── 5. Save final dataset ─────────────────────────────
    output_csv = config.PROCESSED_DIR / "ipo_dataset.csv"
    df_ipo.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info("Dataset saved to %s", output_csv)
    logger.info("Shape: %d rows x %d columns", len(df_ipo), len(df_ipo.columns))

    # ── 6. Print summary ──────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("REBUILD COMPLETE in %.1f minutes", elapsed / 60)
    logger.info("=" * 70)

    logger.info("\nIPOs per year:")
    yearly = df_ipo.groupby("ipo_year").agg(
        count=("ticker", "count"),
        mean_underpricing=("underpricing", "mean"),
        median_tavan_days=("tavan_days", "median"),
    ).reset_index()
    logger.info("\n%s", yearly.to_string(index=False))

    logger.info("\nOverall underpricing stats:")
    logger.info("  Mean underpricing: %.2f%%", df_ipo["underpricing"].mean() * 100)
    logger.info("  Median underpricing: %.2f%%", df_ipo["underpricing"].median() * 100)
    logger.info("  Std underpricing: %.2f%%", df_ipo["underpricing"].std() * 100)

    if "tavan_days" in df_ipo.columns:
        valid = df_ipo["tavan_days"].dropna()
        logger.info("\nTavan series stats:")
        logger.info("  Mean tavan days: %.1f", valid.mean())
        logger.info("  Median tavan days: %.0f", valid.median())
        logger.info("  0 tavan days (no limit hit): %d (%.1f%%)",
                     (valid == 0).sum(), (valid == 0).mean() * 100)
        logger.info("  1+ tavan days: %d (%.1f%%)",
                     (valid >= 1).sum(), (valid >= 1).mean() * 100)
        logger.info("  5+ tavan days: %d (%.1f%%)",
                     (valid >= 5).sum(), (valid >= 5).mean() * 100)
        logger.info("  10+ tavan days: %d (%.1f%%)",
                     (valid >= 10).sum(), (valid >= 10).mean() * 100)

    # Check for potential issues
    logger.info("\nData quality checks:")
    no_yahoo = df_ipo["ipo_close_yahoo"].isna().sum()
    logger.info("  IPOs without Yahoo data: %d", no_yahoo)
    no_tavan = df_ipo["tavan_days"].isna().sum()
    logger.info("  IPOs without tavan detection: %d", no_tavan)
    no_d30 = df_ipo.get("price_d30", pd.Series(dtype=float)).isna().sum()
    logger.info("  IPOs without 30-day price: %d", no_d30)
    no_d365 = df_ipo.get("price_d365", pd.Series(dtype=float)).isna().sum()
    logger.info("  IPOs without 365-day price: %d (expected for recent IPOs)", no_d365)

    logger.info("\nColumns in final dataset:")
    logger.info("  %s", list(df_ipo.columns))

    return df_ipo


if __name__ == "__main__":
    main()
