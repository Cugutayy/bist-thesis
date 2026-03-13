"""
Post-Tavan Contrarian Analysis
================================
In BIST with ±10% daily limits, IPO underpricing unfolds over multiple days
(tavan serisi). During tavan, selling is effectively impossible.

The real investment decision happens AFTER tavan series ends:
  "Should I sell on the first free trading day or hold longer?"

This script computes returns from the first free day to various horizons
using ACTUAL daily returns from Yahoo Finance (split-adjusted but correct
for relative returns).

Output: data/processed/contrarian_results.csv
        data/processed/contrarian_full_table.csv
        data/processed/contrarian_by_horizon.csv
"""
import sys
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_post_tavan_returns(ticker, first_free_day, horizons_days=[5, 10, 20, 30, 60, 90, 180, 365]):
    """
    Compute returns from the first free trading day to various horizons.
    Uses actual Yahoo Finance daily returns (split-unaffected).
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}

    end_dt = pd.Timestamp(first_free_day) + pd.Timedelta(days=max(horizons_days) + 10)
    today = pd.Timestamp.now()
    if end_dt > today:
        end_dt = today

    try:
        data = yf.download(
            ticker, start=first_free_day,
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False,
        )
    except Exception:
        return {}

    if data is None or len(data) < 2:
        return {}

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    result = {}
    base_close = float(close.iloc[0])
    result["first_free_close"] = base_close

    for h in horizons_days:
        target_date = pd.Timestamp(first_free_day) + pd.Timedelta(days=h)
        mask = close.index >= target_date
        if mask.any():
            future_close = float(close.loc[mask].iloc[0])
            # Return from first free day to horizon
            result[f"post_tavan_return_{h}d"] = (future_close / base_close) - 1
        else:
            result[f"post_tavan_return_{h}d"] = None

    return result


def main():
    start_time = time.time()

    # Load IPO dataset with tavan data
    df = pd.read_csv(config.PROCESSED_DIR / "ipo_dataset.csv")
    logger.info("Loaded %d IPOs", len(df))

    # Filter to IPOs with tavan data and first_free_day
    has_tavan = df["tavan_days"].notna() & (df["tavan_days"] > 0)
    df_tavan = df[has_tavan].copy()
    logger.info("IPOs with tavan series (>0 days): %d", len(df_tavan))

    no_tavan = df[~has_tavan & df["tavan_days"].notna()].copy()
    logger.info("IPOs without tavan (0 days): %d", len(no_tavan))

    horizons = [5, 10, 20, 30, 60, 90, 180, 365]

    # ── Compute post-tavan returns for IPOs WITH tavan ────
    logger.info("\n--- Computing post-tavan returns ---")
    all_results = []

    for i, (_, row) in enumerate(df_tavan.iterrows(), 1):
        ticker = row["ticker"]
        ffd = row.get("first_free_day")
        if pd.isna(ffd):
            continue

        logger.info("[%d/%d] %s (tavan=%d days, first_free=%s)",
                    i, len(df_tavan), ticker, row["tavan_days"], ffd)

        post_returns = compute_post_tavan_returns(ticker, ffd, horizons)
        post_returns["ticker"] = ticker
        post_returns["ipo_date"] = row["ipo_date"]
        post_returns["tavan_days"] = row["tavan_days"]
        post_returns["tavan_series_return"] = row["tavan_series_return"]
        post_returns["offer_price"] = row["offer_price"]
        post_returns["ipo_year"] = row.get("ipo_year")
        all_results.append(post_returns)

        if i % 30 == 0:
            time.sleep(1.0)
        else:
            time.sleep(0.3)

    df_post = pd.DataFrame(all_results)

    # ── Analysis: Is it better to sell on first free day or hold? ────
    logger.info("\n--- Contrarian Analysis Results ---")

    # For each horizon, compute:
    # - Average post-tavan return (if you hold from first free day)
    # - % of IPOs where holding was better (positive return)
    # - % where selling was better (negative return)
    # - t-test for significance

    horizon_results = []
    for h in horizons:
        col = f"post_tavan_return_{h}d"
        if col not in df_post.columns:
            continue

        returns = df_post[col].dropna()
        if len(returns) < 2:
            continue

        t_stat, p_val = stats.ttest_1samp(returns, 0)

        result = {
            "horizon_days": h,
            "n_ipos": len(returns),
            "mean_return": returns.mean(),
            "median_return": returns.median(),
            "std_return": returns.std(),
            "pct_positive": (returns > 0).mean(),
            "pct_negative": (returns < 0).mean(),
            "t_stat": t_stat,
            "p_value": p_val,
            "significant": p_val < 0.05,
            "recommendation": "HOLD" if returns.mean() > 0 else "SELL",
        }
        horizon_results.append(result)

        logger.info(
            "  %3dd: mean=%+.1f%%, median=%+.1f%%, pos=%.0f%%, t=%.2f, p=%.3f → %s",
            h, returns.mean()*100, returns.median()*100,
            (returns > 0).mean()*100, t_stat, p_val,
            result["recommendation"]
        )

    df_horizons = pd.DataFrame(horizon_results)

    # ── Summary by tavan length ──────────────────────────
    logger.info("\n--- By Tavan Length ---")
    df_post["tavan_group"] = pd.cut(
        df_post["tavan_days"],
        bins=[0, 2, 5, 10, 50],
        labels=["1-2 days", "3-5 days", "6-10 days", "11+ days"],
        right=True,
    )

    for group in df_post["tavan_group"].dropna().unique():
        subset = df_post[df_post["tavan_group"] == group]
        if len(subset) < 3:
            continue
        logger.info("  %s (n=%d):", group, len(subset))
        for h in [30, 90, 365]:
            col = f"post_tavan_return_{h}d"
            if col in subset.columns:
                returns = subset[col].dropna()
                if len(returns) > 0:
                    logger.info("    %3dd: mean=%+.1f%%, positive=%.0f%%",
                               h, returns.mean()*100, (returns > 0).mean()*100)

    # ── Overall contrarian summary ───────────────────────
    logger.info("\n--- Overall Summary ---")

    # Group: overall, tavan (>0 days), no-tavan (0 days)
    summary_rows = []

    # Tavan IPOs: sell after tavan vs hold
    tavan_30d = df_post["post_tavan_return_30d"].dropna()
    tavan_90d = df_post["post_tavan_return_90d"].dropna()

    summary_rows.append({
        "group": "tavan_ipos",
        "n_ipos": len(df_tavan),
        "avg_tavan_days": df_tavan["tavan_days"].mean(),
        "avg_tavan_return": df_tavan["tavan_series_return"].mean(),
        "post_30d_mean": tavan_30d.mean() if len(tavan_30d) > 0 else None,
        "post_30d_pct_positive": (tavan_30d > 0).mean() if len(tavan_30d) > 0 else None,
        "post_90d_mean": tavan_90d.mean() if len(tavan_90d) > 0 else None,
        "post_90d_pct_positive": (tavan_90d > 0).mean() if len(tavan_90d) > 0 else None,
    })

    summary_rows.append({
        "group": "no_tavan_ipos",
        "n_ipos": len(no_tavan),
        "avg_tavan_days": 0,
        "avg_tavan_return": 0,
        "post_30d_mean": None,
        "post_30d_pct_positive": None,
        "post_90d_mean": None,
        "post_90d_pct_positive": None,
    })

    df_summary = pd.DataFrame(summary_rows)

    # ── Save results ─────────────────────────────────────
    df_horizons.to_csv(
        config.PROCESSED_DIR / "contrarian_by_horizon.csv",
        index=False, encoding="utf-8-sig",
    )
    df_post.to_csv(
        config.PROCESSED_DIR / "contrarian_full_table.csv",
        index=False, encoding="utf-8-sig",
    )
    df_summary.to_csv(
        config.PROCESSED_DIR / "contrarian_results.csv",
        index=False, encoding="utf-8-sig",
    )

    elapsed = time.time() - start_time
    logger.info("\nContrarian analysis complete in %.1f minutes", elapsed / 60)
    logger.info("Files saved to %s", config.PROCESSED_DIR)

    return df_horizons, df_post, df_summary


if __name__ == "__main__":
    main()
