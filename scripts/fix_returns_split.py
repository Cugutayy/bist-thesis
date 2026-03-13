"""
Fix Multi-Period Returns for Split Adjustment
==============================================
The return_dN columns were computed as (price_dN - offer_price) / offer_price,
which mixes Yahoo split-adjusted prices with nominal offer prices.

This script recalculates returns as (price_dN / ipo_close) - 1,
where both are Yahoo split-adjusted → correct relative returns.

Also recalculates BHAR and excess returns accordingly.
"""
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    df = pd.read_csv(config.PROCESSED_DIR / "ipo_dataset.csv")
    logger.info("Loaded %d IPOs", len(df))

    periods = [1, 5, 10, 30, 60, 90, 180, 365]

    # Count how many returns change
    changes = 0

    for p in periods:
        price_col = f"price_d{p}"
        bm_col = f"benchmark_d{p}"
        ret_col = f"return_d{p}"
        bm_ret_col = f"benchmark_return_d{p}"
        excess_col = f"excess_return_d{p}"
        bhar_col = f"bhar_d{p}"

        if price_col not in df.columns:
            continue

        # Recalculate return from ipo_close (Yahoo split-adjusted)
        old_returns = df[ret_col].copy() if ret_col in df.columns else None

        mask = df["ipo_close"].notna() & df[price_col].notna() & (df["ipo_close"] > 0)
        df.loc[mask, ret_col] = (df.loc[mask, price_col] / df.loc[mask, "ipo_close"]) - 1
        df.loc[~mask, ret_col] = None

        # Compare old vs new
        if old_returns is not None:
            diff = (df[ret_col] - old_returns).dropna()
            n_changed = (diff.abs() > 0.001).sum()
            if n_changed > 0:
                logger.info("  %s: %d returns changed (mean delta: %.4f)",
                           ret_col, n_changed, diff.mean())
                changes += n_changed

        # Recalculate benchmark return
        if bm_col in df.columns and "benchmark_ipo" in df.columns:
            bm_mask = df["benchmark_ipo"].notna() & df[bm_col].notna() & (df["benchmark_ipo"] > 0)
            df.loc[bm_mask, bm_ret_col] = (df.loc[bm_mask, bm_col] / df.loc[bm_mask, "benchmark_ipo"]) - 1

        # Recalculate excess return
        if ret_col in df.columns and bm_ret_col in df.columns:
            both_valid = df[ret_col].notna() & df[bm_ret_col].notna()
            df.loc[both_valid, excess_col] = df.loc[both_valid, ret_col] - df.loc[both_valid, bm_ret_col]

        # Recalculate BHAR = (1 + R_stock) / (1 + R_benchmark) - 1
        if ret_col in df.columns and bm_ret_col in df.columns:
            both_valid = df[ret_col].notna() & df[bm_ret_col].notna()
            denom = 1 + df.loc[both_valid, bm_ret_col]
            denom = denom.replace(0, np.nan)  # avoid division by zero
            df.loc[both_valid, bhar_col] = (1 + df.loc[both_valid, ret_col]) / denom - 1

    logger.info("\nTotal return values changed: %d", changes)

    # Save
    df.to_csv(config.PROCESSED_DIR / "ipo_dataset.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved fixed dataset to %s", config.PROCESSED_DIR / "ipo_dataset.csv")

    # Verify: check some known stocks
    logger.info("\nVerification sample:")
    sample = df[["ticker", "offer_price", "ipo_close", "price_d30", "return_d30",
                 "tavan_days", "underpricing"]].dropna(subset=["return_d30"]).head(10)
    for _, row in sample.iterrows():
        expected = (row["price_d30"] / row["ipo_close"]) - 1
        logger.info("  %s: offer=%.2f, ipo_close=%.4f, price_d30=%.4f, return_d30=%.4f (expected=%.4f) %s",
                   row["ticker"], row["offer_price"], row["ipo_close"],
                   row["price_d30"], row["return_d30"], expected,
                   "✓" if abs(row["return_d30"] - expected) < 0.0001 else "✗")


if __name__ == "__main__":
    main()
