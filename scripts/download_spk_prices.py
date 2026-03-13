"""
Download historical price data for all SPK penalty stocks and save to a single CSV cache.

Reads data/processed/spk_penalties.csv for stock tickers and event dates,
downloads daily OHLCV data from Yahoo Finance for [event_date - 180 days, event_date + 90 days],
and saves everything to data/processed/spk_price_cache.csv.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import yfinance as yf

# ─── Setup paths ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SPK_FILE = DATA_DIR / "spk_penalties.csv"
OUTPUT_FILE = DATA_DIR / "spk_price_cache.csv"


def download_spk_prices():
    """Download price data for all SPK penalty stocks."""
    # Load SPK penalties
    if not SPK_FILE.exists():
        print(f"ERROR: {SPK_FILE} not found.")
        sys.exit(1)

    df = pd.read_csv(SPK_FILE, parse_dates=["karar_tarihi"])
    print(f"Loaded {len(df)} SPK penalty cases.")
    print(f"Unique tickers: {df['hisse_kodu'].nunique()}")
    print()

    all_frames = []
    success_count = 0
    fail_count = 0
    failed_tickers = []

    for idx, row in df.iterrows():
        ticker_raw = row["hisse_kodu"]
        ticker_yf = ticker_raw + ".IS"
        event_date = pd.Timestamp(row["karar_tarihi"])
        start = (event_date - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        end = (event_date + pd.Timedelta(days=90)).strftime("%Y-%m-%d")

        print(f"[{idx+1}/{len(df)}] Downloading {ticker_yf} "
              f"(event: {event_date.strftime('%Y-%m-%d')}, range: {start} to {end})...", end=" ")

        try:
            stock_data = yf.download(ticker_yf, start=start, end=end, progress=False)

            # Handle empty result
            if stock_data is None or len(stock_data) == 0:
                print("NO DATA (possibly delisted)")
                fail_count += 1
                failed_tickers.append(ticker_raw)
                time.sleep(1)
                continue

            # Handle MultiIndex columns (newer yfinance returns MultiIndex for single ticker)
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data = stock_data.droplevel(level=1, axis=1)

            # Keep only the columns we need
            cols_needed = ["Open", "High", "Low", "Close", "Volume"]
            available_cols = [c for c in cols_needed if c in stock_data.columns]
            stock_data = stock_data[available_cols].copy()

            # Add ticker column
            stock_data["ticker"] = ticker_raw

            # Reset index to make Date a column
            stock_data = stock_data.reset_index()

            # Rename index column to Date if needed
            if "Date" not in stock_data.columns:
                # The index name might be different
                date_col = stock_data.columns[0]
                stock_data = stock_data.rename(columns={date_col: "Date"})

            all_frames.append(stock_data)
            print(f"OK ({len(stock_data)} rows)")
            success_count += 1

        except Exception as e:
            print(f"ERROR: {e}")
            fail_count += 1
            failed_tickers.append(ticker_raw)

        # Rate limit
        time.sleep(1)

    # Combine all data
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)

        # Reorder columns: ticker first
        col_order = ["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
        col_order = [c for c in col_order if c in combined.columns]
        combined = combined[col_order]

        # Save
        combined.to_csv(OUTPUT_FILE, index=False)
        print()
        print(f"{'='*60}")
        print(f"DONE!")
        print(f"  Success: {success_count}/{len(df)}")
        print(f"  Failed:  {fail_count}/{len(df)}")
        print(f"  Total rows saved: {len(combined)}")
        print(f"  Output: {OUTPUT_FILE}")
        if failed_tickers:
            print(f"  Failed tickers: {', '.join(failed_tickers)}")
    else:
        print("No data was downloaded. Check your internet connection and ticker symbols.")


if __name__ == "__main__":
    download_spk_prices()
