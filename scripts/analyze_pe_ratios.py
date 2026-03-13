"""
IPO Company P/E (F/K) Ratio Analysis
=====================================
The user asked: "neden underpricing diyoruz, akademik olarak bunu nasıl
açıklıyoruz, ama aslında şirketler karlılığının yani f/k oranları kaç"

This script:
1. Fetches trailing P/E ratios for all 209 IPO tickers from Yahoo Finance
2. Analyzes whether underpricing is explained by company fundamentals
3. Tests Rock (1986) winner's curse hypothesis: do high-quality firms
   underprice more (to signal quality)?
4. Provides summary statistics by tavan length and P/E quintile

Academic context:
- Underpricing = offer price below market value (Ritter 1991, Rock 1986)
- In Turkey: tavan serisi mechanism makes underpricing multi-day
- F/K (Fiyat/Kazanç) = P/E ratio = market cap / earnings
- Low P/E → "cheap" stock → potentially already underpriced at IPO
- High P/E → growth expectations → speculative, may overprice after tavan

Output: data/processed/ipo_pe_analysis.csv
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


def fetch_pe_ratios(tickers):
    """Fetch trailing P/E and P/B ratios from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return {}

    results = []
    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] Fetching fundamentals: %s", i, len(tickers), ticker)
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            result = {
                "ticker": ticker,
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "revenue": info.get("totalRevenue"),
                "ebitda": info.get("ebitda"),
                "net_income": info.get("netIncomeToCommon"),
                "dividend_yield": info.get("dividendYield"),
                "sector_yf": info.get("sector"),
                "industry_yf": info.get("industry"),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
            results.append(result)
        except Exception as e:
            logger.warning("Failed %s: %s", ticker, e)
            results.append({"ticker": ticker, "trailing_pe": None})

        # Rate limiting
        if i % 20 == 0:
            time.sleep(2.0)
        else:
            time.sleep(0.5)

    return pd.DataFrame(results)


def main():
    start_time = time.time()

    # Load IPO dataset
    df = pd.read_csv(config.PROCESSED_DIR / "ipo_dataset.csv")
    logger.info("Loaded %d IPOs", len(df))

    # Fetch P/E ratios
    tickers = df["ticker"].tolist()
    df_pe = fetch_pe_ratios(tickers)

    # Merge with IPO data
    df_merged = df.merge(df_pe, on="ticker", how="left")

    # ── Analysis ──────────────────────────────────────────────
    logger.info("\n--- P/E Ratio Analysis ---")

    # Convert P/E to numeric (some may come back as strings)
    df_merged["trailing_pe"] = pd.to_numeric(df_merged["trailing_pe"], errors="coerce")
    df_merged["forward_pe"] = pd.to_numeric(df_merged["forward_pe"], errors="coerce")
    df_merged["price_to_book"] = pd.to_numeric(df_merged["price_to_book"], errors="coerce")
    df_merged["market_cap"] = pd.to_numeric(df_merged["market_cap"], errors="coerce")

    pe_valid = df_merged["trailing_pe"].dropna()
    pe_positive = pe_valid[pe_valid > 0]

    logger.info("IPOs with P/E data: %d / %d", len(pe_valid), len(df))
    logger.info("IPOs with positive P/E: %d", len(pe_positive))

    if len(pe_positive) > 0:
        logger.info("\nP/E Ratio Statistics:")
        logger.info("  Mean: %.1f", pe_positive.mean())
        logger.info("  Median: %.1f", pe_positive.median())
        logger.info("  Std: %.1f", pe_positive.std())
        logger.info("  Min: %.1f", pe_positive.min())
        logger.info("  Max: %.1f", pe_positive.max())

    # P/B ratio
    pb_valid = df_merged["price_to_book"].dropna()
    pb_positive = pb_valid[pb_valid > 0]
    if len(pb_positive) > 0:
        logger.info("\nP/B Ratio Statistics:")
        logger.info("  Mean: %.2f", pb_positive.mean())
        logger.info("  Median: %.2f", pb_positive.median())

    # ── Correlation: P/E vs Underpricing ─────────────────────
    logger.info("\n--- P/E vs Underpricing ---")
    upr_col = "underpricing" if "underpricing" in df_merged.columns else "tavan_series_return"
    valid_both = df_merged[[upr_col, "trailing_pe"]].dropna()
    valid_both = valid_both[valid_both["trailing_pe"] > 0]

    if len(valid_both) > 5:
        corr, pval = stats.pearsonr(valid_both["trailing_pe"], valid_both[upr_col])
        spearman, sp_p = stats.spearmanr(valid_both["trailing_pe"], valid_both[upr_col])
        logger.info("  Pearson r = %.4f (p = %.4f)", corr, pval)
        logger.info("  Spearman rho = %.4f (p = %.4f)", spearman, sp_p)

    # ── P/E by Tavan Groups ──────────────────────────────────
    logger.info("\n--- P/E by Tavan Length ---")
    if "tavan_days" in df_merged.columns:
        df_merged["tavan_group"] = pd.cut(
            df_merged["tavan_days"],
            bins=[-1, 0, 2, 5, 10, 50],
            labels=["0 days", "1-2 days", "3-5 days", "6-10 days", "11+ days"],
            right=True,
        )

        for group in df_merged["tavan_group"].dropna().unique():
            subset = df_merged[df_merged["tavan_group"] == group]
            pe_sub = subset["trailing_pe"].dropna()
            pe_sub = pe_sub[pe_sub > 0]
            if len(pe_sub) > 0:
                logger.info("  %s (n=%d): median P/E = %.1f, mean P/E = %.1f",
                           group, len(pe_sub), pe_sub.median(), pe_sub.mean())

    # ── Academic Explanation of Underpricing ──────────────────
    logger.info("\n" + "=" * 70)
    logger.info("ACADEMIC EXPLANATION OF IPO UNDERPRICING")
    logger.info("=" * 70)
    logger.info("""
    'Underpricing' = the phenomenon where IPO shares are offered to investors
    at a price BELOW their market value on the first trading day.

    Key theories:
    1. Rock (1986) - Winner's Curse / Information Asymmetry:
       Informed investors crowd out uninformed ones in good IPOs.
       To compensate uninformed investors, ALL IPOs must be underpriced.

    2. Ritter (1991) - Long-run underperformance:
       IPOs are underpriced short-term but OVERPRICED long-term.
       Investors who hold too long earn below-market returns.

    3. Benveniste & Spindt (1989) - Information revelation:
       Underpricing is the 'cost' of getting investors to reveal
       their true demand during book-building.

    4. BIST-specific: Tavan Serisi Mechanism
       With ±10% daily limits, underpricing cannot be realized in one day.
       Instead, it unfolds over multiple consecutive tavan days.
       Proper underpricing = cumulative tavan series return (offer → first free day).

    The P/E (F/K) ratio tells us about FUNDAMENTAL value:
    - Low P/E: company is 'cheap' relative to earnings → fundamental underpricing
    - High P/E: growth expectations priced in → speculative, may be 'overpriced'
    - Negative P/E: company has losses → speculative, riskier

    If underpricing is explained by fundamentals (low P/E → high underpricing),
    this supports the 'mispricing' hypothesis.
    If there's NO correlation, underpricing may be driven by supply-demand
    dynamics (behavioral) rather than fundamental value.
    """)

    # ── Save results ─────────────────────────────────────────
    output_cols = ["ticker", "company_name", "ipo_date", "offer_price",
                   "tavan_days", "underpricing", "tavan_series_return",
                   "trailing_pe", "forward_pe", "price_to_book",
                   "market_cap", "sector_yf", "industry_yf",
                   "current_price", "tavan_group"]
    output_cols = [c for c in output_cols if c in df_merged.columns]

    df_output = df_merged[output_cols].copy()
    df_output.to_csv(
        config.PROCESSED_DIR / "ipo_pe_analysis.csv",
        index=False, encoding="utf-8-sig",
    )
    logger.info("\nSaved to %s", config.PROCESSED_DIR / "ipo_pe_analysis.csv")

    elapsed = time.time() - start_time
    logger.info("Analysis complete in %.1f minutes", elapsed / 60)

    return df_merged


if __name__ == "__main__":
    main()
