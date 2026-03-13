"""
BIST Thesis Project — Main Orchestration Script
IPO Fever and the Cost of the Crowd

Usage:
    python run.py --collect          # Collect all data
    python run.py --collect-ipo      # Collect IPO data only
    python run.py --collect-spk      # Collect SPK penalty data only
    python run.py --collect-macro    # Collect macro/inflation data only
    python run.py --analyze          # Run all analyses
    python run.py --analyze-event    # Run event studies only
    python run.py --analyze-herding  # Run CSAD herding analysis
    python run.py --analyze-contrarian # Run contrarian backtest
    python run.py --dashboard        # Launch interactive dashboard
    python run.py --all              # Do everything (collect + analyze + dashboard)
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

# ─── Logging Setup ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("thesis")


def ensure_dirs():
    """Create all necessary directories."""
    for d in [config.RAW_DIR, config.PROCESSED_DIR, config.CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Directories ready")


# ─── Data Collection ─────────────────────────────────────
def collect_ipo():
    """Collect and process IPO data."""
    logger.info("=" * 60)
    logger.info("COLLECTING IPO DATA")
    logger.info("=" * 60)

    from src.data_collection.ipo_data import build_ipo_dataset
    df = build_ipo_dataset()
    output = config.PROCESSED_DIR / "ipo_dataset.csv"
    df.to_csv(output, index=False)
    logger.info(f"IPO dataset saved: {output} ({len(df)} IPOs)")
    return df


def collect_spk():
    """Collect and process SPK penalty data."""
    logger.info("=" * 60)
    logger.info("COLLECTING SPK MANIPULATION DATA")
    logger.info("=" * 60)

    from src.data_collection.spk_data import get_penalties_df
    df = get_penalties_df()
    output = config.PROCESSED_DIR / "spk_penalties.csv"
    df.to_csv(output, index=False)
    logger.info(f"SPK penalties saved: {output} ({len(df)} cases)")
    return df


def collect_macro():
    """Collect and process macro/inflation data."""
    logger.info("=" * 60)
    logger.info("COLLECTING MACRO/INFLATION DATA")
    logger.info("=" * 60)

    from src.data_collection.macro_data import (
        get_tufe_data, fetch_usdtry, fetch_bist100, get_macro_summary
    )

    tufe = get_tufe_data()
    tufe.to_csv(config.PROCESSED_DIR / "tufe_data.csv", index=False)
    logger.info(f"TUFE data saved ({len(tufe)} months)")

    try:
        usdtry = fetch_usdtry(config.START_DATE, config.END_DATE)
        usdtry.to_csv(config.PROCESSED_DIR / "usdtry_data.csv")
        logger.info(f"USD/TRY data saved ({len(usdtry)} days)")
    except Exception as e:
        logger.warning(f"USD/TRY fetch failed: {e}")

    try:
        bist = fetch_bist100(config.START_DATE, config.END_DATE)
        bist.to_csv(config.PROCESSED_DIR / "bist100_data.csv")
        logger.info(f"BIST-100 data saved ({len(bist)} days)")
    except Exception as e:
        logger.warning(f"BIST-100 fetch failed: {e}")

    summary = get_macro_summary()
    summary.to_csv(config.PROCESSED_DIR / "macro_summary.csv", index=False)
    logger.info("Macro summary saved")

    return tufe


def collect_all():
    """Collect all data."""
    ensure_dirs()
    collect_ipo()
    collect_spk()
    collect_macro()
    logger.info("ALL DATA COLLECTION COMPLETE")


# ─── Analysis ────────────────────────────────────────────
def analyze_event_study():
    """Run event studies for SPK manipulation cases."""
    logger.info("=" * 60)
    logger.info("RUNNING EVENT STUDIES")
    logger.info("=" * 60)

    import pandas as pd
    import yfinance as yf
    from datetime import timedelta
    from src.analysis.event_study import run_event_study, aggregate_event_study

    spk = pd.read_csv(config.PROCESSED_DIR / "spk_penalties.csv", parse_dates=["karar_tarihi"])

    # Fetch BIST-100 market data once
    logger.info("Fetching BIST-100 market data...")
    market_raw = yf.download("XU100.IS", start="2018-01-01", end=config.END_DATE, progress=False)
    if market_raw is None or len(market_raw) == 0:
        logger.error("Could not fetch BIST-100 data")
        return []

    market_data = pd.DataFrame(index=market_raw.index.tz_localize(None) if market_raw.index.tz else market_raw.index)
    market_data["close"] = market_raw["Close"].values.flatten()
    market_data["return"] = market_data["close"].pct_change()

    results = []
    for _, case in spk.iterrows():
        ticker = case["hisse_kodu"]
        event_date = case["karar_tarihi"]
        yf_ticker = ticker + ".IS" if not ticker.endswith(".IS") else ticker
        logger.info(f"Event study: {ticker} ({event_date})")

        try:
            # Fetch stock data with enough history for estimation + event window
            fetch_start = (pd.Timestamp(event_date) - timedelta(days=400)).strftime("%Y-%m-%d")
            fetch_end = (pd.Timestamp(event_date) + timedelta(days=120)).strftime("%Y-%m-%d")

            stock_raw = yf.download(yf_ticker, start=fetch_start, end=fetch_end, progress=False)
            if stock_raw is None or len(stock_raw) < 100:
                logger.warning(f"Not enough price data for {ticker}")
                continue

            stock_data = pd.DataFrame(index=stock_raw.index.tz_localize(None) if stock_raw.index.tz else stock_raw.index)
            stock_data["close"] = stock_raw["Close"].values.flatten()
            stock_data["return"] = stock_data["close"].pct_change()

            result = run_event_study(
                stock_data=stock_data,
                market_data=market_data,
                event_date=event_date,
                estimation_window=config.ESTIMATION_WINDOW,
                event_window_pre=config.EVENT_WINDOW_PRE,
                event_window_post=config.EVENT_WINDOW_POST,
            )
            if result is not None:
                result["ticker"] = ticker
                results.append(result)
                logger.info(f"  {ticker}: CAR = {result['car_series'].iloc[-1]:.4f}, t = {result['t_stat']:.3f}")
        except Exception as e:
            logger.warning(f"Event study failed for {ticker}: {e}")

    if results:
        agg = aggregate_event_study(results)

        # Save CAAR as CSV for dashboard
        caar_df = pd.DataFrame({
            "day": agg["caar_series"].index,
            "caar": agg["caar_series"].values,
            "mean_ar": agg["mean_ar_series"].values,
        })
        caar_df.to_csv(config.PROCESSED_DIR / "event_study_results.csv", index=False)

        # Save individual results summary
        individual = []
        for r in results:
            car_final = r["car_series"].iloc[-1] if len(r["car_series"]) > 0 else None
            individual.append({
                "ticker": r["ticker"],
                "event_date": str(r["event_date"]),
                "car": car_final,
                "bhar": r["bhar"],
                "t_stat": r["t_stat"],
                "t_pvalue": r["t_pvalue"],
                "alpha": r["model_params"]["alpha"],
                "beta": r["model_params"]["beta"],
                "r_squared": r["model_params"]["r_squared"],
            })
        pd.DataFrame(individual).to_csv(config.PROCESSED_DIR / "event_study_individual.csv", index=False)

        logger.info(f"Event study results saved ({len(results)} successful)")
        logger.info(f"  CAAR final = {agg['caar_series'].iloc[-1]:.4f}")
        logger.info(f"  t-stat = {agg['t_stat']:.3f}, p-value = {agg['t_pvalue']:.4f}")
    else:
        logger.warning("No successful event studies")

    return results


def analyze_herding():
    """Run CSAD herding analysis."""
    logger.info("=" * 60)
    logger.info("RUNNING CSAD HERDING ANALYSIS")
    logger.info("=" * 60)

    import pandas as pd
    import yfinance as yf
    from src.analysis.csad_herding import (
        calculate_csad, test_herding, rolling_herding, regime_herding
    )

    # Fetch BIST-100 market data
    logger.info("Fetching BIST-100 market data...")
    bist_raw = yf.download("XU100.IS", start=config.START_DATE, end=config.END_DATE, progress=False)
    if bist_raw is None or len(bist_raw) == 0:
        logger.error("Could not fetch BIST-100 data")
        return

    bist_idx = bist_raw.index.tz_localize(None) if bist_raw.index.tz else bist_raw.index
    market_returns = pd.Series(
        bist_raw["Close"].values.flatten(),
        index=bist_idx
    ).pct_change().dropna()

    # Fetch individual stock data for CSAD calculation
    logger.info("Fetching BIST-30 individual stock data for CSAD...")
    bist30_tickers = [
        "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
        "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS",
        "ISCTR.IS", "KCHOL.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS",
        "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
        "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS",
        "TKFEN.IS", "TOASO.IS", "TUPRS.IS", "VESTL.IS", "YKBNK.IS",
    ]

    stock_data = {}
    for ticker in bist30_tickers:
        try:
            data = yf.download(ticker, start=config.START_DATE, end=config.END_DATE, progress=False)
            if data is not None and len(data) > 100:
                idx = data.index.tz_localize(None) if data.index.tz else data.index
                stock_data[ticker] = pd.Series(data["Close"].values.flatten(), index=idx).pct_change().dropna()
                logger.info(f"  {ticker}: {len(stock_data[ticker])} days")
        except Exception as e:
            logger.warning(f"  {ticker}: failed ({e})")

    if len(stock_data) < 10:
        logger.warning("Not enough stock data for herding analysis")
        return

    stock_returns = pd.DataFrame(stock_data)
    logger.info(f"Stock returns matrix: {stock_returns.shape}")

    # Step 1: Calculate CSAD
    csad = calculate_csad(stock_returns, market_returns)
    logger.info(f"CSAD calculated: {len(csad)} observations, mean = {csad.mean():.6f}")

    # Step 2: Main herding test
    herding_result = test_herding(csad, market_returns)
    logger.info(f"Herding test: gamma2 = {herding_result['gamma2']:.6f}, "
                f"t = {herding_result['gamma2_tstat']:.3f}, "
                f"p = {herding_result['gamma2_pvalue']:.4f}")
    logger.info(f"Herding detected: {herding_result['herding_detected']}")

    # Step 3: Rolling herding
    rolling_result = rolling_herding(csad, market_returns, window=config.CSAD_ROLLING_WINDOW)
    herding_pct = rolling_result["herding_flag"].mean()
    logger.info(f"Rolling herding: {herding_pct:.1%} of windows show significant herding")

    # Step 4: Bull vs Bear regime herding
    regime_result = regime_herding(csad, market_returns)
    logger.info(f"Bull gamma2 = {regime_result['comparison']['bull_gamma2']:.6f} "
                f"({'herding' if regime_result['comparison']['bull_herding'] else 'no herding'})")
    logger.info(f"Bear gamma2 = {regime_result['comparison']['bear_gamma2']:.6f} "
                f"({'herding' if regime_result['comparison']['bear_herding'] else 'no herding'})")

    # Save results
    # Main result (remove non-serializable model_summary)
    save_result = {k: v for k, v in herding_result.items() if k != "model_summary"}
    pd.DataFrame([save_result]).to_csv(config.PROCESSED_DIR / "csad_results.csv", index=False)

    rolling_result.to_csv(config.PROCESSED_DIR / "csad_rolling.csv")

    # Save regime results
    regime_save = {
        "bull_gamma2": regime_result["comparison"]["bull_gamma2"],
        "bear_gamma2": regime_result["comparison"]["bear_gamma2"],
        "bull_herding": regime_result["comparison"]["bull_herding"],
        "bear_herding": regime_result["comparison"]["bear_herding"],
        "bull_n": regime_result["comparison"]["bull_n"],
        "bear_n": regime_result["comparison"]["bear_n"],
        "stronger_in": regime_result["comparison"]["stronger_in"],
    }
    pd.DataFrame([regime_save]).to_csv(config.PROCESSED_DIR / "csad_regime.csv", index=False)

    # Save CSAD time series
    csad.to_csv(config.PROCESSED_DIR / "csad_timeseries.csv")

    logger.info("Herding analysis complete and saved")


def analyze_contrarian():
    """Run contrarian strategy backtest."""
    logger.info("=" * 60)
    logger.info("RUNNING CONTRARIAN STRATEGY BACKTEST")
    logger.info("=" * 60)

    import pandas as pd
    from src.analysis.contrarian import ipo_contrarian_strategy

    ipo_df = pd.read_csv(config.PROCESSED_DIR / "ipo_dataset.csv", parse_dates=["ipo_date"])
    logger.info(f"Loaded {len(ipo_df)} IPOs for contrarian analysis")

    # Rename columns: price_dX -> close_Xd (for contrarian module compatibility)
    rename_map = {}
    for col in ipo_df.columns:
        if col.startswith("price_d"):
            days = col.replace("price_d", "")
            try:
                int(days)
                rename_map[col] = f"close_{days}d"
            except ValueError:
                pass
    ipo_df.rename(columns=rename_map, inplace=True)
    logger.info(f"Renamed {len(rename_map)} price columns for contrarian compatibility")

    results = ipo_contrarian_strategy(ipo_df)

    if results is not None:
        # Save by_horizon summary
        if isinstance(results.get("by_horizon"), pd.DataFrame) and len(results["by_horizon"]) > 0:
            results["by_horizon"].to_csv(config.PROCESSED_DIR / "contrarian_by_horizon.csv", index=False)

        # Save overall, hot, cold summaries
        summary_rows = []
        for label, data in [("overall", results["overall"]), ("hot", results["hot_ipos"]), ("cold", results["cold_ipos"])]:
            if data:
                row = {"group": label}
                row.update(data)
                summary_rows.append(row)
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(config.PROCESSED_DIR / "contrarian_results.csv", index=False)

        # Save enriched IPO table with contrarian fields
        if "full_table" in results and results["full_table"] is not None:
            results["full_table"].to_csv(config.PROCESSED_DIR / "contrarian_full_table.csv", index=False)

        # Log key findings
        overall = results["overall"]
        logger.info(f"Overall: {overall['n_ipos']} IPOs")
        logger.info(f"  Avg first-day return: {overall['avg_first_day_return']:.1%}")
        logger.info(f"  Pct positive first day: {overall['pct_positive_first_day']:.0%}")

        for key in sorted(overall.keys()):
            if key.startswith("avg_savings_"):
                horizon = key.replace("avg_savings_", "")
                pct_key = f"pct_better_selling_{horizon}"
                logger.info(f"  Sell Day 1 vs Hold {horizon}: avg savings = {overall[key]:.1%}, "
                           f"better {overall.get(pct_key, 0):.0%} of the time")

        logger.info("Contrarian strategy results saved")

    return results


def analyze_all():
    """Run all analyses."""
    ensure_dirs()
    analyze_event_study()
    analyze_herding()
    analyze_contrarian()
    logger.info("ALL ANALYSES COMPLETE")


# ─── Dashboard ───────────────────────────────────────────
def launch_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = PROJECT_ROOT / "dashboards" / "app.py"
    logger.info(f"Launching dashboard: {dashboard_path}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port=8501",
        "--theme.base=light",
    ])


# ─── Main ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="BIST Thesis Project — IPO Fever and the Cost of the Crowd"
    )
    parser.add_argument("--collect", action="store_true", help="Collect all data")
    parser.add_argument("--collect-ipo", action="store_true", help="Collect IPO data")
    parser.add_argument("--collect-spk", action="store_true", help="Collect SPK data")
    parser.add_argument("--collect-macro", action="store_true", help="Collect macro data")
    parser.add_argument("--analyze", action="store_true", help="Run all analyses")
    parser.add_argument("--analyze-event", action="store_true", help="Run event studies")
    parser.add_argument("--analyze-herding", action="store_true", help="Run CSAD herding")
    parser.add_argument("--analyze-contrarian", action="store_true", help="Run contrarian backtest")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard")
    parser.add_argument("--all", action="store_true", help="Do everything")

    args = parser.parse_args()
    ensure_dirs()

    if args.all:
        collect_all()
        analyze_all()
        launch_dashboard()
    elif args.collect:
        collect_all()
    elif args.collect_ipo:
        collect_ipo()
    elif args.collect_spk:
        collect_spk()
    elif args.collect_macro:
        collect_macro()
    elif args.analyze:
        analyze_all()
    elif args.analyze_event:
        analyze_event_study()
    elif args.analyze_herding:
        analyze_herding()
    elif args.analyze_contrarian:
        analyze_contrarian()
    elif args.dashboard:
        launch_dashboard()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python run.py --collect     # First: collect data")
        print("  python run.py --analyze     # Then: run analyses")
        print("  python run.py --dashboard   # Finally: view results")
        print("  python run.py --all         # Or do everything at once")


if __name__ == "__main__":
    main()
