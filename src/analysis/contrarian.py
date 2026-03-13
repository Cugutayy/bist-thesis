"""
Contrarian Strategy Backtest Module
=====================================
Implements contrarian (reversal) trading strategies for BIST thesis project.
Core idea: past losers outperform past winners over short horizons,
especially in markets with herding and overreaction (e.g., IPO fever).

Also includes an IPO-specific contrarian analysis:
  - "Sell IPO on day 1" vs. "Hold"
  - Grouped by oversubscription level
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ─── Project imports ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


# ─────────────────────────────────────────────────────────────
# Momentum / Reversal Signal
# ─────────────────────────────────────────────────────────────

def calculate_momentum_signal(
    returns_df: pd.DataFrame,
    lookback: int = None,
) -> pd.DataFrame:
    """
    Calculate a past-return momentum signal for each stock.

    Signal_i,t = cumulative return of stock i over [t-lookback, t-1]

    A contrarian strategy will *reverse* this: buy low-signal, sell high-signal.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns (columns = tickers, index = dates).
    lookback : int, optional
        Number of past trading days to measure.
        Defaults to config.CONTRARIAN_LOOKBACK.

    Returns
    -------
    pd.DataFrame
        Signal DataFrame with same shape as returns_df.
        NaN for dates where lookback history is insufficient.
    """
    lookback = lookback or config.CONTRARIAN_LOOKBACK

    # Cumulative return over the lookback window (geometric)
    signal = (1 + returns_df).rolling(window=lookback).apply(
        lambda x: np.prod(x) - 1, raw=True
    )

    # Shift by 1 so signal on day t uses [t-lookback, t-1]
    signal = signal.shift(1)
    signal.columns = returns_df.columns

    return signal


# ─────────────────────────────────────────────────────────────
# Portfolio Formation
# ─────────────────────────────────────────────────────────────

def form_portfolios(
    returns_df: pd.DataFrame,
    signal: pd.DataFrame,
    n_portfolios: int = 5,
) -> pd.DataFrame:
    """
    Sort stocks into quantile portfolios based on signal each period.

    Portfolio 1 = lowest signal (past losers).
    Portfolio N = highest signal (past winners).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns (columns = tickers, index = dates).
    signal : pd.DataFrame
        Signal matrix (same shape as returns_df).
    n_portfolios : int
        Number of quantile buckets.

    Returns
    -------
    pd.DataFrame
        Columns = portfolio labels (1 .. n_portfolios + "L-W"),
        index = dates, values = equal-weighted portfolio returns.
    """
    portfolio_returns = {i: [] for i in range(1, n_portfolios + 1)}
    dates = []

    for date in returns_df.index:
        if date not in signal.index:
            continue

        sig_row = signal.loc[date].dropna()
        ret_row = returns_df.loc[date]

        # Need enough stocks
        valid_tickers = sig_row.index.intersection(ret_row.dropna().index)
        if len(valid_tickers) < n_portfolios:
            continue

        sig_vals = sig_row[valid_tickers]
        ret_vals = ret_row[valid_tickers]

        # Quantile breakpoints
        try:
            bins = pd.qcut(sig_vals, q=n_portfolios, labels=False, duplicates="drop")
        except ValueError:
            continue

        dates.append(date)
        for q in range(n_portfolios):
            tickers_in_q = bins[bins == q].index
            if len(tickers_in_q) > 0:
                portfolio_returns[q + 1].append(ret_vals[tickers_in_q].mean())
            else:
                portfolio_returns[q + 1].append(np.nan)

    result = pd.DataFrame(portfolio_returns, index=dates)
    result.columns = [f"P{i}" for i in range(1, n_portfolios + 1)]

    # L-W (Losers minus Winners) = contrarian spread
    result["L-W"] = result[f"P1"] - result[f"P{n_portfolios}"]

    return result


# ─────────────────────────────────────────────────────────────
# Full Contrarian Backtest
# ─────────────────────────────────────────────────────────────

def contrarian_backtest(
    returns_df: pd.DataFrame,
    lookback: int = None,
    holding: int = None,
    top_n: int = None,
) -> dict:
    """
    Run a contrarian (short-term reversal) backtest.

    Strategy:
        - Each rebalancing date: rank stocks by past `lookback`-day return.
        - Buy the `top_n` worst performers (losers).
        - Sell the `top_n` best performers (winners).
        - Hold for `holding` days, then rebalance.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns (columns = tickers, index = dates).
    lookback : int, optional
        Signal lookback period.  Defaults to config value.
    holding : int, optional
        Holding period.  Defaults to config value.
    top_n : int, optional
        Number of stocks in loser / winner portfolios.
        Defaults to config value.

    Returns
    -------
    dict
        portfolio_returns : pd.DataFrame
            Daily returns of loser, winner, and L-W portfolios.
        cumulative_returns : pd.DataFrame
            Cumulative returns of each portfolio.
        rebalance_dates : list
            Dates when portfolios were rebalanced.
        statistics : dict
            Summary stats from strategy_statistics.
        holdings_log : list[dict]
            Log of holdings at each rebalance.
    """
    lookback = lookback or config.CONTRARIAN_LOOKBACK
    holding = holding or config.CONTRARIAN_HOLDING
    top_n = top_n or config.CONTRARIAN_TOP_N

    signal = calculate_momentum_signal(returns_df, lookback)
    dates = returns_df.index.tolist()

    loser_rets = []
    winner_rets = []
    lmw_rets = []
    ret_dates = []
    rebalance_dates = []
    holdings_log = []

    i = lookback  # Start after enough history
    while i < len(dates):
        date = dates[i]

        # Signal on rebalance date
        sig_row = signal.loc[date].dropna()
        available = sig_row.index.intersection(returns_df.loc[date].dropna().index)

        if len(available) < 2 * top_n:
            i += 1
            continue

        sig_vals = sig_row[available].sort_values()

        losers = sig_vals.index[:top_n].tolist()
        winners = sig_vals.index[-top_n:].tolist()

        rebalance_dates.append(date)
        holdings_log.append({
            "date": date,
            "losers": losers,
            "winners": winners,
        })

        # Hold for `holding` days
        hold_end = min(i + holding, len(dates))
        for j in range(i, hold_end):
            d = dates[j]
            day_ret = returns_df.loc[d]

            loser_ret = day_ret[losers].mean()
            winner_ret = day_ret[winners].mean()

            loser_rets.append(loser_ret)
            winner_rets.append(winner_ret)
            lmw_rets.append(loser_ret - winner_ret)
            ret_dates.append(d)

        i = hold_end  # Next rebalance

    portfolio_returns = pd.DataFrame({
        "Losers": loser_rets,
        "Winners": winner_rets,
        "L-W": lmw_rets,
    }, index=ret_dates)

    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    stats_dict = strategy_statistics(portfolio_returns)

    return {
        "portfolio_returns": portfolio_returns,
        "cumulative_returns": cumulative_returns,
        "rebalance_dates": rebalance_dates,
        "statistics": stats_dict,
        "holdings_log": holdings_log,
    }


# ─────────────────────────────────────────────────────────────
# IPO-Specific Contrarian Strategy
# ─────────────────────────────────────────────────────────────

def ipo_contrarian_strategy(
    ipo_data: pd.DataFrame,
    oversubscription_threshold: float = None,
) -> dict:
    """
    Analyse an IPO-specific contrarian strategy:
      "Sell IPO on day 1" vs. "Hold for N days"

    Parameters
    ----------
    ipo_data : pd.DataFrame
        Must contain columns:
        - ticker : str
        - ipo_date : datetime
        - offer_price : float
        - first_day_close : float
        - oversubscription_ratio : float (times oversubscribed)
        - close_5d, close_10d, close_30d, close_60d, close_90d : float
          (closing prices at various horizons after IPO)
        Missing horizon columns are simply skipped.

    oversubscription_threshold : float, optional
        Threshold for "hot" IPO.  Defaults to config value.

    Returns
    -------
    dict
        overall : summary across all IPOs
        hot_ipos : summary for oversubscribed IPOs
        cold_ipos : summary for undersubscribed IPOs
        by_horizon : return comparison per holding horizon
        full_table : enriched DataFrame with all calculated fields
    """
    oversubscription_threshold = (
        oversubscription_threshold or config.OVERSUBSCRIPTION_THRESHOLD
    )

    df = ipo_data.copy()

    # --- First-day return ---
    df["first_day_return"] = (
        (df["first_day_close"] - df["offer_price"]) / df["offer_price"]
    )

    # --- Returns at various horizons (vs offer price) ---
    horizons = {}
    for col in df.columns:
        if col.startswith("close_") and col.endswith("d"):
            days_str = col.replace("close_", "").replace("d", "")
            try:
                days = int(days_str)
                horizon_col = f"return_{days}d"
                df[horizon_col] = (df[col] - df["offer_price"]) / df["offer_price"]
                horizons[days] = horizon_col
            except ValueError:
                continue

    # --- "Sell day 1" savings vs holding ---
    # If you sell on day 1, you lock in first_day_return.
    # If you hold for N days, your return is return_Nd.
    # "Savings from selling early" = first_day_return - return_Nd
    #   (positive means selling early was better)
    for days, ret_col in horizons.items():
        df[f"savings_{days}d"] = df["first_day_return"] - df[ret_col]

    # --- Hot / Cold IPO split ---
    if "oversubscription_ratio" in df.columns:
        df["is_hot"] = df["oversubscription_ratio"] >= oversubscription_threshold
    else:
        df["is_hot"] = False

    # --- Summary statistics ---
    def _summarise_group(group: pd.DataFrame) -> dict:
        summary = {
            "n_ipos": len(group),
            "avg_first_day_return": group["first_day_return"].mean(),
            "median_first_day_return": group["first_day_return"].median(),
            "pct_positive_first_day": (group["first_day_return"] > 0).mean(),
        }

        for days, ret_col in horizons.items():
            savings_col = f"savings_{days}d"
            if savings_col in group.columns:
                savings = group[savings_col].dropna()
                summary[f"avg_savings_{days}d"] = savings.mean()
                summary[f"median_savings_{days}d"] = savings.median()
                summary[f"pct_better_selling_{days}d"] = (savings > 0).mean()

                # t-test: is average savings significantly > 0?
                if len(savings) > 1 and savings.std() > 0:
                    t_stat, p_val = stats.ttest_1samp(savings, 0)
                    summary[f"savings_tstat_{days}d"] = t_stat
                    summary[f"savings_pvalue_{days}d"] = p_val
                else:
                    summary[f"savings_tstat_{days}d"] = np.nan
                    summary[f"savings_pvalue_{days}d"] = np.nan

        return summary

    overall = _summarise_group(df)
    hot = _summarise_group(df[df["is_hot"]]) if df["is_hot"].any() else {}
    cold = _summarise_group(df[~df["is_hot"]]) if (~df["is_hot"]).any() else {}

    # --- By-horizon comparison table ---
    by_horizon = []
    for days, ret_col in sorted(horizons.items()):
        row = {
            "horizon_days": days,
            "avg_hold_return": df[ret_col].mean(),
            "avg_first_day_return": df["first_day_return"].mean(),
            "avg_savings": df[f"savings_{days}d"].mean(),
            "pct_better_selling": (df[f"savings_{days}d"] > 0).mean(),
        }
        by_horizon.append(row)

    by_horizon_df = pd.DataFrame(by_horizon) if by_horizon else pd.DataFrame()

    return {
        "overall": overall,
        "hot_ipos": hot,
        "cold_ipos": cold,
        "by_horizon": by_horizon_df,
        "full_table": df,
    }


# ─────────────────────────────────────────────────────────────
# Strategy Performance Statistics
# ─────────────────────────────────────────────────────────────

def strategy_statistics(
    portfolio_returns: pd.DataFrame | pd.Series,
    risk_free_rate: float = 0.0,
    annualisation_factor: int = 252,
) -> dict:
    """
    Compute standard strategy performance metrics.

    Parameters
    ----------
    portfolio_returns : pd.DataFrame or pd.Series
        Daily portfolio returns.  If DataFrame, statistics are computed
        for each column independently.
    risk_free_rate : float
        Annualised risk-free rate (default 0).
    annualisation_factor : int
        Trading days per year.

    Returns
    -------
    dict
        Per-column dict with: total_return, annualised_return,
        annualised_vol, sharpe_ratio, max_drawdown, max_drawdown_duration,
        win_rate, avg_win, avg_loss, profit_factor, skewness, kurtosis,
        n_obs
    """
    if isinstance(portfolio_returns, pd.Series):
        portfolio_returns = portfolio_returns.to_frame("Strategy")

    results = {}

    for col in portfolio_returns.columns:
        rets = portfolio_returns[col].dropna()
        if len(rets) == 0:
            results[col] = {}
            continue

        cum = (1 + rets).cumprod()
        total_ret = cum.iloc[-1] - 1

        n_days = len(rets)
        ann_ret = (1 + total_ret) ** (annualisation_factor / max(n_days, 1)) - 1
        ann_vol = rets.std() * np.sqrt(annualisation_factor)

        daily_rf = (1 + risk_free_rate) ** (1 / annualisation_factor) - 1
        excess = rets - daily_rf
        sharpe = (
            excess.mean() / excess.std() * np.sqrt(annualisation_factor)
            if excess.std() > 0 else np.nan
        )

        # Max drawdown
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()

        # Drawdown duration (in trading days)
        is_dd = drawdown < 0
        if is_dd.any():
            dd_groups = (~is_dd).cumsum()
            dd_lengths = is_dd.groupby(dd_groups).sum()
            max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
        else:
            max_dd_duration = 0

        wins = rets[rets > 0]
        losses = rets[rets < 0]
        win_rate = len(wins) / len(rets) if len(rets) > 0 else np.nan
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0

        # Profit factor = gross profits / gross losses
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else np.inf
        )

        results[col] = {
            "total_return": total_ret,
            "annualised_return": ann_ret,
            "annualised_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": max_dd_duration,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "skewness": float(rets.skew()),
            "kurtosis": float(rets.kurtosis()),
            "n_obs": n_days,
        }

    # If only one column, flatten
    if len(results) == 1:
        return list(results.values())[0]

    return results


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_contrarian_results(
    backtest_results: dict = None,
    ipo_results: dict = None,
    title: str = "Contrarian Strategy Results",
) -> go.Figure:
    """
    Visualise contrarian strategy outcomes.

    Panels (as available):
    1. Cumulative returns (loser, winner, L-W)
    2. Rolling spread (L-W)
    3. IPO sell-day-1 savings by horizon
    4. Hot vs Cold IPO comparison

    Parameters
    ----------
    backtest_results : dict, optional
        Output of `contrarian_backtest`.
    ipo_results : dict, optional
        Output of `ipo_contrarian_strategy`.
    title : str
        Overall title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    panels = []
    if backtest_results is not None:
        panels.append("cum_ret")
        panels.append("spread")
    if ipo_results is not None:
        panels.append("ipo_savings")
        if ipo_results.get("hot_ipos") and ipo_results.get("cold_ipos"):
            panels.append("hot_cold")

    n_panels = max(len(panels), 1)

    subplot_titles = []
    for p in panels:
        if p == "cum_ret":
            subplot_titles.append("Cumulative Portfolio Returns")
        elif p == "spread":
            subplot_titles.append("Loser - Winner Spread (L-W)")
        elif p == "ipo_savings":
            subplot_titles.append("Savings from Selling IPO on Day 1")
        elif p == "hot_cold":
            subplot_titles.append("Hot vs Cold IPOs: Sell-Day-1 Savings")

    fig = make_subplots(
        rows=n_panels, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )

    row = 1

    # Panel 1: Cumulative returns
    if "cum_ret" in panels:
        cum = backtest_results["cumulative_returns"]

        color_map = {
            "Losers": config.COLOR_PALETTE["positive"],
            "Winners": config.COLOR_PALETTE["negative"],
            "L-W": config.COLOR_PALETTE["primary"],
        }

        for col in cum.columns:
            fig.add_trace(go.Scatter(
                x=cum.index,
                y=cum[col].values,
                mode="lines",
                name=col,
                line=dict(
                    color=color_map.get(col, config.COLOR_PALETTE["neutral"]),
                    width=2 if col == "L-W" else 1.5,
                    dash="solid" if col == "L-W" else "dot",
                ),
            ), row=row, col=1)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color=config.COLOR_PALETTE["neutral"],
            row=row, col=1,
        )

        # Annotate stats
        lw_stats = backtest_results["statistics"]
        if isinstance(lw_stats, dict) and "L-W" in lw_stats:
            lw_s = lw_stats["L-W"]
        elif isinstance(lw_stats, dict) and "sharpe_ratio" in lw_stats:
            lw_s = lw_stats
        else:
            lw_s = None

        if lw_s:
            annotation_text = (
                f"Sharpe: {lw_s.get('sharpe_ratio', 0):.2f} | "
                f"MaxDD: {lw_s.get('max_drawdown', 0):.1%} | "
                f"Win: {lw_s.get('win_rate', 0):.0%}"
            )
            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10),
                row=row, col=1,
            )

        row += 1

    # Panel 2: L-W spread over time
    if "spread" in panels:
        lmw = backtest_results["portfolio_returns"]["L-W"]
        rolling_lmw = lmw.rolling(20, min_periods=5).mean()

        fig.add_trace(go.Bar(
            x=lmw.index,
            y=lmw.values,
            name="Daily L-W",
            marker_color=[
                config.COLOR_PALETTE["positive"] if v >= 0
                else config.COLOR_PALETTE["negative"]
                for v in lmw.values
            ],
            opacity=0.4,
            showlegend=False,
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=rolling_lmw.index,
            y=rolling_lmw.values,
            mode="lines",
            name="20d Rolling Avg",
            line=dict(color=config.COLOR_PALETTE["primary"], width=2),
        ), row=row, col=1)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color=config.COLOR_PALETTE["neutral"],
            row=row, col=1,
        )
        row += 1

    # Panel 3: IPO sell-day-1 savings by horizon
    if "ipo_savings" in panels:
        bh = ipo_results["by_horizon"]
        if len(bh) > 0:
            x_labels = [f"{int(d)}d" for d in bh["horizon_days"]]
            fig.add_trace(go.Bar(
                x=x_labels,
                y=bh["avg_savings"].values,
                name="Avg Savings (Sell Day 1)",
                marker_color=[
                    config.COLOR_PALETTE["positive"] if v >= 0
                    else config.COLOR_PALETTE["negative"]
                    for v in bh["avg_savings"].values
                ],
                text=[f"{v:.1%}" for v in bh["avg_savings"].values],
                textposition="outside",
            ), row=row, col=1)

            fig.add_hline(
                y=0, line_dash="dash",
                line_color=config.COLOR_PALETTE["neutral"],
                row=row, col=1,
            )
        row += 1

    # Panel 4: Hot vs Cold IPO comparison
    if "hot_cold" in panels:
        hot = ipo_results["hot_ipos"]
        cold = ipo_results["cold_ipos"]

        # Compare key metrics
        categories = ["First Day Return", "Win Rate (Sell Day 1)"]
        hot_vals = [
            hot.get("avg_first_day_return", 0),
            hot.get("pct_positive_first_day", 0),
        ]
        cold_vals = [
            cold.get("avg_first_day_return", 0),
            cold.get("pct_positive_first_day", 0),
        ]

        fig.add_trace(go.Bar(
            x=categories,
            y=hot_vals,
            name=f"Hot IPOs (N={hot.get('n_ipos', 0)})",
            marker_color=config.COLOR_PALETTE["ipo"],
        ), row=row, col=1)

        fig.add_trace(go.Bar(
            x=categories,
            y=cold_vals,
            name=f"Cold IPOs (N={cold.get('n_ipos', 0)})",
            marker_color=config.COLOR_PALETTE["neutral"],
        ), row=row, col=1)

    fig.update_layout(
        title=title,
        template=config.DASHBOARD_THEME,
        height=350 * n_panels,
        hovermode="x unified",
        barmode="group",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig
