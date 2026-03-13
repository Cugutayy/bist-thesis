"""
CSAD Herding Analysis Module
==============================
Cross-Sectional Absolute Deviation (CSAD) herding analysis for BIST.
Based on Chang, Cheng, and Khorana (2000) methodology.

Under rational asset pricing, CSAD should increase linearly with |R_m|.
If CSAD increases at a *decreasing* rate (gamma_2 < 0), this is evidence
of herding -- investors suppress their own beliefs and follow the crowd.

Reference: Chang, E.C., Cheng, J.W. & Khorana, A. (2000)
    "An examination of herd behavior in equity markets"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# ─── Project imports ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


# ─────────────────────────────────────────────────────────────
# CSAD Calculation
# ─────────────────────────────────────────────────────────────

def calculate_csad(
    stock_returns_df: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.Series:
    """
    Calculate Cross-Sectional Absolute Deviation of returns.

    CSAD_t = (1/N) * sum(|R_i,t - R_m,t|)  for i = 1..N

    Parameters
    ----------
    stock_returns_df : pd.DataFrame
        DataFrame of individual stock returns (columns = tickers,
        index = dates).
    market_returns : pd.Series
        Equal-weighted or cap-weighted market return series.

    Returns
    -------
    pd.Series
        CSAD series indexed by date.
    """
    # Align dates
    common_idx = stock_returns_df.index.intersection(market_returns.index)
    stock_ret = stock_returns_df.loc[common_idx]
    mkt_ret = market_returns.loc[common_idx]

    # |R_i,t - R_m,t| for each stock, then average across stocks
    deviations = stock_ret.subtract(mkt_ret, axis=0).abs()
    csad = deviations.mean(axis=1)
    csad.name = "CSAD"

    return csad


# ─────────────────────────────────────────────────────────────
# Herding Test – Main Regression
# ─────────────────────────────────────────────────────────────

def test_herding(
    csad_series: pd.Series,
    market_returns: pd.Series,
    significance_level: float = None,
) -> dict:
    """
    Test for herding using the Chang-Cheng-Khorana (2000) regression.

    Model:
        CSAD_t = alpha + gamma_1 * |R_m,t| + gamma_2 * R_m,t^2 + epsilon_t

    Interpretation:
        gamma_2 < 0  and  statistically significant  -->  herding evidence

    Parameters
    ----------
    csad_series : pd.Series
        CSAD time series.
    market_returns : pd.Series
        Market return series.
    significance_level : float, optional
        Threshold for significance.  Defaults to config value.

    Returns
    -------
    dict
        alpha, gamma1, gamma2, gamma2_tstat, gamma2_pvalue,
        herding_detected (bool), r_squared, n_obs,
        newey_west_se, model_summary
    """
    significance_level = significance_level or config.HERDING_SIGNIFICANCE

    # Align
    combined = pd.DataFrame({
        "csad": csad_series,
        "rm": market_returns,
    }).dropna()

    if len(combined) < 30:
        raise ValueError(f"Too few observations for herding test: {len(combined)}")

    y = combined["csad"]
    abs_rm = combined["rm"].abs()
    rm_sq = combined["rm"] ** 2

    X = pd.DataFrame({
        "abs_rm": abs_rm,
        "rm_sq": rm_sq,
    })
    X = sm.add_constant(X)

    # OLS with Newey-West HAC standard errors for robustness
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    alpha = model.params["const"]
    gamma1 = model.params["abs_rm"]
    gamma2 = model.params["rm_sq"]
    gamma2_tstat = model.tvalues["rm_sq"]
    gamma2_pvalue = model.pvalues["rm_sq"]

    herding_detected = (gamma2 < 0) and (gamma2_pvalue < significance_level)

    return {
        "alpha": alpha,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "gamma2_tstat": gamma2_tstat,
        "gamma2_pvalue": gamma2_pvalue,
        "herding_detected": herding_detected,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": len(combined),
        "newey_west_se": model.bse.to_dict(),
        "model_summary": model.summary(),
    }


# ─────────────────────────────────────────────────────────────
# Rolling Herding – Time-Varying gamma_2
# ─────────────────────────────────────────────────────────────

def rolling_herding(
    csad_series: pd.Series,
    market_returns: pd.Series,
    window: int = None,
) -> pd.DataFrame:
    """
    Estimate the herding coefficient gamma_2 on a rolling basis.

    Parameters
    ----------
    csad_series : pd.Series
        CSAD time series.
    market_returns : pd.Series
        Market return series.
    window : int, optional
        Rolling window size in trading days.
        Defaults to config.CSAD_ROLLING_WINDOW.

    Returns
    -------
    pd.DataFrame
        Columns: gamma2, gamma2_tstat, gamma2_pvalue, herding_flag
        Index: dates
    """
    window = window or config.CSAD_ROLLING_WINDOW

    combined = pd.DataFrame({
        "csad": csad_series,
        "rm": market_returns,
    }).dropna()

    if len(combined) < window + 10:
        raise ValueError(
            f"Need at least {window + 10} observations for rolling herding "
            f"(have {len(combined)})."
        )

    combined["abs_rm"] = combined["rm"].abs()
    combined["rm_sq"] = combined["rm"] ** 2

    y = combined["csad"]
    X = sm.add_constant(combined[["abs_rm", "rm_sq"]])

    rolling_model = RollingOLS(y, X, window=window)
    rolling_results = rolling_model.fit()

    gamma2 = rolling_results.params["rm_sq"]
    gamma2_tstat = rolling_results.tvalues["rm_sq"]

    # p-values from t-distribution (df = window - 3)
    df = window - 3
    gamma2_pvalue = 2 * stats.t.sf(gamma2_tstat.abs(), df=df)

    result = pd.DataFrame({
        "gamma2": gamma2,
        "gamma2_tstat": gamma2_tstat,
        "gamma2_pvalue": gamma2_pvalue,
        "herding_flag": (gamma2 < 0) & (gamma2_pvalue < config.HERDING_SIGNIFICANCE),
    })

    return result.dropna()


# ─────────────────────────────────────────────────────────────
# Regime Herding – Bull vs. Bear
# ─────────────────────────────────────────────────────────────

def regime_herding(
    csad_series: pd.Series,
    market_returns: pd.Series,
    significance_level: float = None,
) -> dict:
    """
    Separate herding analysis for bull (R_m > 0) and bear (R_m < 0) markets.

    Model for each regime:
        CSAD_t = alpha + gamma_1 * |R_m,t| + gamma_2 * R_m,t^2 + epsilon

    Parameters
    ----------
    csad_series : pd.Series
        CSAD time series.
    market_returns : pd.Series
        Market return series.
    significance_level : float, optional
        Threshold for significance.

    Returns
    -------
    dict
        bull: herding test dict for R_m > 0 days
        bear: herding test dict for R_m < 0 days
        comparison: summary comparison of both regimes
    """
    significance_level = significance_level or config.HERDING_SIGNIFICANCE

    combined = pd.DataFrame({
        "csad": csad_series,
        "rm": market_returns,
    }).dropna()

    bull_mask = combined["rm"] > 0
    bear_mask = combined["rm"] < 0

    bull_csad = combined.loc[bull_mask, "csad"]
    bull_rm = combined.loc[bull_mask, "rm"]

    bear_csad = combined.loc[bear_mask, "csad"]
    bear_rm = combined.loc[bear_mask, "rm"]

    bull_result = test_herding(bull_csad, bull_rm, significance_level)
    bear_result = test_herding(bear_csad, bear_rm, significance_level)

    comparison = {
        "bull_gamma2": bull_result["gamma2"],
        "bear_gamma2": bear_result["gamma2"],
        "bull_herding": bull_result["herding_detected"],
        "bear_herding": bear_result["herding_detected"],
        "bull_n": bull_result["n_obs"],
        "bear_n": bear_result["n_obs"],
        "stronger_in": (
            "bear" if abs(bear_result["gamma2"]) > abs(bull_result["gamma2"])
            else "bull"
        ),
    }

    return {
        "bull": bull_result,
        "bear": bear_result,
        "comparison": comparison,
    }


# ─────────────────────────────────────────────────────────────
# Herding Around IPO Dates
# ─────────────────────────────────────────────────────────────

def herding_vs_ipo(
    csad_series: pd.Series,
    market_returns: pd.Series,
    ipo_dates: list | pd.DatetimeIndex,
    window_around_ipo: int = 10,
) -> dict:
    """
    Test whether herding is stronger around IPO dates.

    Compares gamma_2 in periods around IPO dates vs. non-IPO periods.

    Parameters
    ----------
    csad_series : pd.Series
        CSAD time series.
    market_returns : pd.Series
        Market return series.
    ipo_dates : list or DatetimeIndex
        IPO listing dates.
    window_around_ipo : int
        Number of trading days around each IPO to flag as "IPO period".

    Returns
    -------
    dict
        ipo_period: herding test dict
        non_ipo_period: herding test dict
        difference_test: Chow-type comparison
    """
    combined = pd.DataFrame({
        "csad": csad_series,
        "rm": market_returns,
    }).dropna()

    ipo_dates = pd.DatetimeIndex(ipo_dates)

    # Mark IPO-adjacent dates
    ipo_mask = pd.Series(False, index=combined.index)
    for ipo_dt in ipo_dates:
        # Find trading days within window_around_ipo of each IPO
        day_diff = (combined.index - ipo_dt).days
        near_ipo = (day_diff >= -window_around_ipo) & (day_diff <= window_around_ipo)
        ipo_mask = ipo_mask | near_ipo

    ipo_csad = combined.loc[ipo_mask, "csad"]
    ipo_rm = combined.loc[ipo_mask, "rm"]
    non_ipo_csad = combined.loc[~ipo_mask, "csad"]
    non_ipo_rm = combined.loc[~ipo_mask, "rm"]

    ipo_result = None
    non_ipo_result = None
    difference_gamma2 = np.nan
    diff_significant = False

    if len(ipo_csad) >= 30:
        ipo_result = test_herding(ipo_csad, ipo_rm)

    if len(non_ipo_csad) >= 30:
        non_ipo_result = test_herding(non_ipo_csad, non_ipo_rm)

    # Dummy-variable interaction approach for formal comparison
    if ipo_result is not None and non_ipo_result is not None:
        combined["abs_rm"] = combined["rm"].abs()
        combined["rm_sq"] = combined["rm"] ** 2
        combined["ipo_dummy"] = ipo_mask.astype(int)
        combined["ipo_x_rm_sq"] = combined["ipo_dummy"] * combined["rm_sq"]

        y = combined["csad"]
        X = sm.add_constant(combined[["abs_rm", "rm_sq", "ipo_dummy", "ipo_x_rm_sq"]])

        interaction_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        difference_gamma2 = interaction_model.params["ipo_x_rm_sq"]
        diff_pvalue = interaction_model.pvalues["ipo_x_rm_sq"]
        diff_significant = diff_pvalue < config.HERDING_SIGNIFICANCE

        difference_test = {
            "interaction_coeff": difference_gamma2,
            "interaction_pvalue": diff_pvalue,
            "significant": diff_significant,
            "model_summary": interaction_model.summary(),
        }
    else:
        difference_test = {
            "interaction_coeff": np.nan,
            "interaction_pvalue": np.nan,
            "significant": False,
            "model_summary": None,
        }

    return {
        "ipo_period": ipo_result,
        "non_ipo_period": non_ipo_result,
        "ipo_days": int(ipo_mask.sum()),
        "non_ipo_days": int((~ipo_mask).sum()),
        "difference_test": difference_test,
    }


# ─────────────────────────────────────────────────────────────
# Herding in Manipulated vs. Non-Manipulated Stocks
# ─────────────────────────────────────────────────────────────

def herding_in_manipulated_stocks(
    manipulated_tickers: list[str],
    all_stock_returns: pd.DataFrame,
    market_returns: pd.Series,
) -> dict:
    """
    Compare CSAD-based herding in manipulated vs. non-manipulated stocks.

    Parameters
    ----------
    manipulated_tickers : list[str]
        List of ticker symbols flagged as manipulated (SPK sanctions).
    all_stock_returns : pd.DataFrame
        Full DataFrame of stock returns (columns = tickers).
    market_returns : pd.Series
        Market return series.

    Returns
    -------
    dict
        manipulated: herding test on subset
        non_manipulated: herding test on complement
        comparison: summary including difference test
    """
    all_tickers = set(all_stock_returns.columns)
    manip_set = set(manipulated_tickers) & all_tickers
    non_manip_set = all_tickers - manip_set

    if not manip_set:
        raise ValueError(
            "No manipulated tickers found in the stock returns DataFrame."
        )

    manip_returns = all_stock_returns[list(manip_set)]
    non_manip_returns = all_stock_returns[list(non_manip_set)]

    # CSAD for each subset
    csad_manip = calculate_csad(manip_returns, market_returns)
    csad_non_manip = calculate_csad(non_manip_returns, market_returns)

    manip_result = test_herding(csad_manip, market_returns)
    non_manip_result = test_herding(csad_non_manip, market_returns)

    # Mean CSAD comparison (Welch t-test)
    t_mean, p_mean = stats.ttest_ind(
        csad_manip.dropna(), csad_non_manip.dropna(), equal_var=False
    )

    comparison = {
        "manip_gamma2": manip_result["gamma2"],
        "non_manip_gamma2": non_manip_result["gamma2"],
        "manip_herding": manip_result["herding_detected"],
        "non_manip_herding": non_manip_result["herding_detected"],
        "manip_n_stocks": len(manip_set),
        "non_manip_n_stocks": len(non_manip_set),
        "mean_csad_manip": csad_manip.mean(),
        "mean_csad_non_manip": csad_non_manip.mean(),
        "mean_csad_diff_tstat": t_mean,
        "mean_csad_diff_pvalue": p_mean,
    }

    return {
        "manipulated": manip_result,
        "non_manipulated": non_manip_result,
        "csad_manipulated": csad_manip,
        "csad_non_manipulated": csad_non_manip,
        "comparison": comparison,
    }


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_herding_results(
    csad_series: pd.Series = None,
    market_returns: pd.Series = None,
    rolling_df: pd.DataFrame = None,
    regime_results: dict = None,
    title: str = "CSAD Herding Analysis",
) -> go.Figure:
    """
    Comprehensive herding visualisation.

    Up to 4 panels depending on what data is provided:
    1. CSAD over time
    2. CSAD vs R_m^2 scatter (herding regression)
    3. Rolling gamma_2
    4. Bull vs Bear comparison

    Parameters
    ----------
    csad_series : pd.Series, optional
        CSAD time series.
    market_returns : pd.Series, optional
        Market return series.
    rolling_df : pd.DataFrame, optional
        Output of `rolling_herding`.
    regime_results : dict, optional
        Output of `regime_herding`.
    title : str
        Overall title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    # Determine number of subplots
    panels = []
    if csad_series is not None:
        panels.append("csad_ts")
    if csad_series is not None and market_returns is not None:
        panels.append("scatter")
    if rolling_df is not None:
        panels.append("rolling")
    if regime_results is not None:
        panels.append("regime")

    n_panels = max(len(panels), 1)

    subplot_titles = []
    for p in panels:
        if p == "csad_ts":
            subplot_titles.append("CSAD Over Time")
        elif p == "scatter":
            subplot_titles.append("CSAD vs Rm-squared")
        elif p == "rolling":
            subplot_titles.append("Rolling Gamma-2 Coefficient")
        elif p == "regime":
            subplot_titles.append("Bull vs Bear Herding")

    fig = make_subplots(
        rows=n_panels, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )

    row = 1

    # Panel 1: CSAD time series
    if "csad_ts" in panels:
        fig.add_trace(go.Scatter(
            x=csad_series.index,
            y=csad_series.values,
            mode="lines",
            name="CSAD",
            line=dict(color=config.COLOR_PALETTE["primary"], width=1),
        ), row=row, col=1)

        # Add rolling mean
        rolling_mean = csad_series.rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean.values,
            mode="lines",
            name="CSAD (20d MA)",
            line=dict(color=config.COLOR_PALETTE["secondary"], width=2),
        ), row=row, col=1)
        row += 1

    # Panel 2: Scatter -- CSAD vs Rm^2
    if "scatter" in panels:
        combined = pd.DataFrame({
            "csad": csad_series,
            "rm": market_returns,
        }).dropna()
        combined["rm_sq"] = combined["rm"] ** 2

        fig.add_trace(go.Scatter(
            x=combined["rm_sq"],
            y=combined["csad"],
            mode="markers",
            name="Daily Obs.",
            marker=dict(
                color=config.COLOR_PALETTE["primary"],
                size=3,
                opacity=0.5,
            ),
        ), row=row, col=1)

        # Fitted quadratic curve
        x_fit = np.linspace(combined["rm_sq"].min(), combined["rm_sq"].max(), 200)
        herding = test_herding(csad_series, market_returns)
        y_fit = (
            herding["alpha"]
            + herding["gamma1"] * np.sqrt(x_fit)
            + herding["gamma2"] * x_fit
        )
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            name=f"Fit (gamma2={herding['gamma2']:.4f})",
            line=dict(color=config.COLOR_PALETTE["negative"], width=2),
        ), row=row, col=1)
        row += 1

    # Panel 3: Rolling gamma_2
    if "rolling" in panels:
        fig.add_trace(go.Scatter(
            x=rolling_df.index,
            y=rolling_df["gamma2"],
            mode="lines",
            name="Rolling gamma_2",
            line=dict(color=config.COLOR_PALETTE["primary"], width=1.5),
        ), row=row, col=1)

        # Highlight herding episodes
        herding_eps = rolling_df[rolling_df["herding_flag"]]
        if len(herding_eps) > 0:
            fig.add_trace(go.Scatter(
                x=herding_eps.index,
                y=herding_eps["gamma2"],
                mode="markers",
                name="Herding (sig.)",
                marker=dict(
                    color=config.COLOR_PALETTE["negative"],
                    size=4,
                    symbol="circle",
                ),
            ), row=row, col=1)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color=config.COLOR_PALETTE["neutral"],
            row=row, col=1,
        )
        row += 1

    # Panel 4: Bull vs Bear bar chart
    if "regime" in panels:
        comp = regime_results["comparison"]
        categories = ["Bull Market", "Bear Market"]
        gammas = [comp["bull_gamma2"], comp["bear_gamma2"]]
        colors = [config.COLOR_PALETTE["positive"], config.COLOR_PALETTE["negative"]]
        text_labels = [
            f"{'***' if comp['bull_herding'] else 'n.s.'}",
            f"{'***' if comp['bear_herding'] else 'n.s.'}",
        ]

        fig.add_trace(go.Bar(
            x=categories,
            y=gammas,
            marker_color=colors,
            text=text_labels,
            textposition="outside",
            name="gamma_2",
            showlegend=False,
        ), row=row, col=1)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color=config.COLOR_PALETTE["neutral"],
            row=row, col=1,
        )

    fig.update_layout(
        title=title,
        template=config.DASHBOARD_THEME,
        height=350 * n_panels,
        showlegend=True,
        hovermode="x unified",
    )

    return fig
