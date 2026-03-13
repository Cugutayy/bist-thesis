"""
Event Study Analysis Module
============================
Comprehensive event study framework for BIST thesis project.
Implements market model estimation, abnormal return calculation,
cumulative abnormal returns, buy-and-hold abnormal returns,
and multiple significance tests.

Reference: MacKinlay (1997), "Event Studies in Economics and Finance"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm

# ─── Project imports ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


# ─────────────────────────────────────────────────────────────
# Market Model Estimation
# ─────────────────────────────────────────────────────────────

def market_model_estimation(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    estimation_window: int = None,
) -> dict:
    """
    Estimate the market model parameters via OLS regression.

    Model: R_i,t = alpha + beta * R_m,t + epsilon_t

    Parameters
    ----------
    stock_returns : pd.Series
        Daily stock returns during the estimation window.
    market_returns : pd.Series
        Daily market returns during the estimation window.
    estimation_window : int, optional
        Number of observations to use. If None, uses all provided data.
        Defaults to config.ESTIMATION_WINDOW if not specified.

    Returns
    -------
    dict
        Keys: alpha, beta, sigma, r_squared, residuals, model_summary,
              n_obs, estimation_start, estimation_end
    """
    if estimation_window is None:
        estimation_window = config.ESTIMATION_WINDOW

    # Align the two series on their common index
    combined = pd.DataFrame({
        "stock": stock_returns,
        "market": market_returns,
    }).dropna()

    if len(combined) < config.MIN_OBSERVATIONS:
        raise ValueError(
            f"Insufficient observations for estimation: {len(combined)} < "
            f"{config.MIN_OBSERVATIONS} (MIN_OBSERVATIONS)."
        )

    # Trim to the estimation window (most recent observations)
    if len(combined) > estimation_window:
        combined = combined.iloc[-estimation_window:]

    y = combined["stock"]
    X = sm.add_constant(combined["market"])

    model = sm.OLS(y, X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    sigma = np.sqrt(model.mse_resid)

    return {
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "r_squared": model.rsquared,
        "residuals": model.resid,
        "model_summary": model.summary(),
        "n_obs": len(combined),
        "estimation_start": combined.index.min(),
        "estimation_end": combined.index.max(),
    }


# ─────────────────────────────────────────────────────────────
# Abnormal Returns
# ─────────────────────────────────────────────────────────────

def calculate_abnormal_returns(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    alpha: float,
    beta: float,
) -> pd.Series:
    """
    Calculate abnormal returns using the market model.

    AR_t = R_i,t - (alpha + beta * R_m,t)

    Parameters
    ----------
    stock_returns : pd.Series
        Actual stock returns during the event window.
    market_returns : pd.Series
        Market returns during the event window.
    alpha : float
        Intercept from market model estimation.
    beta : float
        Slope from market model estimation.

    Returns
    -------
    pd.Series
        Abnormal returns indexed by date.
    """
    combined = pd.DataFrame({
        "stock": stock_returns,
        "market": market_returns,
    }).dropna()

    expected_returns = alpha + beta * combined["market"]
    abnormal_returns = combined["stock"] - expected_returns
    abnormal_returns.name = "AR"

    return abnormal_returns


# ─────────────────────────────────────────────────────────────
# CAR – Cumulative Abnormal Return
# ─────────────────────────────────────────────────────────────

def calculate_car(
    ar_series: pd.Series,
    start: int = None,
    end: int = None,
) -> pd.Series:
    """
    Calculate Cumulative Abnormal Return over a window.

    CAR(t1, t2) = sum(AR_t) for t in [t1, t2]

    Parameters
    ----------
    ar_series : pd.Series
        Abnormal return series (indexed by date or relative day).
    start : int, optional
        Start index (relative to event). If None, uses the first index.
    end : int, optional
        End index (relative to event). If None, uses the last index.

    Returns
    -------
    pd.Series
        Cumulative abnormal return series.
    """
    if start is not None and end is not None:
        ar_slice = ar_series.iloc[start:end + 1]
    elif start is not None:
        ar_slice = ar_series.iloc[start:]
    elif end is not None:
        ar_slice = ar_series.iloc[:end + 1]
    else:
        ar_slice = ar_series

    car = ar_slice.cumsum()
    car.name = "CAR"
    return car


# ─────────────────────────────────────────────────────────────
# BHAR – Buy-and-Hold Abnormal Return
# ─────────────────────────────────────────────────────────────

def calculate_bhar(
    stock_prices: pd.Series,
    market_prices: pd.Series,
    start: int = None,
    end: int = None,
) -> float:
    """
    Calculate Buy-and-Hold Abnormal Return.

    BHAR = product(1 + R_i,t) - product(1 + R_m,t)  for t in [start, end]

    Parameters
    ----------
    stock_prices : pd.Series
        Stock price series (levels, not returns).
    market_prices : pd.Series
        Market index price series (levels, not returns).
    start : int, optional
        Start index position.  Defaults to first available.
    end : int, optional
        End index position.  Defaults to last available.

    Returns
    -------
    float
        Buy-and-hold abnormal return.
    """
    combined = pd.DataFrame({
        "stock": stock_prices,
        "market": market_prices,
    }).dropna()

    if start is not None and end is not None:
        combined = combined.iloc[start:end + 1]
    elif start is not None:
        combined = combined.iloc[start:]
    elif end is not None:
        combined = combined.iloc[:end + 1]

    if len(combined) < 2:
        raise ValueError("Need at least 2 price observations for BHAR.")

    # Compute returns from prices
    stock_ret = combined["stock"].pct_change().dropna()
    market_ret = combined["market"].pct_change().dropna()

    stock_bh = np.prod(1 + stock_ret) - 1
    market_bh = np.prod(1 + market_ret) - 1
    bhar = stock_bh - market_bh

    return float(bhar)


# ─────────────────────────────────────────────────────────────
# Significance Tests
# ─────────────────────────────────────────────────────────────

def test_significance(
    car_values: np.ndarray,
    sigma: float = None,
    sigmas: np.ndarray = None,
) -> dict:
    """
    Perform multiple significance tests on CAR/CAAR values.

    Tests
    -----
    1. Parametric t-test  : H0: mean(CAR) = 0
    2. Boehmer et al. (1991) standardised cross-sectional test
    3. Non-parametric sign test

    Parameters
    ----------
    car_values : array-like
        Array of CAR values (one per event in a multi-event study,
        or a single-element array for a single event).
    sigma : float, optional
        Single estimation-period standard deviation.  Used for Boehmer
        test normalisation when per-event sigmas are not available,
        and for single-event t-test.
    sigmas : array-like, optional
        Per-event estimation-period standard deviations (one per event).
        Preferred for Boehmer test: each CAR_i is standardised by its
        own sigma_i per Boehmer et al. (1991).

    Returns
    -------
    dict
        t_stat, t_pvalue, boehmer_stat, boehmer_pvalue,
        sign_stat, sign_pvalue, n_events, mean_car, std_car
    """
    car_values = np.asarray(car_values, dtype=float)
    car_values = car_values[~np.isnan(car_values)]
    n = len(car_values)

    if n == 0:
        raise ValueError("No valid CAR values provided.")

    mean_car = np.mean(car_values)
    std_car = np.std(car_values, ddof=1) if n > 1 else 0.0

    # --- 1. Parametric t-test ---
    if n > 1 and std_car > 0:
        t_stat = mean_car / (std_car / np.sqrt(n))
        t_pvalue = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    elif n == 1:
        # Single event: if sigma available, use it
        eff_sigma = sigma
        if sigmas is not None and len(sigmas) == 1:
            eff_sigma = sigmas[0]
        if eff_sigma is not None and eff_sigma > 0:
            t_stat = car_values[0] / eff_sigma
            t_pvalue = 2 * stats.norm.sf(abs(t_stat))
        else:
            t_stat = np.nan
            t_pvalue = np.nan
    else:
        t_stat = np.nan
        t_pvalue = np.nan

    # --- 2. Boehmer et al. (1991) standardised cross-sectional test ---
    # Per Boehmer et al. (1991), each event's CAR should be standardised
    # by its own estimation-period sigma_i before cross-sectional aggregation.
    boehmer_stat = np.nan
    boehmer_pvalue = np.nan

    if n > 1:
        if sigmas is not None:
            # Preferred: per-event standardisation
            sigmas_arr = np.asarray(sigmas, dtype=float)
            # Filter to valid pairs (both CAR and sigma non-NaN and sigma > 0)
            valid = ~np.isnan(sigmas_arr) & (sigmas_arr > 0)
            if valid.sum() > 1:
                scar = car_values[valid] / sigmas_arr[valid]
                mean_scar = np.mean(scar)
                std_scar = np.std(scar, ddof=1)
                if std_scar > 0:
                    n_valid = len(scar)
                    boehmer_stat = mean_scar / (std_scar / np.sqrt(n_valid))
                    boehmer_pvalue = 2 * stats.t.sf(abs(boehmer_stat), df=n_valid - 1)
        elif sigma is not None and sigma > 0:
            # Fallback: single sigma for all events
            scar = car_values / sigma
            mean_scar = np.mean(scar)
            std_scar = np.std(scar, ddof=1)
            if std_scar > 0:
                boehmer_stat = mean_scar / (std_scar / np.sqrt(n))
                boehmer_pvalue = 2 * stats.t.sf(abs(boehmer_stat), df=n - 1)

    # --- 3. Non-parametric sign test ---
    n_positive = np.sum(car_values > 0)
    n_negative = np.sum(car_values < 0)
    n_nonzero = n_positive + n_negative

    if n_nonzero > 0:
        # Under H0 the number of positives ~ Binomial(n_nonzero, 0.5)
        sign_stat = (n_positive - 0.5 * n_nonzero) / (0.5 * np.sqrt(n_nonzero))
        sign_pvalue = 2 * stats.norm.sf(abs(sign_stat))
    else:
        sign_stat = np.nan
        sign_pvalue = np.nan

    return {
        "t_stat": t_stat,
        "t_pvalue": t_pvalue,
        "boehmer_stat": boehmer_stat,
        "boehmer_pvalue": boehmer_pvalue,
        "sign_stat": sign_stat,
        "sign_pvalue": sign_pvalue,
        "n_events": n,
        "mean_car": mean_car,
        "std_car": std_car,
    }


# ─────────────────────────────────────────────────────────────
# Full Event Study Pipeline
# ─────────────────────────────────────────────────────────────

def run_event_study(
    stock_data: pd.DataFrame,
    market_data: pd.DataFrame,
    event_date: str | pd.Timestamp,
    estimation_window: int = None,
    event_window_pre: int = None,
    event_window_post: int = None,
    price_col: str = "close",
    return_col: str = "return",
) -> dict:
    """
    Run a complete event study for a single event.

    Parameters
    ----------
    stock_data : pd.DataFrame
        Stock data with DatetimeIndex. Must contain `price_col` and/or
        `return_col`.  If `return_col` is missing it is computed from
        `price_col`.
    market_data : pd.DataFrame
        Market (benchmark) data with the same requirements.
    event_date : str or pd.Timestamp
        The event date (e.g., IPO listing date).
    estimation_window : int, optional
        Length of the estimation window.  Defaults to config value.
    event_window_pre : int, optional
        Trading days before the event.  Defaults to config value.
    event_window_post : int, optional
        Trading days after the event.  Defaults to config value.
    price_col : str
        Column name for price levels.
    return_col : str
        Column name for daily returns.

    Returns
    -------
    dict
        ar_series, car_series, bhar, t_stat, t_pvalue,
        boehmer_stat, boehmer_pvalue, sign_stat, sign_pvalue,
        confidence_interval_95, model_params, event_date,
        event_window_dates, relative_days
    """
    estimation_window = estimation_window or config.ESTIMATION_WINDOW
    event_window_pre = event_window_pre or config.EVENT_WINDOW_PRE
    event_window_post = event_window_post or config.EVENT_WINDOW_POST

    event_date = pd.Timestamp(event_date)

    # --- Ensure return columns exist ---
    for df, label in [(stock_data, "stock"), (market_data, "market")]:
        if return_col not in df.columns and price_col in df.columns:
            df[return_col] = df[price_col].pct_change()

    if return_col not in stock_data.columns:
        raise ValueError(f"Cannot find '{return_col}' or '{price_col}' in stock_data.")
    if return_col not in market_data.columns:
        raise ValueError(f"Cannot find '{return_col}' or '{price_col}' in market_data.")

    # --- Locate event date in trading calendar ---
    trading_days = stock_data.index.intersection(market_data.index).sort_values()

    # Find the closest trading day on or after the event date
    valid_days = trading_days[trading_days >= event_date]
    if len(valid_days) == 0:
        raise ValueError(f"No trading days found on or after event date {event_date}.")
    event_idx_date = valid_days[0]
    event_pos = trading_days.get_loc(event_idx_date)

    # --- Define windows ---
    est_start = max(0, event_pos - event_window_pre - estimation_window)
    est_end = event_pos - event_window_pre
    evt_start = event_pos - event_window_pre
    evt_end = min(len(trading_days) - 1, event_pos + event_window_post)

    if est_end - est_start < config.MIN_OBSERVATIONS:
        raise ValueError(
            f"Estimation window too short: {est_end - est_start} days "
            f"(need {config.MIN_OBSERVATIONS})."
        )

    estimation_dates = trading_days[est_start:est_end]
    event_dates = trading_days[evt_start:evt_end + 1]

    # --- Market model estimation ---
    est_stock_ret = stock_data.loc[estimation_dates, return_col].dropna()
    est_market_ret = market_data.loc[estimation_dates, return_col].dropna()

    model = market_model_estimation(est_stock_ret, est_market_ret, estimation_window)

    # --- Abnormal returns in event window ---
    evt_stock_ret = stock_data.loc[event_dates, return_col]
    evt_market_ret = market_data.loc[event_dates, return_col]

    ar = calculate_abnormal_returns(
        evt_stock_ret, evt_market_ret, model["alpha"], model["beta"]
    )

    # Create relative day index
    relative_days = np.arange(-event_window_pre, len(ar) - event_window_pre)
    relative_days = relative_days[: len(ar)]
    ar_indexed = ar.copy()
    ar_indexed.index = relative_days[: len(ar_indexed)]

    # --- CAR ---
    car = calculate_car(ar_indexed)

    # --- BHAR ---
    bhar_value = np.nan
    if price_col in stock_data.columns and price_col in market_data.columns:
        try:
            evt_stock_px = stock_data.loc[event_dates, price_col]
            evt_market_px = market_data.loc[event_dates, price_col]
            bhar_value = calculate_bhar(evt_stock_px, evt_market_px)
        except (ValueError, KeyError):
            bhar_value = np.nan

    # --- Significance tests ---
    final_car = car.iloc[-1] if len(car) > 0 else np.nan
    sig = test_significance(np.array([final_car]), sigma=model["sigma"])

    # --- Confidence interval ---
    se = model["sigma"] * np.sqrt(len(ar))
    ci_95 = (final_car - 1.96 * se, final_car + 1.96 * se)

    return {
        "ar_series": ar_indexed,
        "car_series": car,
        "bhar": bhar_value,
        "t_stat": sig["t_stat"],
        "t_pvalue": sig["t_pvalue"],
        "boehmer_stat": sig["boehmer_stat"],
        "boehmer_pvalue": sig["boehmer_pvalue"],
        "sign_stat": sig["sign_stat"],
        "sign_pvalue": sig["sign_pvalue"],
        "confidence_interval_95": ci_95,
        "model_params": {
            "alpha": model["alpha"],
            "beta": model["beta"],
            "sigma": model["sigma"],
            "r_squared": model["r_squared"],
            "n_obs": model["n_obs"],
        },
        "event_date": event_date,
        "event_window_dates": event_dates,
        "relative_days": relative_days,
    }


# ─────────────────────────────────────────────────────────────
# Aggregate Event Study (CAAR)
# ─────────────────────────────────────────────────────────────

def aggregate_event_study(results_list: list[dict]) -> dict:
    """
    Aggregate multiple single-event study results into CAAR analysis.

    CAAR(t1, t2) = (1/N) * sum(CAR_i(t1, t2))

    Parameters
    ----------
    results_list : list[dict]
        List of dicts returned by `run_event_study`.

    Returns
    -------
    dict
        caar_series, mean_ar_series, n_events,
        t_stat, t_pvalue, boehmer_stat, boehmer_pvalue,
        sign_stat, sign_pvalue, individual_cars, individual_bhars
    """
    if not results_list:
        raise ValueError("Empty results list.")

    # Collect all AR and CAR series aligned by relative day
    ar_frames = {}
    car_frames = {}
    final_cars = []
    bhars = []
    sigmas = []

    for i, res in enumerate(results_list):
        ar_frames[i] = res["ar_series"]
        car_frames[i] = res["car_series"]
        if len(res["car_series"]) > 0:
            final_cars.append(res["car_series"].iloc[-1])
        bhars.append(res["bhar"])
        sigmas.append(res["model_params"]["sigma"])

    ar_df = pd.DataFrame(ar_frames)
    car_df = pd.DataFrame(car_frames)

    # Mean AR and CAAR across events
    mean_ar = ar_df.mean(axis=1)
    mean_ar.name = "Mean_AR"

    caar = car_df.mean(axis=1)
    caar.name = "CAAR"

    # Significance on final CAR values
    final_cars = np.array(final_cars)
    sigmas_arr = np.array(sigmas) if sigmas else None

    sig = test_significance(final_cars, sigmas=sigmas_arr)

    return {
        "caar_series": caar,
        "mean_ar_series": mean_ar,
        "n_events": len(results_list),
        "t_stat": sig["t_stat"],
        "t_pvalue": sig["t_pvalue"],
        "boehmer_stat": sig["boehmer_stat"],
        "boehmer_pvalue": sig["boehmer_pvalue"],
        "sign_stat": sig["sign_stat"],
        "sign_pvalue": sig["sign_pvalue"],
        "individual_cars": final_cars,
        "individual_bhars": np.array(bhars),
    }


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_car(
    car_series: pd.Series,
    event_date: str | pd.Timestamp = None,
    title: str = "Cumulative Abnormal Return",
    show_ci: bool = True,
    sigma: float = None,
) -> go.Figure:
    """
    Plot CAR or CAAR series with optional confidence bands.

    Parameters
    ----------
    car_series : pd.Series
        CAR (or CAAR) series indexed by relative event day.
    event_date : str or pd.Timestamp, optional
        The event date for labelling.
    title : str
        Chart title.
    show_ci : bool
        Whether to display 95% confidence bands.
    sigma : float, optional
        Standard deviation for computing confidence bands.
        If None, uses rolling std of the series itself.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    fig = go.Figure()

    x_vals = car_series.index.tolist()
    y_vals = car_series.values

    # Main CAR line
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines+markers",
        name="CAR",
        line=dict(color=config.COLOR_PALETTE["primary"], width=2),
        marker=dict(size=3),
    ))

    # Confidence bands
    if show_ci:
        if sigma is not None:
            # Widen confidence band with sqrt of days from event
            days_from_start = np.arange(1, len(car_series) + 1)
            se = sigma * np.sqrt(days_from_start)
        else:
            # Fallback: expanding std
            se = car_series.expanding(min_periods=2).std().fillna(0).values

        upper = y_vals + 1.96 * se
        lower = y_vals - 1.96 * se

        fig.add_trace(go.Scatter(
            x=x_vals + x_vals[::-1],
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(31,119,180,0.15)",
            line=dict(width=0),
            name="95% CI",
            showlegend=True,
        ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=config.COLOR_PALETTE["neutral"])

    # Event day marker
    if 0 in x_vals:
        fig.add_vline(
            x=0,
            line_dash="dot",
            line_color=config.COLOR_PALETTE["negative"],
            annotation_text="Event Day",
        )

    subtitle = f"Event: {event_date}" if event_date else ""
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>"),
        xaxis_title="Relative Trading Day",
        yaxis_title="Cumulative Abnormal Return",
        template=config.DASHBOARD_THEME,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_aggregate_event_study(agg_results: dict, title: str = None) -> go.Figure:
    """
    Plot the CAAR and mean AR from aggregated event study results.

    Parameters
    ----------
    agg_results : dict
        Output of `aggregate_event_study`.
    title : str, optional
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with two subplots.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Cumulative Average Abnormal Return (CAAR)", "Mean Abnormal Return"),
        vertical_spacing=0.12,
    )

    caar = agg_results["caar_series"]
    mean_ar = agg_results["mean_ar_series"]

    # CAAR
    fig.add_trace(go.Scatter(
        x=caar.index.tolist(),
        y=caar.values,
        mode="lines+markers",
        name="CAAR",
        line=dict(color=config.COLOR_PALETTE["primary"], width=2),
        marker=dict(size=3),
    ), row=1, col=1)

    # Mean AR bar chart
    colors = [
        config.COLOR_PALETTE["positive"] if v >= 0 else config.COLOR_PALETTE["negative"]
        for v in mean_ar.values
    ]
    fig.add_trace(go.Bar(
        x=mean_ar.index.tolist(),
        y=mean_ar.values,
        name="Mean AR",
        marker_color=colors,
    ), row=2, col=1)

    # Zero lines
    fig.add_hline(y=0, line_dash="dash", line_color=config.COLOR_PALETTE["neutral"], row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=config.COLOR_PALETTE["neutral"], row=2, col=1)

    # Event day marker
    fig.add_vline(x=0, line_dash="dot", line_color=config.COLOR_PALETTE["negative"])

    default_title = (
        f"Aggregate Event Study (N={agg_results['n_events']})"
        f"  |  t={agg_results['t_stat']:.2f}, p={agg_results['t_pvalue']:.4f}"
    )

    fig.update_layout(
        title=title or default_title,
        template=config.DASHBOARD_THEME,
        height=700,
        hovermode="x unified",
        xaxis2_title="Relative Trading Day",
        yaxis_title="CAAR",
        yaxis2_title="Mean AR",
    )

    return fig
