"""
Dashboard Page 3: Inflation Illusion Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

st.set_page_config(page_title="Inflation Illusion", page_icon="💰", layout="wide")

# ─── Page Header ──────────────────────────────────────────
st.title("Dimension 3: Inflation Illusion")

st.info(
    "**What is this page about?**\n\n"
    "Turkey experienced extreme inflation between 2020-2025. "
    "This page explores whether investors confuse **nominal gains** (TL numbers going up) "
    "with **real wealth creation** (actual purchasing power). "
    "An IPO return of +50% sounds great — but if inflation was 60% that year, "
    "you actually *lost* purchasing power.\n\n"
    "**Formula:** Real Return = (1 + Nominal Return) / (1 + Inflation) - 1"
)

st.markdown("---")


@st.cache_data(ttl=3600)
def load_data():
    try:
        from src.data_collection.macro_data import get_tufe_data, get_macro_summary
        tufe = get_tufe_data()
        summary = get_macro_summary()
        return tufe, summary
    except Exception as e:
        st.warning(f"Data loading error: {e}")
        return None, None


@st.cache_data(ttl=3600)
def load_ipo_data():
    processed = config.PROCESSED_DIR / "ipo_dataset.csv"
    if processed.exists():
        return pd.read_csv(processed, parse_dates=["ipo_date"])
    return None


@st.cache_data(ttl=3600)
def load_market_data():
    """Load BIST and USD/TRY from yfinance with correct handling."""
    try:
        import yfinance as yf
        bist = yf.download("XU100.IS", start="2020-01-01", end="2025-12-31", progress=False)
        usdtry = yf.download("USDTRY=X", start="2020-01-01", end="2025-12-31", progress=False)

        result = {}
        if bist is not None and len(bist) > 0:
            result["bist_close"] = bist["Close"].squeeze()  # MultiIndex -> Series
        if usdtry is not None and len(usdtry) > 0:
            result["usdtry_close"] = usdtry["Close"].squeeze()
        return result
    except Exception as e:
        st.warning(f"Could not fetch market data: {e}")
        return {}


tufe_df, macro_summary = load_data()
ipo_df = load_ipo_data()
market = load_market_data()

# ─── The Big Picture (ALL from data) ─────────────────────
st.markdown("### The Big Picture: What Did BIST Really Return?")
st.caption(
    "All numbers below are computed from actual data — nothing is hardcoded."
)

bist_close = market.get("bist_close")
usdtry_close = market.get("usdtry_close")

m1, m2, m3, m4 = st.columns(4)

if bist_close is not None:
    bist_pct = (bist_close.iloc[-1] / bist_close.iloc[0] - 1)
    with m1:
        st.metric("BIST-100 Nominal", f"+{bist_pct:.0%}",
                  delta=f"{bist_close.iloc[0]:.0f} → {bist_close.iloc[-1]:.0f}")

if tufe_df is not None and "tufe_index" in tufe_df.columns:
    cpi_first = tufe_df["tufe_index"].iloc[0]
    cpi_last = tufe_df["tufe_index"].iloc[-1]
    cpi_pct = (cpi_last / cpi_first - 1)
    with m2:
        st.metric("CPI Increase", f"+{cpi_pct:.0%}",
                  delta=f"Index: {cpi_first:.0f} → {cpi_last:.0f}")

if bist_close is not None and tufe_df is not None:
    bist_real = (1 + bist_pct) / (1 + cpi_pct) - 1
    with m3:
        st.metric("BIST-100 Real", f"+{bist_real:.0%}", delta="CPI-adjusted")

if bist_close is not None and usdtry_close is not None:
    bist_usd_first = bist_close.iloc[0] / usdtry_close.reindex(bist_close.index, method="ffill").iloc[0]
    bist_usd_last = bist_close.iloc[-1] / usdtry_close.reindex(bist_close.index, method="ffill").iloc[-1]
    bist_usd_pct = (bist_usd_last / bist_usd_first - 1)
    with m4:
        st.metric("BIST-100 in USD", f"+{bist_usd_pct:.0%}",
                  delta=f"${bist_usd_first:.0f} → ${bist_usd_last:.0f}")

if bist_close is not None and tufe_df is not None:
    st.warning(
        f"**Key insight:** The BIST-100 rose **+{bist_pct:.0%}** in nominal TL — but CPI increased **+{cpi_pct:.0%}**. "
        f"After inflation adjustment, the real return was **+{bist_real:.0%}**. "
        f"In USD terms, the gain was only **+{bist_usd_pct:.0%}**. "
        f"The gap between +{bist_pct:.0%} nominal and +{bist_real:.0%} real is the **Inflation Illusion**."
    )

st.markdown("---")

# ─── Inflation Timeline ─────────────────────────────────
if tufe_df is not None:
    st.markdown("### Turkey CPI (TUFE) Timeline")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Monthly YoY Inflation (%)")
        st.caption(
            "How much more expensive things got compared to a year ago. "
            "The red dashed line at 50% marks a critical threshold."
        )
        if "tufe_yoy" in tufe_df.columns:
            peak_val = tufe_df["tufe_yoy"].max()
            peak_date = tufe_df.loc[tufe_df["tufe_yoy"].idxmax(), "date"]
            fig = px.area(
                tufe_df, x="date", y="tufe_yoy",
                color_discrete_sequence=[config.COLOR_PALETTE["inflation"]],
                labels={"tufe_yoy": "YoY Inflation (%)", "date": "Date"},
            )
            fig.add_hline(y=50, line_dash="dash", line_color="red",
                         annotation_text="50% threshold")
            fig.update_layout(template=config.DASHBOARD_THEME, height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Peak: {peak_val:.1f}% in {pd.Timestamp(peak_date).strftime('%b %Y')}")

    with c2:
        st.markdown("#### CPI Index (2003=100)")
        st.caption(
            "Cumulative price increases since 2003. "
            "A steeper curve means prices are rising faster."
        )
        if "tufe_index" in tufe_df.columns:
            fig = px.line(
                tufe_df, x="date", y="tufe_index",
                color_discrete_sequence=[config.COLOR_PALETTE["negative"]],
                labels={"tufe_index": "CPI Index", "date": "Date"},
            )
            fig.update_layout(template=config.DASHBOARD_THEME, height=380)
            st.plotly_chart(fig, use_container_width=True)

# ─── BIST vs Inflation vs USD — ACTUAL PRICES ──────────
st.markdown("---")
st.markdown("### BIST-100 vs Inflation vs USD/TRY")

tab_actual, tab_indexed = st.tabs(["Actual Values", "Indexed (Jan 2020 = 100)"])

if bist_close is not None:
    with tab_actual:
        st.info(
            "**Actual price levels.** The left axis shows BIST-100 index value and CPI. "
            "The right axis shows USD/TRY exchange rate."
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=bist_close.index, y=bist_close.values,
                      name=f"BIST-100 ({bist_close.iloc[0]:.0f} → {bist_close.iloc[-1]:.0f})",
                      line=dict(color="#1f77b4", width=2)),
            secondary_y=False,
        )

        if tufe_df is not None and "tufe_index" in tufe_df.columns:
            cpi = tufe_df.set_index("date")["tufe_index"]
            fig.add_trace(
                go.Scatter(x=cpi.index, y=cpi.values,
                          name=f"CPI Index ({cpi.iloc[0]:.0f} → {cpi.iloc[-1]:.0f})",
                          line=dict(color="#d62728", width=2, dash="dash")),
                secondary_y=False,
            )

        if usdtry_close is not None:
            fig.add_trace(
                go.Scatter(x=usdtry_close.index, y=usdtry_close.values,
                          name=f"USD/TRY ({usdtry_close.iloc[0]:.2f} → {usdtry_close.iloc[-1]:.2f})",
                          line=dict(color="#ff7f0e", width=2, dash="dot")),
                secondary_y=True,
            )

        fig.update_layout(
            template=config.DASHBOARD_THEME,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=550,
        )
        fig.update_yaxes(title_text="BIST-100 / CPI Index", secondary_y=False)
        fig.update_yaxes(title_text="USD/TRY", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab_indexed:
        st.info(
            "**All lines normalized to 100 at January 2020** — this lets you compare growth rates directly. "
            "A value of 500 means 5x growth from the starting point."
        )

        fig = go.Figure()
        bist_norm = bist_close / bist_close.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=bist_norm.index, y=bist_norm.values,
            name=f"BIST-100 Nominal ({bist_norm.iloc[-1]:.0f})",
            line=dict(color="#1f77b4", width=2),
        ))

        if tufe_df is not None and "tufe_index" in tufe_df.columns:
            cpi = tufe_df.set_index("date")["tufe_index"]
            cpi_norm = cpi / cpi.iloc[0] * 100
            cpi_daily = cpi_norm.resample("D").interpolate()
            fig.add_trace(go.Scatter(
                x=cpi_daily.index, y=cpi_daily.values,
                name=f"CPI Index ({cpi_daily.iloc[-1]:.0f})",
                line=dict(color="#d62728", width=2, dash="dash"),
            ))

        if usdtry_close is not None:
            usd_norm = usdtry_close / usdtry_close.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=usd_norm.index, y=usd_norm.values,
                name=f"USD/TRY ({usd_norm.iloc[-1]:.0f})",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
            ))

        if usdtry_close is not None:
            common = bist_close.index.intersection(usdtry_close.index)
            bist_usd = bist_close.loc[common].values / usdtry_close.loc[common].values
            bist_usd_norm = pd.Series(bist_usd / bist_usd[0] * 100, index=common)
            fig.add_trace(go.Scatter(
                x=bist_usd_norm.index, y=bist_usd_norm.values,
                name=f"BIST-100 in USD ({bist_usd_norm.iloc[-1]:.0f})",
                line=dict(color="#2ca02c", width=2),
            ))

        fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5,
                     annotation_text="Starting point (100)")
        fig.update_layout(
            template=config.DASHBOARD_THEME,
            yaxis_title="Growth (Jan 2020 = 100)",
            xaxis_title="Date",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "**How to read:** BIST-100 at 968 means the index grew 9.68x since Jan 2020. "
            "CPI at 694 means prices grew 6.94x. The gap between blue and red is the real return."
        )

# ─── IPO Returns: Nominal vs Real ───────────────────────
if ipo_df is not None and tufe_df is not None:
    st.markdown("---")
    st.markdown("### IPO Returns: Nominal vs Real")

    st.info(
        "**Adjusting for inflation:** We compute each IPO's real return at each horizon "
        "using the actual CPI data for that specific time period — not a blanket estimate."
    )

    # Compute period-specific inflation for each IPO
    if "tufe_index" in tufe_df.columns:
        cpi_ts = tufe_df.set_index("date")["tufe_index"].resample("D").interpolate()

        results_table = []
        upr_col = "underpricing" if "underpricing" in ipo_df.columns else "first_day_return"
        for period, col in [("Tavan Series", upr_col),
                           ("30 Days", "return_d30"), ("90 Days", "return_d90"),
                           ("180 Days", "return_d180"), ("365 Days", "return_d365")]:
            if col in ipo_df.columns:
                valid = ipo_df[["ipo_date", col]].dropna()
                nominal_mean = valid[col].mean()

                # Compute actual inflation for each IPO's specific period
                if period == "Tavan Series":
                    # Use tavan_days to determine the inflation period
                    valid_with_tavan = ipo_df[["ipo_date", col, "tavan_days"]].dropna() if "tavan_days" in ipo_df.columns else valid.copy()
                    valid_with_tavan["_days"] = valid_with_tavan["tavan_days"].apply(lambda x: max(int(x), 1)) if "tavan_days" in valid_with_tavan.columns else 1
                else:
                    valid_with_tavan = valid.copy()
                    valid_with_tavan["_days"] = int(period.split()[0])
                real_returns = []
                for _, row in valid_with_tavan.iterrows():
                    ipo_date = pd.Timestamp(row["ipo_date"])
                    days = int(row.get("_days", 1)) if "_days" in row.index else 1
                    end_date = ipo_date + pd.Timedelta(days=days)
                    try:
                        cpi_start = cpi_ts.asof(ipo_date)
                        cpi_end = cpi_ts.asof(end_date)
                        if pd.notna(cpi_start) and pd.notna(cpi_end) and cpi_start > 0:
                            period_inflation = (cpi_end / cpi_start) - 1
                            real_ret = (1 + row[col]) / (1 + period_inflation) - 1
                            real_returns.append(real_ret)
                    except:
                        pass

                real_mean = np.mean(real_returns) if real_returns else None
                results_table.append({
                    "Period": period,
                    "Avg Nominal Return": nominal_mean,
                    "Avg Period Inflation": nominal_mean - real_mean if real_mean else None,
                    "Avg Real Return": real_mean,
                    "N": len(valid),
                })

        if results_table:
            rt_df = pd.DataFrame(results_table)
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Nominal vs Real Returns by Holding Period")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Nominal Return",
                    x=rt_df["Period"], y=rt_df["Avg Nominal Return"],
                    marker_color="#1f77b4", text=rt_df["Avg Nominal Return"].apply(lambda x: f"{x:+.1%}"),
                    textposition="outside",
                ))
                fig.add_trace(go.Bar(
                    name="Real Return",
                    x=rt_df["Period"], y=rt_df["Avg Real Return"],
                    marker_color="#2ca02c", text=rt_df["Avg Real Return"].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"),
                    textposition="outside",
                ))
                fig.update_layout(
                    template=config.DASHBOARD_THEME, barmode="group",
                    yaxis_tickformat=".0%", yaxis_title="Return", height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("#### Return Comparison Table")
                display_rt = rt_df.copy()
                for c_name in ["Avg Nominal Return", "Avg Period Inflation", "Avg Real Return"]:
                    if c_name in display_rt.columns:
                        display_rt[c_name] = display_rt[c_name].apply(
                            lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"
                        )
                st.dataframe(display_rt, use_container_width=True)
                st.caption(
                    "Inflation is computed individually for each IPO based on the actual CPI "
                    "during its specific holding period, then averaged."
                )

    # Scatter: Nominal vs Real 1-year returns
    if "return_d365" in ipo_df.columns:
        st.markdown("#### IPO 1-Year Returns: Nominal vs Real")
        st.caption(
            "Each dot is one IPO. The **red shaded area** is the 'Illusion Zone' — "
            "IPOs with positive nominal returns but negative real returns."
        )

        plot_df = ipo_df[["ticker", "ipo_date", "return_d365"]].dropna().copy()

        # Compute actual real return for each IPO
        real_rets_365 = []
        for _, row in plot_df.iterrows():
            ipo_date = pd.Timestamp(row["ipo_date"])
            end_date = ipo_date + pd.Timedelta(days=365)
            try:
                cpi_start = cpi_ts.asof(ipo_date)
                cpi_end = cpi_ts.asof(end_date)
                if pd.notna(cpi_start) and pd.notna(cpi_end) and cpi_start > 0:
                    period_inflation = (cpi_end / cpi_start) - 1
                    real_rets_365.append((1 + row["return_d365"]) / (1 + period_inflation) - 1)
                else:
                    real_rets_365.append(np.nan)
            except:
                real_rets_365.append(np.nan)

        plot_df["real_return"] = real_rets_365
        plot_df = plot_df.dropna(subset=["real_return"])
        plot_df["illusion"] = (plot_df["return_d365"] > 0) & (plot_df["real_return"] < 0)

        n_illusion = plot_df["illusion"].sum()
        n_total = len(plot_df)

        if n_total > 0:
            st.error(
                f"**Money Illusion Cases:** {n_illusion}/{n_total} IPOs "
                f"({n_illusion/n_total:.0%}) had positive nominal but NEGATIVE real 1-year returns."
            )

            fig = px.scatter(
                plot_df, x="return_d365", y="real_return",
                color="illusion",
                color_discrete_map={True: config.COLOR_PALETTE["negative"], False: config.COLOR_PALETTE["positive"]},
                hover_data=["ticker"],
                labels={
                    "return_d365": "Nominal 1-Year Return",
                    "real_return": "Real 1-Year Return",
                    "illusion": "Money Illusion Victim",
                },
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
            if n_illusion > 0:
                fig.add_vrect(
                    x0=0, x1=plot_df["return_d365"].max() * 1.1,
                    y0=plot_df["real_return"].min() * 1.1, y1=0,
                    fillcolor="rgba(255,0,0,0.05)", line_width=0,
                    annotation_text="ILLUSION ZONE",
                )
            fig.update_layout(
                template=config.DASHBOARD_THEME,
                xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

# ─── Inflation vs IPO Demand ────────────────────────────
st.markdown("---")
st.markdown("### Does Inflation Drive IPO Demand?")

st.info(
    "**Modigliani & Cohn (1979) Hypothesis:**\n\n"
    "Investors in high-inflation environments tend to use nominal interest rates "
    "to discount future cash flows. This makes stocks appear more attractive, "
    "potentially driving more investment into IPOs during inflationary periods."
)

if ipo_df is not None and tufe_df is not None:
    if "ipo_date" in ipo_df.columns:
        ipo_monthly = ipo_df.copy()
        ipo_monthly["month"] = pd.to_datetime(ipo_monthly["ipo_date"]).dt.to_period("M").dt.to_timestamp()

        monthly_counts = ipo_monthly.groupby("month").agg(
            ipo_count=("ticker", "size"),
        ).reset_index()

        if "date" in tufe_df.columns:
            tufe_monthly = tufe_df.copy()
            tufe_monthly["month"] = pd.to_datetime(tufe_monthly["date"]).dt.to_period("M").dt.to_timestamp()
            merged = monthly_counts.merge(tufe_monthly[["month", "tufe_yoy"]], on="month", how="left")

            if "tufe_yoy" in merged.columns:
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("#### IPO Count vs Inflation Over Time")
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Bar(x=merged["month"], y=merged["ipo_count"],
                               name="IPO Count", marker_color=config.COLOR_PALETTE["ipo"]),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(x=merged["month"], y=merged["tufe_yoy"],
                                   name="Inflation YoY %", line=dict(color="red", width=2)),
                        secondary_y=True,
                    )
                    fig.update_layout(template=config.DASHBOARD_THEME, height=400)
                    fig.update_yaxes(title_text="IPO Count", secondary_y=False)
                    fig.update_yaxes(title_text="Inflation YoY %", secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    st.markdown("#### Correlation: Inflation vs IPO Frequency")
                    from scipy import stats
                    valid = merged.dropna(subset=["tufe_yoy", "ipo_count"])
                    if len(valid) > 5:
                        corr, pval = stats.pearsonr(valid["tufe_yoy"], valid["ipo_count"])

                        fig = px.scatter(
                            valid, x="tufe_yoy", y="ipo_count",
                            trendline="ols",
                            labels={"tufe_yoy": "YoY Inflation %", "ipo_count": "Monthly IPO Count"},
                            color_discrete_sequence=[config.COLOR_PALETTE["inflation"]],
                        )
                        fig.update_layout(template=config.DASHBOARD_THEME, height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown(
                            f"**Pearson r = {corr:.3f}** (p = {pval:.4f}) — "
                            f"{'Significant' if pval < 0.05 else 'Not significant'} at 5% level."
                        )

# ─── Summary Table ───────────────────────────────────────
st.markdown("---")
st.markdown("### Key Statistics (from data)")

if bist_close is not None and tufe_df is not None and usdtry_close is not None:
    bist_pct_val = bist_close.iloc[-1] / bist_close.iloc[0] - 1
    cpi_pct_val = tufe_df["tufe_index"].iloc[-1] / tufe_df["tufe_index"].iloc[0] - 1
    bist_real_val = (1 + bist_pct_val) / (1 + cpi_pct_val) - 1
    usdtry_chg = usdtry_close.iloc[-1] / usdtry_close.iloc[0] - 1

    common = bist_close.index.intersection(usdtry_close.index)
    busd_0 = bist_close.loc[common].iloc[0] / usdtry_close.loc[common].iloc[0]
    busd_n = bist_close.loc[common].iloc[-1] / usdtry_close.loc[common].iloc[-1]
    bist_usd_val = busd_n / busd_0 - 1

    peak_inf = tufe_df["tufe_yoy"].max() if "tufe_yoy" in tufe_df.columns else "?"
    avg_inf = tufe_df["tufe_yoy"].mean() if "tufe_yoy" in tufe_df.columns else "?"

    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | BIST-100 Nominal Return | **+{bist_pct_val:.0%}** ({bist_close.iloc[0]:.0f} → {bist_close.iloc[-1]:.0f}) |
    | BIST-100 Real Return (CPI-adjusted) | **+{bist_real_val:.0%}** |
    | BIST-100 USD Return | **+{bist_usd_val:.0%}** (${busd_0:.0f} → ${busd_n:.0f}) |
    | Cumulative CPI | **+{cpi_pct_val:.0%}** |
    | USD/TRY Depreciation | **+{usdtry_chg:.0%}** ({usdtry_close.iloc[0]:.2f} → {usdtry_close.iloc[-1]:.2f}) |
    | Peak YoY Inflation | **{peak_inf:.1f}%** |
    | Average YoY Inflation | **{avg_inf:.1f}%** |
    """)

st.markdown("---")
st.caption("Data sources: TUIK (CPI data), Yahoo Finance (market data)")
