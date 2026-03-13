"""
Dashboard Page 1: IPO Analysis
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

st.set_page_config(page_title="IPO Analysis", page_icon="📈", layout="wide")

# ─── Page Header ──────────────────────────────────────────
st.title("Dimension 1: IPO Frenzy Analysis")

st.info(
    "**What is this page about?**\n\n"
    "Between 2020 and 2025, Borsa Istanbul experienced an unprecedented IPO boom. "
    "Millions of retail investors rushed to participate in every new listing. "
    "This page analyzes **whether these IPOs were systematically underpriced** "
    "and **whether a contrarian strategy of selling after the tavan series would have been profitable.**\n\n"
    "**Key concept — Tavan Serisi:** BIST has a daily price limit of ±10%. "
    "Most IPOs hit the upper limit (tavan) on the first day and continue for multiple consecutive days. "
    "The proper measure of underpricing is the **actual cumulative return during the entire tavan series** "
    "(from offer price to first free-trading day), computed from real daily closing prices."
)

st.markdown("---")

# ─── Data Loading ────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_ipo_data():
    processed = config.PROCESSED_DIR / "ipo_dataset.csv"
    if processed.exists():
        df = pd.read_csv(processed, parse_dates=["ipo_date"])
        return df
    try:
        from src.data_collection.ipo_data import build_ipo_dataset
        df = build_ipo_dataset()
        return df
    except Exception as e:
        st.warning(f"Could not load IPO data: {e}")
        return None


df = load_ipo_data()

if df is None or len(df) == 0:
    st.error("IPO data not available. Run data collection first: `python run.py --collect-ipo`")
    st.stop()

# ─── Sidebar Filters ────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    st.caption("Use these filters to narrow down the IPO data by year and sector.")

    year_range = st.slider(
        "Year Range",
        min_value=2020,
        max_value=2025,
        value=(2020, 2025),
    )

    if "sector" in df.columns:
        sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("Sector", sectors)
    else:
        selected_sector = "All"

    # Tavan group filter
    tavan_groups = st.multiselect(
        "Tavan Group",
        ["No Tavan (0)", "Short (1-3)", "Medium (4-7)", "Long (8+)"],
        default=["No Tavan (0)", "Short (1-3)", "Medium (4-7)", "Long (8+)"],
    )

# ─── Apply Filters ──────────────────────────────────────
filtered = df.copy()
if "ipo_date" in filtered.columns:
    filtered["year"] = pd.to_datetime(filtered["ipo_date"]).dt.year
    filtered = filtered[
        (filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])
    ]

if selected_sector != "All" and "sector" in filtered.columns:
    filtered = filtered[filtered["sector"] == selected_sector]

# ─── Key Metrics ─────────────────────────────────────────
st.markdown("### Key Metrics")
st.caption(
    "High-level summary of all IPOs in the selected period. "
    "'Avg Underpricing' shows the average cumulative tavan series return from offer price to first free-trading day. "
    "'Avg Tavan Days' shows how many consecutive days the stock hit the +10% daily limit. "
    "'% Hit Tavan' shows the proportion of IPOs that hit the daily price limit on their first day."
)

m1, m2, m3, m4, m5, m6 = st.columns(6)

# Use underpricing (tavan series return) as the primary metric
underpricing_col = "underpricing" if "underpricing" in filtered.columns else "first_day_return"
first_day_col = underpricing_col  # for backward compatibility with charts below

with m1:
    st.metric("Total IPOs", len(filtered))
with m2:
    if underpricing_col in filtered.columns:
        avg_ret = filtered[underpricing_col].mean()
        st.metric("Avg Underpricing", f"{avg_ret:.1%}")
with m3:
    if underpricing_col in filtered.columns:
        median_ret = filtered[underpricing_col].median()
        st.metric("Median Underpricing", f"{median_ret:.1%}")
with m4:
    if "tavan_days" in filtered.columns:
        avg_tavan = filtered["tavan_days"].mean()
        st.metric("Avg Tavan Days", f"{avg_tavan:.1f}")
with m5:
    if "hit_tavan_day1" in filtered.columns:
        pct_tavan = filtered["hit_tavan_day1"].mean()
        st.metric("% Hit Tavan", f"{pct_tavan:.0%}")
with m6:
    if "tavan_days" in filtered.columns:
        max_tavan = filtered["tavan_days"].max()
        st.metric("Max Tavan Days", f"{int(max_tavan)}")

st.markdown("---")

# ─── Charts Row 1: Timeline & Distribution ──────────────
chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("#### IPO Count by Year")
    st.caption(
        "How many companies went public each year? A rising number of IPOs reflects "
        "growing market optimism and retail investor participation."
    )
    if "year" in filtered.columns:
        yearly = filtered.groupby("year").size().reset_index(name="count")
        fig = px.bar(
            yearly, x="year", y="count",
            color_discrete_sequence=[config.COLOR_PALETTE["ipo"]],
            text="count",
        )
        fig.update_layout(
            template=config.DASHBOARD_THEME,
            xaxis_title="Year", yaxis_title="Number of IPOs",
            showlegend=False, height=380,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

with chart2:
    st.markdown("#### Tavan Series Distribution")
    st.caption(
        "How many consecutive tavan (daily limit +10%) days did each IPO have? "
        "More tavan days means higher underpricing from the offer price. "
        "0 = stock did not hit tavan on IPO day."
    )
    if "tavan_days" in filtered.columns:
        fig = px.histogram(
            filtered, x="tavan_days", nbins=22,
            color_discrete_sequence=[config.COLOR_PALETTE["primary"]],
            labels={"tavan_days": "Consecutive Tavan Days"},
        )
        fig.add_vline(
            x=filtered["tavan_days"].mean(),
            line_dash="dot", line_color="green",
            annotation_text=f"Mean: {filtered['tavan_days'].mean():.1f} days",
        )
        fig.update_layout(template=config.DASHBOARD_THEME, height=380)
        st.plotly_chart(fig, use_container_width=True)

# ─── Charts Row 2: Oversubscription & Returns ───────────
st.markdown("---")
chart3, chart4 = st.columns(2)

with chart3:
    st.markdown("#### Offer Price vs Underpricing")
    st.caption(
        "Does the offer price predict underpricing? Each dot is one IPO. "
        "If the line slopes downward, cheaper IPOs tend to have higher underpricing — "
        "evidence that retail-friendly low-priced IPOs attract more speculative demand."
    )
    if first_day_col and "offer_price" in filtered.columns:
        scatter_df = filtered[["ticker", "offer_price", first_day_col]].dropna()
        if len(scatter_df) > 2:
            fig = px.scatter(
                scatter_df, x="offer_price", y=first_day_col,
                hover_data=["ticker"],
                color_discrete_sequence=[config.COLOR_PALETTE["primary"]],
                trendline="ols",
                labels={
                    "offer_price": "Offer Price (TL)",
                    first_day_col: "Underpricing (Tavan Series Return)",
                },
            )
            fig.update_layout(template=config.DASHBOARD_THEME, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for scatter plot.")

with chart4:
    st.markdown("#### Underpricing by Tavan Group")
    st.caption(
        "IPOs are classified by how many consecutive tavan days they had:\n"
        "- **No Tavan (0 days):** Stock did not hit the price limit\n"
        "- **Short (1-3 days):** Brief tavan series\n"
        "- **Medium (4-7 days):** Moderate tavan series\n"
        "- **Long (8+ days):** Extended tavan series\n\n"
        "Box plots show the distribution of underpricing for each group."
    )
    if first_day_col and "tavan_days" in filtered.columns:
        temp_df = filtered.copy()
        temp_df["tavan_group"] = pd.cut(
            temp_df["tavan_days"].fillna(0),
            bins=[-1, 0, 3, 7, float("inf")],
            labels=["No Tavan (0)", "Short (1-3)", "Medium (4-7)", "Long (8+)"],
        )
        fig = px.box(
            temp_df.dropna(subset=["tavan_group", first_day_col]),
            x="tavan_group", y=first_day_col,
            color="tavan_group",
            color_discrete_map={
                "No Tavan (0)": "#95a5a6",
                "Short (1-3)": "#3498db",
                "Medium (4-7)": "#f39c12",
                "Long (8+)": "#e74c3c",
            },
            labels={first_day_col: "Underpricing (Tavan Series Return)"},
        )
        fig.update_layout(template=config.DASHBOARD_THEME, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# ─── Multi-Period Return Analysis ────────────────────────
st.markdown("---")
st.markdown("### Multi-Period Return Analysis")
st.info(
    "**Does the IPO 'pop' last, or does it fade over time?**\n\n"
    "We track IPO returns from the IPO close price out to 365 days. These are returns based on "
    "Yahoo Finance split-adjusted prices (relative returns are correct). If returns decline over longer periods, "
    "it means initial excitement fades. This complements our **post-tavan contrarian analysis** above."
)

return_period_cols = {
    "return_d1": "1 Day",
    "return_d5": "5 Days",
    "return_d10": "10 Days",
    "return_d30": "30 Days",
    "return_d60": "60 Days",
    "return_d90": "90 Days",
    "return_d180": "180 Days",
    "return_d365": "365 Days",
}

available_returns = {k: v for k, v in return_period_cols.items() if k in filtered.columns}
if available_returns:
    avg_returns = {v: filtered[k].mean() for k, v in available_returns.items()}
    med_returns = {v: filtered[k].median() for k, v in available_returns.items()}

    period_df = pd.DataFrame({
        "Period": list(avg_returns.keys()),
        "Mean Return": list(avg_returns.values()),
        "Median Return": list(med_returns.values()),
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Mean Return", x=period_df["Period"], y=period_df["Mean Return"],
        marker_color=config.COLOR_PALETTE["primary"],
    ))
    fig.add_trace(go.Bar(
        name="Median Return", x=period_df["Period"], y=period_df["Median Return"],
        marker_color=config.COLOR_PALETTE["secondary"],
    ))
    fig.update_layout(
        template=config.DASHBOARD_THEME,
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_title="Return",
        xaxis_title="Holding Period",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Blue bars = average (mean) return across all IPOs. Orange bars = median return. "
        "If bars get smaller for longer periods, it means early gains tend to erode over time."
    )

# ─── Contrarian Strategy Results ─────────────────────────
st.markdown("---")
st.markdown("### Contrarian Strategy: Sell After Tavan or Hold?")

st.info(
    "**Adapted for BIST daily limits:**\n\n"
    "In BIST, during the tavan series you effectively cannot sell (no buyers at limit price). "
    "The real investment decision happens **after the tavan series ends** (first free trading day).\n\n"
    "We compare: Should you sell on the **first free day** or hold longer?\n"
    "- **Short-term (5-10 days):** Post-tavan returns are negative on average (mean-reversion)\n"
    "- **Long-term (60+ days):** Holding tends to be profitable, especially in a bull market\n"
    "- **Tavan length matters:** Longer tavan series = more speculative = worse post-tavan performance"
)

contrarian_file = config.PROCESSED_DIR / "contrarian_by_horizon.csv"
if contrarian_file.exists():
    df_contrarian = pd.read_csv(contrarian_file)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Post-tavan returns by holding period:**")
        for _, row in df_contrarian.iterrows():
            h = int(row["horizon_days"])
            mean_r = row["mean_return"]
            pct_pos = row["pct_positive"]
            sig = "**" if row["p_value"] < 0.05 else ""
            icon = "+" if mean_r > 0 else ""
            st.metric(
                f"{h}-Day Post-Tavan Return",
                f"{icon}{mean_r:.1%}",
                delta=f"{pct_pos:.0%} positive | p={row['p_value']:.3f}" if h <= 90 else None,
            )

    with c2:
        st.markdown("**Post-tavan returns by tavan length:**")
        st.caption(
            "Longer tavan series indicate more speculative demand. "
            "Do they mean-revert more aggressively?"
        )
        contrarian_full = config.PROCESSED_DIR / "contrarian_full_table.csv"
        if contrarian_full.exists():
            df_full = pd.read_csv(contrarian_full)
            if "tavan_group" in df_full.columns and "post_tavan_return_30d" in df_full.columns:
                group_stats = df_full.groupby("tavan_group", observed=True).agg(
                    mean_30d=("post_tavan_return_30d", "mean"),
                    mean_90d=("post_tavan_return_90d", "mean"),
                    count=("ticker", "count"),
                ).reset_index()

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="30-Day", x=group_stats["tavan_group"],
                    y=group_stats["mean_30d"],
                    marker_color=config.COLOR_PALETTE["primary"],
                ))
                if "mean_90d" in group_stats.columns:
                    fig.add_trace(go.Bar(
                        name="90-Day", x=group_stats["tavan_group"],
                        y=group_stats["mean_90d"],
                        marker_color=config.COLOR_PALETTE["secondary"],
                    ))
                fig.update_layout(
                    template=config.DASHBOARD_THEME,
                    barmode="group", yaxis_tickformat=".0%",
                    yaxis_title="Post-Tavan Return",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    # Summary
    short_term = df_contrarian[df_contrarian["horizon_days"] <= 10]
    long_term = df_contrarian[df_contrarian["horizon_days"] >= 60]
    if len(short_term) > 0 and short_term["mean_return"].mean() < 0:
        st.success(
            "**Result:** Post-tavan, IPO stocks show short-term mean reversion "
            f"(5-10 day avg: {short_term['mean_return'].mean():.1%}). "
            f"Selling on the first free day beats holding short-term. "
            f"However, long-term holding (60+ days) tends to be profitable "
            f"(avg: {long_term['mean_return'].mean():.1%}), likely driven by the bull market."
        )
else:
    st.warning("Post-tavan contrarian results not available. Run: `python scripts/rerun_contrarian.py`")

# ─── Sector Analysis ────────────────────────────────────
# Sector data comes from Yahoo Finance (ipo_pe_analysis.csv), not ipo_dataset.csv
st.markdown("---")
st.markdown("### IPO Performance by Sector")
st.caption(
    "Which sectors had the highest underpricing (tavan series return)? "
    "Sectors are sourced from Yahoo Finance classification. "
    "Color intensity shows the number of IPOs — sectors with fewer IPOs may have less reliable statistics."
)

_pe_sector_file = config.PROCESSED_DIR / "ipo_pe_analysis.csv"
if _pe_sector_file.exists() and first_day_col:
    _pe_df = pd.read_csv(_pe_sector_file)
    # Merge sector_yf into filtered data
    _sector_merged = filtered.merge(
        _pe_df[["ticker", "sector_yf"]].dropna(subset=["sector_yf"]),
        on="ticker", how="inner",
    )
    if len(_sector_merged) > 0 and "sector_yf" in _sector_merged.columns:
        sector_stats = _sector_merged.groupby("sector_yf").agg(
            count=("ticker", "size"),
            avg_return=(first_day_col, "mean"),
        ).reset_index().sort_values("avg_return", ascending=False)

        fig = px.bar(
            sector_stats, x="sector_yf", y="avg_return",
            color="count", color_continuous_scale="viridis",
            text=sector_stats["avg_return"].apply(lambda x: f"{x:.0%}"),
            labels={
                "avg_return": "Average Underpricing",
                "count": "Number of IPOs",
                "sector_yf": "Sector (Yahoo Finance)",
            },
        )
        fig.update_layout(
            template=config.DASHBOARD_THEME,
            xaxis_title="Sector",
            yaxis_title="Average Underpricing",
            yaxis_tickformat=".0%",
            height=450,
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sector data not available for the selected filter.")
else:
    st.info("Sector analysis requires ipo_pe_analysis.csv. Run: `python scripts/analyze_pe_ratios.py`")

# ─── P/E (F/K) Ratio Analysis ──────────────────────────
st.markdown("---")
st.markdown("### Fundamental Valuation: P/E (F/K) Ratio Analysis")
st.info(
    "**Are IPOs underpriced because of fundamentals?**\n\n"
    "The F/K (Fiyat/Kazanç, P/E) ratio measures how 'expensive' a stock is relative to its earnings. "
    "Low F/K = 'cheap' stock, High F/K = growth expectations priced in.\n\n"
    "If underpricing is explained by fundamentals (low F/K → high underpricing), "
    "this supports the 'mispricing' hypothesis. If there's NO correlation, "
    "underpricing may be driven by **supply-demand dynamics** (behavioral) rather than fundamental value.\n\n"
    "**Note:** F/K values shown here are **current trailing P/E** from Yahoo Finance, not F/K at the time of IPO. "
    "This is a limitation — ideally we would use IPO-time valuations."
)

pe_file = config.PROCESSED_DIR / "ipo_pe_analysis.csv"
if pe_file.exists():
    df_pe = pd.read_csv(pe_file)
    df_pe["trailing_pe"] = pd.to_numeric(df_pe["trailing_pe"], errors="coerce")
    df_pe["forward_pe"] = pd.to_numeric(df_pe["forward_pe"], errors="coerce")
    df_pe["price_to_book"] = pd.to_numeric(df_pe["price_to_book"], errors="coerce")
    df_pe["market_cap"] = pd.to_numeric(df_pe["market_cap"], errors="coerce")

    # Filter to valid positive finite P/E
    pe_valid = df_pe[df_pe["trailing_pe"].notna() & (df_pe["trailing_pe"] > 0) & (df_pe["trailing_pe"] < 1e6)]

    pe1, pe2, pe3, pe4 = st.columns(4)
    with pe1:
        st.metric("IPOs with P/E Data", f"{len(pe_valid)} / {len(df_pe)}")
    with pe2:
        st.metric("Median P/E", f"{pe_valid['trailing_pe'].median():.1f}")
    with pe3:
        pb_valid = df_pe["price_to_book"].dropna()
        pb_pos = pb_valid[pb_valid > 0]
        st.metric("Median P/B", f"{pb_pos.median():.2f}" if len(pb_pos) > 0 else "N/A")
    with pe4:
        mc_valid = df_pe["market_cap"].dropna()
        if len(mc_valid) > 0:
            st.metric("Median Market Cap", f"₺{mc_valid.median()/1e9:.1f}B")

    pe_chart1, pe_chart2 = st.columns(2)

    with pe_chart1:
        st.markdown("#### P/E Ratio Distribution")
        st.caption("Distribution of trailing P/E ratios across IPO companies (capped at 100 for visibility).")
        pe_capped = pe_valid[pe_valid["trailing_pe"] <= 100]
        if len(pe_capped) > 0:
            fig = px.histogram(
                pe_capped, x="trailing_pe", nbins=25,
                color_discrete_sequence=[config.COLOR_PALETTE["secondary"]],
                labels={"trailing_pe": "Trailing P/E Ratio"},
            )
            fig.add_vline(
                x=pe_capped["trailing_pe"].median(),
                line_dash="dot", line_color="red",
                annotation_text=f"Median: {pe_capped['trailing_pe'].median():.1f}",
            )
            fig.update_layout(template=config.DASHBOARD_THEME, height=380)
            st.plotly_chart(fig, use_container_width=True)

    with pe_chart2:
        st.markdown("#### P/E vs Underpricing")
        st.caption(
            "Does fundamental value (P/E) explain underpricing? If the correlation is weak, "
            "underpricing is behavioral, not fundamental."
        )
        # Merge PE with underpricing
        pe_upr = pe_valid.copy()
        if "underpricing" in pe_upr.columns:
            upr_pe_col = "underpricing"
        elif "tavan_series_return" in pe_upr.columns:
            upr_pe_col = "tavan_series_return"
        else:
            upr_pe_col = None

        if upr_pe_col and len(pe_upr[pe_upr[upr_pe_col].notna()]) > 5:
            pe_scatter = pe_upr[pe_upr[upr_pe_col].notna() & (pe_upr["trailing_pe"] <= 100)].copy()
            if len(pe_scatter) > 5:
                from scipy import stats as sp_stats
                rho, rho_p = sp_stats.spearmanr(pe_scatter["trailing_pe"], pe_scatter[upr_pe_col])
                fig = px.scatter(
                    pe_scatter, x="trailing_pe", y=upr_pe_col,
                    hover_data=["ticker"],
                    color_discrete_sequence=[config.COLOR_PALETTE["primary"]],
                    trendline="ols",
                    labels={
                        "trailing_pe": "Trailing P/E Ratio",
                        upr_pe_col: "Underpricing (Tavan Series Return)",
                    },
                )
                fig.update_layout(template=config.DASHBOARD_THEME, height=380)
                st.plotly_chart(fig, use_container_width=True)
                sig_text = "**significant**" if rho_p < 0.05 else "**not significant**"
                st.caption(f"Spearman ρ = {rho:.3f} (p = {rho_p:.3f}) — {sig_text}")

    # P/E by tavan group
    if "tavan_group" in pe_valid.columns:
        st.markdown("#### Median P/E by Tavan Length")
        st.caption(
            "Do IPOs with longer tavan series have different P/E profiles? "
            "Higher P/E in long-tavan groups suggests speculative (growth) stocks drive longer tavan series."
        )
        tavan_pe = pe_valid.groupby("tavan_group", observed=True).agg(
            median_pe=("trailing_pe", "median"),
            count=("ticker", "count"),
        ).reset_index()
        if len(tavan_pe) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tavan_pe["tavan_group"].astype(str),
                y=tavan_pe["median_pe"],
                text=[f"{v:.1f}\n(n={n})" for v, n in zip(tavan_pe["median_pe"], tavan_pe["count"])],
                textposition="outside",
                marker_color=config.COLOR_PALETTE["ipo"],
            ))
            fig.update_layout(
                template=config.DASHBOARD_THEME,
                xaxis_title="Tavan Group",
                yaxis_title="Median P/E Ratio",
                height=380,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Academic Interpretation"):
        st.markdown("""
        **Key Finding:** The Spearman correlation between P/E ratio and underpricing is
        weak and statistically not significant. This means:

        1. **Underpricing is NOT driven by fundamental value.** Stocks with low P/E (fundamentally "cheap")
           do not systematically exhibit more underpricing than expensive ones.

        2. **Behavioral explanation dominates.** Underpricing in BIST IPOs is better explained by:
           - **Supply-demand dynamics** (Rock 1986 — Winner's Curse)
           - **Retail investor herding** (our CSAD analysis)
           - **Tavan mechanism** amplifying initial demand-supply imbalance

        3. **Tavan group pattern:** Higher P/E stocks tend to have longer tavan series,
           suggesting that speculative/growth stocks attract more investor enthusiasm
           and sustain longer ceiling-hitting periods.

        This supports our thesis: IPO "fever" is a **behavioral phenomenon**, not a
        rational response to fundamental mispricing.
        """)
else:
    st.warning("P/E analysis not available. Run: `python scripts/analyze_pe_ratios.py`")

# ─── Detailed IPO Table ─────────────────────────────────
st.markdown("---")
st.markdown("### IPO Database")
st.caption("Full list of all IPOs in the dataset. Click column headers to sort.")

display_cols = ["ticker", "company_name", "ipo_date", "offer_price",
                "tavan_days", "underpricing", "tavan_series_return", "ipo_year"]
display_cols = [c for c in display_cols if c in filtered.columns]

if display_cols:
    display_df = filtered[display_cols].copy()
    if first_day_col in display_df.columns:
        display_df[first_day_col] = display_df[first_day_col].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    st.dataframe(display_df, height=400)

# ─── Statistical Tests ──────────────────────────────────
st.markdown("---")
with st.expander("Statistical Tests (click to expand)"):
    st.info(
        "**What are these tests?** Statistical tests help us determine whether our findings "
        "are 'real' or could have happened by random chance.\n\n"
        "- **t-test:** Tests if the average underpricing (tavan series return) is significantly different from 0. "
        "A very small p-value (< 0.05) means the underpricing is statistically significant.\n"
        "- **Wilcoxon test:** Same idea, but doesn't assume returns follow a normal distribution.\n"
        "- **Correlation:** Tests whether oversubscription predicts underpricing."
    )

    from scipy import stats

    if first_day_col:
        returns = filtered[first_day_col].dropna()
        t_stat, p_val = stats.ttest_1samp(returns, 0)
        st.markdown(f"""
        **H0: Mean underpricing (tavan series return) = 0**
        - t-statistic: {t_stat:.3f}
        - p-value: {p_val:.6f}
        - Result: {'**Reject H0** — Significant underpricing exists' if p_val < 0.05 else 'Cannot reject H0'}
        - N = {len(returns)}, Mean = {returns.mean():.4f}, Std = {returns.std():.4f}
        """)

        w_stat, w_pval = stats.wilcoxon(returns)
        st.markdown(f"""
        **Wilcoxon Signed-Rank Test (non-parametric):**
        - W-statistic: {w_stat:.3f}
        - p-value: {w_pval:.6f}
        - Result: {'**Reject H0**' if w_pval < 0.05 else 'Cannot reject H0'}
        """)

        if "oversubscription_ratio" in filtered.columns:
            os_data = filtered[["oversubscription_ratio", first_day_col]].dropna()
            if len(os_data) >= 2:
                corr, corr_p = stats.pearsonr(os_data["oversubscription_ratio"], os_data[first_day_col])
                spearman, sp_p = stats.spearmanr(os_data["oversubscription_ratio"], os_data[first_day_col])
                st.markdown(f"""
                **Oversubscription vs Underpricing Correlation:**
                - Pearson r = {corr:.4f} (p = {corr_p:.4f}) — {'Significant' if corr_p < 0.05 else 'Not significant'}
                - Spearman rho = {spearman:.4f} (p = {sp_p:.4f}) — {'Significant' if sp_p < 0.05 else 'Not significant'}
                """)
                st.caption(
                    "Pearson r measures linear correlation. Spearman rho measures monotonic correlation "
                    "(robust to outliers). Values closer to +1 mean stronger positive relationship."
                )
            else:
                st.info("Not enough oversubscription data for correlation analysis (need at least 2 data points).")
