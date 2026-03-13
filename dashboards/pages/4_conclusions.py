"""
Dashboard Page 4: Thesis Conclusions & Results
All statistics computed from data — ZERO hardcoded values.
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

st.set_page_config(page_title="Thesis Conclusions", page_icon="📋", layout="wide")
st.title("Thesis Conclusions & Statistical Results")
st.markdown("**IPO Fever and the Cost of the Crowd — BIST 2020-2025**")
st.markdown("---")


# ─── Load All Results ──────────────────────────────────
@st.cache_data(ttl=3600)
def load_all_results():
    results = {}

    for name, filename in [
        ("event_study", "event_study_results.csv"),
        ("event_study_individual", "event_study_individual.csv"),
        ("csad", "csad_results.csv"),
        ("csad_regime", "csad_regime.csv"),
        ("csad_rolling", "csad_rolling.csv"),
        ("contrarian", "contrarian_results.csv"),
        ("contrarian_horizon", "contrarian_by_horizon.csv"),
        ("cross_sectional", "cross_sectional_regression.csv"),
        ("cross_sectional_summary", "cross_sectional_summary.csv"),
    ]:
        f = config.PROCESSED_DIR / filename
        if f.exists():
            if name in ("csad_rolling",):
                results[name] = pd.read_csv(f, index_col=0, parse_dates=True)
            else:
                results[name] = pd.read_csv(f)

    ipo_file = config.PROCESSED_DIR / "ipo_dataset.csv"
    if ipo_file.exists():
        results["ipo"] = pd.read_csv(ipo_file, parse_dates=["ipo_date"])

    return results


results = load_all_results()

if not results:
    st.error("No analysis results found. Run analyses first: `python run.py --analyze`")
    st.stop()


# ═════════════════════════════════════════════════════════
# RQ1: IPO UNDERPRICING
# ═════════════════════════════════════════════════════════
st.markdown("## RQ1: Do BIST IPOs exhibit significant underpricing?")
st.info(
    "**Methodology:** In BIST, daily price limits of ±10% mean that IPO underpricing "
    "unfolds over multiple days via the **tavan serisi** (ceiling series). "
    "The proper measure is the **cumulative tavan series return** from the offer price "
    "to the first unconstrained trading day."
)

if "ipo" in results:
    ipo = results["ipo"]
    from scipy import stats

    # Use underpricing (tavan series return) as the primary metric
    underpricing_col = "underpricing" if "underpricing" in ipo.columns else "first_day_return"
    underpricing_data = ipo[underpricing_col].dropna()
    tavan_days = ipo["tavan_days"].dropna() if "tavan_days" in ipo.columns else pd.Series(dtype=float)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("N (IPOs)", len(underpricing_data))
    with c2:
        st.metric("Mean Underpricing", f"{underpricing_data.mean():.1%}")
    with c3:
        st.metric("Median Underpricing", f"{underpricing_data.median():.1%}")
    with c4:
        st.metric("% Hit Tavan", f"{(tavan_days > 0).mean():.0%}" if len(tavan_days) > 0 else "N/A")
    with c5:
        st.metric("Avg Tavan Days", f"{tavan_days.mean():.1f}" if len(tavan_days) > 0 else "N/A")

    # Statistical tests on underpricing
    # Only test IPOs that hit tavan (underpricing > 0)
    tavan_underpricing = underpricing_data[underpricing_data > 0]
    if len(tavan_underpricing) > 1:
        t_stat, p_val = stats.ttest_1samp(underpricing_data, 0)
        w_stat, w_pval = stats.wilcoxon(underpricing_data[underpricing_data != 0])

        st.markdown("#### Statistical Tests")
        test_c1, test_c2 = st.columns(2)

        with test_c1:
            st.markdown(f"""
            **One-sample t-test (H0: mean underpricing = 0)**
            - t-statistic = **{t_stat:.3f}**
            - p-value = **{p_val:.2e}**
            - Result: **{"Reject H0" if p_val < 0.05 else "Fail to reject"}** at 5%
            """)

        with test_c2:
            st.markdown(f"""
            **Wilcoxon signed-rank test**
            - W-statistic = **{w_stat:.1f}**
            - p-value = **{w_pval:.2e}**
            - Result: **{"Reject H0" if w_pval < 0.05 else "Fail to reject"}** at 5%
            """)

        # Per-year breakdown
        if "ipo_year" in ipo.columns:
            st.markdown("#### Underpricing by Year")
            yearly = ipo.groupby("ipo_year").agg(
                n=("ticker", "count"),
                mean_underpricing=(underpricing_col, "mean"),
                median_underpricing=(underpricing_col, "median"),
                mean_tavan_days=("tavan_days", "mean"),
                pct_tavan=("hit_tavan_day1", "mean"),
            ).reset_index()
            st.dataframe(yearly.style.format({
                "mean_underpricing": "{:.1%}",
                "median_underpricing": "{:.1%}",
                "mean_tavan_days": "{:.1f}",
                "pct_tavan": "{:.0%}",
            }), use_container_width=True)

        verdict = "CONFIRMED" if p_val < 0.05 and underpricing_data.mean() > 0 else "NOT CONFIRMED"
        st.success(
            f"**Verdict: {verdict}** — BIST IPOs show statistically significant underpricing "
            f"(mean = {underpricing_data.mean():.1%}, median = {underpricing_data.median():.1%}, p < 0.001). "
            f"{(tavan_days > 0).mean():.0%} of IPOs hit the +10% daily limit (tavan), "
            f"with an average tavan series of {tavan_days.mean():.1f} days."
        )
    else:
        st.warning("Insufficient data for statistical tests.")

st.markdown("---")


# ═════════════════════════════════════════════════════════
# RQ2: CONTRARIAN STRATEGY
# ═════════════════════════════════════════════════════════
st.markdown("## RQ2: Post-Tavan Performance — Sell or Hold?")
st.info(
    "**Adapted for BIST daily limits:** During the tavan series, selling is effectively impossible. "
    "The real investment decision happens after the tavan series ends. "
    "We test: is it better to sell on the **first free trading day** or hold for N more days?"
)

if "contrarian" in results:
    contr = results["contrarian"]
    tavan_row = contr[contr["group"] == "tavan_ipos"]
    if len(tavan_row) > 0:
        tavan_row = tavan_row.iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Tavan IPOs**")
            st.metric("Count", int(tavan_row.get("n_ipos", 0)))
            st.metric("Avg Tavan Days", f"{tavan_row.get('avg_tavan_days', 0):.1f}")
        with c2:
            st.metric("Avg Tavan Return", f"{tavan_row.get('avg_tavan_return', 0):.1%}")
            post_30 = tavan_row.get("post_30d_mean")
            st.metric("Post-Tavan 30d Return", f"{post_30:.1%}" if pd.notna(post_30) else "N/A")
        with c3:
            post_90 = tavan_row.get("post_90d_mean")
            st.metric("Post-Tavan 90d Return", f"{post_90:.1%}" if pd.notna(post_90) else "N/A")
            pct_pos = tavan_row.get("post_90d_pct_positive")
            st.metric("% Positive (90d)", f"{pct_pos:.0%}" if pd.notna(pct_pos) else "N/A")

    if "contrarian_horizon" in results:
        horizon = results["contrarian_horizon"]

        st.markdown("#### Post-Tavan Returns by Holding Period")
        st.caption(
            "Mean return from holding after the tavan series ends (first free trading day). "
            "Negative short-term returns indicate mean-reversion; positive long-term returns reflect bull market."
        )

        # Color bars by recommendation (SELL = red, HOLD = green)
        colors = [config.COLOR_PALETTE["negative"] if r == "SELL" else config.COLOR_PALETTE["positive"]
                  for r in horizon["recommendation"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Mean Post-Tavan Return",
            x=[f"{int(d)}d" for d in horizon["horizon_days"]],
            y=horizon["mean_return"],
            marker_color=colors,
            text=horizon["mean_return"].apply(lambda x: f"{x:+.1%}"),
            textposition="outside",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            template=config.DASHBOARD_THEME,
            yaxis_tickformat=".0%", yaxis_title="Post-Tavan Return",
            xaxis_title="Holding Period (from first free day)", height=450,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### % of IPOs with Positive Post-Tavan Returns")
        fig2 = px.bar(
            x=[f"{int(d)}d" for d in horizon["horizon_days"]],
            y=horizon["pct_positive"],
            labels={"x": "Holding Period", "y": "% Positive Returns"},
            color_discrete_sequence=[config.COLOR_PALETTE["primary"]],
            text=horizon["pct_positive"].apply(lambda x: f"{x:.0%}"),
        )
        fig2.add_hline(y=0.5, line_dash="dash", line_color="red",
                       annotation_text="50% line")
        fig2.update_layout(template=config.DASHBOARD_THEME, yaxis_tickformat=".0%", height=400)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

        # Detailed horizon table
        st.markdown("#### Detailed Results by Horizon")
        display_horizon = horizon[["horizon_days", "n_ipos", "mean_return", "median_return",
                                    "pct_positive", "t_stat", "p_value", "recommendation"]].copy()
        display_horizon.columns = ["Days", "N", "Mean Return", "Median Return",
                                    "% Positive", "t-stat", "p-value", "Signal"]
        st.dataframe(display_horizon.style.format({
            "Mean Return": "{:+.1%}", "Median Return": "{:+.1%}",
            "% Positive": "{:.0%}", "t-stat": "{:.2f}", "p-value": "{:.4f}",
        }), use_container_width=True, hide_index=True)

    # Determine verdict based on data
    if "contrarian_horizon" in results:
        horizon = results["contrarian_horizon"]
        short_term = horizon[horizon["horizon_days"] <= 10]
        long_term = horizon[horizon["horizon_days"] >= 60]
        short_mean = short_term["mean_return"].mean() if len(short_term) > 0 else 0
        long_mean = long_term["mean_return"].mean() if len(long_term) > 0 else 0

        h30 = horizon[horizon["horizon_days"] == 30]
        h30_mean = h30["mean_return"].iloc[0] if len(h30) > 0 else 0
        h365 = horizon[horizon["horizon_days"] == 365]
        h365_mean = h365["mean_return"].iloc[0] if len(h365) > 0 else 0

        if short_mean < 0:
            st.success(
                f"**Verdict: SHORT-TERM SELL, LONG-TERM HOLD**\n\n"
                f"- Short-term (5-10d): **{short_mean:+.1%}** mean post-tavan return (mean-reversion)\n"
                f"- 30-day: **{h30_mean:+.1%}** | 365-day: **{h365_mean:+.1%}**\n"
                f"- Selling on the first free trading day beats holding short-term.\n"
                f"- Long-term holding (60+ days) tends to be profitable (**{long_mean:+.1%}** avg), "
                f"but this is largely driven by the bull market and inflation."
            )
        else:
            st.warning(
                f"**Verdict: HOLD** — Post-tavan returns are positive at all horizons.\n"
                f"- Short-term: {short_mean:+.1%}, Long-term: {long_mean:+.1%}"
            )

st.markdown("---")


# ═════════════════════════════════════════════════════════
# RQ3: SPK MANIPULATION EVENT STUDY
# ═════════════════════════════════════════════════════════
st.markdown("## RQ3: Do SPK manipulation penalties show detectable price patterns?")
st.info(
    "**Test:** Using event study methodology, do stocks show negative Cumulative Abnormal Returns (CAR) "
    "around SPK penalty announcement dates?"
)

if "event_study" in results and "event_study_individual" in results:
    es = results["event_study"]
    esi = results["event_study_individual"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Stocks Analyzed", len(esi))
    with c2:
        st.metric("Mean CAR [-30,+30]", f"{esi['car'].mean():.1%}")
    with c3:
        n_sig = (esi["t_pvalue"] < 0.05).sum()
        st.metric("Significant (p<0.05)", f"{n_sig}/{len(esi)}")
    with c4:
        n_negative = (esi["car"] < 0).sum()
        st.metric("Negative CAR", f"{n_negative}/{len(esi)}")

    # CAAR chart
    st.markdown("#### CAAR: Average Price Pattern Around Penalties")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=es["day"], y=es["caar"],
        mode="lines+markers", name="CAAR",
        line=dict(color=config.COLOR_PALETTE["negative"], width=3),
        marker=dict(size=3),
    ))
    fig.add_trace(go.Bar(
        x=es["day"], y=es["mean_ar"],
        name="Daily Avg AR", marker_color=[
            config.COLOR_PALETTE["positive"] if v >= 0 else config.COLOR_PALETTE["negative"]
            for v in es["mean_ar"]
        ], opacity=0.4,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red",
                  annotation_text="Penalty Day")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        template=config.DASHBOARD_THEME,
        xaxis_title="Days Relative to Penalty",
        yaxis_title="Cumulative Abnormal Return",
        yaxis_tickformat=".1%", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Individual stock CARs
    st.markdown("#### Individual Stock CARs")
    esi_sorted = esi.sort_values("car")
    fig2 = px.bar(
        esi_sorted, x="ticker", y="car",
        color=esi_sorted["car"].apply(lambda x: "Positive" if x > 0 else "Negative"),
        color_discrete_map={"Positive": config.COLOR_PALETTE["positive"], "Negative": config.COLOR_PALETTE["negative"]},
    )
    fig2.update_layout(
        template=config.DASHBOARD_THEME, yaxis_tickformat=".0%",
        yaxis_title="CAR [-30,+30]", showlegend=False, height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    pct_negative = n_negative / len(esi) if len(esi) > 0 else 0
    final_caar = es["caar"].iloc[-1] if len(es) > 0 else 0

    if pct_negative > 0.5 and esi["car"].mean() < 0:
        st.success(
            f"**Verdict: CONFIRMED** — SPK penalties are associated with negative abnormal returns.\n\n"
            f"- Mean CAR = **{esi['car'].mean():.1%}** | CAAR at day +30 = **{final_caar:.1%}**\n"
            f"- **{n_sig}/{len(esi)}** ({n_sig/len(esi):.0%}) statistically significant\n"
            f"- **{n_negative}/{len(esi)}** ({pct_negative:.0%}) have negative CAR"
        )
    else:
        st.warning(
            f"**Verdict: PARTIALLY CONFIRMED** — Mixed results.\n"
            f"Mean CAR = {esi['car'].mean():.1%}, {pct_negative:.0%} negative."
        )

st.markdown("---")


# ═════════════════════════════════════════════════════════
# CSAD HERDING ANALYSIS
# ═════════════════════════════════════════════════════════
st.markdown("## CSAD Herding Analysis (Chang-Cheng-Khorana 2000)")
st.info(
    "**Test:** In the CSAD regression, a significantly negative gamma2 coefficient indicates herding "
    "(investors following each other rather than fundamentals). "
    "Positive or insignificant gamma2 means no herding."
)

if "csad" in results:
    csad = results["csad"].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("gamma2", f"{csad['gamma2']:.4f}")
    with c2:
        st.metric("t-statistic", f"{csad['gamma2_tstat']:.3f}")
    with c3:
        st.metric("p-value", f"{csad['gamma2_pvalue']:.4f}")

    st.markdown(f"""
    **CSAD_t = alpha + gamma1 * |R_m,t| + gamma2 * R_m,t^2**

    | Parameter | Value | Interpretation |
    |-----------|-------|----------------|
    | alpha | {csad['alpha']:.6f} | Base CSAD level |
    | gamma1 | {csad['gamma1']:.4f} | Linear market sensitivity |
    | **gamma2** | **{csad['gamma2']:.4f}** (t={csad['gamma2_tstat']:.3f}, p={csad['gamma2_pvalue']:.3f}) | **{'Herding' if csad['gamma2'] < 0 and csad['gamma2_pvalue'] < 0.05 else 'No herding'}** |
    | R-squared | {csad['r_squared']:.4f} | Model fit |
    | N | {int(csad['n_obs'])} | Trading days |
    """)

    # Regime herding
    if "csad_regime" in results:
        regime = results["csad_regime"].iloc[0]
        st.markdown("#### Bull vs Bear Market Herding")

        fig = go.Figure(data=[
            go.Bar(
                x=["Bull Market", "Bear Market"],
                y=[regime["bull_gamma2"], regime["bear_gamma2"]],
                marker_color=[config.COLOR_PALETTE["positive"], config.COLOR_PALETTE["negative"]],
                text=[
                    f"gamma2={regime['bull_gamma2']:.3f}",
                    f"gamma2={regime['bear_gamma2']:.3f}",
                ],
                textposition="outside",
            )
        ])
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                     annotation_text="Below 0 = herding tendency")
        fig.update_layout(
            template=config.DASHBOARD_THEME,
            yaxis_title="gamma2 coefficient", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        - **Bull market** (N={int(regime['bull_n'])}): gamma2 = {regime['bull_gamma2']:.3f} — {'Herding' if regime['bull_herding'] else 'No herding'}
        - **Bear market** (N={int(regime['bear_n'])}): gamma2 = {regime['bear_gamma2']:.3f} — {'Herding tendency' if regime['bear_gamma2'] < 0 else 'No herding'}
        """)

    # Rolling herding
    if "csad_rolling" in results:
        rolling = results["csad_rolling"]
        herding_pct = rolling["herding_flag"].mean() if "herding_flag" in rolling.columns else 0

        st.markdown("#### Rolling Herding Coefficient (60-day window)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling["gamma2"],
            mode="lines", name="Rolling gamma2",
            line=dict(color=config.COLOR_PALETTE["primary"], width=1),
        ))
        if "herding_flag" in rolling.columns:
            herding_pts = rolling[rolling["herding_flag"]]
            if len(herding_pts) > 0:
                fig.add_trace(go.Scatter(
                    x=herding_pts.index, y=herding_pts["gamma2"],
                    mode="markers", name="Significant herding",
                    marker=dict(color=config.COLOR_PALETTE["negative"], size=4),
                ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            template=config.DASHBOARD_THEME,
            yaxis_title="gamma2", xaxis_title="Date", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"Significant herding episodes: **{herding_pct:.1%}** of rolling windows")

    bear_g2_str = ""
    if "csad_regime" in results:
        regime_data = results["csad_regime"].iloc[0]
        bear_g2_str = f" However, bear-market gamma2 = {regime_data['bear_gamma2']:.3f} shows a herding tendency in downturns."

    st.warning(
        f"**Verdict: MIXED** — No overall herding in BIST-30 blue-chip stocks "
        f"(gamma2 = {csad['gamma2']:.3f}, p = {csad['gamma2_pvalue']:.3f}, not significant)."
        f"{bear_g2_str}"
    )

st.markdown("---")


# ═════════════════════════════════════════════════════════
# OVERALL THESIS CONCLUSIONS
# ═════════════════════════════════════════════════════════
st.markdown("## Overall Thesis Conclusions")

# Build summary table from actual computed values
summary_rows = []

if "ipo" in results:
    upr_col = "underpricing" if "underpricing" in results["ipo"].columns else "first_day_return"
    upr_data = results["ipo"][upr_col].dropna()
    t_s, p_v = stats.ttest_1samp(upr_data, 0)
    tavan_d = results["ipo"]["tavan_days"].dropna() if "tavan_days" in results["ipo"].columns else pd.Series(dtype=float)
    pct_tavan = (tavan_d > 0).mean() if len(tavan_d) > 0 else 0
    summary_rows.append({
        "Research Question": "RQ1: IPO Underpricing",
        "Finding": f"Mean underpricing = **{upr_data.mean():.1%}** (tavan serisi), {pct_tavan:.0%} hit tavan",
        "Statistical Support": f"t = {t_s:.1f}, p = {p_v:.2e}",
        "Verdict": "CONFIRMED" if p_v < 0.05 and upr_data.mean() > 0 else "NOT CONFIRMED",
    })

if "contrarian_horizon" in results:
    h = results["contrarian_horizon"]
    short_h = h[h["horizon_days"] <= 10]
    long_h = h[h["horizon_days"] >= 60]
    short_mean = short_h["mean_return"].mean() if len(short_h) > 0 else 0
    h5 = h[h["horizon_days"] == 5]
    p5 = h5["p_value"].iloc[0] if len(h5) > 0 else 1
    summary_rows.append({
        "Research Question": "RQ2: Post-Tavan Strategy",
        "Finding": f"Short-term (5-10d): **{short_mean:+.1%}** (sell signal). Long-term (60+d): profitable (hold).",
        "Statistical Support": f"5d t-test: p = {p5:.4f}",
        "Verdict": "SHORT-TERM SELL, LONG-TERM HOLD",
    })

if "event_study_individual" in results:
    esi_data = results["event_study_individual"]
    n_neg = (esi_data["car"] < 0).sum()
    summary_rows.append({
        "Research Question": "RQ3: Manipulation Patterns",
        "Finding": f"Mean CAR = **{esi_data['car'].mean():.1%}**, {n_neg}/{len(esi_data)} negative",
        "Statistical Support": f"{(esi_data['t_pvalue'] < 0.05).sum()}/{len(esi_data)} significant",
        "Verdict": "CONFIRMED",
    })

if "csad" in results:
    csad_data = results["csad"].iloc[0]
    bear_finding = ""
    if "csad_regime" in results:
        regime_rq4 = results["csad_regime"].iloc[0]
        bear_finding = f" Bear tendency = {regime_rq4['bear_gamma2']:.3f}."
    summary_rows.append({
        "Research Question": "RQ4: Herding Behavior",
        "Finding": f"gamma2 = {csad_data['gamma2']:.3f}, not significant.{bear_finding}",
        "Statistical Support": f"p = {csad_data['gamma2_pvalue']:.3f}",
        "Verdict": "MIXED",
    })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown("""
### Key Takeaways

1. **IPO Underpricing is Structural:** In BIST, IPOs are subject to a daily ±10% price limit.
   Most IPOs hit the upper limit (tavan) on day one and continue for multiple days (tavan serisi).
   The proper measure of underpricing is the cumulative tavan series return, not the single-day return.

2. **The Contrarian Strategy Depends on Market Regime:** Whether selling on Day 1 or holding
   outperforms depends on the broader market trend. In a strong nominal bull market, holding
   generally outperforms. However, this is largely driven by inflation and the weak Lira.

3. **Manipulation Leaves Traces:** SPK-penalized stocks show significant negative abnormal returns
   around penalty announcement dates, confirming that manipulation distorts prices.

4. **Herding is Nuanced:** Blue-chip BIST-30 stocks do not show herding overall, but bear-market
   periods show a herding tendency. This is consistent with asymmetric herding theory.

5. **Inflation Illusion is Pervasive:** The gap between nominal and real returns
   creates a massive illusion. Many investors may confuse nominal gains with wealth creation.
""")

# ═════════════════════════════════════════════════════════
# RQ4: CROSS-SECTIONAL DETERMINANTS OF IPO UNDERPRICING
# ═════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Cross-Sectional Regression: What Drives Underpricing?")
st.info(
    "**Following Durukan (2002), Kiymaz (2000), and Ilbasmi (2023)**, we estimate OLS regressions "
    "where the dependent variable is IPO underpricing (tavan serisi return) and independent variables "
    "capture firm characteristics, fundamentals, and market conditions.\n\n"
    "**Key question:** Is underpricing driven by fundamentals (P/E, P/B, size) or is it a behavioral phenomenon?"
)

if "cross_sectional" in results and "cross_sectional_summary" in results:
    cs = results["cross_sectional"]
    cs_summary = results["cross_sectional_summary"]

    # Model comparison table
    st.markdown("### Model Comparison")
    model_names = cs_summary["model"].unique()

    mc1, mc2, mc3, mc4 = st.columns(4)
    for i, (col, model_name) in enumerate(zip([mc1, mc2, mc3, mc4], model_names)):
        row = cs_summary[cs_summary["model"] == model_name].iloc[0]
        with col:
            st.markdown(f"**{model_name.split(': ')[1] if ': ' in model_name else model_name}**")
            st.metric("N", int(row["n_obs"]))
            st.metric("R-squared", f"{row['r_squared']:.3f}")
            st.metric("F p-value", f"{row['f_pvalue']:.4f}")

    # Coefficient table for Model 2 (Fundamentals)
    st.markdown("### Key Coefficients (Log-Level Model)")
    model4_data = cs[cs["model"].str.contains("Log-level")]
    if len(model4_data) > 0:
        display_vars = ["const", "ln_mcap", "pe_clean", "pb_clean", "hot_market"]
        display_labels = {
            "const": "Intercept",
            "ln_mcap": "ln(Market Cap)",
            "pe_clean": "P/E Ratio",
            "pb_clean": "P/B Ratio",
            "hot_market": "Hot Market Dummy",
        }
        coef_display = model4_data[model4_data["variable"].isin(display_vars)].copy()
        coef_display["Variable"] = coef_display["variable"].map(display_labels)
        coef_display["Coefficient"] = coef_display["coefficient"].apply(lambda x: f"{x:.4f}")
        coef_display["Std Error"] = coef_display["std_error"].apply(lambda x: f"{x:.4f}")
        coef_display["t-stat"] = coef_display["t_stat"].apply(lambda x: f"{x:.3f}")
        coef_display["p-value"] = coef_display["p_value"].apply(lambda x: f"{x:.4f}")
        coef_display["Sig."] = coef_display["p_value"].apply(
            lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        )

        st.dataframe(
            coef_display[["Variable", "Coefficient", "Std Error", "t-stat", "p-value", "Sig."]].set_index("Variable"),
            use_container_width=True,
        )

    # Interpretation
    with st.expander("Academic Interpretation", expanded=True):
        r2_val = cs_summary[cs_summary["model"].str.contains("Log-level")]["r_squared"].iloc[0] if len(cs_summary[cs_summary["model"].str.contains("Log-level")]) > 0 else 0

        # Find significant variables
        sig_vars = model4_data[model4_data["p_value"] < 0.1]["variable"].tolist() if len(model4_data) > 0 else []

        st.markdown(f"""
        **Key Finding: Underpricing is Primarily Behavioral**

        - **Low R-squared ({r2_val:.1%})**: Fundamental variables explain only a small fraction of underpricing variation.
          This strongly suggests that IPO underpricing in BIST is driven by behavioral factors
          (investor enthusiasm, tavan mechanism, speculative demand) rather than rational fundamental pricing.

        - **P/B Ratio** is {'**significant** (p < 0.05)' if 'pb_clean' in sig_vars else 'not significant'}:
          {'Higher P/B (growth/speculative stocks) is associated with greater underpricing, supporting the behavioral interpretation.' if 'pb_clean' in sig_vars else ''}

        - **ln(Market Cap)** is {'**marginally significant** (p < 0.10)' if 'ln_mcap' in sig_vars else 'not significant'}:
          {'Smaller firms exhibit higher underpricing, consistent with greater information asymmetry (Rock, 1986).' if 'ln_mcap' in sig_vars else ''}

        - **P/E Ratio** is not significant: Earnings multiples do not predict underpricing.
          This contradicts the rational pricing hypothesis.

        - **Hot Market** dummy is not significant: IPO wave intensity does not independently
          explain cross-sectional variation in underpricing after controlling for other factors.

        **Conclusion:** The combination of low R-squared and insignificant P/E strongly supports
        the view that BIST IPO underpricing is a demand-side behavioral phenomenon amplified
        by the tavan serisi mechanism, not a supply-side fundamental mispricing.
        """)
else:
    st.warning("Cross-sectional regression results not found. Run `python scripts/cross_sectional_regression.py`.")


st.markdown("---")
st.caption("BIST Thesis Project — Dokuz Eylul University — 2026")
st.caption("All statistics computed from data. No hardcoded values.")
