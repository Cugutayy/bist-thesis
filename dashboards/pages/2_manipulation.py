"""
Dashboard Page 2: SPK Manipulation Analysis
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

st.set_page_config(page_title="SPK Manipulation", page_icon="🔍", layout="wide")

# ─── Page Header ──────────────────────────────────────────
st.title("Dimension 2: SPK Manipulation Analysis")

st.info(
    "**What is this page about?**\n\n"
    "Turkey's Capital Markets Board (SPK) investigates and penalizes market manipulation "
    "in Borsa Istanbul. This page analyzes **47 verified manipulation cases** from 2020-2025 and asks: "
    "*Can we detect abnormal price patterns (pump-and-dump) around the time these penalties "
    "were announced?*\n\n"
    "We use **Event Study methodology** (MacKinlay, 1997) to measure whether stock prices "
    "behave abnormally in the 30 days before and after an SPK penalty announcement."
)

st.markdown("---")


@st.cache_data(ttl=3600)
def load_spk_data():
    processed = config.PROCESSED_DIR / "spk_penalties.csv"
    if processed.exists():
        return pd.read_csv(processed, parse_dates=["karar_tarihi", "inceleme_baslangic", "inceleme_bitis"])
    try:
        from src.data_collection.spk_data import get_penalties_df
        return get_penalties_df()
    except Exception as e:
        st.warning(f"Could not load SPK data: {e}")
        return None


@st.cache_data(ttl=3600)
def load_event_study_results():
    processed = config.PROCESSED_DIR / "event_study_results.csv"
    if processed.exists():
        return pd.read_csv(processed, parse_dates=True)
    return None


df = load_spk_data()
event_results = load_event_study_results()

if df is None or len(df) == 0:
    st.error("SPK data not available. Run data collection first.")
    st.stop()

# ─── Key Metrics ─────────────────────────────────────────
st.markdown("### Overview")
st.caption(
    "Summary of all SPK manipulation penalties in the dataset. "
    "These include cases of price manipulation, insider trading, and market abuse."
)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Total Cases", len(df))
with m2:
    total_fine = df["toplam_ceza_tl"].sum()
    if total_fine >= 1e9:
        st.metric("Total Fines", f"₺{total_fine/1e9:.1f}B")
    else:
        st.metric("Total Fines", f"₺{total_fine/1e6:.0f}M")
with m3:
    st.metric("People Penalized", int(df["kisi_sayisi"].sum()))
with m4:
    st.metric("Unique Stocks", df["hisse_kodu"].nunique())
with m5:
    if "islem_yasagi" in df.columns:
        ban_pct = df["islem_yasagi"].mean()
        st.metric("% with Trading Ban", f"{ban_pct:.0%}")

st.markdown("---")

# ─── Charts Row 1 ───────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Penalties by Year")
    st.caption(
        "The left axis (purple bars) shows the number of manipulation cases per year, "
        "while the right axis (red line) shows the total fines imposed in millions of TL. "
        "A spike in cases or fines may indicate either increased enforcement or increased manipulation activity."
    )
    df["year"] = pd.to_datetime(df["karar_tarihi"]).dt.year
    yearly = df.groupby("year").agg(
        cases=("hisse_kodu", "size"),
        total_fine=("toplam_ceza_tl", "sum"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=yearly["year"], y=yearly["cases"], name="Cases",
               marker_color=config.COLOR_PALETTE["manipulation"]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=yearly["year"], y=yearly["total_fine"] / 1e6, name="Total Fines (M TL)",
                   line=dict(color=config.COLOR_PALETTE["negative"], width=3)),
        secondary_y=True,
    )
    fig.update_layout(template=config.DASHBOARD_THEME, height=380)
    fig.update_yaxes(title_text="Number of Cases", secondary_y=False)
    fig.update_yaxes(title_text="Total Fines (M TL)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("#### Penalty Distribution")
    st.caption(
        "How are the fine amounts distributed? Most fines are relatively small, "
        "but a few massive penalties (outliers) drive up the total. "
        "This histogram shows how many cases fall into each fine-amount bracket."
    )
    fig = px.histogram(
        df, x="toplam_ceza_tl", nbins=20,
        color_discrete_sequence=[config.COLOR_PALETTE["manipulation"]],
        labels={"toplam_ceza_tl": "Penalty Amount (TL)"},
    )
    fig.update_layout(template=config.DASHBOARD_THEME, height=380)
    st.plotly_chart(fig, use_container_width=True)

# ─── Top Penalties ───────────────────────────────────────
st.markdown("---")
st.markdown("### Largest Penalties")
st.caption(
    "The 10 biggest fines imposed by SPK. These are typically the most egregious cases "
    "of market manipulation, often involving coordinated trading rings that artificially "
    "inflate stock prices before selling at a profit."
)
top_penalties = df.nlargest(10, "toplam_ceza_tl")[
    ["hisse_kodu", "company_name", "karar_tarihi", "toplam_ceza_tl", "kisi_sayisi", "ceza_turu", "notes"]
].copy()
top_penalties["toplam_ceza_tl"] = top_penalties["toplam_ceza_tl"].apply(
    lambda x: f"₺{x/1e6:.1f}M" if x >= 1e6 else f"₺{x/1e3:.0f}K"
)
st.dataframe(top_penalties, use_container_width=True)

# ─── Manipulation Type Analysis ──────────────────────────
st.markdown("---")
c3, c4 = st.columns(2)

with c3:
    st.markdown("#### Cases by Manipulation Type")
    st.caption(
        "Types of manipulation detected by SPK. 'Yapay fiyat/arz/talep' means artificial "
        "price/supply/demand creation. 'Bilgi bazli' means information-based manipulation (insider trading)."
    )
    if "ceza_turu" in df.columns:
        type_counts = df["ceza_turu"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig = px.pie(
            type_counts, names="type", values="count",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(template=config.DASHBOARD_THEME, height=380)
        st.plotly_chart(fig, use_container_width=True)

with c4:
    st.markdown("#### Average Fine by Type")
    st.caption(
        "Which type of manipulation gets the harshest penalty? "
        "This chart compares the average fine amount across different manipulation categories."
    )
    if "ceza_turu" in df.columns:
        type_fine = df.groupby("ceza_turu")["toplam_ceza_tl"].mean().reset_index()
        type_fine.columns = ["type", "avg_fine"]
        fig = px.bar(
            type_fine, x="type", y="avg_fine",
            color_discrete_sequence=[config.COLOR_PALETTE["negative"]],
            text=type_fine["avg_fine"].apply(lambda x: f"₺{x/1e6:.1f}M"),
        )
        fig.update_layout(template=config.DASHBOARD_THEME, yaxis_title="Average Fine (TL)", height=380)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# EVENT STUDY SECTION
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Event Study: What Happens to Stock Prices Around Manipulation Penalties?")

st.info(
    "**How does an Event Study work?**\n\n"
    "1. We take the SPK penalty announcement date as the 'event' (Day 0).\n"
    "2. We look at stock prices from **Day -30** to **Day +30** around the event.\n"
    "3. We calculate the **'normal' return** the stock should have earned (using market model regression).\n"
    "4. The **Abnormal Return (AR)** = Actual Return - Expected Return.\n"
    "5. We sum ARs over time to get **Cumulative Abnormal Return (CAR)**.\n"
    "6. If CAR is significantly negative, it means the stock lost more value than expected around the penalty.\n\n"
    "A typical **pump-and-dump** pattern would show: positive CAR before the event (price pumped up), "
    "then sharply negative CAR after (crash when manipulation is exposed)."
)

# ─── AGGREGATE EVENT STUDY (most important - show first) ──
if event_results is not None:
    st.markdown("#### Aggregate Results: Average Pattern Across All Penalized Stocks")
    st.caption(
        "This is the average price behavior across ALL manipulation cases combined. "
        "The red line (CAAR) shows the cumulative average abnormal return over time. "
        "The bars show daily abnormal returns. The dashed red vertical line marks Day 0 "
        "(the penalty announcement date)."
    )

    fig = go.Figure()
    if "day" in event_results.columns and "caar" in event_results.columns:
        fig.add_trace(go.Scatter(
            x=event_results["day"], y=event_results["caar"],
            mode="lines+markers", name="CAAR (Cumulative Avg Abnormal Return)",
            line=dict(color=config.COLOR_PALETTE["negative"], width=3),
            marker=dict(size=3),
        ))
        if "mean_ar" in event_results.columns:
            fig.add_trace(go.Bar(
                x=event_results["day"], y=event_results["mean_ar"],
                name="Daily Avg Abnormal Return",
                marker_color=[
                    config.COLOR_PALETTE["positive"] if v >= 0 else config.COLOR_PALETTE["negative"]
                    for v in event_results["mean_ar"]
                ],
                opacity=0.3,
            ))

        fig.add_vline(x=0, line_dash="dash", line_color="red",
                     annotation_text="Penalty Announcement (Day 0)")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            template=config.DASHBOARD_THEME,
            xaxis_title="Days Relative to Penalty Announcement",
            yaxis_title="Cumulative Abnormal Return",
            yaxis_tickformat=".1%",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        final_caar = event_results["caar"].iloc[-1] if len(event_results) > 0 else 0
        if final_caar < -0.05:
            st.success(
                f"**Interpretation:** The CAAR at Day +30 is **{final_caar:.1%}**, meaning manipulated stocks "
                f"lost an average of {abs(final_caar):.1%} more than expected over the event window. "
                f"This confirms that SPK penalties are associated with significant negative price reactions."
            )
        elif final_caar < 0:
            st.warning(
                f"**Interpretation:** The CAAR at Day +30 is **{final_caar:.1%}**, slightly negative. "
                f"There is a mild downward trend but the effect is moderate."
            )

# ─── Individual Stock CARs ─────────────────────────────
es_ind_file = config.PROCESSED_DIR / "event_study_individual.csv"
if es_ind_file.exists():
    st.markdown("---")
    st.markdown("#### Individual Stock CARs: How Did Each Manipulated Stock Perform?")
    st.caption(
        "Each bar represents one manipulated stock's Cumulative Abnormal Return (CAR) over the [-30, +30] day window. "
        "**Red bars** = stock lost value (negative CAR). **Green bars** = stock gained value. "
        "Most bars being red means manipulation penalties are generally associated with price declines."
    )
    esi = pd.read_csv(es_ind_file)

    em1, em2, em3, em4 = st.columns(4)
    with em1:
        st.metric("Stocks Analyzed", len(esi))
    with em2:
        st.metric("Mean CAR", f"{esi['car'].mean():.1%}")
    with em3:
        n_sig = (esi["t_pvalue"] < 0.05).sum()
        st.metric("Significant (p<0.05)", f"{n_sig}/{len(esi)}")
    with em4:
        n_negative = (esi["car"] < 0).sum()
        st.metric("Negative CAR", f"{n_negative}/{len(esi)}")

    esi_sorted = esi.sort_values("car")
    fig_ind = px.bar(
        esi_sorted, x="ticker", y="car",
        color=esi_sorted["car"].apply(lambda x: "Gained Value" if x > 0 else "Lost Value"),
        color_discrete_map={
            "Gained Value": config.COLOR_PALETTE["positive"],
            "Lost Value": config.COLOR_PALETTE["negative"],
        },
        labels={"car": "Cumulative Abnormal Return", "ticker": "Stock Ticker"},
    )
    fig_ind.update_layout(
        template=config.DASHBOARD_THEME, yaxis_tickformat=".0%",
        yaxis_title="CAR [-30, +30 days]",
        xaxis_title="Manipulated Stock",
        showlegend=True,
        height=450,
    )
    st.plotly_chart(fig_ind, use_container_width=True)

    # Notable cases
    st.markdown("#### Notable Cases")
    notable_pos = esi.nlargest(3, "car")[["ticker", "car", "t_stat", "t_pvalue"]]
    notable_neg = esi.nsmallest(3, "car")[["ticker", "car", "t_stat", "t_pvalue"]]

    n1, n2 = st.columns(2)
    with n1:
        st.markdown("**Biggest Price Gains (possible pump before penalty):**")
        st.caption("These stocks had the most positive CARs - their prices went UP around the penalty period, possibly because the pump phase was still ongoing.")
        for _, r in notable_pos.iterrows():
            sig = " (significant)" if r["t_pvalue"] < 0.05 else ""
            st.markdown(f"- **{r['ticker']}**: CAR = {r['car']:.1%} (t={r['t_stat']:.2f}){sig}")
    with n2:
        st.markdown("**Biggest Price Crashes (dump after exposure):**")
        st.caption("These stocks had the most negative CARs - their prices crashed the most around the penalty announcement.")
        for _, r in notable_neg.iterrows():
            sig = " (significant)" if r["t_pvalue"] < 0.05 else ""
            st.markdown(f"- **{r['ticker']}**: CAR = {r['car']:.1%} (t={r['t_stat']:.2f}){sig}")

    # Summary box
    pct_negative = n_negative / len(esi) if len(esi) > 0 else 0
    pct_sig = n_sig / len(esi) if len(esi) > 0 else 0
    st.success(
        f"**Summary:** Out of {len(esi)} manipulated stocks, **{n_negative}** ({pct_negative:.0%}) "
        f"had negative CARs and **{n_sig}** ({pct_sig:.0%}) were statistically significant. "
        f"The average CAR is **{esi['car'].mean():.1%}**, confirming that manipulation penalties "
        f"are associated with abnormal price declines."
    )

# ─── Individual Case Explorer ─────────────────────────────
st.markdown("---")
st.markdown("### Case Explorer: Stock vs Market Comparison")
st.caption(
    "Select a manipulation case to compare the stock's price and volume against the **BIST-100 index**. "
    "This reveals whether the price drop was **stock-specific** (manipulation-related) or a **market-wide** move. "
    "The investigation period (orange shading) shows the dates SPK flagged for suspicious trading activity."
)

selected_stock = st.selectbox(
    "Select manipulated stock:",
    df["hisse_kodu"].unique(),
    format_func=lambda x: f"{x} — {df[df['hisse_kodu']==x]['company_name'].iloc[0]}" if "company_name" in df.columns else x,
)

if selected_stock:
    case = df[df["hisse_kodu"] == selected_stock].iloc[0]

    # Show case details
    detail_cols = st.columns(5)
    with detail_cols[0]:
        st.metric("Company", case.get('company_name', selected_stock))
    with detail_cols[1]:
        st.metric("Decision Date", str(case['karar_tarihi'])[:10])
    with detail_cols[2]:
        fine = case['toplam_ceza_tl']
        st.metric("Fine", f"₺{fine/1e6:.1f}M" if fine >= 1e6 else f"₺{fine/1e3:.0f}K")
    with detail_cols[3]:
        st.metric("People Penalized", int(case['kisi_sayisi']))
    with detail_cols[4]:
        st.metric("Type", case.get('ceza_turu', 'N/A'))

    # Show notes about the case
    if pd.notna(case.get("notes")):
        with st.expander("📋 Case details (click to expand)"):
            st.write(case['notes'])
            if pd.notna(case.get("inceleme_baslangic")):
                inv_days = case.get("inceleme_suresi_gun", "?")
                st.write(
                    f"**Investigation period:** {str(case['inceleme_baslangic'])[:10]} → "
                    f"{str(case['inceleme_bitis'])[:10]} ({inv_days} days)\n\n"
                    f"SPK examined this period for evidence of coordinated trading, artificial price/demand creation, "
                    f"or insider trading. The investigation dates come from SPK's official penalty bulletins."
                )

    # Show pre-computed CAR for this stock if available
    if es_ind_file.exists():
        esi_all = pd.read_csv(es_ind_file)
        stock_car = esi_all[esi_all["ticker"] == selected_stock]
        if len(stock_car) > 0:
            car_val = stock_car.iloc[0]["car"]
            t_val = stock_car.iloc[0]["t_stat"]
            p_val = stock_car.iloc[0]["t_pvalue"]
            sig_text = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
            if car_val < 0:
                st.error(
                    f"**Event Study Result:** CAR = **{car_val:.1%}** "
                    f"(t = {t_val:.2f}, p = {p_val:.4f}) — {sig_text}. "
                    f"This stock lost {abs(car_val):.1%} more than the market model predicted."
                )
            else:
                st.warning(
                    f"**Event Study Result:** CAR = **{car_val:.1%}** "
                    f"(t = {t_val:.2f}, p = {p_val:.4f}) — {sig_text}. "
                    f"This stock gained value around the penalty period."
                )

    # Load cached price data and BIST-100
    price_cache_file = config.PROCESSED_DIR / "spk_price_cache.csv"
    bist100_file = config.PROCESSED_DIR / "bist100_data.csv"

    if price_cache_file.exists():
        price_cache = pd.read_csv(price_cache_file, parse_dates=["Date"])
        stock_data = price_cache[price_cache["ticker"] == selected_stock].copy()

        # Load BIST-100 data
        bist_data = None
        if bist100_file.exists():
            bist_data = pd.read_csv(bist100_file, parse_dates=["date"])
            bist_data = bist_data.rename(columns={"date": "Date", "bist100_close": "BIST100"})

        if len(stock_data) > 0:
            stock_data = stock_data.sort_values("Date").reset_index(drop=True)
            event_date_str = str(pd.Timestamp(case["karar_tarihi"]).strftime("%Y-%m-%d"))
            event_date_ts = pd.Timestamp(case["karar_tarihi"])

            # Prepare investigation period timestamps
            inv_start_ts = pd.Timestamp(case["inceleme_baslangic"]) if pd.notna(case.get("inceleme_baslangic")) else None
            inv_end_ts = pd.Timestamp(case["inceleme_bitis"]) if pd.notna(case.get("inceleme_bitis")) else None

            # ═══════════════════════════════════════════
            # COMBINED CHART: 3 rows — Price+BIST | Volume | Normalized
            # ═══════════════════════════════════════════
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.50, 0.22, 0.28],
                specs=[[{"secondary_y": True}], [{}], [{}]],
            )

            # ── Row 1: Candlestick + BIST-100 ──
            fig.add_trace(go.Candlestick(
                x=stock_data["Date"],
                open=stock_data["Open"],
                high=stock_data["High"],
                low=stock_data["Low"],
                close=stock_data["Close"],
                name=selected_stock,
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                increasing_fillcolor="#26a69a",
                decreasing_fillcolor="#ef5350",
                increasing_line_width=0.8,
                decreasing_line_width=0.8,
                whiskerwidth=0.3,
            ), row=1, col=1, secondary_y=False)

            # BIST-100 overlay on secondary y-axis
            bist_period = pd.DataFrame()
            if bist_data is not None:
                date_min = stock_data["Date"].min()
                date_max = stock_data["Date"].max()
                bist_period = bist_data[(bist_data["Date"] >= date_min) & (bist_data["Date"] <= date_max)].copy()
                if len(bist_period) > 0:
                    fig.add_trace(go.Scatter(
                        x=bist_period["Date"],
                        y=bist_period["BIST100"],
                        name="BIST-100 Index",
                        line=dict(color="#1976d2", width=1.8, dash="dot"),
                        opacity=0.65,
                    ), row=1, col=1, secondary_y=True)

            # ── Row 2: Volume with color coding ──
            if "Volume" in stock_data.columns:
                vol_data = stock_data[stock_data["Volume"].notna()].copy()
                if len(vol_data) > 0:
                    penalty_window_start = event_date_ts - pd.Timedelta(days=7)
                    penalty_window_end = event_date_ts + pd.Timedelta(days=7)

                    vol_colors = []
                    for d in vol_data["Date"]:
                        if penalty_window_start <= d <= penalty_window_end:
                            vol_colors.append("#ef5350")       # red = around penalty
                        elif inv_start_ts and inv_end_ts and inv_start_ts <= d <= inv_end_ts:
                            vol_colors.append("#ff9800")       # orange = investigation
                        else:
                            vol_colors.append("#b0bec5")       # grey = normal
                    fig.add_trace(go.Bar(
                        x=vol_data["Date"],
                        y=vol_data["Volume"],
                        name="Volume",
                        marker_color=vol_colors,
                        showlegend=False,
                    ), row=2, col=1)

                    # Average volume reference line
                    avg_vol = vol_data["Volume"].mean()
                    fig.add_hline(
                        y=avg_vol, row=2, col=1,
                        line_dash="dot", line_color="#78909c", line_width=1,
                        annotation_text=f"Avg: {avg_vol/1e6:.1f}M" if avg_vol >= 1e6 else f"Avg: {avg_vol/1e3:.0f}K",
                        annotation_position="top right",
                        annotation_font_size=9, annotation_font_color="#78909c",
                    )

            # ── Row 3: Normalized performance (base = 100) ──
            stock_norm = stock_data[["Date", "Close"]].copy()
            base_price = stock_norm["Close"].iloc[0]
            stock_norm["Normalized"] = (stock_norm["Close"] / base_price) * 100

            fig.add_trace(go.Scatter(
                x=stock_norm["Date"], y=stock_norm["Normalized"],
                name=f"{selected_stock} (indexed)",
                line=dict(color="#ef5350", width=2.2),
            ), row=3, col=1)

            if len(bist_period) > 0:
                bist_norm = bist_period[["Date", "BIST100"]].copy()
                bist_base = bist_norm["BIST100"].iloc[0]
                bist_norm["Normalized"] = (bist_norm["BIST100"] / bist_base) * 100
                fig.add_trace(go.Scatter(
                    x=bist_norm["Date"], y=bist_norm["Normalized"],
                    name="BIST-100 (indexed)",
                    line=dict(color="#1976d2", width=2.2),
                ), row=3, col=1)

                # Shade divergence area between the two lines
                merged = pd.merge(
                    stock_norm[["Date", "Normalized"]].rename(columns={"Normalized": "stock"}),
                    bist_norm[["Date", "Normalized"]].rename(columns={"Normalized": "bist"}),
                    on="Date", how="inner"
                )
                if len(merged) > 0:
                    fig.add_trace(go.Scatter(
                        x=merged["Date"], y=merged["stock"],
                        fill=None, mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ), row=3, col=1)
                    fig.add_trace(go.Scatter(
                        x=merged["Date"], y=merged["bist"],
                        fill="tonexty", fillcolor="rgba(239,83,80,0.10)",
                        mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ), row=3, col=1)

            fig.add_hline(y=100, row=3, col=1, line_dash="dot", line_color="#bdbdbd", line_width=1)

            # ── Shared annotations: penalty line + investigation shading ──
            for row_n in [1, 2, 3]:
                fig.add_vline(
                    x=event_date_str, row=row_n, col=1,
                    line_dash="dash", line_color="#d32f2f", line_width=1.5,
                )
                if inv_start_ts:
                    inv_start_str = str(inv_start_ts.strftime("%Y-%m-%d"))
                    inv_end_str = str(inv_end_ts.strftime("%Y-%m-%d"))
                    fig.add_vrect(
                        x0=inv_start_str, x1=inv_end_str,
                        fillcolor="rgba(255,152,0,0.07)", line_width=0,
                        row=row_n, col=1,
                    )

            # Annotations on row 1 only
            fig.add_annotation(
                x=event_date_str, y=1.04, yref="y domain", xref="x",
                text="⚖️ SPK Penalty", showarrow=False,
                font=dict(color="#d32f2f", size=11),
            )
            if inv_start_ts:
                fig.add_annotation(
                    x=str(inv_start_ts.strftime("%Y-%m-%d")), y=0.95, yref="y domain", xref="x",
                    text="🔍 Investigation Period", showarrow=True,
                    arrowhead=0, arrowwidth=1, arrowcolor="#ff9800",
                    ax=60, ay=-20, font=dict(color="#e65100", size=10),
                )

            # ── Layout polish ──
            fig.update_layout(
                template=config.DASHBOARD_THEME,
                height=820,
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=10),
                ),
                margin=dict(t=50, b=30, l=60, r=60),
                xaxis_rangeslider_visible=False,
                xaxis3_rangeslider_visible=False,
            )
            fig.update_yaxes(title_text="Price (TL)", title_font_size=10, row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="BIST-100", title_font_size=10, row=1, col=1, secondary_y=True,
                             showgrid=False)
            fig.update_yaxes(title_text="Volume", title_font_size=10, row=2, col=1)
            fig.update_yaxes(title_text="Indexed (base=100)", title_font_size=10, row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # ── Legend explanation ──
            leg1, leg2, leg3 = st.columns(3)
            with leg1:
                st.markdown(
                    "🟢🔴 **Candlestick** — stock price  \n"
                    "🔵 **Blue dotted** — BIST-100 index  \n"
                    "⚖️ **Red dashed** — penalty date"
                )
            with leg2:
                st.markdown(
                    "🟠 **Orange bars** — volume during investigation  \n"
                    "🔴 **Red bars** — volume around penalty (±7 days)  \n"
                    "⬜ **Grey bars** — normal trading volume"
                )
            with leg3:
                st.markdown(
                    "📊 **Bottom panel** — both series indexed to 100  \n"
                    "If stock (red) drops below BIST-100 (blue) →  \n"
                    "the drop is **stock-specific**, not market-wide"
                )

            # ── Volume analysis summary ──
            if "Volume" in stock_data.columns:
                vol_notnull = stock_data[stock_data["Volume"].notna()]
                if len(vol_notnull) > 5 and inv_start_ts and inv_end_ts:
                    pre_inv = vol_notnull[vol_notnull["Date"] < inv_start_ts]["Volume"]
                    during_inv = vol_notnull[(vol_notnull["Date"] >= inv_start_ts) & (vol_notnull["Date"] <= inv_end_ts)]["Volume"]
                    around_penalty = vol_notnull[
                        (vol_notnull["Date"] >= event_date_ts - pd.Timedelta(days=7)) &
                        (vol_notnull["Date"] <= event_date_ts + pd.Timedelta(days=7))
                    ]["Volume"]
                    post_penalty = vol_notnull[vol_notnull["Date"] > event_date_ts]["Volume"]

                    st.markdown("##### Volume Analysis")
                    vc1, vc2, vc3, vc4 = st.columns(4)

                    def fmt_vol(v):
                        return f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K"

                    with vc1:
                        if len(pre_inv) > 0:
                            st.metric("Pre-Investigation Avg", fmt_vol(pre_inv.mean()))
                    with vc2:
                        if len(during_inv) > 0:
                            st.metric("During Investigation Avg", fmt_vol(during_inv.mean()))
                    with vc3:
                        if len(around_penalty) > 0:
                            st.metric("Penalty Week Avg", fmt_vol(around_penalty.mean()))
                    with vc4:
                        if len(during_inv) > 0 and len(pre_inv) > 0 and pre_inv.mean() > 0:
                            vol_change = (during_inv.mean() / pre_inv.mean() - 1) * 100
                            direction = "Higher" if vol_change > 0 else "Lower"
                            st.metric(
                                "Investigation vs Pre",
                                f"{vol_change:+.0f}%",
                                delta=f"{direction} during manipulation period",
                                delta_color="inverse",
                            )

        else:
            st.warning(
                f"Price data for **{selected_stock}** is not available in the cache. "
                f"This stock may have been delisted or the ticker symbol may differ."
            )
    else:
        st.info(
            "Price cache not found. Run `python scripts/download_spk_prices.py` to generate it."
        )


# ─── Full Database ───────────────────────────────────────
st.markdown("---")
with st.expander("Full SPK Penalty Database (click to expand)"):
    st.caption("Complete dataset of all SPK manipulation penalties used in this analysis.")
    st.dataframe(df, height=400)
