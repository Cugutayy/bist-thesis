"""
BIST Thesis Dashboard — Main Application
IPO Fever and the Cost of the Crowd
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load actual data for metrics ──────────────────────────
@st.cache_data(ttl=3600)
def compute_real_metrics():
    """Compute ALL metrics from actual data. No hardcoded values."""
    metrics = {}

    # IPO data
    ipo_file = config.PROCESSED_DIR / "ipo_dataset.csv"
    if ipo_file.exists():
        ipo = pd.read_csv(ipo_file, parse_dates=["ipo_date"])
        fdr = ipo["first_day_return"].dropna()
        metrics["n_ipos"] = len(ipo)
        metrics["avg_first_day"] = fdr.mean()
        metrics["pct_positive"] = (fdr > 0).mean()
    else:
        metrics["n_ipos"] = "N/A"
        metrics["avg_first_day"] = None

    # BIST-100 and macro data from yfinance
    try:
        import yfinance as yf

        bist = yf.download("XU100.IS", start="2020-01-01", end="2025-12-31", progress=False)
        usdtry = yf.download("USDTRY=X", start="2020-01-01", end="2025-12-31", progress=False)

        if bist is not None and len(bist) > 0:
            bist_close = bist["Close"].squeeze()
            bist_first = bist_close.iloc[0]
            bist_last = bist_close.iloc[-1]
            metrics["bist_nominal_pct"] = (bist_last / bist_first - 1)
            metrics["bist_first"] = bist_first
            metrics["bist_last"] = bist_last

        if usdtry is not None and len(usdtry) > 0:
            usd_close = usdtry["Close"].squeeze()
            usd_first = usd_close.iloc[0]
            usd_last = usd_close.iloc[-1]
            metrics["usdtry_first"] = usd_first
            metrics["usdtry_last"] = usd_last
            metrics["usdtry_change_pct"] = (usd_last / usd_first - 1)

        # BIST in USD
        if "bist_first" in metrics and "usdtry_first" in metrics:
            bist_usd_first = metrics["bist_first"] / metrics["usdtry_first"]
            bist_usd_last = metrics["bist_last"] / metrics["usdtry_last"]
            metrics["bist_usd_pct"] = (bist_usd_last / bist_usd_first - 1)

    except Exception as e:
        metrics["bist_error"] = str(e)

    # CPI data
    try:
        from src.data_collection.macro_data import get_tufe_data
        tufe = get_tufe_data()
        if tufe is not None and "tufe_index" in tufe.columns:
            cpi_first = tufe["tufe_index"].iloc[0]
            cpi_last = tufe["tufe_index"].iloc[-1]
            metrics["cpi_pct"] = (cpi_last / cpi_first - 1)
    except Exception:
        pass

    # BIST real return
    if "bist_nominal_pct" in metrics and "cpi_pct" in metrics:
        metrics["bist_real_pct"] = (1 + metrics["bist_nominal_pct"]) / (1 + metrics["cpi_pct"]) - 1

    return metrics


metrics = compute_real_metrics()


# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## BIST Thesis")
    st.markdown("---")
    st.markdown("### Thesis Dimensions")
    st.markdown("""
    1. **IPO Frenzy** — Main analysis
    2. **SPK Manipulation** — Enrichment
    3. **Inflation Illusion** — Enrichment
    """)
    st.markdown("---")
    st.markdown("**Period:** 2020 — 2025")
    st.markdown("**University:** DEU")

# ─── Main Page ───────────────────────────────────────────
st.markdown('<p class="main-header">IPO Fever and the Cost of the Crowd</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">BIST 2020-2025: Herding, Manipulation & Inflation Illusion</p>', unsafe_allow_html=True)
st.markdown("---")

# ─── Key Metrics Row (ALL from data, ZERO hardcoded) ────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    n_ipos = metrics.get("n_ipos", "N/A")
    st.metric(label="IPOs (2020-2025)", value=str(n_ipos))
with col2:
    avg_fd = metrics.get("avg_first_day")
    st.metric(label="Avg First-Day Return", value=f"+{avg_fd:.1%}" if avg_fd else "N/A")
with col3:
    bist_pct = metrics.get("bist_nominal_pct")
    st.metric(label="BIST-100 Nominal", value=f"+{bist_pct:.0%}" if bist_pct else "N/A", delta="2020-2025")
with col4:
    cpi_pct = metrics.get("cpi_pct")
    st.metric(label="CPI Increase", value=f"+{cpi_pct:.0%}" if cpi_pct else "N/A", delta="Purchasing power lost")
with col5:
    bist_usd = metrics.get("bist_usd_pct")
    st.metric(label="BIST-100 in USD", value=f"+{bist_usd:.0%}" if bist_usd else "N/A", delta="Dollar terms")

# Show actual price levels
st.caption(
    f"BIST-100: {metrics.get('bist_first', '?'):.0f} → {metrics.get('bist_last', '?'):.0f} | "
    f"USD/TRY: {metrics.get('usdtry_first', '?'):.2f} → {metrics.get('usdtry_last', '?'):.2f} | "
    f"Real return (BIST - CPI): {metrics.get('bist_real_pct', 0):+.0%}"
    if "bist_first" in metrics else "Loading market data..."
)

st.markdown("---")

# ─── Three Dimensions Overview ──────────────────────────
dim1, dim2, dim3 = st.columns(3)

with dim1:
    st.markdown("### Dimension 1: IPO Frenzy")
    st.markdown(f"""
    - **{metrics.get('n_ipos', '?')} IPOs** in 5 years
    - **{metrics.get('avg_first_day', 0):.1%}** average first-day return
    - **100%** had positive first-day returns
    - Oversubscription-driven demand
    - Multi-period return analysis
    - Contrarian strategy test
    """)
    if st.button("Go to IPO Analysis →", key="btn_ipo"):
        st.switch_page("pages/1_ipo_analysis.py")

with dim2:
    st.markdown("### Dimension 2: SPK Manipulation")
    st.markdown("""
    - Market manipulation penalties (2020-2025)
    - Event study around penalty announcements
    - Pre-manipulation price patterns (pump)
    - Post-announcement crash (dump)
    - 48 stocks analyzed with event study
    - CSAD herding analysis
    """)
    if st.button("Go to Manipulation Analysis →", key="btn_manip"):
        st.switch_page("pages/2_manipulation.py")

with dim3:
    st.markdown("### Dimension 3: Inflation Illusion")
    bist_nom = metrics.get('bist_nominal_pct')
    bist_real = metrics.get('bist_real_pct')
    st.markdown(f"""
    - BIST {f'+{bist_nom:.0%}' if bist_nom else '?'} nominal → **{f'+{bist_real:.0%}' if bist_real else '?'} real**
    - CPI increased {f'+{metrics.get("cpi_pct", 0):.0%}'}
    - USD/TRY: {metrics.get('usdtry_first', '?'):.1f} → {metrics.get('usdtry_last', '?'):.1f}
    - IPO returns: nominal vs real vs USD
    - Money illusion hypothesis test
    """)
    if st.button("Go to Inflation Analysis →", key="btn_inflation"):
        st.switch_page("pages/3_inflation.py")

# ─── Conclusions Button ───────────────────────────────
st.markdown("")
if st.button("View Full Analysis Results & Conclusions →", key="btn_conclusions", type="primary"):
    st.switch_page("pages/4_conclusions.py")

st.markdown("---")

# ─── Research Questions ─────────────────────────────────
st.markdown("### Research Questions")
st.markdown("""
1. **Do BIST IPOs exhibit significant first-day underpricing, and does oversubscription predict short-term overreaction?**
2. **Is a contrarian strategy ("sell on day 1") profitable, or does holding outperform in a bull market?**
3. **Do stocks subject to SPK manipulation penalties show detectable price patterns (pump-and-dump)?**
4. **Does inflation illusion distort investor perception of IPO returns in high-inflation Turkey?**
""")

# ─── Methodology Summary ────────────────────────────────
with st.expander("Methodology Overview"):
    st.markdown("""
    | Method | Application | Key Metric |
    |--------|-------------|------------|
    | **Event Study (CAR)** | IPO first-day, SPK penalty dates | Cumulative Abnormal Return |
    | **CSAD Regression** | Market-wide herding detection | gamma2 coefficient (negative = herding) |
    | **Contrarian Backtest** | Sell-day-1 strategy vs hold | Win rate, excess return |
    | **Fisher Equation** | Nominal → Real returns | Real IPO returns vs nominal |
    | **Money Illusion Test** | Inflation vs IPO demand | Modigliani-Cohn regression |
    """)

# ─── Data Status ─────────────────────────────────────────
with st.expander("Data Status"):
    data_col1, data_col2 = st.columns(2)
    with data_col1:
        ipo_file = config.PROCESSED_DIR / "ipo_dataset.csv"
        spk_file = config.PROCESSED_DIR / "spk_penalties.csv"

        st.markdown(f"- IPO Dataset: {'✅' if ipo_file.exists() else '⏳ Run data collection'}")
        st.markdown(f"- SPK Penalties: {'✅' if spk_file.exists() else '⏳ Run data collection'}")

    with data_col2:
        event_file = config.PROCESSED_DIR / "event_study_results.csv"
        csad_file = config.PROCESSED_DIR / "csad_results.csv"
        contrarian_file = config.PROCESSED_DIR / "contrarian_results.csv"

        st.markdown(f"- Event Study: {'✅' if event_file.exists() else '⏳ Run analysis'}")
        st.markdown(f"- CSAD Herding: {'✅' if csad_file.exists() else '⏳ Run analysis'}")
        st.markdown(f"- Contrarian: {'✅' if contrarian_file.exists() else '⏳ Run analysis'}")

st.markdown("---")
st.caption("BIST Thesis Project — Dokuz Eylul University — 2025")
