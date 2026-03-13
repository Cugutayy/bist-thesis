"""
BIST Thesis Project Configuration
IPO Fever and the Cost of the Crowd
"""
from pathlib import Path
from datetime import datetime

# ─── Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# ─── Date Range ──────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
ANALYSIS_START = "2020-01-01"
ANALYSIS_END = "2025-06-30"

# ─── BIST Benchmark ─────────────────────────────────────
BIST100_TICKER = "XU100.IS"
BIST_CURRENCY = "TRY"
USD_TRY_TICKER = "USDTRY=X"

# ─── Event Study Parameters ─────────────────────────────
EVENT_WINDOW_PRE = 30      # days before event
EVENT_WINDOW_POST = 30     # days after event
ESTIMATION_WINDOW = 120    # days for normal return estimation
MIN_OBSERVATIONS = 60      # minimum obs for estimation

# ─── IPO Analysis Parameters ────────────────────────────
IPO_FIRST_DAY_WINDOW = 1
IPO_SHORT_TERM_WINDOWS = [5, 10, 30, 60, 90]
IPO_LONG_TERM_WINDOWS = [180, 365]
OVERSUBSCRIPTION_THRESHOLD = 5.0  # times oversubscribed = "hot" IPO

# ─── CSAD Herding Parameters ────────────────────────────
CSAD_ROLLING_WINDOW = 60   # days for rolling herding
HERDING_SIGNIFICANCE = 0.05

# ─── Contrarian Strategy ────────────────────────────────
CONTRARIAN_LOOKBACK = 5    # days
CONTRARIAN_HOLDING = 5     # days
CONTRARIAN_TOP_N = 10      # number of stocks in each portfolio

# ─── Inflation / Macro ──────────────────────────────────
TUFE_BASE_YEAR = 2003      # TUIK base year
CPI_BASE_DATE = "2020-01-01"

# ─── SPK Manipulation ───────────────────────────────────
SPK_BULLETIN_URL = "https://spk.gov.tr/spk-bultenleri"
SPK_SANCTIONS_URL = "https://idariyaptirimlar.spk.gov.tr"

# ─── Dashboard ───────────────────────────────────────────
DASHBOARD_TITLE = "BIST Thesis: IPO Fever & the Cost of the Crowd"
DASHBOARD_THEME = "plotly_white"
COLOR_PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "ipo": "#9467bd",
    "manipulation": "#e377c2",
    "inflation": "#bcbd22",
}
