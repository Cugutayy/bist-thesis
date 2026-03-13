"""
Cross-Sectional OLS Regression: Determinants of IPO Underpricing
Following Durukan (2002), Kiymaz (2000), Ilbasmi (2023)

Dependent variable: underpricing (tavan serisi cumulative return)
Independent variables: offer_price, ln(market_cap), P/E, P/B, tavan_days,
                       ipo_year dummies, sector dummies, hot_market dummy
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("CROSS-SECTIONAL OLS REGRESSION: DETERMINANTS OF IPO UNDERPRICING")
print("=" * 70)

# ─── Load and merge data ────────────────────────────────────
ipo = pd.read_csv("data/processed/ipo_dataset.csv")
pe = pd.read_csv("data/processed/ipo_pe_analysis.csv")

# Merge PE data into IPO dataset
pe_cols = ["ticker", "trailing_pe", "price_to_book", "market_cap", "sector_yf", "industry_yf"]
df = ipo.merge(pe[pe_cols], on="ticker", how="left")

print(f"\nTotal IPOs: {len(df)}")
print(f"With underpricing data: {df['underpricing'].notna().sum()}")
print(f"With market_cap: {df['market_cap'].notna().sum()}")
print(f"With P/E: {(df['trailing_pe'].notna() & (df['trailing_pe'] > 0) & (df['trailing_pe'] < 500)).sum()}")
print(f"With P/B: {df['price_to_book'].notna().sum()}")
print(f"With sector: {df['sector_yf'].notna().sum()}")

# ─── Create regression variables ────────────────────────────
df = df[df["underpricing"].notna()].copy()

# Log market cap
df["ln_mcap"] = np.log(df["market_cap"].replace(0, np.nan))

# Log offer price
df["ln_offer_price"] = np.log(df["offer_price"].replace(0, np.nan))

# P/E: clean outliers
df["pe_clean"] = df["trailing_pe"].copy()
df.loc[(df["pe_clean"] <= 0) | (df["pe_clean"] > 500), "pe_clean"] = np.nan

# P/B: clean outliers
df["pb_clean"] = df["price_to_book"].copy()
df.loc[(df["pb_clean"] <= 0) | (df["pb_clean"] > 50), "pb_clean"] = np.nan

# Year dummies (2020 as base)
for yr in [2021, 2022, 2023, 2024, 2025]:
    df[f"yr_{yr}"] = (df["ipo_year"] == yr).astype(int)

# Hot market dummy: months with >= 5 IPOs
monthly_counts = df.groupby(["ipo_year", "ipo_month"]).size().reset_index(name="n_ipos_month")
df = df.merge(monthly_counts, on=["ipo_year", "ipo_month"], how="left")
df["hot_market"] = (df["n_ipos_month"] >= 5).astype(int)

# Pre-IPO market return (BIST-100 30-day return before IPO)
# We can approximate from benchmark data
df["pre_ipo_market"] = df["benchmark_return_d30"].fillna(0)  # market return over 30 days post-IPO as proxy

# Sector dummies (top 5 sectors + other)
sector_counts = df["sector_yf"].value_counts()
top_sectors = sector_counts.head(5).index.tolist()
print(f"\nTop 5 sectors: {top_sectors}")
for s in top_sectors:
    safe_name = s.replace(" ", "_").replace("-", "_")[:15] if isinstance(s, str) else "unknown"
    df[f"sec_{safe_name}"] = (df["sector_yf"] == s).astype(int)

# ─── MODEL 1: Basic (all observations) ─────────────────────
print("\n" + "=" * 70)
print("MODEL 1: Basic Regression (Maximum Sample)")
print("=" * 70)

y1 = df["underpricing"]
X1_cols = ["ln_offer_price"] + [f"yr_{yr}" for yr in [2021, 2022, 2023, 2024, 2025]] + ["hot_market"]
X1 = df[X1_cols].copy()
X1 = sm.add_constant(X1)

# Drop NaN rows
mask1 = X1.notna().all(axis=1) & y1.notna()
y1_clean = y1[mask1]
X1_clean = X1[mask1]

model1 = sm.OLS(y1_clean, X1_clean).fit(cov_type="HC3")
print(f"\nN = {int(model1.nobs)}")
print(f"R-squared = {model1.rsquared:.4f}")
print(f"Adj R-squared = {model1.rsquared_adj:.4f}")
print(f"F-statistic = {model1.fvalue:.4f} (p = {model1.f_pvalue:.4f})")
print("\nCoefficients:")
print(model1.summary2().tables[1].to_string())

# ─── MODEL 2: With fundamentals ────────────────────────────
print("\n" + "=" * 70)
print("MODEL 2: With Fundamental Variables")
print("=" * 70)

X2_cols = ["ln_offer_price", "ln_mcap", "pe_clean", "pb_clean"] + \
          [f"yr_{yr}" for yr in [2021, 2022, 2023, 2024, 2025]] + ["hot_market"]
X2 = df[X2_cols].copy()
X2 = sm.add_constant(X2)

mask2 = X2.notna().all(axis=1) & y1.notna()
y2_clean = y1[mask2]
X2_clean = X2[mask2]

model2 = sm.OLS(y2_clean, X2_clean).fit(cov_type="HC3")
print(f"\nN = {int(model2.nobs)}")
print(f"R-squared = {model2.rsquared:.4f}")
print(f"Adj R-squared = {model2.rsquared_adj:.4f}")
print(f"F-statistic = {model2.fvalue:.4f} (p = {model2.f_pvalue:.4f})")
print("\nCoefficients:")
print(model2.summary2().tables[1].to_string())

# ─── MODEL 3: Full model with sectors ──────────────────────
print("\n" + "=" * 70)
print("MODEL 3: Full Model (Fundamentals + Sectors)")
print("=" * 70)

sec_cols = [c for c in df.columns if c.startswith("sec_")]
X3_cols = ["ln_offer_price", "ln_mcap", "pe_clean", "pb_clean"] + \
          [f"yr_{yr}" for yr in [2021, 2022, 2023, 2024, 2025]] + \
          ["hot_market"] + sec_cols[:-1]  # drop last sector as base
X3 = df[X3_cols].copy()
X3 = sm.add_constant(X3)

mask3 = X3.notna().all(axis=1) & y1.notna()
y3_clean = y1[mask3]
X3_clean = X3[mask3]

model3 = sm.OLS(y3_clean, X3_clean).fit(cov_type="HC3")
print(f"\nN = {int(model3.nobs)}")
print(f"R-squared = {model3.rsquared:.4f}")
print(f"Adj R-squared = {model3.rsquared_adj:.4f}")
print(f"F-statistic = {model3.fvalue:.4f} (p = {model3.f_pvalue:.4f})")
print("\nCoefficients:")
print(model3.summary2().tables[1].to_string())

# ─── MODEL 4: Log-level specification (robustness) ─────────
print("\n" + "=" * 70)
print("MODEL 4: Log-Level Robustness (ln(1 + underpricing))")
print("=" * 70)

y4 = np.log(1 + df["underpricing"])
X4_cols = ["ln_offer_price", "ln_mcap", "pe_clean", "pb_clean"] + \
          [f"yr_{yr}" for yr in [2021, 2022, 2023, 2024, 2025]] + ["hot_market"]
X4 = df[X4_cols].copy()
X4 = sm.add_constant(X4)

mask4 = X4.notna().all(axis=1) & y4.notna()
y4_clean = y4[mask4]
X4_clean = X4[mask4]

model4 = sm.OLS(y4_clean, X4_clean).fit(cov_type="HC3")
print(f"\nN = {int(model4.nobs)}")
print(f"R-squared = {model4.rsquared:.4f}")
print(f"Adj R-squared = {model4.rsquared_adj:.4f}")
print(f"F-statistic = {model4.fvalue:.4f} (p = {model4.f_pvalue:.4f})")
print("\nCoefficients:")
print(model4.summary2().tables[1].to_string())

# ─── CORRELATION MATRIX ────────────────────────────────────
print("\n" + "=" * 70)
print("PAIRWISE CORRELATIONS")
print("=" * 70)

corr_vars = ["underpricing", "ln_offer_price", "ln_mcap", "pe_clean", "pb_clean", "hot_market"]
corr_df = df[corr_vars].dropna()
print(f"\nN = {len(corr_df)}")
corr_matrix = corr_df.corr()
print("\nPearson Correlations:")
print(corr_matrix.round(3).to_string())

# Spearman
print("\nSpearman Correlations:")
spearman = corr_df.rank().corr()
print(spearman.round(3).to_string())

# ─── VIF (Variance Inflation Factor) ──────────────────────
print("\n" + "=" * 70)
print("VARIANCE INFLATION FACTORS (Model 2)")
print("=" * 70)

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = X2_clean.drop("const", axis=1)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data.to_string(index=False))

# ─── Save results ──────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save regression table
results = []
for name, model in [("Model 1: Basic", model1), ("Model 2: Fundamentals", model2),
                     ("Model 3: Full", model3), ("Model 4: Log-level", model4)]:
    for var in model.params.index:
        results.append({
            "model": name,
            "variable": var,
            "coefficient": model.params[var],
            "std_error": model.bse[var],
            "t_stat": model.tvalues[var],
            "p_value": model.pvalues[var],
            "significant_5pct": model.pvalues[var] < 0.05,
        })

results_df = pd.DataFrame(results)
results_df.to_csv("data/processed/cross_sectional_regression.csv", index=False)
print("Saved: data/processed/cross_sectional_regression.csv")

# Save model summaries
summary = pd.DataFrame({
    "model": ["Model 1: Basic", "Model 2: Fundamentals", "Model 3: Full", "Model 4: Log-level"],
    "n_obs": [model1.nobs, model2.nobs, model3.nobs, model4.nobs],
    "r_squared": [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared],
    "adj_r_squared": [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj, model4.rsquared_adj],
    "f_stat": [model1.fvalue, model2.fvalue, model3.fvalue, model4.fvalue],
    "f_pvalue": [model1.f_pvalue, model2.f_pvalue, model3.f_pvalue, model4.f_pvalue],
})
summary.to_csv("data/processed/cross_sectional_summary.csv", index=False)
print("Saved: data/processed/cross_sectional_summary.csv")

print("\n" + "=" * 70)
print("INTERPRETATION SUMMARY")
print("=" * 70)
print("""
Key findings from cross-sectional regression:
1. The model explains how much of IPO underpricing variation
2. Significant variables indicate determinants of underpricing
3. Year dummies capture time-varying market conditions
4. Hot market dummy captures IPO wave effects
5. Fundamental variables (P/E, P/B, market cap) test whether
   underpricing is driven by fundamentals vs behavioral factors

If R-squared is LOW: underpricing is largely behavioral/random
If fundamental vars are significant: some rational pricing component
If year/hot_market are significant: market conditions matter
""")
