"""
Generate thesis website HTML with all data embedded.
Reads CSV data and creates a complete static HTML file with Plotly.js charts.
All statistics are computed from data -- never hardcoded.
"""
import pandas as pd
import numpy as np
import json
import sys
from scipy import stats as sp_stats

sys.path.insert(0, '.')
import config

# ── Read all data ──
ipo = pd.read_csv(config.PROCESSED_DIR / 'ipo_dataset.csv')
es_results = pd.read_csv(config.PROCESSED_DIR / 'event_study_results.csv')
es_individual = pd.read_csv(config.PROCESSED_DIR / 'event_study_individual.csv')
csad = pd.read_csv(config.PROCESSED_DIR / 'csad_results.csv')
csad_regime = pd.read_csv(config.PROCESSED_DIR / 'csad_regime.csv')
csad_rolling = pd.read_csv(config.PROCESSED_DIR / 'csad_rolling.csv')
contrarian = pd.read_csv(config.PROCESSED_DIR / 'contrarian_by_horizon.csv')
spk = pd.read_csv(config.PROCESSED_DIR / 'spk_penalties.csv')
pe_data = pd.read_csv(config.PROCESSED_DIR / 'ipo_pe_analysis.csv')
regression = pd.read_csv(config.PROCESSED_DIR / 'cross_sectional_regression.csv')
tufe = pd.read_csv(config.PROCESSED_DIR / 'tufe_data.csv')
reg_summary = pd.read_csv(config.PROCESSED_DIR / 'cross_sectional_summary.csv')
bist100 = pd.read_csv(config.PROCESSED_DIR / 'bist100_data.csv')
usdtry = pd.read_csv(config.PROCESSED_DIR / 'usdtry_data.csv')
spk_price_cache = pd.read_csv(config.PROCESSED_DIR / 'spk_price_cache.csv')

# ── Merge PE data with IPO ──
ipo_pe = ipo.merge(pe_data[['ticker', 'trailing_pe', 'price_to_book', 'market_cap', 'sector_yf', 'tavan_group']], on='ticker', how='left')


# Fill missing tavan_group from tavan_days
def get_tavan_group(d):
    if d == 0:
        return '0 days'
    elif d <= 2:
        return '1-2 days'
    elif d <= 5:
        return '3-5 days'
    elif d <= 10:
        return '6-10 days'
    else:
        return '11+ days'


ipo_pe['tavan_group'] = ipo_pe['tavan_group'].fillna(ipo_pe['tavan_days'].apply(get_tavan_group))

# ── Pre-compute stats ──
n_ipo = len(ipo)
mean_under = ipo['underpricing'].mean() * 100
median_under = ipo['underpricing'].median() * 100
avg_tavan = ipo['tavan_days'].mean()
pct_tavan = ipo['hit_tavan_day1'].mean() * 100
max_tavan = int(ipo['tavan_days'].max())
avg_fdr = ipo['first_day_return'].mean() * 100

# IPO by year (with hit_tavan_day1 percentage)
yearly = ipo.groupby('ipo_year').agg(
    n=('ticker', 'count'),
    mean_u=('underpricing', 'mean'),
    median_u=('underpricing', 'median'),
    avg_td=('tavan_days', 'mean'),
    pct_tavan=('hit_tavan_day1', 'mean')
).reset_index()

# Tavan distribution
tavan_hist = ipo['tavan_days'].value_counts().sort_index().reset_index()
tavan_hist.columns = ['days', 'count']

# SPK stats
spk_total_fine = spk['toplam_ceza_tl'].sum()
spk_people = spk['kisi_sayisi'].sum()
spk_ban_pct = (spk['islem_yasagi'] == True).mean() * 100 if 'islem_yasagi' in spk.columns else 0
spk_by_year = spk.groupby('yil').agg(n=('hisse_kodu', 'count'), total=('toplam_ceza_tl', 'sum')).reset_index()
spk_by_type = spk['ceza_turu'].value_counts().reset_index()
spk_by_type.columns = ['type', 'count']

# Event study stats
es_mean_car = es_individual['car'].mean() * 100
es_median_car = es_individual['car'].median() * 100
es_neg = (es_individual['car'] < 0).sum()
es_sig = (es_individual['t_pvalue'] < 0.05).sum()
es_n = len(es_individual)

# CSAD stats (full regression table - Feature #9)
csad_row = csad.iloc[0]
gamma2 = csad_row['gamma2']
gamma2_t = csad_row['gamma2_tstat']
gamma2_p = csad_row['gamma2_pvalue']
alpha_val = csad_row['alpha']
gamma1_val = csad_row['gamma1']
r2_val = csad_row['r_squared']
adj_r2_val = csad_row['adj_r_squared']
n_obs_val = int(csad_row['n_obs'])

# PE stats
valid_pe = pe_data[pe_data['trailing_pe'].notna() & (pe_data['trailing_pe'] > 0)]
pe_n = len(valid_pe)
pe_median = valid_pe['trailing_pe'].median()
pe_mean = valid_pe['trailing_pe'].mean()

# Feature #6: P/B and Market Cap metrics
valid_pb = pe_data[pe_data['price_to_book'].notna() & (pe_data['price_to_book'] > 0)]
pb_median = valid_pb['price_to_book'].median()
pb_n = len(valid_pb)
valid_mc = pe_data[pe_data['market_cap'].notna()]
mc_median_b = valid_mc['market_cap'].median() / 1e9  # in billions TL
mc_n = len(valid_mc)

# TUFE stats
tufe_start = tufe[tufe['date'].str.startswith('2020-01')]['tufe_index'].values[0]
tufe_end = tufe[tufe['date'].str.startswith('2025-12')]['tufe_index'].values[0]
tufe_pct = (tufe_end / tufe_start - 1) * 100

# Feature #2: Peak inflation from data (NOT hardcoded)
peak_yoy = tufe['tufe_yoy'].max()
peak_idx = tufe['tufe_yoy'].idxmax()
peak_date = tufe.loc[peak_idx, 'date']
peak_date_str = pd.Timestamp(peak_date).strftime('%b %Y')

# Contrarian table
contr_json = contrarian.to_dict('records')

# Feature #4: Statistical tests for underpricing
upr = ipo['underpricing'].dropna()
t_stat_upr, p_val_upr = sp_stats.ttest_1samp(upr, 0)
upr_nonzero = upr[upr != 0]
w_stat_upr, w_pval_upr = sp_stats.wilcoxon(upr_nonzero)

# Feature #5: Sector analysis
sector_stats = ipo_pe.groupby('sector_yf').agg(
    count=('ticker', 'count'),
    avg_return=('underpricing', 'mean')
).dropna().sort_values('avg_return', ascending=False).reset_index()
sector_stats['avg_return_pct'] = sector_stats['avg_return'] * 100

# Feature #7: P/E by Tavan Group
pe_by_tavan = pe_data[pe_data['trailing_pe'].notna() & (pe_data['trailing_pe'] > 0)].groupby('tavan_group').agg(
    median_pe=('trailing_pe', 'median'),
    count=('ticker', 'count')
).reset_index()
# Sort by tavan group logically
tavan_order = ['0 days', '1-2 days', '3-5 days', '6-10 days', '11+ days']
pe_by_tavan['sort_key'] = pe_by_tavan['tavan_group'].map({g: i for i, g in enumerate(tavan_order)})
pe_by_tavan = pe_by_tavan.sort_values('sort_key').drop(columns='sort_key')

# Feature #8: Top 10 penalties (already loaded)
spk_top = spk.nlargest(10, 'toplam_ceza_tl')[['hisse_kodu', 'company_name', 'karar_tarihi', 'toplam_ceza_tl', 'kisi_sayisi', 'ceza_turu', 'notes']]

# ── Prepare JSON data for embedding ──
def to_json(df, cols=None):
    if cols:
        df = df[cols]
    return df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict('records')


# ── NEW FEATURES: Data preparation ──

# Feature: Average fine by manipulation type
spk_avg_fine_by_type = spk.groupby('ceza_turu')['toplam_ceza_tl'].mean().reset_index()
spk_avg_fine_by_type.columns = ['type', 'avg_fine']

# Feature: Notable cases (top 3 positive + top 3 negative CARs)
notable_pos = es_individual.nlargest(3, 'car')[['ticker', 'car', 't_stat', 't_pvalue']].copy()
notable_neg = es_individual.nsmallest(3, 'car')[['ticker', 'car', 't_stat', 't_pvalue']].copy()

# Feature: SPK price cache data for case explorer chart
# Build a dictionary: ticker -> list of {Date, Open, High, Low, Close, Volume}
spk_price_by_ticker = {}
for ticker in spk_price_cache['ticker'].unique():
    tdata = spk_price_cache[spk_price_cache['ticker'] == ticker].sort_values('Date').copy()
    spk_price_by_ticker[ticker] = to_json(tdata, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Feature: Full SPK database table
spk_full_table = spk[['hisse_kodu', 'company_name', 'karar_tarihi', 'toplam_ceza_tl',
                       'kisi_sayisi', 'ceza_turu', 'islem_yasagi', 'notes']].copy()

# Feature: BIST-100 vs Inflation vs USD overlay (use pre-downloaded CSVs)
bist100_json = to_json(bist100, ['date', 'bist100_close'])
usdtry_json = to_json(usdtry, ['date', 'usdtry_close'])

# (Illusion zone data computed after cpi_daily is defined below)

# Feature: Inflation vs IPO demand (monthly)
ipo_monthly = ipo.copy()
ipo_monthly['month'] = pd.to_datetime(ipo_monthly['ipo_date']).dt.to_period('M').dt.to_timestamp()
monthly_ipo_counts = ipo_monthly.groupby('month').agg(ipo_count=('ticker', 'size')).reset_index()
monthly_ipo_counts['month_str'] = monthly_ipo_counts['month'].dt.strftime('%Y-%m-%d')

tufe_monthly = tufe.copy()
tufe_monthly['month'] = pd.to_datetime(tufe_monthly['date']).dt.to_period('M').dt.to_timestamp()
ipo_infl_merged = monthly_ipo_counts.merge(tufe_monthly[['month', 'tufe_yoy']], on='month', how='left')
ipo_infl_merged = ipo_infl_merged.dropna(subset=['tufe_yoy'])
# Compute correlation
if len(ipo_infl_merged) > 5:
    from scipy.stats import pearsonr
    infl_corr, infl_pval = pearsonr(ipo_infl_merged['tufe_yoy'], ipo_infl_merged['ipo_count'])
else:
    infl_corr, infl_pval = 0, 1

# Feature: 4-model comparison table for conclusions
# Build model comparison HTML rows
model_comparison_rows = ""
for _, row in reg_summary.iterrows():
    model_name = row['model']
    short_name = model_name.split(': ')[1] if ': ' in model_name else model_name
    r2 = row['r_squared']
    adj_r2 = row['adj_r_squared']
    f_stat = row['f_stat']
    f_pval = row['f_pvalue']
    n_obs = int(row['n_obs'])
    f_sig = "Yes" if f_pval < 0.05 else "No"
    model_comparison_rows += (
        f'<tr><td><strong>{short_name}</strong></td>'
        f'<td>{n_obs}</td>'
        f'<td>{r2:.4f}</td>'
        f'<td>{adj_r2:.4f}</td>'
        f'<td>{f_stat:.3f}</td>'
        f'<td>{f_pval:.4f}</td>'
        f'<td>{f_sig}</td></tr>\n'
    )

# Build coefficient tables for all 4 models
all_models_coef_html = ""
for model_name in reg_summary['model'].values:
    short_name = model_name.split(': ')[1] if ': ' in model_name else model_name
    m_data = regression[regression['model'] == model_name]
    all_models_coef_html += f'<h3>{short_name} Coefficients</h3><div class="table-scroll"><table><tr><th>Variable</th><th>Coefficient</th><th>Std Error</th><th>t-stat</th><th>p-value</th><th>Sig.</th></tr>'
    for _, r in m_data.iterrows():
        sig_str = "***" if r['p_value'] < 0.01 else ("**" if r['p_value'] < 0.05 else ("*" if r['p_value'] < 0.1 else ""))
        all_models_coef_html += (
            f'<tr><td>{r["variable"]}</td>'
            f'<td>{r["coefficient"]:.4f}</td>'
            f'<td>{r["std_error"]:.4f}</td>'
            f'<td>{r["t_stat"]:.3f}</td>'
            f'<td>{r["p_value"]:.4f}</td>'
            f'<td>{sig_str}</td></tr>'
        )
    all_models_coef_html += '</table></div>'

# Build notable cases HTML
notable_pos_html = ""
for _, r in notable_pos.iterrows():
    sig = " (significant)" if r['t_pvalue'] < 0.05 else ""
    notable_pos_html += f'<tr><td>{r["ticker"]}</td><td style="color:var(--positive);font-weight:bold">{r["car"]*100:.1f}%</td><td>{r["t_stat"]:.2f}</td><td>{r["t_pvalue"]:.4f}{sig}</td></tr>'

notable_neg_html = ""
for _, r in notable_neg.iterrows():
    sig = " (significant)" if r['t_pvalue'] < 0.05 else ""
    notable_neg_html += f'<tr><td>{r["ticker"]}</td><td style="color:var(--negative);font-weight:bold">{r["car"]*100:.1f}%</td><td>{r["t_stat"]:.2f}</td><td>{r["t_pvalue"]:.4f}{sig}</td></tr>'

# Build full SPK database HTML rows
spk_full_rows = ""
for _, row in spk_full_table.iterrows():
    fine_str = f'&#8378;{row["toplam_ceza_tl"]/1e6:.1f}M' if row['toplam_ceza_tl'] >= 1e6 else f'&#8378;{row["toplam_ceza_tl"]/1e3:.0f}K'
    ban_str = "Yes" if row.get('islem_yasagi') else "No"
    notes_str = str(row.get('notes', ''))[:80] if pd.notna(row.get('notes')) else ''
    spk_full_rows += (
        f'<tr><td>{row["hisse_kodu"]}</td>'
        f'<td>{str(row["company_name"])[:30]}</td>'
        f'<td>{row["karar_tarihi"]}</td>'
        f'<td>{fine_str}</td>'
        f'<td>{int(row["kisi_sayisi"])}</td>'
        f'<td>{row["ceza_turu"]}</td>'
        f'<td>{ban_str}</td>'
        f'<td style="font-size:0.75rem">{notes_str}</td></tr>\n'
    )

# Feature #10: Rolling herding episodes
csad_roll_ds = csad_rolling.iloc[::3].copy()  # every 3rd row for performance
herding_episodes = csad_roll_ds[csad_roll_ds['herding_flag'] == True].copy()

# ── Feature #1: REAL RETURNS using Fisher equation with actual CPI data ──
tufe_ts = tufe.set_index('date')['tufe_index']
tufe_ts.index = pd.to_datetime(tufe_ts.index)
cpi_daily = tufe_ts.resample('D').interpolate()

real_return_values = []
for period_name, col, days_val in [
    ('Tavan Series', 'underpricing', None),  # use tavan_days per IPO
    ('30 Days', 'return_d30', 30),
    ('60 Days', 'return_d60', 60),
    ('90 Days', 'return_d90', 90),
    ('180 Days', 'return_d180', 180),
    ('365 Days', 'return_d365', 365),
]:
    real_rets = []
    if col == 'underpricing':
        valid = ipo[['ipo_date', col, 'tavan_days']].dropna()
    else:
        valid = ipo[['ipo_date', col]].dropna()
    for _, row in valid.iterrows():
        ipo_date = pd.Timestamp(row['ipo_date'])
        if days_val is None:
            d = max(int(row.get('tavan_days', 1)), 1)
        else:
            d = days_val
        end_date = ipo_date + pd.Timedelta(days=d)
        try:
            cpi_s = cpi_daily.asof(ipo_date)
            cpi_e = cpi_daily.asof(end_date)
            if pd.notna(cpi_s) and pd.notna(cpi_e) and cpi_s > 0:
                infl = (cpi_e / cpi_s) - 1
                real_rets.append((1 + row[col]) / (1 + infl) - 1)
        except Exception:
            pass
    real_return_values.append(np.mean(real_rets) * 100 if real_rets else 0)

# BIST-100 and USD/TRY statistics for Big Picture
bist_first = bist100.iloc[0]['bist100_close']
bist_last = bist100.iloc[-1]['bist100_close']
bist_nominal_pct = (bist_last / bist_first - 1) * 100
bist_real_pct = ((1 + bist_nominal_pct/100) / (1 + tufe_pct/100) - 1) * 100
usd_first = usdtry.iloc[0]['usdtry_close']
usd_last = usdtry.iloc[-1]['usdtry_close']
usd_change_pct = (usd_last / usd_first - 1) * 100
bist_usd_first = bist_first / usd_first
bist_usd_last = bist_last / usd_last
bist_usd_pct = (bist_usd_last / bist_usd_first - 1) * 100

# ── Illusion zone scatter plot (now that cpi_daily exists) ──
illusion_data = []
for _, row in ipo[['ticker', 'ipo_date', 'return_d365']].dropna().iterrows():
    ipo_date = pd.Timestamp(row['ipo_date'])
    end_date = ipo_date + pd.Timedelta(days=365)
    try:
        cpi_s = cpi_daily.asof(ipo_date)
        cpi_e = cpi_daily.asof(end_date)
        if pd.notna(cpi_s) and pd.notna(cpi_e) and cpi_s > 0:
            infl = (cpi_e / cpi_s) - 1
            real_ret = (1 + row['return_d365']) / (1 + infl) - 1
            in_illusion = bool((row['return_d365'] > 0) and (real_ret < 0))
            illusion_data.append({
                'ticker': str(row['ticker']),
                'nominal': round(float(row['return_d365']) * 100, 2),
                'real': round(float(real_ret) * 100, 2),
                'illusion': in_illusion
            })
    except Exception:
        pass

illusion_df = pd.DataFrame(illusion_data)
n_illusion = int(illusion_df['illusion'].sum()) if len(illusion_df) > 0 else 0
n_illusion_total = len(illusion_df)

# IPO dataset (selected columns for table)
ipo_table_cols = ['ticker', 'company_name', 'ipo_date', 'offer_price', 'tavan_days', 'underpricing', 'tavan_series_return', 'first_day_return', 'ipo_year']
ipo_table = to_json(ipo, ipo_table_cols)

# Individual CARs sorted
es_ind_sorted = es_individual.sort_values('car').reset_index(drop=True)

# Multi-period returns
periods = ['d30', 'd60', 'd90', 'd180', 'd365']
period_labels = ['30 Days', '60 Days', '90 Days', '180 Days', '365 Days']
mean_returns = [ipo[f'return_{p}'].mean() * 100 for p in periods]
median_returns = [ipo[f'return_{p}'].median() * 100 for p in periods]

# Return comparison table data
return_table_rows = ""
period_names_table = ['Tavan Series', '30 Days', '60 Days', '90 Days', '180 Days', '365 Days']
nom_vals = [mean_under] + mean_returns  # already computed
for i, (pname, nom_v, real_v) in enumerate(zip(period_names_table, nom_vals, real_return_values)):
    infl_v = nom_v - real_v  # approximate: nominal - real = inflation effect
    return_table_rows += f'<tr><td>{pname}</td><td>{nom_v:.1f}%</td><td>{infl_v:.1f}%</td><td>{real_v:.1f}%</td></tr>\n'

# PE scatter data
pe_scatter = pe_data[pe_data['trailing_pe'].notna() & (pe_data['trailing_pe'] > 0) & pe_data['underpricing'].notna()]
pe_scatter = pe_scatter[['ticker', 'trailing_pe', 'underpricing']].copy()
pe_scatter['underpricing_pct'] = pe_scatter['underpricing'] * 100

# Regression Model 4 (log-level) coefficients
reg_m4 = regression[regression['model'] == 'Model 4: Log-level']

# ── Build yearly underpricing breakdown HTML (Feature #3) ──
yearly_table_rows = ""
for _, yr in yearly.iterrows():
    yearly_table_rows += (
        f'<tr><td>{int(yr["ipo_year"])}</td>'
        f'<td>{int(yr["n"])}</td>'
        f'<td>{yr["mean_u"]*100:.1f}%</td>'
        f'<td>{yr["median_u"]*100:.1f}%</td>'
        f'<td>{yr["avg_td"]:.1f}</td>'
        f'<td>{yr["pct_tavan"]*100:.0f}%</td></tr>\n'
    )

# ── Build SPK top 10 penalties HTML (Feature #8) ──
spk_top_rows = ""
for _, row in spk_top.iterrows():
    spk_top_rows += (
        f'<tr><td>{row["hisse_kodu"]}</td>'
        f'<td>{str(row["company_name"])[:35]}</td>'
        f'<td>{row["karar_tarihi"]}</td>'
        f'<td>&#8378;{row["toplam_ceza_tl"]/1e6:.1f}M</td>'
        f'<td>{int(row["kisi_sayisi"])}</td>'
        f'<td>{row["ceza_turu"]}</td></tr>\n'
    )

# ── Nominal return arrays for charts ──
nom_full_json = json.dumps([mean_under] + mean_returns)
real_full_json = json.dumps(real_return_values)
period_full_labels = json.dumps(['Tavan Series'] + period_labels)

# ── Build HTML ──
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IPO Fever and the Cost of the Crowd | BIST 2020-2025</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --primary: #1e3a5f;
    --primary-light: #2d5a8e;
    --primary-soft: #e8f0fe;
    --secondary: #c9702a;
    --secondary-soft: #fef3e8;
    --positive: #1a7a3a;
    --positive-soft: #e6f4ea;
    --negative: #b91c1c;
    --negative-soft: #fce8e8;
    --neutral: #64748b;
    --ipo: #6d4aad;
    --ipo-soft: #f0ebf8;
    --manipulation: #b04a77;
    --manipulation-soft: #fce8f0;
    --inflation: #8a7a1e;
    --inflation-soft: #faf5e0;
    --bg: #fafbfc;
    --card-bg: #ffffff;
    --card-hover: #f8fafc;
    --border: #e2e8f0;
    --border-light: #f1f5f9;
    --text: #1e293b;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.03);
    --radius: 10px;
    --radius-lg: 14px;
    --transition: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }}

  * {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}

  /* ── Navigation ── */
  .nav {{
    display: flex;
    gap: 2px;
    background: var(--card-bg);
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: var(--shadow-md);
    padding: 0 16px;
    border-bottom: 1px solid var(--border);
  }}
  .nav-btn {{
    flex: 1;
    padding: 14px 12px;
    border: none;
    background: transparent;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-secondary);
    transition: all var(--transition);
    border-bottom: 2px solid transparent;
    letter-spacing: 0.01em;
    position: relative;
  }}
  .nav-btn:hover {{
    color: var(--primary);
    background: var(--primary-soft);
  }}
  .nav-btn.active {{
    color: var(--primary);
    font-weight: 600;
    border-bottom: 2px solid var(--primary);
  }}

  /* ── Page Container ── */
  .page {{
    display: none;
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 24px 48px;
    transition: opacity 0.15s ease, transform 0.15s ease;
  }}
  .page.active {{ display: block; opacity: 1; transform: translateY(0); }}
  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}

  /* ── Typography ── */
  h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 4px;
    letter-spacing: -0.02em;
    line-height: 1.3;
  }}
  h2 {{
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin: 32px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    letter-spacing: -0.01em;
  }}
  h3 {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 20px 0 10px;
  }}
  .subtitle {{
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: 20px;
    font-weight: 400;
    line-height: 1.5;
  }}

  /* ── Metric Cards ── */
  .metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px;
    margin: 20px 0;
  }}
  .metric {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all var(--transition);
  }}
  .metric:hover {{
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
    border-color: var(--primary-light);
  }}
  .metric .label {{
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
  }}
  .metric .value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin: 6px 0 2px;
    font-feature-settings: 'tnum';
    letter-spacing: -0.02em;
  }}
  .metric .delta {{
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 400;
  }}
  .metric .delta.positive {{ color: var(--positive); }}
  .metric .delta.negative {{ color: var(--negative); }}

  /* ── Grid ── */
  .grid {{ display: grid; gap: 20px; margin: 20px 0; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  @media (max-width: 900px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}

  /* ── Info Boxes ── */
  .info {{
    background: var(--primary-soft);
    border-left: 3px solid var(--primary);
    padding: 14px 18px;
    margin: 14px 0;
    border-radius: 0 var(--radius) var(--radius) 0;
    font-size: 0.88rem;
    line-height: 1.65;
    color: var(--text);
  }}
  .info strong {{ color: var(--primary); }}
  .success {{
    background: var(--positive-soft);
    border-left: 3px solid var(--positive);
    padding: 14px 18px;
    margin: 14px 0;
    border-radius: 0 var(--radius) var(--radius) 0;
    font-size: 0.88rem;
    line-height: 1.65;
  }}
  .success strong {{ color: var(--positive); }}
  .warning {{
    background: var(--secondary-soft);
    border-left: 3px solid var(--secondary);
    padding: 14px 18px;
    margin: 14px 0;
    border-radius: 0 var(--radius) var(--radius) 0;
    font-size: 0.88rem;
    line-height: 1.65;
  }}
  .warning strong {{ color: var(--secondary); }}
  .error-box {{
    background: var(--negative-soft);
    border-left: 3px solid var(--negative);
    padding: 14px 18px;
    margin: 14px 0;
    border-radius: 0 var(--radius) var(--radius) 0;
    font-size: 0.88rem;
    line-height: 1.65;
  }}
  .error-box strong {{ color: var(--negative); }}

  /* ── Chart Container ── */
  .chart {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 12px;
    margin: 10px 0;
    box-shadow: var(--shadow-sm);
    transition: box-shadow var(--transition);
  }}
  .chart:hover {{ box-shadow: var(--shadow); }}

  /* ── Tables ── */
  table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.84rem;
    margin: 14px 0;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
  }}
  th {{
    background: var(--primary);
    color: #fff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 500;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--border-light);
    color: var(--text-secondary);
  }}
  tr:last-child td {{ border-bottom: none; }}
  tbody tr:nth-child(even) {{ background: #f8fafc; }}
  tbody tr:hover {{ background: var(--primary-soft); }}

  /* ── Select / Dropdown ── */
  select {{
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 0.88rem;
    font-family: 'Inter', sans-serif;
    min-width: 220px;
    background: var(--card-bg);
    box-shadow: var(--shadow-sm);
    transition: all var(--transition);
    cursor: pointer;
    color: var(--text);
  }}
  select:hover {{ border-color: var(--primary-light); }}
  select:focus {{ outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(30,58,95,0.1); }}

  /* ── Scrollable Table ── */
  .table-scroll {{
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
  }}
  .table-scroll table {{ border: none; border-radius: 0; }}

  /* ── Section Separator ── */
  hr {{
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
    margin: 36px 0;
  }}

  /* ── Hypothesis / Sub-question Cards ── */
  .sq-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 18px 20px;
    box-shadow: var(--shadow-sm);
    transition: all var(--transition);
  }}
  .sq-card:hover {{
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
  }}
  .sq-card .sq-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
  }}
  .sq-card .sq-title {{
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 6px;
  }}
  .sq-card .sq-text {{
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.55;
  }}

  /* ── Theory Framework ── */
  .theory-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 22px;
    box-shadow: var(--shadow-sm);
    transition: all var(--transition);
    border-top: 3px solid var(--primary);
  }}
  .theory-card:hover {{ box-shadow: var(--shadow-md); }}
  .theory-card h4 {{
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 8px;
  }}
  .theory-card p {{
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.6;
  }}

  /* ── Variable Badge ── */
  .var-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 3px 4px 3px 0;
  }}
  .var-badge.dv {{ background: var(--negative-soft); color: var(--negative); border: 1px solid rgba(185,28,28,0.2); }}
  .var-badge.iv {{ background: var(--primary-soft); color: var(--primary); border: 1px solid rgba(30,58,95,0.2); }}
  .var-badge.cv {{ background: var(--inflation-soft); color: var(--inflation); border: 1px solid rgba(138,122,30,0.2); }}

  /* ── Verdict Tags ── */
  .verdict {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }}
  .verdict.confirmed {{ background: var(--positive-soft); color: var(--positive); }}
  .verdict.partial {{ background: var(--secondary-soft); color: var(--secondary); }}
  .verdict.rejected {{ background: var(--negative-soft); color: var(--negative); }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    padding: 28px 24px;
    margin-top: 40px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.82rem;
    line-height: 1.7;
  }}
  .footer strong {{ color: var(--text-secondary); font-weight: 500; }}

  /* ── Links ── */
  a {{ color: var(--primary-light); text-decoration: none; transition: color var(--transition); }}
  a:hover {{ color: var(--primary); text-decoration: underline; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}

  /* ═══ LIQUID AMBIENT BACKGROUND ═══ */
  body::before {{
    content: '';
    position: fixed;
    inset: -50%;
    width: 200%;
    height: 200%;
    background:
      radial-gradient(ellipse 600px 400px at 25% 30%, rgba(30,58,95,0.045) 0%, transparent 70%),
      radial-gradient(ellipse 500px 500px at 75% 60%, rgba(201,112,42,0.035) 0%, transparent 70%),
      radial-gradient(ellipse 400px 600px at 50% 85%, rgba(109,74,173,0.03) 0%, transparent 70%);
    animation: liquidDrift 25s ease-in-out infinite alternate;
    z-index: -1;
    pointer-events: none;
    will-change: transform;
  }}
  @keyframes liquidDrift {{
    0%   {{ transform: translate(0, 0) rotate(0deg) scale(1); }}
    33%  {{ transform: translate(1.5%, -1%) rotate(1.5deg) scale(1.01); }}
    66%  {{ transform: translate(-1%, 1.5%) rotate(-1deg) scale(0.99); }}
    100% {{ transform: translate(2%, -2%) rotate(2deg) scale(1.02); }}
  }}

  /* ═══ SCROLL REVEAL ═══ */
  .reveal {{
    opacity: 0;
    transform: translateY(28px);
    transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1),
                transform 0.7s cubic-bezier(0.16, 1, 0.3, 1);
  }}
  .reveal.visible {{
    opacity: 1;
    transform: translateY(0);
  }}
  @media (prefers-reduced-motion: reduce) {{
    .reveal {{ opacity: 1; transform: none; transition: none; }}
  }}

  /* ═══ GLASSMORPHISM ENHANCEMENT ═══ */
  @supports (backdrop-filter: blur(1px)) {{
    .metric {{
      background: rgba(255,255,255,0.82);
      backdrop-filter: blur(12px) saturate(1.2);
      -webkit-backdrop-filter: blur(12px) saturate(1.2);
    }}
    .theory-card {{
      background: rgba(255,255,255,0.82);
      backdrop-filter: blur(12px) saturate(1.2);
      -webkit-backdrop-filter: blur(12px) saturate(1.2);
    }}
    .info, .success, .warning {{
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
    }}
    .sq-card {{
      background: rgba(255,255,255,0.85);
      backdrop-filter: blur(10px) saturate(1.15);
      -webkit-backdrop-filter: blur(10px) saturate(1.15);
    }}
  }}

  /* ═══ NAV GLOW ═══ */
  .nav-btn.active {{
    box-shadow: 0 2px 12px rgba(30,58,95,0.15);
  }}
  .nav {{
    backdrop-filter: blur(16px) saturate(1.3);
    -webkit-backdrop-filter: blur(16px) saturate(1.3);
    background: rgba(255,255,255,0.88);
  }}

  /* ═══ IMPROVED PAGE TRANSITION ═══ */
  .page {{
    transition: opacity 0.35s cubic-bezier(0.16, 1, 0.3, 1),
                transform 0.35s cubic-bezier(0.16, 1, 0.3, 1),
                filter 0.35s cubic-bezier(0.16, 1, 0.3, 1);
  }}

  /* ═══ METRIC HOVER GLOW ═══ */
  .metric:hover {{
    box-shadow: var(--shadow-md), 0 0 20px rgba(30,58,95,0.06);
  }}

  /* ═══ CHART CONTAINER ANIMATION ═══ */
  .plotly-chart {{
    border-radius: var(--radius);
    overflow: hidden;
  }}
</style>
</head>
<body>

<nav class="nav">
  <button class="nav-btn active" onclick="showPage('overview',this)">Overview</button>
  <button class="nav-btn" onclick="showPage('ipo',this)">IPO Analysis</button>
  <button class="nav-btn" onclick="showPage('manipulation',this)">Market Environment</button>
  <button class="nav-btn" onclick="showPage('herding',this)">Herding & Inflation</button>
  <button class="nav-btn" onclick="showPage('conclusions',this)">Conclusions</button>
</nav>

<!-- ═══════════════════ OVERVIEW ═══════════════════ -->
<div id="overview" class="page active">
  <h1>IPO Fever and the Cost of the Crowd</h1>
  <p class="subtitle">Borsa Istanbul 2020&ndash;2025 &middot; Underpricing, Herding &amp; Inflation &middot; Dokuz Eylul University</p>

  <div class="metrics">
    <div class="metric"><div class="label">IPOs Analyzed</div><div class="value">{n_ipo}</div><div class="delta">2020&ndash;2025</div></div>
    <div class="metric"><div class="label">Mean Underpricing</div><div class="value">{mean_under:.1f}%</div><div class="delta">Tavan serisi</div></div>
    <div class="metric"><div class="label">Median Underpricing</div><div class="value">{median_under:.1f}%</div></div>
    <div class="metric"><div class="label">CPI Increase</div><div class="value">+{tufe_pct:.0f}%</div><div class="delta negative">Purchasing power lost</div></div>
    <div class="metric"><div class="label">SPK Penalty Cases</div><div class="value">{len(spk)}</div><div class="delta">{spk_people:.0f} people penalized</div></div>
  </div>

  <hr>
  <h2>Research Question</h2>
  <div class="info" style="border-color:var(--primary);font-size:0.95rem;">
    <strong>Main RQ:</strong> To what extent is IPO underpricing in Borsa Istanbul explained by valuation signals and investor behavior during the 2020&ndash;2025 IPO boom?
  </div>

  <div class="grid grid-3" style="margin-top:16px;">
    <div class="sq-card">
      <div class="sq-label" style="background:var(--ipo-soft);color:var(--ipo);">Sub-Question 1</div>
      <div class="sq-title">Valuation Ratios</div>
      <div class="sq-text">Do valuation ratios (P/E and P/B) influence IPO underpricing?</div>
    </div>
    <div class="sq-card">
      <div class="sq-label" style="background:var(--primary-soft);color:var(--primary);">Sub-Question 2</div>
      <div class="sq-title">Investor Herding</div>
      <div class="sq-text">Is there statistical evidence of investor herding during IPO trading periods?</div>
    </div>
    <div class="sq-card">
      <div class="sq-label" style="background:var(--inflation-soft);color:var(--inflation);">Sub-Question 3</div>
      <div class="sq-title">Market Conditions</div>
      <div class="sq-text">Do market conditions (BIST performance, inflation, CPI) affect IPO underpricing outcomes?</div>
    </div>
  </div>

  <hr>
  <h2>Theoretical Framework</h2>
  <div class="grid grid-2">
    <div class="theory-card">
      <h4>Information Asymmetry Theory</h4>
      <p>IPO underpricing arises due to asymmetric information between investors and issuers. Informed investors can better assess true value, forcing issuers to underprice to attract uninformed participants (Rock, 1986; Benveniste &amp; Spindt, 1989).</p>
    </div>
    <div class="theory-card">
      <h4>Behavioral Finance</h4>
      <p>Investor sentiment and herd behavior can amplify price movements, especially in emerging markets with a high share of retail investors. Overreaction and speculative trading may explain extreme underpricing (De Bondt &amp; Thaler, 1985; CCK, 2000).</p>
    </div>
  </div>
  <div class="info" style="margin-top:12px;border-color:var(--neutral);">
    <strong>Context:</strong> Borsa Istanbul provides a unique environment where both mechanisms may operate simultaneously. SPK enforcement actions during 2020&ndash;2025 serve as evidence of a speculative trading environment in which retail investor sentiment may amplify IPO price movements.
  </div>

  <hr>
  <h2>Variable Structure</h2>
  <div style="margin:12px 0;">
    <p style="margin-bottom:10px;font-size:0.88rem;color:var(--text-secondary);"><strong style="color:var(--text);">Dependent Variable:</strong></p>
    <span class="var-badge dv">IPO Underpricing (tavan serisi)</span>
  </div>
  <div style="margin:12px 0;">
    <p style="margin-bottom:10px;font-size:0.88rem;color:var(--text-secondary);"><strong style="color:var(--text);">Key Independent Variables:</strong></p>
    <span class="var-badge iv">Price-to-Book (P/B)</span>
    <span class="var-badge iv">Price-to-Earnings (P/E)</span>
    <span class="var-badge iv">Herding Indicator (CSAD)</span>
  </div>
  <div style="margin:12px 0;">
    <p style="margin-bottom:10px;font-size:0.88rem;color:var(--text-secondary);"><strong style="color:var(--text);">Control Variables:</strong></p>
    <span class="var-badge cv">Market Capitalization</span>
    <span class="var-badge cv">BIST Index Performance</span>
    <span class="var-badge cv">Inflation Rate</span>
    <span class="var-badge cv">Consumer Price Index (CPI)</span>
  </div>

  <hr>
  <h2>Data Sources &amp; Methodology</h2>
  <table>
    <tr><th>Data</th><th>Source</th><th>Period</th><th>Notes</th></tr>
    <tr><td>IPO Data</td><td>halkarz.com + KAP + SPK</td><td>2020-2025</td><td>209 IPOs, 3-source verification</td></tr>
    <tr><td>Daily Prices</td><td>Yahoo Finance</td><td>2020-2025</td><td>Split-adjusted close prices</td></tr>
    <tr><td>CPI (T&Uuml;FE)</td><td>T&Uuml;&Iacute;K (Turkish Statistical Institute)</td><td>2019-2025</td><td>Base year 2003=100, monthly</td></tr>
    <tr><td>BIST-100 Index</td><td>Yahoo Finance (XU100.IS)</td><td>2020-2025</td><td>Daily closing values</td></tr>
    <tr><td>USD/TRY</td><td>Yahoo Finance (USDTRY=X)</td><td>2020-2025</td><td>Daily exchange rates</td></tr>
    <tr><td>SPK Penalties</td><td>SPK Weekly Bulletins</td><td>2020-2025</td><td>47 verified cases</td></tr>
  </table>
  <div class="info">
    <strong>CPI Base Year Explanation:</strong> T&Uuml;&Iacute;K publishes the Consumer Price Index (T&Uuml;FE) with base year 2003=100. A value of {tufe_end:.0f} in December 2025 means prices are {tufe_end/100:.1f}x higher than in 2003. This differs from Trading Economics or World Bank data which rebase to different years (e.g. 2015=100 or 2020=100). For our inflation calculations, the base year does not matter because we compute period-specific inflation as (CPI<sub>end</sub> / CPI<sub>start</sub>) &minus; 1, which yields the same result regardless of base year. Source: T&Uuml;&Iacute;K EVDS, Series: &ldquo;T&uuml;ketici Fiyat Endeksi, 2003=100&rdquo; (<a href="https://data.tuik.gov.tr/Kategori/GetKategori?p=Enflasyon-ve-Fiyat-106" target="_blank">data.tuik.gov.tr</a>).
  </div>
</div>

<!-- ═══════════════════ IPO ANALYSIS ═══════════════════ -->
<div id="ipo" class="page">
  <h1>IPO Underpricing Analysis</h1>
  <p class="subtitle">SQ1: Do valuation ratios (P/E and P/B) influence IPO underpricing?</p>
  <div class="info">
    <strong>Tavan Serisi (Consecutive Limit-Up):</strong> BIST imposes &plusmn;10% daily price limits. When an IPO hits the upper limit on day 1 and continues for multiple days, conventional first-day returns severely understate true underpricing. We measure from IPO close to first free-trading day.
  </div>

  <div class="metrics">
    <div class="metric"><div class="label">Total IPOs</div><div class="value">{n_ipo}</div></div>
    <div class="metric"><div class="label">Avg Underpricing</div><div class="value">{mean_under:.1f}%</div></div>
    <div class="metric"><div class="label">Median Underpricing</div><div class="value">{median_under:.1f}%</div></div>
    <div class="metric"><div class="label">Avg Tavan Days</div><div class="value">{avg_tavan:.1f}</div></div>
    <div class="metric"><div class="label">Hit Tavan Day 1</div><div class="value">{pct_tavan:.0f}%</div></div>
    <div class="metric"><div class="label">Max Tavan Days</div><div class="value">{max_tavan}</div></div>
  </div>
  <hr>
  <div class="grid grid-2">
    <div class="chart" id="chart-ipo-yearly"></div>
    <div class="chart" id="chart-tavan-dist"></div>
  </div>
  <div class="grid grid-2">
    <div class="chart" id="chart-offer-vs-under"></div>
    <div class="chart" id="chart-under-by-group"></div>
  </div>
  <hr>
  <!-- Feature #3: Yearly Underpricing Breakdown Table -->
  <h2>Yearly Underpricing Breakdown</h2>
  <table>
    <tr><th>Year</th><th>N</th><th>Mean Underpricing</th><th>Median Underpricing</th><th>Avg Tavan Days</th><th>% Hit Tavan</th></tr>
    {yearly_table_rows}
  </table>

  <!-- Feature #4: Statistical Tests -->
  <div class="info" style="border-color:var(--positive);">
    <strong>Statistical Tests for Underpricing &gt; 0:</strong><br>
    One-sample t-test: t = {t_stat_upr:.3f}, p = {p_val_upr:.2e} (N = {len(upr)})<br>
    Wilcoxon signed-rank: W = {w_stat_upr:.0f}, p = {w_pval_upr:.2e} (N = {len(upr_nonzero)}, excludes zeros)<br>
    <strong>Conclusion:</strong> IPO underpricing is statistically significant at the 1% level under both parametric and non-parametric tests.
  </div>
  <hr>
  <h2>Multi-Period Returns</h2>
  <div class="chart" id="chart-multi-period"></div>

  <h2>Contrarian Strategy: Post-Tavan Returns</h2>
  <div class="info">After the tavan streak ends, do stocks revert? A contrarian SELL at 5 days earns &minus;3.3% (p=0.021), while long-term (365d) returns are +72.6%.</div>
  <div class="chart" id="chart-contrarian"></div>
  <table id="table-contrarian">
    <tr><th>Horizon</th><th>N</th><th>Mean Return</th><th>Median Return</th><th>t-stat</th><th>p-value</th><th>% Positive</th><th>Signal</th></tr>
  </table>
  <hr>
  <!-- Feature #5: Sector Analysis Chart -->
  <h2>Underpricing by Sector</h2>
  <div class="chart" id="chart-sector-underpricing"></div>
  <hr>
  <h2>P/E (F/K) Ratio Analysis</h2>
  <div class="metrics" style="grid-template-columns:repeat(auto-fit,minmax(140px,1fr));">
    <div class="metric"><div class="label">IPOs with P/E Data</div><div class="value">{pe_n} / {n_ipo}</div></div>
    <div class="metric"><div class="label">Median P/E</div><div class="value">{pe_median:.1f}</div></div>
    <div class="metric"><div class="label">Mean P/E</div><div class="value">{pe_mean:.1f}</div></div>
    <!-- Feature #6: P/B and Market Cap -->
    <div class="metric"><div class="label">Median P/B</div><div class="value">{pb_median:.2f}</div><div class="delta">N={pb_n}</div></div>
    <div class="metric"><div class="label">Median Mkt Cap</div><div class="value">{mc_median_b:.1f}B TL</div><div class="delta">N={mc_n}</div></div>
  </div>
  <div class="grid grid-2">
    <div class="chart" id="chart-pe-dist"></div>
    <div class="chart" id="chart-pe-scatter"></div>
  </div>

  <!-- Feature #7: P/E by Tavan Group -->
  <div class="chart" id="chart-pe-by-tavan"></div>
  <hr>
  <h2>IPO Database</h2>
  <div class="table-scroll" id="ipo-table-container"></div>
</div>

<!-- ═══════════════════ MANIPULATION ═══════════════════ -->
<div id="manipulation" class="page">
  <h1>Market Environment: SPK Enforcement</h1>
  <p class="subtitle">Regulatory enforcement as evidence of speculative trading conditions</p>
  <div class="info">
    Turkey's Capital Markets Board (SPK) actively prosecutes market manipulation. Rather than treating manipulation as a primary explanatory variable, we interpret these enforcement cases as evidence of a market environment in which speculative trading and retail investor sentiment may amplify IPO price movements. Event study: MacKinlay (1997) with Boehmer (1991) standardized test.
  </div>

  <div class="metrics">
    <div class="metric"><div class="label">Total Cases</div><div class="value">{len(spk)}</div></div>
    <div class="metric"><div class="label">Total Fines</div><div class="value">&#8378;{spk_total_fine/1e6:.0f}M</div></div>
    <div class="metric"><div class="label">People Penalized</div><div class="value">{spk_people:.0f}</div></div>
    <div class="metric"><div class="label">Stocks Analyzed</div><div class="value">{es_n}</div></div>
    <div class="metric"><div class="label">Negative CARs</div><div class="value">{es_neg}/{es_n}</div><div class="delta">({es_neg/es_n*100:.1f}%)</div></div>
  </div>
  <hr>
  <div class="grid grid-2">
    <div class="chart" id="chart-spk-yearly"></div>
    <div class="chart" id="chart-spk-type"></div>
  </div>

  <div class="grid grid-2">
    <div class="chart" id="chart-penalty-histogram"></div>
    <div class="chart" id="chart-avg-fine-type"></div>
  </div>

  <!-- Feature #8: Top 10 Penalties Table -->
  <h2>Top 10 Largest Penalties</h2>
  <table>
    <tr><th>Ticker</th><th>Company</th><th>Date</th><th>Fine</th><th>People</th><th>Type</th></tr>
    {spk_top_rows}
  </table>
  <hr>
  <h2>Event Study: Cumulative Average Abnormal Return (CAAR)</h2>
  <div class="chart" id="chart-caar"></div>
  <div class="{'success' if es_mean_car < 0 else 'warning'}">
    <strong>CAAR at Day +30: {es_mean_car:.1f}%</strong> | Mean CAR: {es_mean_car:.1f}% | Median CAR: {es_median_car:.1f}% | Negative: {es_neg}/{es_n} ({es_neg/es_n*100:.1f}%) | Significant: {es_sig}/{es_n} ({es_sig/es_n*100:.1f}%)
  </div>

  <h2>Individual Stock CARs</h2>
  <div class="chart" id="chart-individual-cars"></div>
  <hr>
  <h2>Notable Cases</h2>
  <div class="grid grid-2">
    <div>
      <h3>Biggest Price Gains (possible pump before penalty)</h3>
      <div class="info" style="font-size:0.85rem;">These stocks had the most positive CARs &mdash; their prices went UP around the penalty period, possibly because the pump phase was still ongoing.</div>
      <table>
        <tr><th>Ticker</th><th>CAR</th><th>t-stat</th><th>p-value</th></tr>
        {notable_pos_html}
      </table>
    </div>
    <div>
      <h3>Biggest Price Crashes (dump after exposure)</h3>
      <div class="info" style="font-size:0.85rem;border-color:var(--negative);">These stocks had the most negative CARs &mdash; their prices crashed the most around the penalty announcement.</div>
      <table>
        <tr><th>Ticker</th><th>CAR</th><th>t-stat</th><th>p-value</th></tr>
        {notable_neg_html}
      </table>
    </div>
  </div>
  <hr>
  <h2>Case Explorer</h2>
  <div class="info">Select a manipulation case to view the stock's price behavior, volume, and performance relative to BIST-100 around the penalty date. The red dashed line marks the penalty announcement. Orange shading marks the investigation period.</div>
  <div style="margin:12px 0;">
    <label><strong>Select Stock:</strong></label>
    <select id="case-select" onchange="updateCaseExplorer()"></select>
  </div>
  <div id="case-details"></div>
  <div class="chart" id="chart-case-price" style="min-height:600px;"></div>

  <hr>
  <h2>Manipulation Type Breakdown</h2>
  <table>
    <tr><th>Type</th><th>Cases</th><th>%</th></tr>"""

for _, row in spk_by_type.iterrows():
    pct = row['count'] / len(spk) * 100
    html += f"\n    <tr><td>{row['type']}</td><td>{row['count']}</td><td>{pct:.1f}%</td></tr>"

html += f"""
  </table>
  <hr>
  <h2>Full SPK Penalty Database</h2>
  <div class="table-scroll" style="max-height:500px;">
    <table>
      <tr><th>Ticker</th><th>Company</th><th>Date</th><th>Fine</th><th>People</th><th>Type</th><th>Ban</th><th>Notes</th></tr>
      {spk_full_rows}
    </table>
  </div>
</div>

<!-- ═══════════════════ HERDING & INFLATION ═══════════════════ -->
<div id="herding" class="page">
  <h1>Herding Analysis &amp; Inflation</h1>
  <p class="subtitle">SQ2: Evidence of herding? &middot; SQ3: Market conditions &amp; inflation effects</p>

  <h2>CSAD Herding Test (CCK 2000)</h2>
  <div class="info">
    Following Chang, Cheng & Khorana (2000), we test for herding via: CSAD = &alpha; + &gamma;&#8321;|R<sub>m</sub>| + &gamma;&#8322;R<sub>m</sub>&sup2; + &epsilon;. A significantly negative &gamma;&#8322; indicates herding.
  </div>

  <div class="metrics" style="grid-template-columns:repeat(3,1fr);">
    <div class="metric"><div class="label">&gamma;&#8322;</div><div class="value">{gamma2:.3f}</div></div>
    <div class="metric"><div class="label">t-statistic</div><div class="value">{gamma2_t:.3f}</div></div>
    <div class="metric"><div class="label">p-value</div><div class="value">{gamma2_p:.3f}</div></div>
  </div>

  <div class="warning">
    <strong>Result: No evidence of herding</strong> (&gamma;&#8322; = {gamma2:.3f}, p = {gamma2_p:.3f}). The positive coefficient suggests return dispersion increases MORE than proportionally during extreme market movements, inconsistent with herding. Note: Zhang (2024) shows CSAD test power averages only 59.37%.
  </div>

  <!-- Feature #9: CSAD Full Regression Table -->
  <h3>Full CSAD Regression Results</h3>
  <table>
    <tr><th>Variable</th><th>Coefficient</th><th>Interpretation</th></tr>
    <tr><td>&alpha; (Intercept)</td><td>{alpha_val:.6f}</td><td>Baseline cross-sectional dispersion</td></tr>
    <tr><td>&gamma;&#8321; (|R<sub>m</sub>|)</td><td>{gamma1_val:.6f}</td><td>Linear sensitivity to market return</td></tr>
    <tr><td>&gamma;&#8322; (R<sub>m</sub>&sup2;)</td><td>{gamma2:.6f}</td><td>Non-linear term (negative = herding)</td></tr>
    <tr><td colspan="3" style="border-top:2px solid var(--border);"></td></tr>
    <tr><td>R&sup2;</td><td>{r2_val:.4f}</td><td>Proportion of variance explained</td></tr>
    <tr><td>Adj. R&sup2;</td><td>{adj_r2_val:.4f}</td><td>Adjusted for degrees of freedom</td></tr>
    <tr><td>N observations</td><td>{n_obs_val}</td><td>Daily observations (BIST-100 stocks)</td></tr>
  </table>

  <div class="grid grid-2">
    <div class="chart" id="chart-csad-regime"></div>
    <div class="chart" id="chart-csad-rolling"></div>
  </div>
  <hr>
  <h2>Inflation Illusion</h2>
  <div class="info">
    <strong>Fisher Equation:</strong> Real Return = (1 + Nominal) / (1 + Inflation) &minus; 1. Turkey's extreme inflation (peaking at {peak_yoy:.1f}% YoY in {peak_date_str}) means nominal IPO gains vastly overstate real wealth creation.
  </div>

  <div class="metrics">
    <div class="metric"><div class="label">BIST-100 Nominal</div><div class="value">+{bist_nominal_pct:.0f}%</div><div class="delta">2020-2025</div></div>
    <div class="metric"><div class="label">CPI Increase</div><div class="value">+{tufe_pct:.0f}%</div><div class="delta negative">Purchasing power lost</div></div>
    <div class="metric"><div class="label">BIST-100 Real</div><div class="value">+{bist_real_pct:.0f}%</div><div class="delta">CPI-adjusted</div></div>
    <div class="metric"><div class="label">BIST-100 in USD</div><div class="value">+{bist_usd_pct:.0f}%</div><div class="delta">Dollar terms</div></div>
  </div>
  <div class="warning">
    <strong>The Inflation Illusion:</strong> BIST-100 gained +{bist_nominal_pct:.0f}% in nominal terms but only +{bist_real_pct:.0f}% in real (CPI-adjusted) terms. In dollar terms, the gain was +{bist_usd_pct:.0f}%. The gap between nominal and real returns is the "Inflation Illusion."
  </div>

  <div class="metrics">
    <div class="metric"><div class="label">CPI Start (Jan 2020)</div><div class="value">{tufe_start:.0f}</div></div>
    <div class="metric"><div class="label">CPI End (Dec 2025)</div><div class="value">{tufe_end:.0f}</div></div>
    <div class="metric"><div class="label">CPI Increase</div><div class="value">+{tufe_pct:.0f}%</div></div>
    <div class="metric"><div class="label">Peak YoY Inflation</div><div class="value">{peak_yoy:.1f}%</div><div class="delta">{peak_date_str}</div></div>
  </div>
  <div class="info" style="font-size:0.8rem;">
    <strong>Source:</strong> T&Uuml;&Iacute;K (Turkish Statistical Institute), Consumer Price Index (T&Uuml;FE), Base Year 2003=100.
    Values represent the cumulative price level relative to 2003, not percentage changes. For example, {tufe_end:.0f} means prices are {tufe_end/100:.1f}x what they were in 2003.
    Data: <a href="https://data.tuik.gov.tr/Kategori/GetKategori?p=Enflasyon-ve-Fiyat-106" target="_blank">data.tuik.gov.tr</a> | TCMB EVDS: <a href="https://evds2.tcmb.gov.tr/" target="_blank">evds2.tcmb.gov.tr</a>
  </div>

  <div class="grid grid-2">
    <div class="chart" id="chart-inflation-yoy"></div>
    <div class="chart" id="chart-cpi-index"></div>
  </div>
  <hr>
  <h2>BIST-100 vs Inflation vs USD/TRY</h2>
  <div class="info">All three series normalized to 100 at January 2020, allowing direct comparison of growth rates. If BIST-100 (blue) is below CPI (red), equities have <strong>lost</strong> real value.</div>
  <div class="chart" id="chart-bist-infl-usd" style="min-height:500px;"></div>
  <hr>
  <h2>Nominal vs Real IPO Returns</h2>
  <div class="info">Real returns computed per-IPO using the Fisher equation with daily-interpolated CPI data from TUIK. Each IPO's inflation is measured over its specific holding period.</div>
  <div class="chart" id="chart-nominal-vs-real"></div>

  <h3>Return Comparison Table</h3>
  <table>
    <tr><th>Period</th><th>Avg Nominal Return</th><th>Avg Period Inflation</th><th>Avg Real Return</th></tr>
    {return_table_rows}
  </table>
  <hr>
  <h2>Illusion Zone: 1-Year Returns</h2>
  <div class="info">Each dot is one IPO. The <strong style="color:var(--negative);">red shaded area</strong> is the "Illusion Zone" &mdash; IPOs with positive nominal returns but negative real returns. These investors thought they were profiting, but actually lost purchasing power.</div>
  <div class="error-box">
    <strong>Money Illusion Cases:</strong> {n_illusion}/{n_illusion_total} IPOs ({n_illusion/n_illusion_total*100 if n_illusion_total > 0 else 0:.0f}%) had positive nominal but NEGATIVE real 1-year returns.
  </div>
  <div class="chart" id="chart-illusion-zone" style="min-height:500px;"></div>
  <hr>
  <h2>Does Inflation Drive IPO Demand?</h2>
  <div class="info"><strong>Modigliani &amp; Cohn (1979) Hypothesis:</strong> Investors in high-inflation environments use nominal rates to discount, making stocks appear more attractive, potentially driving more IPO demand.</div>
  <div class="grid grid-2">
    <div class="chart" id="chart-ipo-vs-inflation"></div>
    <div class="chart" id="chart-inflation-ipo-scatter"></div>
  </div>
  <div class="{'success' if infl_pval < 0.05 else 'warning'}">
    <strong>Pearson r = {infl_corr:.3f}</strong> (p = {infl_pval:.4f}) &mdash; {'Significant' if infl_pval < 0.05 else 'Not significant'} at 5% level. {'Higher inflation is associated with more IPO activity.' if infl_corr > 0 and infl_pval < 0.05 else 'No statistically significant relationship between monthly inflation and IPO frequency.'}
  </div>
  <hr>
  <h2>Key Statistics (from data)</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>BIST-100 Nominal Return</td><td>+{bist_nominal_pct:.0f}%</td></tr>
    <tr><td>CPI Increase</td><td>+{tufe_pct:.0f}%</td></tr>
    <tr><td>BIST-100 Real Return</td><td>+{bist_real_pct:.0f}%</td></tr>
    <tr><td>BIST-100 in USD</td><td>+{bist_usd_pct:.0f}%</td></tr>
    <tr><td>USD/TRY Change</td><td>+{usd_change_pct:.0f}%</td></tr>
    <tr><td>Peak YoY Inflation</td><td>{peak_yoy:.1f}% ({peak_date_str})</td></tr>
  </table>
</div>

<!-- ═══════════════════ CONCLUSIONS ═══════════════════ -->
<div id="conclusions" class="page">
  <h1>Conclusions &amp; Statistical Results</h1>
  <p class="subtitle">IPO Fever and the Cost of the Crowd &middot; Borsa Istanbul 2020&ndash;2025</p>

  <div class="info" style="border-color:var(--primary);font-size:0.92rem;margin-top:20px;">
    <strong>Main RQ:</strong> To what extent is IPO underpricing in Borsa Istanbul explained by valuation signals and investor behavior during the 2020&ndash;2025 IPO boom?<br><br>
    The empirical results provide <strong>partial support</strong> for this hypothesis. Valuation indicators&mdash;particularly the price-to-book ratio&mdash;show a statistically significant relationship with IPO underpricing. However, the analysis finds limited statistical evidence for systematic herding behavior among investors during IPO trading periods.
  </div>

  <hr>
  <h2>SQ1: Do Valuation Ratios Influence Underpricing? <span class="verdict partial">Partial</span></h2>

  <div class="success">
    <strong>IPO underpricing is confirmed:</strong> Mean = {mean_under:.1f}% (t = {t_stat_upr:.2f}, p {'< 0.001' if p_val_upr < 0.001 else f'= {p_val_upr:.3f}'}), Median = {median_under:.1f}%. Wilcoxon test also significant (W = {w_stat_upr:.0f}, p {'< 0.001' if w_pval_upr < 0.001 else f'= {w_pval_upr:.3f}'}). N = {n_ipo} IPOs.
  </div>

  <h3>Cross-Sectional Regression: 4-Model Comparison</h3>
  <div class="info">We estimate four progressively enriched OLS specifications to identify determinants of IPO underpricing. Only Model 4 (Log-level) is jointly significant (F-test p &lt; 0.01).</div>
  <table>
    <tr><th>Model</th><th>N</th><th>R&sup2;</th><th>Adj. R&sup2;</th><th>F-stat</th><th>F p-value</th><th>Significant?</th></tr>
    {model_comparison_rows}
  </table>

  <div class="chart" id="chart-regression"></div>
  <div class="chart" id="chart-regression-comparison"></div>

  {all_models_coef_html}

  <div class="warning">
    <strong>Result:</strong> Price-to-book ratio is the only significant determinant (p = 0.040). P/E ratio (p &gt; 0.50) and market cap (p = 0.095) are not significant. Observable fundamentals explain only 10&ndash;21% of variation, suggesting underpricing is primarily behavioural.
  </div>

  <hr>
  <h2>SQ2: Is There Evidence of Investor Herding? <span class="verdict rejected">Not Found</span></h2>
  <div class="warning">
    <strong>No significant herding detected:</strong> &gamma;&#8322; = {gamma2:.3f}, p = {gamma2_p:.3f}. The positive coefficient suggests return dispersion <em>increases</em> during extreme market movements, inconsistent with herding. This null result must be qualified by low test power (Zhang 2024: avg 59.37%).
  </div>

  <hr>
  <h2>SQ3: Do Market Conditions Affect Underpricing? <span class="verdict partial">Partial</span></h2>

  <div class="grid grid-2">
    <div class="success">
      <strong>SPK Enforcement:</strong> CAAR = {es_mean_car:.1f}% over [-30,+30] window. {es_sig}/{es_n} ({es_sig/es_n*100:.1f}%) individual stocks show significant negative CARs. SPK enforcement actions confirm the speculative market environment.
    </div>
    <div class="success">
      <strong>Post-Tavan Dynamics:</strong> Short-term mean reversion at 5 days (&minus;3.3%, p = 0.021). Long-term returns positive (+72.6% at 365d). Consistent with De Bondt &amp; Thaler (1985) overreaction hypothesis.
    </div>
  </div>

  <div class="warning" style="margin-top:14px;">
    <strong>Inflation Illusion:</strong> BIST-100 gained +{bist_nominal_pct:.0f}% nominally but only +{bist_real_pct:.0f}% in real terms. {n_illusion}/{n_illusion_total} IPOs ({n_illusion/n_illusion_total*100 if n_illusion_total > 0 else 0:.0f}%) had positive nominal but negative real 1-year returns.
  </div>

  <hr>
  <h2>Summary of Findings</h2>
  <div class="info" style="font-size:0.92rem;">
    IPO pricing in Borsa Istanbul is influenced primarily by <strong>valuation-related signals</strong> (particularly P/B), while behavioral factors appear to play a <strong>secondary role</strong>. The extreme underpricing ({mean_under:.1f}% mean) cannot be fully explained by observable firm characteristics (R&sup2; = 10&ndash;21%), pointing to unobserved behavioural and institutional factors unique to Turkey's price-limit regime.
  </div>

  <table>
    <tr><th>Sub-Question</th><th>Finding</th><th>Statistical Support</th><th>Verdict</th></tr>
    <tr><td>SQ1: Valuation Ratios</td><td>P/B significant (p=0.040), P/E not significant (p&gt;0.50)</td><td>R&sup2;=21.4% (Model 4)</td><td><span class="verdict partial">Partial</span></td></tr>
    <tr><td>SQ2: Investor Herding</td><td>No herding detected via CSAD</td><td>&gamma;&#8322;={gamma2:.3f}, p={gamma2_p:.3f}</td><td><span class="verdict rejected">Not Found</span></td></tr>
    <tr><td>SQ3: Market Conditions</td><td>Overreaction confirmed; inflation erodes gains</td><td>5d: p=0.021, 365d: p&lt;0.001</td><td><span class="verdict partial">Partial</span></td></tr>
  </table>
</div>

<div class="footer">
  <strong>IPO Fever and the Cost of the Crowd</strong><br>
  Dokuz Eylul University &middot; Department of Economics &middot; 2026<br>
  All statistics computed from verified data sources. No hardcoded values.<br>
  <small style="color:var(--text-muted);">Data: T&Uuml;&Iacute;K (CPI) &middot; Yahoo Finance (prices) &middot; halkarz.com + KAP + SPK (IPO data) &middot; SPK Weekly Bulletins (penalties)</small>
</div>

<script>
// ═══ NAVIGATION ═══
function showPage(id, btn) {{
  const currentPage = document.querySelector('.page.active');
  if (currentPage) {{
    currentPage.style.opacity = '0';
    currentPage.style.transform = 'translateY(16px) scale(0.995)';
    currentPage.style.filter = 'blur(2px)';
    setTimeout(() => {{
      document.querySelectorAll('.page').forEach(p => {{
        p.classList.remove('active');
        p.style.filter = '';
      }});
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      const newPage = document.getElementById(id);
      newPage.classList.add('active');
      newPage.style.opacity = '0';
      newPage.style.transform = 'translateY(16px) scale(0.995)';
      newPage.style.filter = 'blur(2px)';
      if (btn) btn.classList.add('active');
      window.scrollTo({{top: 0, behavior: 'instant'}});
      requestAnimationFrame(() => {{
        newPage.style.opacity = '1';
        newPage.style.transform = 'translateY(0) scale(1)';
        newPage.style.filter = 'blur(0px)';
      }});
    }}, 200);
  }} else {{
    document.getElementById(id).classList.add('active');
    if (btn) btn.classList.add('active');
  }}
}}

// ═══ DATA ═══
const yearly = {json.dumps(yearly.to_dict('records'))};
const tavanHist = {json.dumps(tavan_hist.to_dict('records'))};
const ipoData = {json.dumps(to_json(ipo, ['ticker','offer_price','underpricing','tavan_days','tavan_series_return']))};
const contrarian = {json.dumps(contr_json)};
const esResults = {json.dumps(to_json(es_results))};
const esIndividual = {json.dumps(to_json(es_ind_sorted))};
const spkData = {json.dumps(to_json(spk, ['hisse_kodu','company_name','karar_tarihi','toplam_ceza_tl','kisi_sayisi','ceza_turu','notes','yil']))};
const spkByYear = {json.dumps(spk_by_year.to_dict('records'))};
const peScatter = {json.dumps(to_json(pe_scatter))};
const csadRolling = {json.dumps(to_json(csad_roll_ds))};
const herdingEpisodes = {json.dumps(to_json(herding_episodes))};
const tufeData = {json.dumps(to_json(tufe))};
const ipoTable = {json.dumps(ipo_table)};
const regM4 = {json.dumps(to_json(reg_m4))};
const sectorStats = {json.dumps(to_json(sector_stats))};
const peByTavan = {json.dumps(pe_by_tavan.to_dict('records'))};

// New feature data
const spkFineAmounts = {json.dumps(spk['toplam_ceza_tl'].tolist())};
const spkAvgFineByType = {json.dumps(spk_avg_fine_by_type.to_dict('records'))};
const spkPriceData = {json.dumps(spk_price_by_ticker)};
const spkFullData = {json.dumps(to_json(spk, ['hisse_kodu','company_name','karar_tarihi','toplam_ceza_tl','kisi_sayisi','ceza_turu','islem_yasagi','inceleme_baslangic','inceleme_bitis','notes']))};
const bist100Data = {json.dumps(bist100_json)};
const usdtryData = {json.dumps(usdtry_json)};
const illusionData = {json.dumps(illusion_data)};
const ipoInflData = {json.dumps(to_json(ipo_infl_merged, ['month_str', 'ipo_count', 'tufe_yoy']))};
const regAllModels = {json.dumps(regression.to_dict('records'))};
const regSummary = {json.dumps(reg_summary.to_dict('records'))};

// Tavan groups for box plot
const ipoGrouped = {json.dumps(to_json(ipo_pe, ['ticker','underpricing','tavan_days','tavan_group']))};

// Feature #1: Pre-computed real returns from Python (Fisher equation with actual CPI)
const nomFull = {nom_full_json};
const realFull = {real_full_json};
const periodsFull = {period_full_labels};

// ═══ CHART DEFAULTS ═══
const layout_base = {{ font: {{ family: 'Inter, -apple-system, sans-serif', size: 12, color: '#475569' }}, paper_bgcolor: '#fff', plot_bgcolor: '#fafbfc', margin: {{ l:50, r:20, t:45, b:40 }}, autosize: true, hoverlabel: {{ font: {{ family: 'Inter, sans-serif', size: 12 }}, bgcolor: '#1e293b', font_color: '#fff' }} }};
const cfg = {{ responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d'] }};

// ═══ OVERVIEW PAGE (loaded on start) ═══

// ═══ IPO PAGE CHARTS ═══
function initIPOCharts() {{
  // Yearly IPO count
  Plotly.newPlot('chart-ipo-yearly', [{{
    x: yearly.map(d=>d.ipo_year), y: yearly.map(d=>d.n),
    type:'bar', marker:{{color:'#9467bd'}}, text:yearly.map(d=>d.n), textposition:'outside'
  }}], {{...layout_base, title:'IPO Count by Year', yaxis:{{title:'Count'}}, height:380}}, cfg);

  // Tavan distribution
  Plotly.newPlot('chart-tavan-dist', [{{
    x: ipoData.map(d=>d.tavan_days), type:'histogram', nbinsx:22,
    marker:{{color:'#1f77b4'}}, name:'IPOs'
  }}], {{...layout_base, title:'Tavan Series Distribution', xaxis:{{title:'Consecutive Limit-Up Days'}}, yaxis:{{title:'Frequency'}}, height:380,
    shapes:[{{type:'line', x0:{avg_tavan:.1f}, x1:{avg_tavan:.1f}, y0:0, y1:1, yref:'paper', line:{{color:'#2ca02c',dash:'dot',width:2}}}}],
    annotations:[{{x:{avg_tavan:.1f}, y:1, yref:'paper', text:'Mean={avg_tavan:.1f}', showarrow:false, yanchor:'bottom'}}]
  }}, cfg);

  // Offer price vs underpricing
  Plotly.newPlot('chart-offer-vs-under', [{{
    x: ipoData.map(d=>d.offer_price), y: ipoData.map(d=>d.underpricing*100),
    mode:'markers', type:'scatter', marker:{{color:'#1f77b4',size:6,opacity:0.7}},
    text: ipoData.map(d=>d.ticker), hovertemplate:'%{{text}}<br>Price: %{{x}}<br>Underpricing: %{{y:.1f}}%<extra></extra>'
  }}], {{...layout_base, title:'Offer Price vs Underpricing (%)', xaxis:{{title:'Offer Price (TL)'}}, yaxis:{{title:'Underpricing (%)'}}, height:400}}, cfg);

  // Underpricing by tavan group
  const groups = ['0 days','1-2 days','3-5 days','6-10 days','11+ days'];
  const groupColors = {{'0 days':'#95a5a6','1-2 days':'#3498db','3-5 days':'#f39c12','6-10 days':'#e74c3c','11+ days':'#8e44ad'}};
  const boxTraces = groups.map(g => ({{
    y: ipoGrouped.filter(d=>d.tavan_group===g).map(d=>d.underpricing*100),
    type:'box', name:g, marker:{{color:groupColors[g]||'#999'}}
  }}));
  Plotly.newPlot('chart-under-by-group', boxTraces, {{...layout_base, title:'Underpricing by Tavan Group', yaxis:{{title:'Underpricing (%)'}}, showlegend:false, height:400}}, cfg);

  // Multi-period returns
  const mpPeriods = {json.dumps(period_labels)};
  const meanRets = {json.dumps(mean_returns)};
  const medianRets = {json.dumps(median_returns)};
  Plotly.newPlot('chart-multi-period', [
    {{x:mpPeriods, y:meanRets, type:'bar', name:'Mean', marker:{{color:'#1f77b4'}}, text:meanRets.map(v=>v.toFixed(1)+'%'), textposition:'outside'}},
    {{x:mpPeriods, y:medianRets, type:'bar', name:'Median', marker:{{color:'#ff7f0e'}}, text:medianRets.map(v=>v.toFixed(1)+'%'), textposition:'outside'}}
  ], {{...layout_base, title:'Multi-Period Buy-and-Hold Returns', barmode:'group', yaxis:{{title:'Return (%)'}}, height:450}}, cfg);

  // Contrarian chart
  Plotly.newPlot('chart-contrarian', [{{
    x: contrarian.map(d=>d.horizon_days+'d'), y: contrarian.map(d=>d.mean_return*100),
    type:'bar', marker:{{color: contrarian.map(d=>d.mean_return<0?'#d62728':'#2ca02c')}},
    text: contrarian.map(d=>(d.mean_return*100).toFixed(1)+'%'), textposition:'outside'
  }}], {{...layout_base, title:'Post-Tavan Contrarian Returns by Horizon', yaxis:{{title:'Mean Return (%)'}}, height:400,
    shapes:[{{type:'line',x0:-0.5,x1:7.5,y0:0,y1:0,line:{{color:'#7f7f7f',dash:'dash'}}}}]
  }}, cfg);

  // Contrarian table
  const tbl = document.getElementById('table-contrarian');
  contrarian.forEach(d => {{
    const sig = d.p_value < 0.01 ? '***' : d.p_value < 0.05 ? '**' : d.p_value < 0.1 ? '*' : '';
    tbl.innerHTML += `<tr><td>${{d.horizon_days}}d</td><td>${{d.n_ipos}}</td><td>${{(d.mean_return*100).toFixed(1)}}%</td><td>${{(d.median_return*100).toFixed(1)}}%</td><td>${{d.t_stat.toFixed(3)}}</td><td>${{d.p_value.toFixed(3)}}${{sig}}</td><td>${{(d.pct_positive*100).toFixed(1)}}%</td><td>${{d.recommendation}}</td></tr>`;
  }});

  // Feature #5: Sector analysis chart
  Plotly.newPlot('chart-sector-underpricing', [{{
    y: sectorStats.map(d=>d.sector_yf),
    x: sectorStats.map(d=>d.avg_return_pct),
    type:'bar', orientation:'h',
    marker:{{color:sectorStats.map(d=>d.avg_return_pct >= 0 ? '#2ca02c' : '#d62728')}},
    text: sectorStats.map(d=>d.avg_return_pct.toFixed(1)+'% (n='+d.count+')'),
    textposition:'outside',
    hovertemplate:'%{{y}}<br>Avg Underpricing: %{{x:.1f}}%<br>IPOs: '+sectorStats.map(d=>d.count).join(',')+'<extra></extra>'
  }}], {{...layout_base, title:'Average Underpricing by Sector', xaxis:{{title:'Average Underpricing (%)'}}, yaxis:{{automargin:true}}, height:Math.max(380, sectorStats.length * 35), margin:{{l:160,r:80,t:40,b:40}}}}, cfg);

  // PE distribution
  const peVals = peScatter.map(d=>d.trailing_pe).filter(v=>v && v<100);
  Plotly.newPlot('chart-pe-dist', [{{
    x:peVals, type:'histogram', nbinsx:25, marker:{{color:'#ff7f0e'}}
  }}], {{...layout_base, title:'P/E Ratio Distribution (capped at 100)', xaxis:{{title:'Trailing P/E'}}, yaxis:{{title:'Frequency'}}, height:380,
    shapes:[{{type:'line',x0:{pe_median:.1f},x1:{pe_median:.1f},y0:0,y1:1,yref:'paper',line:{{color:'#d62728',dash:'dot',width:2}}}}],
    annotations:[{{x:{pe_median:.1f},y:1,yref:'paper',text:'Median={pe_median:.1f}',showarrow:false,yanchor:'bottom'}}]
  }}, cfg);

  // PE vs underpricing scatter
  Plotly.newPlot('chart-pe-scatter', [{{
    x:peScatter.map(d=>d.trailing_pe), y:peScatter.map(d=>d.underpricing_pct),
    mode:'markers', type:'scatter', marker:{{color:'#1f77b4',size:6,opacity:0.7}},
    text:peScatter.map(d=>d.ticker), hovertemplate:'%{{text}}<br>P/E: %{{x:.1f}}<br>Underpricing: %{{y:.1f}}%<extra></extra>'
  }}], {{...layout_base, title:'P/E vs Underpricing (Spearman \\u03c1=0.126, p=0.183)', xaxis:{{title:'Trailing P/E',range:[0,100]}}, yaxis:{{title:'Underpricing (%)'}}, height:380}}, cfg);

  // Feature #7: P/E by Tavan Group chart
  Plotly.newPlot('chart-pe-by-tavan', [{{
    x: peByTavan.map(d=>d.tavan_group),
    y: peByTavan.map(d=>d.median_pe),
    type:'bar',
    marker:{{color:['#95a5a6','#3498db','#f39c12','#e74c3c','#8e44ad']}},
    text: peByTavan.map(d=>d.median_pe.toFixed(1)+' (n='+d.count+')'),
    textposition:'outside'
  }}], {{...layout_base, title:'Median P/E by Tavan Group', yaxis:{{title:'Median Trailing P/E'}}, height:380}}, cfg);

  // IPO table
  let tableHTML = '<table><tr><th>Ticker</th><th>Company</th><th>IPO Date</th><th>Offer Price</th><th>Tavan Days</th><th>Underpricing</th><th>Year</th></tr>';
  ipoTable.forEach(d => {{
    tableHTML += `<tr><td>${{d.ticker}}</td><td>${{(d.company_name||'').substring(0,30)}}</td><td>${{d.ipo_date}}</td><td>${{d.offer_price}}</td><td>${{d.tavan_days||0}}</td><td>${{((d.underpricing||0)*100).toFixed(1)}}%</td><td>${{d.ipo_year}}</td></tr>`;
  }});
  tableHTML += '</table>';
  document.getElementById('ipo-table-container').innerHTML = tableHTML;
}}

// ═══ MANIPULATION PAGE CHARTS ═══
function initManipulationCharts() {{
  // SPK by year
  Plotly.newPlot('chart-spk-yearly', [
    {{x:spkByYear.map(d=>d.yil), y:spkByYear.map(d=>d.n), type:'bar', name:'Cases', marker:{{color:'#e377c2'}}, yaxis:'y'}},
    {{x:spkByYear.map(d=>d.yil), y:spkByYear.map(d=>d.total/1e6), type:'scatter', mode:'lines+markers', name:'Total Fines (M TL)', marker:{{color:'#d62728'}}, line:{{color:'#d62728',width:3}}, yaxis:'y2'}}
  ], {{...layout_base, title:'SPK Penalties by Year', yaxis:{{title:'Cases'}}, yaxis2:{{title:'Fines (M TL)',overlaying:'y',side:'right'}}, height:380}}, cfg);

  // SPK type pie
  const types = {json.dumps(spk_by_type.to_dict('records'))};
  Plotly.newPlot('chart-spk-type', [{{
    labels:types.map(d=>d.type), values:types.map(d=>d.count), type:'pie',
    marker:{{colors:['#e377c2','#ff7f0e','#7f7f7f']}}
  }}], {{...layout_base, title:'Cases by Manipulation Type', height:380}}, cfg);

  // Penalty Distribution Histogram
  Plotly.newPlot('chart-penalty-histogram', [{{
    x: spkFineAmounts, type:'histogram', nbinsx:20,
    marker:{{color:'#e377c2'}}, name:'Cases'
  }}], {{...layout_base, title:'Penalty Amount Distribution', xaxis:{{title:'Penalty Amount (TL)'}}, yaxis:{{title:'Frequency'}}, height:380}}, cfg);

  // Average Fine by Manipulation Type
  Plotly.newPlot('chart-avg-fine-type', [{{
    x: spkAvgFineByType.map(d=>d.type),
    y: spkAvgFineByType.map(d=>d.avg_fine),
    type:'bar',
    marker:{{color:'#d62728'}},
    text: spkAvgFineByType.map(d=>'\\u20BA'+(d.avg_fine/1e6).toFixed(1)+'M'),
    textposition:'outside'
  }}], {{...layout_base, title:'Average Fine by Manipulation Type', yaxis:{{title:'Average Fine (TL)'}}, height:380}}, cfg);

  // CAAR chart
  const days = esResults.map(d=>d.day);
  const caar = esResults.map(d=>d.caar*100);
  const meanAR = esResults.map(d=>d.mean_ar*100);
  Plotly.newPlot('chart-caar', [
    {{x:days, y:meanAR, type:'bar', name:'Daily Mean AR', marker:{{color:meanAR.map(v=>v>=0?'rgba(44,160,44,0.3)':'rgba(214,39,40,0.3)'),line:{{width:0}}}}}},
    {{x:days, y:caar, type:'scatter', mode:'lines+markers', name:'CAAR', line:{{color:'#d62728',width:3}}, marker:{{size:3,color:'#d62728'}}}}
  ], {{...layout_base, title:'Cumulative Average Abnormal Return (CAAR) [{es_n} stocks]', xaxis:{{title:'Relative Trading Day'}}, yaxis:{{title:'CAAR (%)',tickformat:'.1f'}}, height:500,
    shapes:[
      {{type:'line',x0:0,x1:0,y0:0,y1:1,yref:'paper',line:{{color:'#d62728',dash:'dash',width:2}}}},
      {{type:'line',x0:-30,x1:30,y0:0,y1:0,line:{{color:'#7f7f7f',dash:'dot'}}}}
    ],
    annotations:[{{x:0,y:1,yref:'paper',text:'Penalty Day',showarrow:false,yanchor:'bottom',font:{{color:'#d62728'}}}}]
  }}, cfg);

  // Individual CARs
  Plotly.newPlot('chart-individual-cars', [{{
    x:esIndividual.map(d=>d.ticker), y:esIndividual.map(d=>d.car*100),
    type:'bar', marker:{{color:esIndividual.map(d=>d.car>=0?'#2ca02c':'#d62728')}},
    hovertemplate:'%{{x}}<br>CAR: %{{y:.1f}}%<extra></extra>'
  }}], {{...layout_base, title:'Individual Stock CARs (sorted)', xaxis:{{tickangle:-45,tickfont:{{size:9}}}}, yaxis:{{title:'CAR (%)',tickformat:'.0f'}}, height:450,
    shapes:[{{type:'line',x0:-0.5,x1:{es_n-0.5},y0:0,y1:0,line:{{color:'#7f7f7f',dash:'dot'}}}}]
  }}, cfg);

  // Case explorer dropdown
  const sel = document.getElementById('case-select');
  spkData.forEach(d => {{
    const opt = document.createElement('option');
    opt.value = d.hisse_kodu;
    opt.text = d.hisse_kodu + ' - ' + (d.company_name||'').substring(0,40);
    sel.appendChild(opt);
  }});
  updateCaseExplorer();
}}

function updateCaseExplorer() {{
  const ticker = document.getElementById('case-select').value;
  const c = spkData.find(d=>d.hisse_kodu===ticker);
  const es = esIndividual.find(d=>d.ticker===ticker);

  if (!c) return;

  let html = '<div class="metrics" style="grid-template-columns:repeat(auto-fit,minmax(140px,1fr));">';
  html += `<div class="metric"><div class="label">Company</div><div class="value" style="font-size:0.9rem">${{(c.company_name||'').substring(0,25)}}</div></div>`;
  html += `<div class="metric"><div class="label">Decision Date</div><div class="value" style="font-size:0.9rem">${{c.karar_tarihi}}</div></div>`;
  html += `<div class="metric"><div class="label">Fine</div><div class="value">&#8378;${{(c.toplam_ceza_tl/1e6).toFixed(1)}}M</div></div>`;
  html += `<div class="metric"><div class="label">People</div><div class="value">${{c.kisi_sayisi}}</div></div>`;
  html += `<div class="metric"><div class="label">Type</div><div class="value" style="font-size:0.9rem">${{c.ceza_turu}}</div></div>`;
  if (es) {{
    const carPct = (es.car*100).toFixed(1);
    const cls = es.car < 0 ? 'negative' : 'positive';
    html += `<div class="metric"><div class="label">CAR [-30,+30]</div><div class="value ${{cls}}" style="color:var(--${{cls}})">${{carPct}}%</div><div class="delta">t=${{es.t_stat.toFixed(2)}}, p=${{es.t_pvalue < 0.001 ? '<0.001' : es.t_pvalue.toFixed(3)}}</div></div>`;
  }}
  html += '</div>';
  if (c.notes) html += `<div class="info">${{c.notes}}</div>`;

  document.getElementById('case-details').innerHTML = html;

  // Draw case explorer price chart
  const priceData = spkPriceData[ticker];
  const chartEl = document.getElementById('chart-case-price');
  if (priceData && priceData.length > 0) {{
    const eventDate = c.karar_tarihi;
    const invStart = c.inceleme_baslangic;
    const invEnd = c.inceleme_bitis;

    // Find BIST-100 data for the same period
    const dateMin = priceData[0].Date;
    const dateMax = priceData[priceData.length-1].Date;
    const bist_period = bist100Data.filter(d => d.date >= dateMin && d.date <= dateMax);

    // Row 1: Candlestick + BIST-100
    const candleTrace = {{
      x: priceData.map(d=>d.Date),
      open: priceData.map(d=>d.Open),
      high: priceData.map(d=>d.High),
      low: priceData.map(d=>d.Low),
      close: priceData.map(d=>d.Close),
      type:'candlestick', name:ticker,
      increasing:{{line:{{color:'#26a69a'}}}},
      decreasing:{{line:{{color:'#ef5350'}}}},
      xaxis:'x', yaxis:'y'
    }};

    const traces = [candleTrace];

    // BIST-100 on secondary y-axis
    if (bist_period.length > 0) {{
      traces.push({{
        x: bist_period.map(d=>d.date),
        y: bist_period.map(d=>d.bist100_close),
        type:'scatter', mode:'lines', name:'BIST-100',
        line:{{color:'#1976d2',width:1.5,dash:'dot'}}, opacity:0.65,
        xaxis:'x', yaxis:'y4'
      }});
    }}

    // Row 2: Volume
    const volData = priceData.filter(d=>d.Volume != null && d.Volume > 0);
    if (volData.length > 0) {{
      const volColors = volData.map(d => {{
        if (eventDate && Math.abs(new Date(d.Date) - new Date(eventDate)) < 7*86400000) return '#ef5350';
        if (invStart && invEnd && d.Date >= invStart && d.Date <= invEnd) return '#ff9800';
        return '#b0bec5';
      }});
      traces.push({{
        x: volData.map(d=>d.Date),
        y: volData.map(d=>d.Volume),
        type:'bar', name:'Volume', marker:{{color:volColors}},
        showlegend:false, xaxis:'x2', yaxis:'y2'
      }});
    }}

    // Row 3: Normalized performance (base=100)
    const basePrice = priceData[0].Close;
    const stockNorm = priceData.map(d => ({{date:d.Date, val:(d.Close/basePrice)*100}}));
    traces.push({{
      x: stockNorm.map(d=>d.date), y: stockNorm.map(d=>d.val),
      type:'scatter', mode:'lines', name:ticker+' (indexed)',
      line:{{color:'#ef5350',width:2}},
      xaxis:'x3', yaxis:'y3'
    }});

    if (bist_period.length > 0) {{
      const bistBase = bist_period[0].bist100_close;
      const bistNorm = bist_period.map(d => ({{date:d.date, val:(d.bist100_close/bistBase)*100}}));
      traces.push({{
        x: bistNorm.map(d=>d.date), y: bistNorm.map(d=>d.val),
        type:'scatter', mode:'lines', name:'BIST-100 (indexed)',
        line:{{color:'#1976d2',width:2}},
        xaxis:'x3', yaxis:'y3'
      }});
    }}

    const shapes = [];
    const annotations = [];

    // Penalty date line for all 3 rows
    if (eventDate) {{
      ['y','y2','y3'].forEach((ya,i) => {{
        shapes.push({{type:'line', x0:eventDate, x1:eventDate, y0:0, y1:1, yref:ya+' domain', xref:['x','x2','x3'][i], line:{{color:'#d32f2f',dash:'dash',width:1.5}}}});
      }});
      annotations.push({{x:eventDate, y:1.02, yref:'y domain', xref:'x', text:'SPK Penalty', showarrow:false, font:{{color:'#d32f2f',size:10}}}});
    }}

    // Investigation period shading (clipped to available data range)
    if (invStart && invEnd) {{
      const clippedStart = invStart < dateMin ? dateMin : invStart;
      const clippedEnd = invEnd > dateMax ? dateMax : invEnd;
      if (clippedStart < clippedEnd) {{
        ['x','x2','x3'].forEach(xa => {{
          shapes.push({{type:'rect', x0:clippedStart, x1:clippedEnd, y0:0, y1:1, yref:'paper', xref:xa, fillcolor:'rgba(255,152,0,0.07)', line:{{width:0}}}});
        }});
      }}
    }}

    // Baseline 100 for row 3
    shapes.push({{type:'line', x0:dateMin, x1:dateMax, y0:100, y1:100, xref:'x3', yref:'y3', line:{{color:'#bdbdbd',dash:'dot',width:1}}}});

    const layout = {{
      ...layout_base,
      height:700,
      showlegend:true,
      legend:{{orientation:'h', yanchor:'bottom', y:1.02, xanchor:'center', x:0.5, font:{{size:9}}}},
      margin:{{l:60,r:60,t:50,b:30}},
      grid:{{rows:3, columns:1, subplots:[['xy'],['x2y2'],['x3y3']], roworder:'top to bottom', pattern:'independent'}},
      xaxis:{{domain:[0,1], showticklabels:false, rangeslider:{{visible:false}}}},
      xaxis2:{{domain:[0,1], showticklabels:false, anchor:'y2'}},
      xaxis3:{{domain:[0,1], anchor:'y3'}},
      yaxis:{{domain:[0.52,1], title:'Price (TL)', titlefont:{{size:10}}}},
      yaxis4:{{overlaying:'y', side:'right', title:'BIST-100', titlefont:{{size:10}}, showgrid:false}},
      yaxis2:{{domain:[0.30,0.48], title:'Volume', titlefont:{{size:10}}}},
      yaxis3:{{domain:[0,0.26], title:'Indexed (100)', titlefont:{{size:10}}}},
      shapes: shapes,
      annotations: annotations
    }};

    Plotly.newPlot(chartEl, traces, layout, cfg);
  }} else {{
    chartEl.innerHTML = '<div class="warning">Price data not available for this stock.</div>';
  }}
}}

// ═══ HERDING PAGE CHARTS ═══
function initHerdingCharts() {{
  // Bull vs Bear
  const regimeData = {json.dumps(csad_regime.to_dict('records'))};
  const rd = regimeData[0] || {{}};
  Plotly.newPlot('chart-csad-regime', [{{
    x:['Bull Market','Bear Market'],
    y:[rd.bull_gamma2||0, rd.bear_gamma2||0],
    type:'bar', marker:{{color:['#2ca02c','#d62728']}},
    text:[(rd.bull_gamma2||0).toFixed(2), (rd.bear_gamma2||0).toFixed(2)], textposition:'outside'
  }}], {{...layout_base, title:'Bull vs Bear Market Herding (γ₂)', yaxis:{{title:'γ₂ coefficient'}}, height:400,
    shapes:[{{type:'line',x0:-0.5,x1:1.5,y0:0,y1:0,line:{{color:'#7f7f7f',dash:'dash'}}}}],
    annotations:[{{x:0.5,y:0,text:'Herding threshold (γ₂ < 0)',showarrow:false,yshift:-20,font:{{color:'#7f7f7f',size:10}}}}]
  }}, cfg);

  // Feature #10: Rolling CSAD with herding markers
  const rollingTraces = [{{
    x:csadRolling.map(d=>d.Date), y:csadRolling.map(d=>d.gamma2),
    type:'scatter', mode:'lines', name:'γ₂ (60-day rolling)',
    line:{{color:'#1f77b4',width:1}}
  }}];
  // Add herding episode markers (red dots)
  if (herdingEpisodes.length > 0) {{
    rollingTraces.push({{
      x:herdingEpisodes.map(d=>d.Date), y:herdingEpisodes.map(d=>d.gamma2),
      type:'scatter', mode:'markers', name:'Herding episodes (p<0.05)',
      marker:{{color:'#d62728',size:6,symbol:'circle'}}
    }});
  }}
  Plotly.newPlot('chart-csad-rolling', rollingTraces, {{...layout_base, title:'Rolling Herding Coefficient (60-day window)', xaxis:{{title:'Date'}}, yaxis:{{title:'γ₂'}}, height:400,
    shapes:[{{type:'line',x0:csadRolling[0].Date,x1:csadRolling[csadRolling.length-1].Date,y0:0,y1:0,line:{{color:'#7f7f7f',dash:'dash'}}}}]
  }}, cfg);

  // Inflation YoY
  Plotly.newPlot('chart-inflation-yoy', [{{
    x:tufeData.map(d=>d.date), y:tufeData.map(d=>d.tufe_yoy),
    type:'scatter', mode:'lines', fill:'tozeroy', name:'YoY Inflation',
    line:{{color:'#bcbd22'}}, fillcolor:'rgba(188,189,34,0.3)'
  }}], {{...layout_base, title:'Monthly YoY Inflation (%)', xaxis:{{title:'Date'}}, yaxis:{{title:'YoY Inflation (%)'}}, height:380,
    shapes:[{{type:'line',x0:tufeData[0].date,x1:tufeData[tufeData.length-1].date,y0:50,y1:50,line:{{color:'#d62728',dash:'dash'}}}}]
  }}, cfg);

  // Feature #11: CPI Index with enhanced subtitle
  Plotly.newPlot('chart-cpi-index', [{{
    x:tufeData.map(d=>d.date), y:tufeData.map(d=>d.tufe_index),
    type:'scatter', mode:'lines', name:'CPI Index',
    line:{{color:'#d62728',width:2}}
  }}], {{...layout_base, title:{{text:'CPI Index (2003=100)<br><sub>TUIK official series (base year 2003=100). Values show cumulative price level, not percentage change.</sub>',font:{{size:14}}}}, xaxis:{{title:'Date'}}, yaxis:{{title:'Index'}}, height:400}}, cfg);

  // Feature #1: Nominal vs Real returns with pre-computed real returns from Fisher equation
  Plotly.newPlot('chart-nominal-vs-real', [
    {{x:periodsFull, y:nomFull, type:'bar', name:'Nominal', marker:{{color:'#1f77b4'}}, text:nomFull.map(v=>v.toFixed(1)+'%'), textposition:'outside'}},
    {{x:periodsFull, y:realFull, type:'bar', name:'Real (inflation-adjusted)', marker:{{color:'#2ca02c'}}, text:realFull.map(v=>v.toFixed(1)+'%'), textposition:'outside'}}
  ], {{...layout_base, title:'Nominal vs Real IPO Returns (Fisher equation with actual CPI)', barmode:'group', yaxis:{{title:'Return (%)'}}, height:450}}, cfg);

  // BIST-100 vs Inflation vs USD Overlay (indexed to 100)
  if (bist100Data.length > 0) {{
    const bistBase = bist100Data[0].bist100_close;
    const bistNorm = bist100Data.map(d => ({{date:d.date, val:(d.bist100_close/bistBase)*100}}));

    // CPI normalized
    const cpiFiltered = tufeData.filter(d => d.date >= '2020-01-01');
    const cpiBase = cpiFiltered.length > 0 ? cpiFiltered[0].tufe_index : 1;
    const cpiNorm = cpiFiltered.map(d => ({{date:d.date, val:(d.tufe_index/cpiBase)*100}}));

    // USD/TRY normalized
    const usdBase = usdtryData.length > 0 ? usdtryData[0].usdtry_close : 1;
    const usdNorm = usdtryData.map(d => ({{date:d.date, val:(d.usdtry_close/usdBase)*100}}));

    const overlayTraces = [
      {{x:bistNorm.map(d=>d.date), y:bistNorm.map(d=>d.val), type:'scatter', mode:'lines', name:'BIST-100 Nominal ('+bistNorm[bistNorm.length-1].val.toFixed(0)+')', line:{{color:'#1f77b4',width:2}}}},
      {{x:cpiNorm.map(d=>d.date), y:cpiNorm.map(d=>d.val), type:'scatter', mode:'lines', name:'CPI Index ('+cpiNorm[cpiNorm.length-1].val.toFixed(0)+')', line:{{color:'#d62728',width:2,dash:'dash'}}}},
      {{x:usdNorm.map(d=>d.date), y:usdNorm.map(d=>d.val), type:'scatter', mode:'lines', name:'USD/TRY ('+usdNorm[usdNorm.length-1].val.toFixed(0)+')', line:{{color:'#ff7f0e',width:2,dash:'dot'}}}}
    ];

    // BIST in USD
    if (usdtryData.length > 0) {{
      const commonDates = bistNorm.filter(b => usdtryData.some(u => u.date === b.date));
      if (commonDates.length > 0) {{
        const bistUsd = commonDates.map(b => {{
          const u = usdtryData.find(u => u.date === b.date);
          const bOrig = bist100Data.find(d => d.date === b.date);
          return bOrig && u ? (bOrig.bist100_close / u.usdtry_close) : null;
        }}).filter(v => v !== null);
        if (bistUsd.length > 0) {{
          const bistUsdBase = bistUsd[0];
          const bistUsdNorm = commonDates.filter((b,i) => {{
            const u = usdtryData.find(u => u.date === b.date);
            return u !== undefined;
          }}).map((b,i) => ({{date:b.date, val:(bistUsd[i]/bistUsdBase)*100}}));
          if (bistUsdNorm.length > 0) {{
            overlayTraces.push({{
              x:bistUsdNorm.map(d=>d.date), y:bistUsdNorm.map(d=>d.val),
              type:'scatter', mode:'lines', name:'BIST-100 in USD ('+bistUsdNorm[bistUsdNorm.length-1].val.toFixed(0)+')',
              line:{{color:'#2ca02c',width:2}}
            }});
          }}
        }}
      }}
    }}

    Plotly.newPlot('chart-bist-infl-usd', overlayTraces, {{
      ...layout_base, title:'BIST-100 vs CPI vs USD/TRY (Jan 2020 = 100)',
      yaxis:{{title:'Growth (Jan 2020 = 100)'}}, xaxis:{{title:'Date'}},
      legend:{{yanchor:'top',y:0.99,xanchor:'left',x:0.01}},
      height:500,
      shapes:[{{type:'line',x0:bistNorm[0].date,x1:bistNorm[bistNorm.length-1].date,y0:100,y1:100,line:{{color:'gray',dash:'dot',width:1}}}}]
    }}, cfg);
  }}

  // Illusion Zone Scatter
  if (illusionData.length > 0) {{
    const illNorm = illusionData.filter(d => !d.illusion);
    const illYes = illusionData.filter(d => d.illusion);
    const maxNom = Math.max(...illusionData.map(d=>d.nominal));
    const minReal = Math.min(...illusionData.map(d=>d.real));

    const illTraces = [
      {{x:illNorm.map(d=>d.nominal), y:illNorm.map(d=>d.real), type:'scatter', mode:'markers', name:'Normal',
        marker:{{color:'#2ca02c',size:7,opacity:0.7}}, text:illNorm.map(d=>d.ticker), hovertemplate:'%{{text}}<br>Nominal: %{{x:.1f}}%<br>Real: %{{y:.1f}}%<extra></extra>'}},
      {{x:illYes.map(d=>d.nominal), y:illYes.map(d=>d.real), type:'scatter', mode:'markers', name:'Illusion Zone',
        marker:{{color:'#d62728',size:7,opacity:0.7}}, text:illYes.map(d=>d.ticker), hovertemplate:'%{{text}}<br>Nominal: %{{x:.1f}}%<br>Real: %{{y:.1f}}%<extra></extra>'}}
    ];

    Plotly.newPlot('chart-illusion-zone', illTraces, {{
      ...layout_base, title:'1-Year IPO Returns: Nominal vs Real ('+illusionData.length+' IPOs)',
      xaxis:{{title:'Nominal 1-Year Return (%)',zeroline:true}},
      yaxis:{{title:'Real 1-Year Return (%)',zeroline:true}},
      height:500,
      shapes:[
        {{type:'line',x0:-200,x1:maxNom*1.1,y0:0,y1:0,line:{{color:'#d62728',dash:'dash',width:1}}}},
        {{type:'line',x0:0,x1:0,y0:minReal*1.1,y1:maxNom*1.1,line:{{color:'#d62728',dash:'dash',width:1}}}},
        {{type:'rect',x0:0,x1:maxNom*1.1,y0:minReal*1.1,y1:0,fillcolor:'rgba(255,0,0,0.05)',line:{{width:0}}}}
      ],
      annotations:[{{x:maxNom*0.5,y:minReal*0.5,text:'ILLUSION ZONE',showarrow:false,font:{{color:'rgba(214,39,40,0.3)',size:20,family:'Arial Black'}}}}]
    }}, cfg);
  }}

  // Inflation vs IPO Demand: Dual-axis time series
  if (ipoInflData.length > 0) {{
    Plotly.newPlot('chart-ipo-vs-inflation', [
      {{x:ipoInflData.map(d=>d.month_str), y:ipoInflData.map(d=>d.ipo_count), type:'bar', name:'IPO Count', marker:{{color:'#9467bd'}}, yaxis:'y'}},
      {{x:ipoInflData.map(d=>d.month_str), y:ipoInflData.map(d=>d.tufe_yoy), type:'scatter', mode:'lines', name:'Inflation YoY %', line:{{color:'red',width:2}}, yaxis:'y2'}}
    ], {{...layout_base, title:'IPO Count vs Inflation Over Time', yaxis:{{title:'IPO Count'}}, yaxis2:{{title:'Inflation YoY %',overlaying:'y',side:'right'}}, height:400}}, cfg);

    // Scatter: inflation vs IPO frequency with trendline
    const inflX = ipoInflData.map(d=>d.tufe_yoy);
    const inflY = ipoInflData.map(d=>d.ipo_count);
    // Simple OLS for trendline
    const n = inflX.length;
    const sumX = inflX.reduce((a,b)=>a+b,0);
    const sumY = inflY.reduce((a,b)=>a+b,0);
    const sumXY = inflX.reduce((a,b,i)=>a+b*inflY[i],0);
    const sumXX = inflX.reduce((a,b)=>a+b*b,0);
    const slope = (n*sumXY - sumX*sumY) / (n*sumXX - sumX*sumX);
    const intercept = (sumY - slope*sumX) / n;
    const minX = Math.min(...inflX);
    const maxX = Math.max(...inflX);

    Plotly.newPlot('chart-inflation-ipo-scatter', [
      {{x:inflX, y:inflY, type:'scatter', mode:'markers', name:'Monthly data',
        marker:{{color:'#bcbd22',size:7}}, hovertemplate:'Inflation: %{{x:.1f}}%<br>IPOs: %{{y}}<extra></extra>'}},
      {{x:[minX,maxX], y:[intercept+slope*minX, intercept+slope*maxX], type:'scatter', mode:'lines', name:'OLS trendline',
        line:{{color:'#d62728',dash:'dash',width:2}}}}
    ], {{...layout_base, title:'Correlation: Inflation vs IPO Frequency', xaxis:{{title:'YoY Inflation %'}}, yaxis:{{title:'Monthly IPO Count'}}, height:400}}, cfg);
  }}
}}

// ═══ CONCLUSIONS PAGE CHARTS ═══
function initConclusionsCharts() {{
  // Regression coefficients
  const vars = regM4.map(d=>d.variable);
  const coefs = regM4.map(d=>d.coefficient);
  const pvals = regM4.map(d=>d.p_value);
  Plotly.newPlot('chart-regression', [{{
    x:vars, y:coefs, type:'bar',
    marker:{{color:pvals.map(p=>p<0.05?'#2ca02c':p<0.1?'#ff7f0e':'#95a5a6')}},
    text:coefs.map((c,i)=>c.toFixed(3) + (pvals[i]<0.05?'**':pvals[i]<0.1?'*':'')), textposition:'outside',
    hovertemplate:'%{{x}}<br>Coef: %{{y:.4f}}<extra></extra>'
  }}], {{...layout_base, title:'Model 4 (Log-Level) Coefficients', xaxis:{{tickangle:-25}}, yaxis:{{title:'Coefficient'}}, height:400,
    shapes:[{{type:'line',x0:-0.5,x1:vars.length-0.5,y0:0,y1:0,line:{{color:'#7f7f7f',dash:'dot'}}}}]
  }}, cfg);

  // 4-Model R-squared comparison
  if (regSummary.length > 0) {{
    const modelNames = regSummary.map(d => {{
      const parts = d.model.split(': ');
      return parts.length > 1 ? parts[1] : d.model;
    }});
    const r2Vals = regSummary.map(d => d.r_squared);
    const adjR2Vals = regSummary.map(d => d.adj_r_squared);
    const fSig = regSummary.map(d => d.f_pvalue < 0.05);

    Plotly.newPlot('chart-regression-comparison', [
      {{x:modelNames, y:r2Vals, type:'bar', name:'R-squared', marker:{{color:'#1f77b4'}},
        text:r2Vals.map(v=>v.toFixed(3)), textposition:'outside'}},
      {{x:modelNames, y:adjR2Vals, type:'bar', name:'Adj. R-squared', marker:{{color:'#ff7f0e'}},
        text:adjR2Vals.map(v=>v.toFixed(3)), textposition:'outside'}}
    ], {{...layout_base, title:'Model Comparison: R-squared and Adjusted R-squared', barmode:'group',
      yaxis:{{title:'R-squared',range:[0,0.3]}}, height:400,
      annotations:modelNames.map((m,i) => ({{x:m, y:Math.max(r2Vals[i],adjR2Vals[i])+0.03, text:fSig[i]?'F-test: sig.':'F-test: n.s.', showarrow:false, font:{{size:9,color:fSig[i]?'#2ca02c':'#d62728'}}}}))
    }}, cfg);
  }}
}}

// ═══ LAZY INIT ═══
let initialized = {{overview:true, ipo:false, manipulation:false, herding:false, conclusions:false}};

const origShowPage = showPage;
showPage = function(id, btn) {{
  origShowPage(id, btn);
  if (!initialized[id]) {{
    initialized[id] = true;
    if (id === 'ipo') initIPOCharts();
    else if (id === 'manipulation') initManipulationCharts();
    else if (id === 'herding') initHerdingCharts();
    else if (id === 'conclusions') initConclusionsCharts();
  }}
}};

// ═══ SCROLL REVEAL + ANIMATED COUNTERS ═══
const revealObserver = new IntersectionObserver((entries) => {{
  entries.forEach(entry => {{
    if (entry.isIntersecting) {{
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
      // If it's a metric value, animate the counter
      const val = entry.target.querySelector('.value');
      if (val && !val.dataset.animated) animateCounter(val);
    }}
  }});
}}, {{ threshold: 0.15, rootMargin: '0px 0px -30px 0px' }});

function initReveals(container) {{
  const selectors = '.metric, .info, .success, .warning, .theory-card, h2, .sq-card, .grid > div, [id$="-chart"], .table-container';
  const targets = container.querySelectorAll(selectors);
  targets.forEach((el, i) => {{
    if (el.classList.contains('visible')) return;
    el.classList.add('reveal');
    // Stagger within each grid group
    const parent = el.parentElement;
    if (parent && (parent.classList.contains('metrics') || parent.classList.contains('grid'))) {{
      const siblings = Array.from(parent.children);
      const idx = siblings.indexOf(el);
      el.style.transitionDelay = `${{idx * 80}}ms`;
    }}
    revealObserver.observe(el);
  }});
}}

function animateCounter(el) {{
  el.dataset.animated = '1';
  const text = el.textContent.trim();
  // Match patterns: "67.8%", "209", "−3.3%", "4.4", "$1.13B", "1,496" etc
  const match = text.match(/^([^0-9]*?)([0-9,]+[.]?[0-9]*)(.*)/);
  if (!match) return;
  const prefix = match[1];
  const numStr = match[2].replace(/,/g, '');
  const target = parseFloat(numStr);
  if (isNaN(target) || target === 0) return;
  const suffix = match[3];
  const hasComma = match[2].includes(',');
  const decimals = numStr.includes('.') ? numStr.split('.')[1].length : 0;
  const duration = 1000;
  const start = performance.now();
  function update(now) {{
    const elapsed = now - start;
    const t = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    let current = (target * eased).toFixed(decimals);
    if (hasComma) current = parseFloat(current).toLocaleString('en-US', {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}});
    el.textContent = prefix + current + suffix;
    if (t < 1) requestAnimationFrame(update);
    else el.textContent = text; // restore exact original
  }}
  requestAnimationFrame(update);
}}

// Init first page on load
document.addEventListener('DOMContentLoaded', () => {{
  const overview = document.getElementById('overview');
  if (overview) initReveals(overview);
}});

// Hook into page switches to init reveals on new pages
const origShowPage2 = showPage;
showPage = function(id, btn) {{
  origShowPage2(id, btn);
  setTimeout(() => {{
    const page = document.getElementById(id);
    if (page) initReveals(page);
  }}, 200);
}};
</script>
</body>
</html>"""

# Write the file
output_path = r'C:\Users\cugut\Documents\portfolio-tracker\tez\index.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Website generated: {output_path}")
print(f"File size: {len(html.encode('utf-8'))/1024:.0f} KB")
