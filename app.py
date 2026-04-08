import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="India Financial Analyst",
    page_icon="📊",
    layout="wide"
)

# ── Gemini setup ──────────────────────────────────────────────────
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# ── Helpers ───────────────────────────────────────────────────────
def safe(val):
    """Return val if it is not None and not NaN, else None."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val

def loc_safe(frame, row_key, col_key):
    """Safe df.loc — returns None if index/column missing or value is NaN."""
    if row_key not in frame.index:
        return None
    if col_key not in frame.columns:
        return None
    return safe(frame.loc[row_key, col_key])

def first_valid(frame, keys, col_key):
    """Try row-keys in order; return first non-None value found."""
    for k in keys:
        v = loc_safe(frame, k, col_key)
        if v is not None:
            return v
    return None

# ── Ticker auto-detect (NSE first, then BSE) ─────────────────────
def resolve_ticker(raw: str):
    """
    Returns (yf.Ticker, resolved_symbol) or (None, None).
    Tries .NS suffix first, falls back to .BO.
    If user already provided a suffix, validates it directly.
    """
    raw = raw.strip().upper()
    if raw.endswith(".NS") or raw.endswith(".BO"):
        t    = yf.Ticker(raw)
        info = t.info or {}
        if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"):
            return t, raw
        return None, None

    for suffix in [".NS", ".BO"]:
        symbol = raw + suffix
        try:
            t    = yf.Ticker(symbol)
            info = t.info or {}
            if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"):
                return t, symbol
        except Exception:
            continue
    return None, None

# ── Data fetching ─────────────────────────────────────────────────
def fetch_financial_data(tickers, num_years):
    all_data = []
    failed   = []

    for ticker_str in tickers:
        ticker_str = ticker_str.strip().upper()
        t, resolved = resolve_ticker(ticker_str)
        if t is None:
            failed.append(ticker_str)
            continue

        # Display name: strip exchange suffix
        display_name = ticker_str.replace(".NS", "").replace(".BO", "")

        try:
            inc  = t.financials     # rows = metrics, cols = fiscal year end dates
            cf   = t.cashflow
            bs   = t.balance_sheet
            info = t.info or {}

            if inc.empty or cf.empty or bs.empty:
                failed.append(ticker_str)
                continue

            # yfinance returns columns newest-first — take first num_years
            cols            = inc.columns[:num_years]
            most_recent_col = cols[0]   # newest fiscal year

            # ── Live valuation (current market price, point-in-time) ──
            pe_ratio  = safe(info.get("trailingPE"))
            pb_ratio  = safe(info.get("priceToBook"))
            ev_ebitda = safe(info.get("enterpriseToEbitda"))

            for col in cols:
                year = col.year

                # ── Income statement ──────────────────────────────
                revenue    = loc_safe(inc, 'Total Revenue', col)
                net_income = loc_safe(inc, 'Net Income',    col)
                gross_prof = loc_safe(inc, 'Gross Profit',  col)
                ebit       = loc_safe(inc, 'EBIT',          col)

                # EBITDA: direct lookup first, then reconstruct from EBIT + D&A
                ebitda = first_valid(inc, ['EBITDA', 'Normalized EBITDA'], col)
                if ebitda is None and ebit is not None:
                    da = first_valid(cf, [
                        'Reconciled Depreciation',
                        'Depreciation And Amortization',
                        'Depreciation Amortization Depletion',
                        'Depreciation And Amortization In Income Statement',
                    ], col)
                    if da is not None:
                        # D&A sign varies across yfinance versions; abs() ensures correct addition
                        ebitda = ebit + abs(da)

                # Interest expense (yfinance reports as negative — cash outflow)
                interest_exp = first_valid(inc, [
                    'Interest Expense',
                    'Interest Expense Non Operating',
                ], col)

                # ── Cash flow ─────────────────────────────────────
                ocf = first_valid(cf, [
                    'Operating Cash Flow',
                    'Cash Flow From Continuing Operating Activities',
                    'Total Cash From Operating Activities',
                ], col)

                # Capex: yfinance returns as negative (cash outflow)
                capex_raw = first_valid(cf, [
                    'Capital Expenditure',
                    'Purchase Of PPE',
                    'Capital Expenditures',
                ], col)

                # ── Balance sheet ─────────────────────────────────
                total_assets   = first_valid(bs, ['Total Assets'], col)
                total_liab     = first_valid(bs, [
                    'Total Liabilities Net Minority Interest',
                    'Total Liabilities',
                ], col)
                total_equity   = first_valid(bs, [
                    'Stockholders Equity',
                    'Total Equity Gross Minority Interest',
                    'Common Stock Equity',
                ], col)
                current_assets = first_valid(bs, ['Current Assets', 'Total Current Assets'], col)
                current_liab   = first_valid(bs, ['Current Liabilities', 'Total Current Liabilities'], col)
                total_debt     = first_valid(bs, [
                    'Total Debt',
                    'Long Term Debt And Capital Lease Obligation',
                    'Long Term Debt',
                ], col)

                # ── Skip row if any core field is missing ─────────
                if any(v is None for v in [revenue, net_income, ocf, total_assets, total_liab]):
                    continue
                if revenue == 0:   # guard against division-by-zero on margin calcs
                    continue

                # ── Free Cash Flow ────────────────────────────────
                # FCF = OCF − |Capex|
                # yfinance capex is usually negative; abs() normalises regardless of sign convention
                fcf = (ocf - abs(capex_raw)) if capex_raw is not None else None

                # ── ROCE ──────────────────────────────────────────
                # Standard: EBIT / Capital Employed
                # Capital Employed = Total Assets − Current Liabilities
                # Fallback if current_liab missing: Equity + Total Debt
                roce = None
                if ebit is not None:
                    if current_liab is not None and (total_assets - current_liab) != 0:
                        roce = round(ebit / (total_assets - current_liab) * 100, 2)
                    elif total_equity is not None and total_debt is not None:
                        ce = total_equity + total_debt
                        if ce != 0:
                            roce = round(ebit / ce * 100, 2)

                # ── ROE ───────────────────────────────────────────
                # ROE = Net Income / Total Equity
                roe = None
                if total_equity is not None and total_equity != 0:
                    roe = round(net_income / total_equity * 100, 2)

                # ── Interest Coverage ─────────────────────────────
                # EBITDA / |Interest Expense|
                # Guard: both must exist, interest_exp must be non-zero (and non-NaN already handled by safe())
                int_coverage = None
                if ebitda is not None and interest_exp is not None and interest_exp != 0:
                    int_coverage = round(ebitda / abs(interest_exp), 2)

                # ── Current Ratio ─────────────────────────────────
                # Current Assets / Current Liabilities
                current_ratio = None
                if current_assets is not None and current_liab is not None and current_liab != 0:
                    current_ratio = round(current_assets / current_liab, 2)

                # ── Build row ─────────────────────────────────────
                # All raw yfinance values are in INR units; 1 Crore = 1e7
                row = {
                    'Company':                  display_name,
                    'Year':                     year,

                    # Absolute (₹ Crore)
                    'Revenue (₹Cr)':            round(revenue      / 1e7, 2),
                    'Net Income (₹Cr)':         round(net_income   / 1e7, 2),
                    'EBITDA (₹Cr)':             round(ebitda       / 1e7, 2) if ebitda      is not None else None,
                    'Operating CF (₹Cr)':       round(ocf          / 1e7, 2),
                    'Free CF (₹Cr)':            round(fcf          / 1e7, 2) if fcf         is not None else None,
                    'Total Assets (₹Cr)':       round(total_assets / 1e7, 2),
                    'Total Liabilities (₹Cr)':  round(total_liab   / 1e7, 2),
                    'Total Equity (₹Cr)':       round(total_equity / 1e7, 2) if total_equity is not None else None,
                    'Total Debt (₹Cr)':         round(total_debt   / 1e7, 2) if total_debt   is not None else None,

                    # Profitability margins
                    'Gross Margin (%)':         round(gross_prof   / revenue * 100, 2) if gross_prof is not None else None,
                    'EBITDA Margin (%)':        round(ebitda       / revenue * 100, 2) if ebitda     is not None else None,
                    'Net Profit Margin (%)':    round(net_income   / revenue * 100, 2),
                    'Cash Flow Margin (%)':     round(ocf          / revenue * 100, 2),

                    # Return metrics
                    'ROE (%)':                  roe,
                    'ROCE (%)':                 roce,

                    # Leverage
                    'Debt-to-Asset Ratio':      round(total_liab   / total_assets, 2),
                    'Interest Coverage (x)':    int_coverage,

                    # Liquidity
                    'Current Ratio':            current_ratio,

                    # Efficiency
                    'Asset Turnover (x)':       round(revenue      / total_assets, 2),

                    # Valuation — only on most recent year row; NaN for historical rows
                    # (these are live price-based multiples, not historical)
                    'P/E Ratio':    (round(pe_ratio,  2) if pe_ratio  is not None else None) if col == most_recent_col else None,
                    'P/B Ratio':    (round(pb_ratio,  2) if pb_ratio  is not None else None) if col == most_recent_col else None,
                    'EV/EBITDA (x)':(round(ev_ebitda, 2) if ev_ebitda is not None else None) if col == most_recent_col else None,
                }

                all_data.append(row)

        except Exception:
            failed.append(ticker_str)

    if not all_data:
        return None, failed

    df = pd.DataFrame(all_data).sort_values(['Company', 'Year']).reset_index(drop=True)

    # YoY growth — fill_method=None prevents pct_change from bridging NaN gaps
    df['Revenue Growth (%)']    = df.groupby('Company')['Revenue (₹Cr)'].pct_change(fill_method=None).mul(100).round(2)
    df['Net Income Growth (%)'] = df.groupby('Company')['Net Income (₹Cr)'].pct_change(fill_method=None).mul(100).round(2)
    df['EBITDA Growth (%)']     = df.groupby('Company')['EBITDA (₹Cr)'].pct_change(fill_method=None).mul(100).round(2)

    return df, failed


# ── Charts ────────────────────────────────────────────────────────

def plot_revenue(df):
    """Grouped bar chart — Revenue in ₹'000 Cr."""
    companies = df['Company'].unique()
    years     = sorted(df['Year'].unique())
    fig, ax   = plt.subplots(figsize=(10, 4))
    x         = range(len(years))
    width     = 0.8 / max(len(companies), 1)

    for i, company in enumerate(companies):
        vals = []
        for y in years:
            r = df[(df['Company'] == company) & (df['Year'] == y)]
            # ₹Cr ÷ 1000 = ₹'000 Cr  (thousands of crores, i.e. ₹ lakh crore scale)
            vals.append(r['Revenue (₹Cr)'].values[0] / 1000 if not r.empty else 0)
        bars = ax.bar([p + i * width for p in x], vals, width=width, label=company)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + h * 0.01,
                    f'₹{h:.1f}K Cr', ha='center', va='bottom', fontsize=7)

    ax.set_xticks([p + width * (len(companies) - 1) / 2 for p in x])
    ax.set_xticklabels(years)
    ax.set_title("Total Revenue (₹'000 Cr)", fontweight='bold')
    ax.set_ylabel("Revenue (₹'000 Cr)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_margins(df):
    """3-panel line chart: Gross / EBITDA / Net Profit margins."""
    specs  = [('Gross Margin (%)', 'Gross Margin'),
              ('EBITDA Margin (%)', 'EBITDA Margin'),
              ('Net Profit Margin (%)', 'Net Profit Margin')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (col, title) in zip(axes, specs):
        for company in df['Company'].unique():
            subset = df[df['Company'] == company].dropna(subset=[col])
            if subset.empty:
                continue
            ax.plot(subset['Year'], subset[col], marker='o', linewidth=2, label=company)
            for _, r in subset.iterrows():
                ax.annotate(f"{r[col]:.1f}%", (r['Year'], r[col]),
                            textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')
        ax.set_title(f'{title} (%)', fontweight='bold')
        ax.set_ylabel('%')
        ax.set_xticks(sorted(df['Year'].unique()))
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_roe_roce(df):
    """Side-by-side line charts: ROE and ROCE."""
    specs     = [('ROE (%)', 'Return on Equity (ROE)'),
                 ('ROCE (%)', 'Return on Capital Employed (ROCE)')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (col, title) in zip(axes, specs):
        for company in df['Company'].unique():
            subset = df[df['Company'] == company].dropna(subset=[col])
            if subset.empty:
                continue
            ax.plot(subset['Year'], subset[col], marker='o', linewidth=2, label=company)
            for _, r in subset.iterrows():
                ax.annotate(f"{r[col]:.1f}%", (r['Year'], r[col]),
                            textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')
        ax.set_title(f'{title} (%)', fontweight='bold')
        ax.set_ylabel('%')
        ax.set_xticks(sorted(df['Year'].unique()))
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_heatmap(df):
    """Debt-to-Asset heatmap — companies × years."""
    pivot = df.pivot_table(index='Company', columns='Year', values='Debt-to-Asset Ratio')
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, vmin=0.2, vmax=0.9,
                cbar_kws={'label': 'Higher = more leveraged'})
    ax.set_title('Debt-to-Asset Ratio Heatmap', fontweight='bold')
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('')
    plt.tight_layout()
    return fig


def plot_valuation(df):
    """Bar chart of current valuation multiples (most recent year per company)."""
    metrics   = ['P/E Ratio', 'P/B Ratio', 'EV/EBITDA (x)']
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if not available:
        return None

    latest = (
        df.dropna(subset=available, how='all')
          .sort_values('Year')
          .groupby('Company')
          .last()
          .reset_index()
    )
    if latest.empty:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        data = latest[['Company', metric]].dropna()
        if data.empty:
            ax.set_visible(False)
            continue
        bars = ax.bar(data['Company'], data[metric],
                      color=sns.color_palette("Set2", len(data)))
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + h * 0.01,
                    f'{h:.1f}x', ha='center', va='bottom', fontsize=9)
        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel('Multiple')
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Valuation Multiples (Current Market Price)', fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────
st.title("📊 India Financial Analyst")
st.caption("Powered by Gemini 2.5 Flash  |  Live data via Yahoo Finance  |  NSE / BSE auto-detect")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Enter **NSE/BSE tickers** — no suffix needed, exchange is auto-detected.")
    st.markdown("_e.g. RELIANCE, TCS, INFY, HDFCBANK, WIPRO_")
    ticker_input = st.text_input(
        "Ticker symbols (comma separated)",
        placeholder="e.g. RELIANCE, TCS, INFY",
        value="RELIANCE, TCS, HDFCBANK"
    )
    num_years = st.slider("Number of fiscal years", min_value=1, max_value=5, value=3)
    load_btn  = st.button("🔄 Load Data", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**💡 Try asking:**")
    st.markdown("""
    - What is TCS's EBITDA margin trend?
    - Which company has the best ROE?
    - Compare ROCE across all companies
    - Which stock looks cheapest on P/E?
    - Flag companies with weak interest coverage
    - What's driving Reliance's revenue growth?
    - Which company has the strongest free cash flow?
    - Compare debt levels across all companies
    """)

# ── Load data ─────────────────────────────────────────────────────
if load_btn:
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker symbol.")
        st.stop()

    with st.spinner(f"Fetching live data for {', '.join(tickers)}  (NSE → BSE auto-detect)..."):
        df, failed = fetch_financial_data(tickers, num_years)

    if df is None:
        st.error("Could not fetch data for any of the tickers. Please verify the symbols.")
        st.stop()

    if failed:
        st.warning(f"⚠️ Could not fetch: {', '.join(failed)}. Proceeding with available data.")

    st.session_state.df                   = df
    st.session_state.messages             = []
    st.session_state.conversation_history = []
    st.success(f"✅ Loaded: {', '.join(df['Company'].unique())}")

# ── Main content ──────────────────────────────────────────────────
if "df" in st.session_state:
    df                = st.session_state.df
    financial_context = df.to_json(orient='records', indent=2)

    tab1, tab2 = st.tabs(["💬 Chat", "📈 Analysis"])

    # ── Analysis tab ─────────────────────────────────────────────
    with tab2:
        st.subheader("📋 Financial Summary")
        display_cols = [
            'Company', 'Year',
            'Revenue (₹Cr)', 'Revenue Growth (%)',
            'EBITDA (₹Cr)', 'EBITDA Margin (%)', 'EBITDA Growth (%)',
            'Net Income (₹Cr)', 'Net Profit Margin (%)', 'Net Income Growth (%)',
            'ROE (%)', 'ROCE (%)',
            'Free CF (₹Cr)', 'Cash Flow Margin (%)',
            'Total Debt (₹Cr)', 'Debt-to-Asset Ratio',
            'Current Ratio', 'Interest Coverage (x)', 'Asset Turnover (x)',
            'P/E Ratio', 'P/B Ratio', 'EV/EBITDA (x)',
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols].round(2), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📊 Revenue")
        st.pyplot(plot_revenue(df))

        st.subheader("📈 Margin Analysis")
        st.pyplot(plot_margins(df))

        st.subheader("💰 Return Metrics")
        st.pyplot(plot_roe_roce(df))

        st.subheader("🏦 Leverage")
        hm = plot_heatmap(df)
        if hm:
            st.pyplot(hm)

        vf = plot_valuation(df)
        if vf:
            st.subheader("📐 Valuation Multiples (Current)")
            st.pyplot(vf)

    # ── Chat tab ─────────────────────────────────────────────────
    with tab1:
        SYSTEM_PROMPT = f"""
You are a senior equity research analyst specialising in Indian listed companies (NSE/BSE).
You have access to the following verified financial dataset (all figures in ₹ Crore):

{financial_context}

NOTE ON VALUATION MULTIPLES:
P/E, P/B, and EV/EBITDA are CURRENT (live) market-price-based multiples.
They appear only against the most recent fiscal year row per company.
Do NOT treat them as historical multiples for older year rows.

ANALYTICAL FRAMEWORK — apply this to every response:
1. TREND     : Is the metric improving, deteriorating, or stable across the available years?
2. BENCHMARK : How does it rank across all companies in the dataset?
3. FLAGS     : Proactively call out if:
   - Any metric shows >20% YoY swing
   - EBITDA Margin < 15%        → margin pressure
   - Interest Coverage < 2x     → solvency risk
   - Current Ratio < 1          → near-term liquidity concern
   - ROE < 10%                  → weak equity returns
   - ROCE < 12%                 → sub-par capital deployment
   - Debt-to-Asset > 0.65       → high leverage
4. INSIGHT   : One forward-looking or contextual observation.

RESPONSE FORMAT (strict):
- Line 1   : Direct answer in one sentence.
- Lines 2–5: 2–4 bullet points — each citing [Company | FY Year | figure with unit].
- Last line: **Analyst's Take:** one sharp, opinionated sentence.
- Max length: 160 words.
- Units: ₹ Crore for absolutes | % for margins & returns | x for multiples & ratios.

RULES:
- Never fabricate or extrapolate — only use numbers present in the dataset.
- If a metric is unavailable for a company, state that explicitly.
- Comparisons: always rank companies (1st, 2nd, 3rd…) and state the spread.
- Out-of-scope questions: say what you cannot answer, then offer what you CAN.
- Greetings / off-topic: one-sentence reply, then redirect to financial analysis.
"""

        if "messages" not in st.session_state:
            st.session_state.messages             = []
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything about the loaded companies..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.conversation_history.append(f"User: {prompt}")

            history_text = "\n".join(st.session_state.conversation_history[-8:])
            full_prompt  = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Conversation so far:\n{history_text}\n\n"
                f"Respond as the senior equity analyst:"
            )

            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    response = model.generate_content(full_prompt)
                    reply    = response.text.strip()
                    st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.conversation_history.append(f"Analyst: {reply}")

        if st.session_state.get("messages"):
            if st.button("🔄 Reset conversation"):
                st.session_state.messages             = []
                st.session_state.conversation_history = []
                st.rerun()

else:
    st.info("👈 Enter NSE/BSE ticker symbols in the sidebar and click **Load Data** to begin.")
