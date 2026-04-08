import streamlit as st
import google.generativeai as genai
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="India Financial Analyst",
    page_icon="📊",
    layout="wide"
)

# ── Gemini setup ─────────────────────────────────────────────────
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# ── Ticker auto-detect (NSE first, BSE fallback) ──────────────────
def resolve_ticker(raw: str):
    """Try NSE (.NS) first, then BSE (.BO). Return (yf.Ticker, resolved_symbol) or (None, None)."""
    raw = raw.strip().upper()
    # If user already added suffix, use as-is
    if raw.endswith(".NS") or raw.endswith(".BO"):
        t = yf.Ticker(raw)
        info = t.info
        if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"):
            return t, raw
        return None, None

    for suffix in [".NS", ".BO"]:
        symbol = raw + suffix
        t = yf.Ticker(symbol)
        try:
            info = t.info
            if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"):
                return t, symbol
        except Exception:
            continue
    return None, None

# ── Data fetching ─────────────────────────────────────────────────
def fetch_financial_data(tickers, num_years):
    all_data = []
    failed = []
    resolved_map = {}  # raw -> resolved symbol

    for ticker_str in tickers:
        ticker_str = ticker_str.strip().upper()
        t, resolved = resolve_ticker(ticker_str)
        if t is None:
            failed.append(ticker_str)
            continue

        resolved_map[ticker_str] = resolved
        display_name = ticker_str.replace(".NS", "").replace(".BO", "")

        try:
            inc = t.financials
            cf  = t.cashflow
            bs  = t.balance_sheet
            info = t.info

            if inc.empty or cf.empty or bs.empty:
                failed.append(ticker_str)
                continue

            cols = inc.columns[:num_years]

            # ── Live valuation metrics (point-in-time from info) ──
            pe_ratio    = info.get("trailingPE")
            pb_ratio    = info.get("priceToBook")
            ev_ebitda   = info.get("enterpriseToEbitda")
            market_cap  = info.get("marketCap")
            shares_out  = info.get("sharesOutstanding")

            for col in cols:
                year = col.year

                # ── Income statement ──
                revenue    = inc.loc['Total Revenue', col]         if 'Total Revenue'    in inc.index else None
                net_income = inc.loc['Net Income', col]            if 'Net Income'        in inc.index else None
                gross_prof = inc.loc['Gross Profit', col]          if 'Gross Profit'      in inc.index else None
                ebit       = inc.loc['EBIT', col]                  if 'EBIT'              in inc.index else None

                # EBITDA: try direct key, else EBIT + D&A
                ebitda = None
                for eb_key in ['EBITDA', 'Normalized EBITDA']:
                    if eb_key in inc.index:
                        ebitda = inc.loc[eb_key, col]
                        break
                if ebitda is None and ebit is not None:
                    for da_key in ['Reconciled Depreciation', 'Depreciation And Amortization',
                                   'Depreciation Amortization Depletion']:
                        if da_key in cf.index and col in cf.columns:
                            da = cf.loc[da_key, col]
                            if da is not None:
                                ebitda = ebit + abs(da)
                                break

                interest_exp = None
                for ie_key in ['Interest Expense', 'Interest Expense Non Operating']:
                    if ie_key in inc.index:
                        interest_exp = inc.loc[ie_key, col]
                        break

                # ── Cash flow ──
                ocf = None
                for ocf_key in ['Operating Cash Flow',
                                 'Cash Flow From Continuing Operating Activities',
                                 'Total Cash From Operating Activities']:
                    if ocf_key in cf.index and col in cf.columns:
                        ocf = cf.loc[ocf_key, col]
                        if ocf is not None:
                            break

                capex = None
                for cx_key in ['Capital Expenditure', 'Purchase Of PPE',
                                'Capital Expenditures']:
                    if cx_key in cf.index and col in cf.columns:
                        capex = cf.loc[cx_key, col]
                        if capex is not None:
                            break

                # ── Balance sheet ──
                def bs_get(keys):
                    for k in keys:
                        if k in bs.index and col in bs.columns:
                            v = bs.loc[k, col]
                            if v is not None:
                                return v
                    return None

                total_assets  = bs_get(['Total Assets'])
                total_liab    = bs_get(['Total Liabilities Net Minority Interest', 'Total Liabilities'])
                total_equity  = bs_get(['Stockholders Equity', 'Total Equity Gross Minority Interest',
                                        'Common Stock Equity'])
                current_assets = bs_get(['Current Assets', 'Total Current Assets'])
                current_liab   = bs_get(['Current Liabilities', 'Total Current Liabilities'])
                inventory      = bs_get(['Inventory'])
                total_debt     = bs_get(['Total Debt', 'Long Term Debt And Capital Lease Obligation'])

                if any(v is None for v in [revenue, net_income, ocf, total_assets, total_liab]):
                    continue

                # ── FCF ──
                fcf = (ocf + capex) if (ocf is not None and capex is not None) else None

                row = {
                    'Company':                        display_name,
                    'Year':                           year,
                    # Core financials (₹ Cr)
                    'Revenue (₹Cr)':                  round(revenue / 1e7, 2),
                    'Net Income (₹Cr)':               round(net_income / 1e7, 2),
                    'EBITDA (₹Cr)':                   round(ebitda / 1e7, 2) if ebitda is not None else None,
                    'Operating CF (₹Cr)':             round(ocf / 1e7, 2),
                    'Free CF (₹Cr)':                  round(fcf / 1e7, 2) if fcf is not None else None,
                    'Total Assets (₹Cr)':             round(total_assets / 1e7, 2),
                    'Total Liabilities (₹Cr)':        round(total_liab / 1e7, 2),
                    'Total Equity (₹Cr)':             round(total_equity / 1e7, 2) if total_equity is not None else None,
                    # Profitability
                    'Gross Margin (%)':               round(gross_prof / revenue * 100, 2) if gross_prof is not None else None,
                    'EBITDA Margin (%)':              round(ebitda / revenue * 100, 2) if ebitda is not None else None,
                    'Net Profit Margin (%)':          round(net_income / revenue * 100, 2),
                    'Cash Flow Margin (%)':           round(ocf / revenue * 100, 2),
                    # Leverage
                    'Debt-to-Asset Ratio':            round(total_liab / total_assets, 2),
                    'Interest Coverage (x)':          round(ebitda / abs(interest_exp), 2) if (ebitda is not None and interest_exp is not None and interest_exp != 0) else None,
                    # Liquidity
                    'Current Ratio':                  round(current_assets / current_liab, 2) if (current_assets is not None and current_liab is not None and current_liab != 0) else None,
                    # Efficiency
                    'Asset Turnover (x)':             round(revenue / total_assets, 2),
                }

                # ROE and ROCE
                if total_equity is not None and total_equity != 0:
                    row['ROE (%)'] = round(net_income / total_equity * 100, 2)
                else:
                    row['ROE (%)'] = None

                if total_equity is not None and total_debt is not None and (total_equity + total_debt) != 0:
                    row['ROCE (%)'] = round(ebit / (total_equity + total_debt) * 100, 2) if ebit is not None else None
                else:
                    row['ROCE (%)'] = None

                # Valuation (live, same for all years of this ticker)
                row['P/E Ratio']      = round(pe_ratio, 2)   if pe_ratio  is not None else None
                row['P/B Ratio']      = round(pb_ratio, 2)   if pb_ratio  is not None else None
                row['EV/EBITDA (x)']  = round(ev_ebitda, 2)  if ev_ebitda is not None else None

                all_data.append(row)

        except Exception as e:
            failed.append(ticker_str)

    if not all_data:
        return None, failed

    df = pd.DataFrame(all_data).sort_values(['Company', 'Year']).reset_index(drop=True)

    # YoY growth
    df['Revenue Growth (%)']    = df.groupby('Company')['Revenue (₹Cr)'].pct_change().mul(100).round(2)
    df['Net Income Growth (%)'] = df.groupby('Company')['Net Income (₹Cr)'].pct_change().mul(100).round(2)
    df['EBITDA Growth (%)']     = df.groupby('Company')['EBITDA (₹Cr)'].pct_change().mul(100).round(2)

    return df, failed

# ── Charts ────────────────────────────────────────────────────────
def plot_revenue(df):
    companies = df['Company'].unique()
    years = sorted(df['Year'].unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(years))
    width = 0.8 / len(companies)
    for i, company in enumerate(companies):
        vals = []
        for y in years:
            row = df[(df['Company'] == company) & (df['Year'] == y)]
            vals.append(row['Revenue (₹Cr)'].values[0] / 100 if not row.empty else 0)  # ₹Cr → ₹000Cr
        bars = ax.bar([p + i*width for p in x], vals, width=width, label=company)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'₹{bar.get_height():.0f}K Cr', ha='center', va='bottom', fontsize=7)
    ax.set_xticks([p + width*(len(companies)-1)/2 for p in x])
    ax.set_xticklabels(years)
    ax.set_title('Total Revenue (₹000 Cr)', fontweight='bold')
    ax.set_ylabel('Revenue (₹000 Cr)')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_margins(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    margin_cols = ['Gross Margin (%)', 'EBITDA Margin (%)', 'Net Profit Margin (%)']
    titles = ['Gross Margin', 'EBITDA Margin', 'Net Profit Margin']
    for ax, col, title in zip(axes, margin_cols, titles):
        for company in df['Company'].unique():
            subset = df[df['Company'] == company].dropna(subset=[col])
            if subset.empty:
                continue
            ax.plot(subset['Year'], subset[col], marker='o', linewidth=2, label=company)
            for _, row in subset.iterrows():
                ax.annotate(f"{row[col]:.1f}%", (row['Year'], row[col]),
                            textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')
        ax.set_title(f'{title} (%)', fontweight='bold')
        ax.set_ylabel('%')
        ax.set_xticks(sorted(df['Year'].unique()))
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

def plot_roe_roce(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ['ROE (%)', 'ROCE (%)'], ['Return on Equity (ROE)', 'Return on Capital Employed (ROCE)']):
        for company in df['Company'].unique():
            subset = df[df['Company'] == company].dropna(subset=[col])
            if subset.empty:
                continue
            ax.plot(subset['Year'], subset[col], marker='o', linewidth=2, label=company)
            for _, row in subset.iterrows():
                ax.annotate(f"{row[col]:.1f}%", (row['Year'], row[col]),
                            textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')
        ax.set_title(f'{title} (%)', fontweight='bold')
        ax.set_ylabel('%')
        ax.set_xticks(sorted(df['Year'].unique()))
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

def plot_heatmap(df):
    pivot = df.pivot_table(index='Company', columns='Year', values='Debt-to-Asset Ratio')
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, vmin=0.2, vmax=0.9,
                cbar_kws={'label': 'Higher = more leveraged'})
    ax.set_title('Debt-to-Asset Ratio Heatmap', fontweight='bold')
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('')
    plt.tight_layout()
    return fig

def plot_valuation(df):
    """Latest year valuation multiples — bar chart."""
    latest = df.sort_values('Year').groupby('Company').last().reset_index()
    metrics = ['P/E Ratio', 'P/B Ratio', 'EV/EBITDA (x)']
    available = [m for m in metrics if latest[m].notna().any()]
    if not available:
        return None
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        data = latest[['Company', metric]].dropna()
        bars = ax.bar(data['Company'], data[metric], color=sns.color_palette("Set2", len(data)))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{bar.get_height():.1f}x', ha='center', va='bottom', fontsize=9)
        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel('Multiple')
    plt.suptitle('Valuation Multiples (Current)', fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

# ── UI ────────────────────────────────────────────────────────────
st.title("📊 India Financial Analyst")
st.caption("Powered by Gemini 2.5 Flash  |  Live data via Yahoo Finance  |  NSE/BSE auto-detect")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Enter **NSE/BSE tickers** — suffix optional. App auto-detects.")
    st.markdown("_e.g. RELIANCE, TCS, INFY, HDFCBANK_")
    ticker_input = st.text_input(
        "Ticker symbols (comma separated)",
        placeholder="e.g. RELIANCE, TCS, INFY",
        value="RELIANCE, TCS, HDFCBANK"
    )
    num_years = st.slider("Number of fiscal years", min_value=1, max_value=5, value=3)
    load_btn = st.button("🔄 Load Data", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**💡 Try asking:**")
    st.markdown("""
    - What is TCS's EBITDA margin trend?
    - Which company has the best ROE?
    - Compare ROCE across all companies
    - Which stock looks cheapest on P/E?
    - Flag any companies with weak interest coverage
    - What's driving Reliance's revenue growth?
    - Which company has the strongest free cash flow?
    """)

# ── Load data ─────────────────────────────────────────────────────
if load_btn:
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker symbol.")
        st.stop()

    with st.spinner(f"Fetching live data for {', '.join(tickers)} (auto-detecting NSE/BSE)..."):
        df, failed = fetch_financial_data(tickers, num_years)

    if df is None:
        st.error("Could not fetch data for any tickers. Please check the symbols.")
        st.stop()

    if failed:
        st.warning(f"Could not fetch data for: {', '.join(failed)}. Proceeding with available data.")

    st.session_state.df = df
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.success(f"✅ Loaded: {', '.join(df['Company'].unique())}")

# ── Main content ──────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state.df
    financial_context = df.to_json(orient='records', indent=2)

    tab1, tab2 = st.tabs(["💬 Chat", "📈 Analysis"])

    # ── Analysis tab ──
    with tab2:
        st.subheader("📋 Financial Summary")

        display_cols = [
            'Company', 'Year',
            'Revenue (₹Cr)', 'Revenue Growth (%)',
            'EBITDA (₹Cr)', 'EBITDA Margin (%)', 'EBITDA Growth (%)',
            'Net Income (₹Cr)', 'Net Profit Margin (%)',
            'ROE (%)', 'ROCE (%)',
            'Free CF (₹Cr)', 'Cash Flow Margin (%)',
            'Debt-to-Asset Ratio', 'Current Ratio', 'Interest Coverage (x)',
            'P/E Ratio', 'P/B Ratio', 'EV/EBITDA (x)'
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
        st.pyplot(plot_heatmap(df))

        val_fig = plot_valuation(df)
        if val_fig:
            st.subheader("📐 Valuation Multiples (Current)")
            st.pyplot(val_fig)

    # ── Chat tab ──
    with tab1:

        # ── Smarter system prompt ──
        SYSTEM_PROMPT = f"""
You are a senior equity research analyst specialising in Indian listed companies (NSE/BSE).
You have access to the following audited financial dataset (figures in ₹ Crore):

{financial_context}

ANALYTICAL FRAMEWORK — apply this thinking in every response:
1. TREND: Is the metric improving, deteriorating, or stable over the years?
2. BENCHMARK: How does it compare across companies in the dataset?
3. FLAGS: Highlight if any metric shows >20% YoY change, or if ratios breach key thresholds:
   - EBITDA Margin <15%  → margin pressure
   - Interest Coverage <2x → leverage risk
   - Current Ratio <1 → liquidity concern
   - ROE <10% → weak capital efficiency
   - Debt-to-Asset >0.7 → high leverage
4. INSIGHT: Give one forward-looking or contextual observation where relevant.

RESPONSE STYLE:
- Lead with the direct answer in 1 sentence
- Use 2–4 bullet points for supporting data (cite company name + year + figure)
- End with one "Analyst's Take" sentence — a sharp, opinionated insight
- Keep total response under 150 words
- Use ₹ Crore for figures, % for margins, x for multiples
- Never make up numbers — only use what is in the dataset
- If asked something outside the dataset, clearly say so and state what you CAN answer

COMPARISON REQUESTS:
- Always rank companies explicitly (1st, 2nd, 3rd...)
- Mention the spread between best and worst

VALUATION QUESTIONS:
- Note that P/E, P/B, EV/EBITDA are current (live) multiples, not historical

If the user greets you or asks a non-financial question, respond briefly and redirect to financial analysis.
"""

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []

        # Render chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything about the loaded companies..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.conversation_history.append(f"User: {prompt}")

            # Build full prompt with rolling 8-turn memory
            history_text = "\n".join(st.session_state.conversation_history[-8:])
            full_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Conversation so far:\n{history_text}\n\n"
                f"Respond as the senior equity analyst:"
            )

            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    response = model.generate_content(full_prompt)
                    reply = response.text.strip()
                    st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.conversation_history.append(f"Analyst: {reply}")

        if st.session_state.get("messages"):
            if st.button("🔄 Reset conversation"):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()

else:
    st.info("👈 Enter NSE/BSE ticker symbols in the sidebar and click **Load Data** to begin.")
