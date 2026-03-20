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
    page_title="Financial Analyst Chatbot",
    page_icon="📊",
    layout="wide"
)

# ── Gemini setup ─────────────────────────────────────────────────
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# ── Data fetching ─────────────────────────────────────────────────
def fetch_financial_data(tickers, num_years):
    all_data = []
    failed = []

    for ticker_str in tickers:
        ticker_str = ticker_str.strip().upper()
        try:
            t = yf.Ticker(ticker_str)
            inc = t.financials
            cf  = t.cashflow
            bs  = t.balance_sheet

            if inc.empty or cf.empty or bs.empty:
                failed.append(ticker_str)
                continue

            # Take only the requested number of years
            cols = inc.columns[:num_years]

            for col in cols:
                year = col.year

                # Income statement
                revenue   = inc.loc['Total Revenue', col]          if 'Total Revenue'   in inc.index else None
                net_income= inc.loc['Net Income', col]             if 'Net Income'       in inc.index else None

                # Cash flow
                ocf = None
                for ocf_key in ['Operating Cash Flow',
                                 'Cash Flow From Continuing Operating Activities',
                                 'Total Cash From Operating Activities']:
                    if ocf_key in cf.index:
                        ocf = cf.loc[ocf_key, col] if col in cf.columns else None
                        if ocf is not None:
                            break

                # Balance sheet
                total_assets = None
                for ta_key in ['Total Assets']:
                    if ta_key in bs.index:
                        total_assets = bs.loc[ta_key, col] if col in bs.columns else None
                        break

                total_liab = None
                for tl_key in ['Total Liabilities Net Minority Interest',
                                'Total Liabilities']:
                    if tl_key in bs.index:
                        total_liab = bs.loc[tl_key, col] if col in bs.columns else None
                        if total_liab is not None:
                            break

                if any(v is None for v in [revenue, net_income, ocf, total_assets, total_liab]):
                    continue

                all_data.append({
                    'Company':    ticker_str,
                    'Year':       year,
                    'Total Revenue ($mm)':                    round(revenue / 1e6, 2),
                    'Net Income ($mm)':                       round(net_income / 1e6, 2),
                    'Total Assets ($mm)':                     round(total_assets / 1e6, 2),
                    'Total Liabilities ($mm)':                round(total_liab / 1e6, 2),
                    'Operating Cash Flow ($mm)':              round(ocf / 1e6, 2),
                })

        except Exception:
            failed.append(ticker_str)

    if not all_data:
        return None, failed

    df = pd.DataFrame(all_data).sort_values(['Company', 'Year']).reset_index(drop=True)

    # Derived metrics
    df['Net Profit Margin (%)']  = (df['Net Income ($mm)'] / df['Total Revenue ($mm)'] * 100).round(2)
    df['Cash Flow Margin (%)']   = (df['Operating Cash Flow ($mm)'] / df['Total Revenue ($mm)'] * 100).round(2)
    df['Debt-to-Asset Ratio']    = (df['Total Liabilities ($mm)'] / df['Total Assets ($mm)']).round(2)
    df['Revenue Growth (%)']     = df.groupby('Company')['Total Revenue ($mm)'].pct_change().mul(100).round(2)
    df['Net Income Growth (%)']  = df.groupby('Company')['Net Income ($mm)'].pct_change().mul(100).round(2)

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
            vals.append(row['Total Revenue ($mm)'].values[0] / 1000 if not row.empty else 0)
        bars = ax.bar([p + i*width for p in x], vals, width=width, label=company)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'${bar.get_height():.0f}B', ha='center', va='bottom', fontsize=7)
    ax.set_xticks([p + width*(len(companies)-1)/2 for p in x])
    ax.set_xticklabels(years)
    ax.set_title('Total Revenue ($B)', fontweight='bold')
    ax.set_ylabel('Revenue ($ Billions)')
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}B'))
    plt.tight_layout()
    return fig

def plot_margins(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    for company in df['Company'].unique():
        subset = df[df['Company'] == company]
        ax.plot(subset['Year'], subset['Net Profit Margin (%)'],
                marker='o', linewidth=2, label=company)
        for _, row in subset.iterrows():
            ax.annotate(f"{row['Net Profit Margin (%)']:.1f}%",
                        (row['Year'], row['Net Profit Margin (%)']),
                        textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')
    ax.set_title('Net Profit Margin Trend (%)', fontweight='bold')
    ax.set_ylabel('Net Profit Margin (%)')
    ax.set_xticks(sorted(df['Year'].unique()))
    ax.legend()
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

# ── UI ────────────────────────────────────────────────────────────
st.title("📊 Financial Analyst Chatbot")
st.caption("Powered by Gemini 2.5 Flash  |  Live data via Yahoo Finance  |  Any publicly traded company")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_input = st.text_input(
        "Enter ticker symbols (comma separated)",
        placeholder="e.g. AAPL, MSFT, TSLA, NVDA",
        value="AAPL, MSFT, TSLA"
    )
    num_years = st.slider("Number of fiscal years", min_value=1, max_value=5, value=3)
    load_btn = st.button("🔄 Load Data", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**💡 Try asking:**")
    st.markdown("""
    - What was [TICKER]'s revenue last year?
    - Compare margins across all companies
    - Which company is most leveraged?
    - What's the revenue growth trend?
    - Which company has the best cash flow?
    """)

# ── Load data on button click ─────────────────────────────────────
if load_btn or "df" not in st.session_state:
    if load_btn:
        tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker symbol.")
            st.stop()

        with st.spinner(f"Fetching live financial data for {', '.join(tickers)}..."):
            df, failed = fetch_financial_data(tickers, num_years)

        if df is None:
            st.error("Could not fetch data for any of the provided tickers. Please check the symbols and try again.")
            st.stop()

        if failed:
            st.warning(f"Could not fetch data for: {', '.join(failed)}. Proceeding with available data.")

        st.session_state.df = df
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.success(f"✅ Loaded data for: {', '.join(df['Company'].unique())}")

# ── Main content ──────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state.df
    financial_context = df.to_json(orient='records', indent=2)

    # Tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📈 Analysis"])

    with tab2:
        st.subheader("Financial Overview")
        st.dataframe(df[['Company','Year','Total Revenue ($mm)','Net Income ($mm)',
                          'Net Profit Margin (%)','Cash Flow Margin (%)','Debt-to-Asset Ratio',
                          'Revenue Growth (%)']].round(2),
                     use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_revenue(df))
        with col2:
            st.pyplot(plot_margins(df))
        st.pyplot(plot_heatmap(df))

    with tab1:
        SYSTEM_PROMPT = f"""
You are a financial analyst chatbot with access to live SEC financial data.
Here is the complete financial dataset you must use:
{financial_context}

RULES:
- Only answer using the data provided
- Always mention company name and fiscal year
- Keep responses concise and consultant-style (2-4 sentences max)
- If a metric shows >20% change, flag it as noteworthy
- If asked to compare, rank companies clearly
- If question is outside dataset say what you CAN answer
- Never make up numbers
"""

        if "messages" not in st.session_state:
            st.session_state.messages = []
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

            history_text = "\n".join(st.session_state.conversation_history[-6:])
            full_prompt = f"{SYSTEM_PROMPT}\n\nConversation:\n{history_text}\n\nRespond as the financial analyst:"

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = model.generate_content(full_prompt)
                    reply = response.text.strip()
                    st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.conversation_history.append(f"Bot: {reply}")

        if st.session_state.get("messages"):
            if st.button("🔄 Reset conversation"):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()

else:
    st.info("👈 Enter ticker symbols in the sidebar and click **Load Data** to get started.")
