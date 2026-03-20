import streamlit as st
import google.generativeai as genai
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="BCG Financial Analyst Chatbot",
    page_icon="📊",
    layout="centered"
)

# ── Data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        'Company': ['Apple','Apple','Apple',
                    'Microsoft','Microsoft','Microsoft',
                    'Tesla','Tesla','Tesla'],
        'Year': [2022,2023,2024,
                 2022,2023,2024,
                 2022,2023,2024],
        'Total Revenue':                      [394328,383285,391035,
                                               198270,211915,245122,
                                                81462, 96773, 97690],
        'Net Income':                          [99803,96995,93736,
                                                72738,72361,88136,
                                                12556,14997, 7091],
        'Total Assets':                        [352755,352583,364980,
                                                364840,411976,512163,
                                                 82338,106618,122070],
        'Total Liabilities':                   [302083,290437,308030,
                                                198298,205753,243686,
                                                 36440, 43009, 48390],
        'Cash Flow from Operating Activities': [122151,110543,118254,
                                                 89035, 87582,118548,
                                                 14724, 13256, 14923],
    }
    df = pd.DataFrame(data)
    df = df.sort_values(['Company','Year']).reset_index(drop=True)
    df['Net Profit Margin (%)']  = (df['Net Income']/df['Total Revenue']*100).round(2)
    df['Cash Flow Margin (%)']   = (df['Cash Flow from Operating Activities']/df['Total Revenue']*100).round(2)
    df['Debt-to-Asset Ratio']    = (df['Total Liabilities']/df['Total Assets']).round(2)
    df['Revenue Growth (%)']     = df.groupby('Company')['Total Revenue'].pct_change().mul(100).round(2)
    df['Net Income Growth (%)']  = df.groupby('Company')['Net Income'].pct_change().mul(100).round(2)
    return df

df = load_data()
financial_context = df.to_json(orient='records', indent=2)

# ── Gemini setup ─────────────────────────────────────────────────
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

SYSTEM_PROMPT = """
You are a financial analyst chatbot built by a BCG GenAI consulting team.
You have access to SEC 10-K financial data for Apple, Microsoft, and Tesla
for fiscal years 2022, 2023, and 2024.

Here is the complete financial dataset:
{context}

RULES:
- Only answer using the data provided above
- Always mention company name and fiscal year
- Keep responses concise and consultant-style (2-4 sentences max)
- If a metric shows >20% change, flag it as noteworthy
- If asked to compare, rank companies clearly
- If question is outside dataset say: "That's outside my dataset. I can answer
  questions about revenue, net income, assets, liabilities, cash flow, margins,
  and debt ratios for Apple, Microsoft, and Tesla (FY2022-2024)."
- Never make up numbers
""".format(context=financial_context)

# ── UI ───────────────────────────────────────────────────────────
st.title("📊 BCG Financial Analyst Chatbot")
st.caption("Powered by Gemini 2.5 Flash  |  Data: SEC EDGAR 10-K Filings  |  Apple · Microsoft · Tesla  |  FY2022–2024")

with st.expander("💡 Try asking..."):
    st.markdown("""
    - What was Apple's revenue in 2024?
    - What about net income?
    - Compare margins across all three companies
    - Which company is most leveraged?
    - What happened to Tesla's profitability?
    - Which company has the strongest growth profile?
    """)

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Apple, Microsoft, or Tesla financials..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append(f"User: {prompt}")

    # Build full prompt with history
    history_text = "\n".join(st.session_state.conversation_history[-6:])
    full_prompt = f"{SYSTEM_PROMPT}\n\nConversation so far:\n{history_text}\n\nRespond as the financial analyst chatbot:"

    # Call Gemini
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = model.generate_content(full_prompt)
            reply = response.text.strip()
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.conversation_history.append(f"Bot: {reply}")

# Reset button
if st.session_state.messages:
    if st.button("🔄 Reset conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
