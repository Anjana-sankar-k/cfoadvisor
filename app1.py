import streamlit as st
import pandas as pd
from pipeline.parser import parse_file, df_to_text_chunks
from pipeline.embedder import embed_and_store, retrieve
from pipeline.kpis import calculate_kpis
from pipeline.llm import ask_llm

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(page_title="CFO Copilot", page_icon="📊", layout="wide")

# ── Minimal Styling ────────────────────────────────────────────
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.stMetric {
    background-color: #0E1117;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("CFO Copilot")
st.caption("Your AI-powered financial command center")

# ── Session State ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "df" not in st.session_state:
    st.session_state.df = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Data")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    st.caption("Don't have data? Use the sample file in sample_data/")

    if uploaded:
        with st.spinner("Parsing and embedding your data..."):
            df = parse_file(uploaded)
            if df is not None:
                chunks = df_to_text_chunks(df)
                embed_and_store(chunks)

                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.chat_history = []
                st.session_state.summary = None

                st.success(f"Loaded {len(df)} months of data.")
                st.toast("Data processed successfully")

# ── Stop if no data ────────────────────────────────────────────
if not st.session_state.data_loaded:
    st.info("Upload your financial data in the sidebar to get started.")
    st.stop()

df = st.session_state.df

# ── KPI Section ────────────────────────────────────────────────
kpis = calculate_kpis(df)

st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    val = f"₹{kpis['burn_rate']:,.0f}/mo" if kpis.get("burn_rate") else "N/A"
    st.metric("Burn Rate", val)

with col2:
    val = f"{kpis['runway_months']} months" if kpis.get("runway_months") else "N/A"
    st.metric("Runway", val)

with col3:
    val = f"{kpis['gross_margin_pct']}%" if kpis.get("gross_margin_pct") else "N/A"
    st.metric("Gross Margin", val)

with col4:
    val = f"₹{kpis['ebitda']:,.0f}" if kpis.get("ebitda") is not None else "N/A"
    delta = f"{kpis['revenue_trend_pct']}% vs prior 3mo" if kpis.get("revenue_trend_pct") else None
    st.metric("EBITDA", val, delta=delta)

# ── Financial Summary (Jarvis moment) ──────────────────────────
st.subheader("Financial Health Summary")

if st.session_state.summary is None:
    with st.spinner("Analyzing your company..."):
        summary_prompt = """
        Give a CFO-style summary:
        - Overall financial health
        - Burn and runway insight
        - Key trend
        Keep it under 4 lines.
        """
        context = retrieve("overall financial health summary")
        summary = ask_llm(summary_prompt, context, [])
        st.session_state.summary = summary

st.info(st.session_state.summary)

st.divider()

# ── Insights Section ───────────────────────────────────────────
st.subheader("Key Insights")

insight_qs = [
    "Where am I spending the most?",
    "What is the biggest financial risk?",
    "What trend should I be worried about?"
]

cols = st.columns(3)

for i, q in enumerate(insight_qs):
    with cols[i]:
        with st.spinner("..."):
            context = retrieve(q)
            ans = ask_llm(q, context, [])
        st.markdown(f"**{q}**")
        st.write(ans)

st.divider()

# ── Charts ─────────────────────────────────────────────────────
st.subheader("Trends")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if "total_revenue" in df.columns and "total_expenses" in df.columns:
        chart_df = df.set_index("month")[["total_revenue", "total_expenses"]]
        st.line_chart(chart_df)

with chart_col2:
    if "net_cash_flow" in df.columns:
        st.bar_chart(df.set_index("month")["net_cash_flow"])

st.divider()

# ── Chat Section ───────────────────────────────────────────────
st.subheader("Ask Your CFO Copilot")

st.markdown("""
**Try asking:**
- What is my burn rate?
- Which month had highest expenses?
- How long can I survive?
- What should I be worried about?
""")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask anything about your company finances...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # deterministic override for accuracy
            if "burn rate" in question.lower() and kpis.get("burn_rate"):
                answer = f"Your burn rate is ₹{kpis['burn_rate']:,.0f} per month."

            elif "runway" in question.lower() and kpis.get("runway_months"):
                answer = f"Your current runway is approximately {kpis['runway_months']} months."

            else:
                context = retrieve(question)
                answer = ask_llm(question, context, st.session_state.chat_history)

                with st.expander("See retrieved data"):
                    st.write(context)

        st.write(answer)

    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})