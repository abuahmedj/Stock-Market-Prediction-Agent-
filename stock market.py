import streamlit as st
import yfinance as yf
import pandas as pd
import time
import plotly.graph_objects as go # Added for charts
from dataclasses import dataclass, field
from typing import List
from openai import OpenAI

# ================= CONFIG ================= #
# (Make sure to set your API key in your environment or below)
OPENROUTER_API_KEY = ""

SENTIMENT_MODELS = [
    "z-ai/glm-4.5-air:free",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct:free"
]

DECISION_MODELS = [
    "z-ai/glm-4.5-air:free",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct:free"
]

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ================= STATE ================= #
@dataclass
class AgentState:
    symbol: str
    price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    news: str = ""
    sentiment: str = ""
    decision: str = ""
    # Store history for charting
    history_df: pd.DataFrame = field(default_factory=pd.DataFrame) 
    memory: List[str] = field(default_factory=list)

# ================= SAFE LLM CALL ================= #
def call_llm(models, prompt, retries=2):
    for model in models:
        for _ in range(retries):
            try:
                res = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return res.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2)
                    continue
                else:
                    return f"LLM Error: {e}"
    return "LLM Error: All free models are rate-limited."

# ================= DATA AGENT (REAL DATA) ================= #
def data_agent(state):
    try:
        ticker = yf.Ticker(state.symbol)
        # Fetching 1 month for better charting, though we focus on last 5d for metrics
        hist = ticker.history(period="1mo")

        if hist.empty:
            raise Exception("No market data")

        state.history_df = hist
        state.price = round(hist["Close"].iloc[-1], 2)
        state.high = round(hist["High"].iloc[-1], 2)
        state.low = round(hist["Low"].iloc[-1], 2)

        news_items = ticker.news[:5] if ticker.news else []
        state.news = " ".join(
            [n.get("title", "") for n in news_items]
        )

        if not state.news:
            state.news = "No major market-moving news today."

    except Exception as e:
        state.price = state.high = state.low = 0.0
        state.news = f"Market data unavailable: {e}"

# ================= AGENTS ================= #
def sentiment_agent(state):
    prompt = f"""
    Analyze the stock sentiment.
    Stock: {state.symbol}
    Price: {state.price}
    High: {state.high}
    Low: {state.low}
    News: {state.news}
    Respond with exactly one word: Positive, Neutral, or Negative. 
    Then add a short explanation on a new line.
    """
    state.sentiment = call_llm(SENTIMENT_MODELS, prompt)

def decision_agent(state):
    prompt = f"""
    You are a trading decision agent.
    Sentiment: {state.sentiment}
    Price: {state.price}
    High: {state.high}
    Low: {state.low}
    Choose one: BUY, HOLD, or SELL.
    Explain briefly.
    """
    state.decision = call_llm(DECISION_MODELS, prompt)
    state.memory.append(state.decision)

def backtest_agent(state):
    if len(state.memory) < 1:
        return None
    return pd.DataFrame({
        "Run": range(1, len(state.memory) + 1),
        "Decision": state.memory
    })

# ================= IMPROVED STREAMLIT UI ================= #
st.set_page_config(page_title="Agentic AI Trader", page_icon="📈", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    symbol = st.text_input("Stock Symbol", "AAPL").upper()
    st.divider()
    st.markdown("### 🤖 Agents Active")
    st.info("Market Data Agent")
    st.info("Sentiment Analysis Agent")
    st.info("Decision Engine Agent")
    
    if st.button("🚀 Run Analysis", type="primary"):
        run_analysis = True
    else:
        run_analysis = False

# --- Main Content ---
st.title("📈 Agentic AI Stock Market Predictor")
st.markdown(f"Real-time analysis for **{symbol}**")

if "memory" not in st.session_state:
    st.session_state.memory = []

if run_analysis:
    state = AgentState(symbol=symbol, memory=st.session_state.memory)

    # 1. Fetch Data
    with st.status("🤖 AI Agents working...", expanded=True) as status:
        st.write("Fetching real-time market data...")
        data_agent(state)
        st.write("Analyzing news sentiment...")
        sentiment_agent(state)
        st.write("Calculating trading decision...")
        decision_agent(state)
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    st.session_state.memory = state.memory
    
    # 2. Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${state.price}")
    with col2:
        st.metric("Daily High", f"${state.high}")
    with col3:
        st.metric("Daily Low", f"${state.low}")
    with col4:
        # Dynamic color for decision
        decision_short = state.decision.split()[0].upper()
        if "BUY" in decision_short:
            st.metric("AI Recommendation", "BUY", delta="Strong Buy", delta_color="normal")
        elif "SELL" in decision_short:
            st.metric("AI Recommendation", "SELL", delta="-Strong Sell", delta_color="inverse")
        else:
            st.metric("AI Recommendation", "HOLD", delta_color="off")

    st.divider()

    # 3. Tabs for Details
    tab_overview, tab_analysis, tab_history = st.tabs(["📊 Market Overview", "🧠 AI Logic", "📜 Backtest Log"])

    with tab_overview:
        # Plotly Candlestick Chart
        if not state.history_df.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=state.history_df.index,
                open=state.history_df['Open'],
                high=state.history_df['High'],
                low=state.history_df['Low'],
                close=state.history_df['Close']
            )])
            fig.update_layout(title=f"{symbol} - 1 Month Price Action", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No historical data to display.")

    with tab_analysis:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📰 Market News")
            st.info(state.news)
        with c2:
            st.subheader("🧠 Sentiment Analysis")
            if "Positive" in state.sentiment:
                st.success(state.sentiment)
            elif "Negative" in state.sentiment:
                st.error(state.sentiment)
            else:
                st.warning(state.sentiment)
            
            st.subheader("📈 Trading Strategy")
            st.write(state.decision)

    with tab_history:
        st.subheader("Agent Memory")
        df = backtest_agent(state)
        if df is not None:
            # Highlight Buy/Sell rows
            def highlight_decision(val):
                color = 'green' if 'BUY' in str(val).upper() else 'red' if 'SELL' in str(val).upper() else 'orange'
                return f'color: {color}'
            
            st.dataframe(df.style.map(highlight_decision, subset=['Decision']), use_container_width=True)
        else:
            st.info("Run the agent multiple times to build a decision history.")

else:
    st.info("👈 Enter a stock symbol in the sidebar and click 'Run Analysis' to start.")