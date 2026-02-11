import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="🧞‍♂️ Geenie - Core Risk + Monte Carlo",
    layout="wide"
)

st.title("🧞‍♂️ Geenie - Core Risk Engine with Monte Carlo Projections")
st.caption("Institutional-grade analytics and forward-looking scenario simulation by Jamshid Farook")

# -----------------------------
# How to Use Geenie (User Guide)
# -----------------------------
with open("how_to_use_geenie.pdf", "rb") as pdf_file:
    st.download_button(
        label="How to Use Geenie - Download User Guide (PDF)",
        data=pdf_file,
        file_name="Geenie_User_Guide.pdf",
        mime="application/pdf",
    )

# -----------------------------
# CSV Loader
# -----------------------------
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)

    # Detect date column
    date_col = None
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            if parsed.notna().sum() > len(df) * 0.5:
                date_col = col
                break
        except:
            continue
    if date_col is None:
        raise ValueError("No valid date column detected.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Detect numeric price columns
    price_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not price_cols:
        raise ValueError("No numeric price columns found.")

    return df, price_cols

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded_file is None:
    st.info("Upload a CSV to start analysis.")
    st.stop()

# Load data
df, price_cols = load_csv(uploaded_file)

# Date selection
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date, end_date = st.sidebar.date_input(
    "Analysis Window",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Convert to Timestamps to match index type
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

df = df.loc[start_date:end_date]

# Prices series
prices = df[price_cols].dropna()
if len(price_cols) == 1:
    asset_name = price_cols[0]
    price_series = prices[asset_name]
else:
    asset_name = "Equal-Weighted Portfolio"
    price_series = prices.mean(axis=1)

# -----------------------------
# Core Risk Metrics
# -----------------------------
returns = price_series.pct_change().dropna()
annual_return = (1 + returns.mean()) ** 252 - 1
annual_volatility = returns.std() * np.sqrt(252)

drawdown = price_series / price_series.cummax() - 1
max_drawdown = drawdown.min()
down_days_pct = (returns < 0).mean() * 100

# -----------------------------
# Rolling Volatility & Regime
# -----------------------------
rolling_vol = returns.rolling(252).std() * np.sqrt(252)
vol_threshold = rolling_vol.median()
regime = np.where(rolling_vol > vol_threshold, "High Volatility", "Low Volatility")
regime_df = pd.DataFrame({"Returns": returns, "Volatility": rolling_vol, "Regime": regime}).dropna()
regime_stats = regime_df.groupby("Regime")["Returns"].agg(["mean", "std", "count"])

# -----------------------------
# Stress Periods (>30%)
# -----------------------------
stress_periods = drawdown[drawdown < -0.3]

# -----------------------------
# Display Metrics
# -----------------------------
st.subheader(f"📊 Asset Overview - {asset_name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Annualized Return", f"{annual_return*100:.2f}%")
c2.metric("Annualized Volatility", f"{annual_volatility*100:.2f}%")
c3.metric("Maximum Drawdown", f"{max_drawdown*100:.2f}%")
c4.metric("Downside Frequency", f"{down_days_pct:.1f}%")

st.subheader("🧠 Risk Behavior Summary")
st.write(
    f"""
- {asset_name} experienced negative returns on **{down_days_pct:.1f}%** of trading days.
- The largest drawdown observed was **{max_drawdown*100:.2f}%**.
- Volatility indicates this asset behaves as a **{
        'high-risk' if annual_volatility > 0.25 else
        'moderate-risk' if annual_volatility > 0.15 else
        'low-risk'
    }** instrument.
"""
)

st.subheader("📈 Market Regime Analysis")
st.write("Returns by volatility regime:")
st.dataframe(regime_stats.style.format("{:.4f}"))

st.subheader("📉 Rolling Volatility (252-day)")
st.line_chart(rolling_vol.dropna())

st.subheader("⚠️ Stress Periods (Drawdown < -30%)")
st.write(f"Number of stress days: {len(stress_periods)}")
if len(stress_periods) > 0:
    st.line_chart(stress_periods)

# -----------------------------
# Price & Drawdown Charts
# -----------------------------
st.subheader("📈 Price Evolution")
st.line_chart(price_series)

st.subheader("📉 Drawdown Profile")
st.area_chart(drawdown)

# -----------------------------
# Monte Carlo Scenario Testing Explanation
# -----------------------------
st.subheader("🎲 Monte Carlo Scenario Testing - How to Read Charts")
st.markdown(
    """
Each Monte Carlo chart shows **simulated future price paths** based on historical returns:
- **50th percentile (median)** → Expected central path of prices
- **5th percentile** → Worst-case scenario (low end)
- **95th percentile** → Best-case scenario (high end)
These charts give a **range of possible outcomes**, not a single prediction.
"""
)

# -----------------------------
# Monte Carlo Multi-Horizon Forecasts
# -----------------------------
st.subheader("📊 Multi-Horizon Forward Price Projections")

num_simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000)
last_price = price_series.iloc[-1]
mean_return = returns.mean()
vol = returns.std()

forecast_horizons = {
    "Next Day": 1,
    "Next Month (~21 trading days)": 21,
    "Next Year (~252 trading days)": 252,
    "Next 5 Years (~1260 trading days)": 252*5
}

for label, days in forecast_horizons.items():
    all_paths = []
    for _ in range(num_simulations):
        simulated_returns = np.random.normal(mean_return, vol, days)
        path = last_price * np.cumprod(1 + simulated_returns)
        all_paths.append(path)

    all_paths = np.array(all_paths)
    p5 = np.percentile(all_paths, 5, axis=0)
    p50 = np.percentile(all_paths, 50, axis=0)
    p95 = np.percentile(all_paths, 95, axis=0)

    if label == "Next Day":
        # Display as text
        st.markdown(f"### {label} Price Projection (Text)")
        st.write(
            f"**Median next-day price:** {p50[-1]:.2f}  \n"
            f"**5th percentile (worst-case):** {p5[-1]:.2f}  \n"
            f"**95th percentile (best-case):** {p95[-1]:.2f}"
        )
    else:
        # Display as chart
        st.markdown(f"### {label} Price Projection (Chart)")
        st.line_chart(pd.DataFrame({"5th %": p5, "50th %": p50, "95th %": p95}))

        # Show key stats below chart
        expected_price = p50[-1]
        worst_case = p5[-1]
        best_case = p95[-1]

        st.write(
            f"Median projected price: **{expected_price:.2f}**, "
            f"5th percentile: **{worst_case:.2f}**, "
            f"95th percentile: **{best_case:.2f}**"
        )

# -----------------------------
# Data Coverage
# -----------------------------
st.subheader("🧾 Dataset Coverage")
start_str = price_series.index.min().strftime("%Y-%m-%d")
end_str = price_series.index.max().strftime("%Y-%m-%d")
c5, c6, c7 = st.columns(3)
c5.metric("Observations", len(price_series))
c6.metric("Start Date", start_str)
c7.metric("End Date", end_str)

with st.expander("🔍 Raw Data Preview"):
    st.dataframe(df.tail(10))

# -----------------------------
# Footer / About the Maker
# -----------------------------
st.markdown(
    """
    <hr>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        <p><b>🧞‍♂️ Geenie - Core Risk Engine & Monte Carlo Projections</b></p>
        <p>&copy; 2026 Jamshid Farook</p>
        <br>
        <p><b>About the Maker</b></p>
        <p>
        Jamshid Farook<br>
        Data Analyst
        </p>
        <p style="max-width:600px; margin:auto;">
        Detail-oriented Data Analyst with hands-on experience in Python, SQL, Excel,
        and data visualization tools. IBM Data Analyst Certified with proven
        experience in dashboards, real-world datasets, and actionable insights.
        </p>
        <p>
        <a href="https://jamshidfarook.github.io/" target="_blank">Portfolio</a>
        &nbsp;|&nbsp;
        <a href="https://www.linkedin.com/in/jamshidfarook/" target="_blank">LinkedIn</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
