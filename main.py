import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from datetime import timedelta
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Streamlit page configurations
st.set_page_config(
    page_title="Street Quant",
    layout="wide"
)

# Constants
RISK_FREE_RATE = 0.02 
TRADING_DAYS = 252     

@st.cache_data(ttl=3600, show_spinner="Fetching market data")
def fetch_data(ticker, start_date, end_date):
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            # Fetching primary data
            data = yf.download(ticker, start=start_date, end=end_date, 
                              auto_adjust=True, progress=False, threads=True)
            
            # Fetch benchmark data
            benchmark = yf.download('SPY', start=start_date, end=end_date,
                                  auto_adjust=True, progress=False, threads=True)
            
            if not data.empty and not benchmark.empty:
                # Returns and risk metrics
                data['Daily_Return'] = data['Close'].pct_change()
                benchmark['Daily_Return'] = benchmark['Close'].pct_change()
                
                return data, benchmark, None
            
            # Fallback to Ticker API if download fails
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            if not data.empty:
                data['Daily_Return'] = data['Close'].pct_change()
                return data, None, None
                
            return None, None, "No data returned for the given ticker and date range."
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return None, None, "Rate limited by Yahoo Finance. Try again later."
            return None, None, f"Error: {str(e)}"

def calculate_metrics(data, benchmark=None):
    if data is None or data.empty:
        return {}
    
    metrics = {}
    returns = data['Close'].pct_change().dropna()
    
    # Basic metrics 
    metrics['Cumulative_Return'] = float((1 + returns).prod() - 1)
    metrics['Annualized_Return'] = float((1 + metrics['Cumulative_Return'])**(TRADING_DAYS/len(data)) - 1)
    metrics['Annualized_Volatility'] = float(returns.std() * np.sqrt(TRADING_DAYS))
    
    # Fixed Sharpe ratio calculation
    annual_vol = metrics['Annualized_Volatility']
    if isinstance(annual_vol, (int, float)) and annual_vol != 0:
        metrics['Sharpe_Ratio'] = float((metrics['Annualized_Return'] - RISK_FREE_RATE) / annual_vol)
    else:
        metrics['Sharpe_Ratio'] = 0.0
    
    # Risk metrics
    metrics['Max_Drawdown'] = float((1 - data['Close'] / data['Close'].cummax()).max())
    metrics['VaR_95'] = float(returns.quantile(0.05))
    metrics['CVaR_95'] = float(returns[returns <= metrics['VaR_95']].mean())
    
    # Statistical tests
    adf_result = adfuller(data['Close'].dropna())
    metrics['ADF_Statistic'] = float(adf_result[0])
    metrics['ADF_pvalue'] = float(adf_result[1])
    
    # Correlation with benchmark
    if benchmark is not None and not benchmark.empty and 'Daily_Return' in benchmark.columns:
        merged = pd.merge(data['Daily_Return'], benchmark['Daily_Return'], 
                         left_index=True, right_index=True, how='inner')
        corr_matrix = merged.corr()
        if not corr_matrix.empty and len(corr_matrix) > 1:
            beta = float(corr_matrix.iloc[0,1] * (data['Daily_Return'].std() / benchmark['Daily_Return'].std())) if benchmark['Daily_Return'].std() != 0 else 0.0
            metrics['Beta'] = beta
            benchmark_annual_return = float(benchmark['Daily_Return'].mean() * TRADING_DAYS)
            metrics['Alpha'] = float(metrics['Annualized_Return'] - (RISK_FREE_RATE + beta * 
                                                             (benchmark_annual_return - RISK_FREE_RATE)))
    
    return metrics

def plot_interactive_chart(data):
    if isinstance(data['Close'], pd.DataFrame):
        close_series = data['Close'].iloc[:, 0]
    else:
        close_series = data['Close']

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=close_series,
        increasing_line_color='#2ECC71',
        decreasing_line_color='#E74C3C'
    )])

    # Moving averages
    for window in [20, 50]:
        ma_col = f'MA_{window}'
        data[ma_col] = close_series.rolling(window).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[ma_col],
            name=f'{window}-Day MA',
            line=dict(width=1)
        ))

    # Bollinger Bands
    data['MA_20'] = close_series.rolling(20).mean()
    rolling_std = close_series.rolling(20).std()
    data['Upper_Band'] = data['MA_20'] + 2 * rolling_std
    data['Lower_Band'] = data['MA_20'] - 2 * rolling_std

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Upper_Band'],
        name='Upper Bollinger Band',
        line=dict(color='rgba(200, 200, 200, 0.5)', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Lower_Band'],
        name='Lower Bollinger Band',
        line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
        fill='tonexty'
    ))

    fig.update_layout(
        title=f'{ticker} Price Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_dark'
    )

    return fig

def plot_returns_distribution(data):
    returns = data['Daily_Return'].dropna()
    
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Returns Distribution', 'Q-Q Plot'),
                       vertical_spacing=0.15)
    
    # Histogram with KDE
    fig.add_trace(go.Histogram(
        x=returns, 
        name='Returns',
        nbinsx=50,
        marker_color='#3498DB',
        opacity=0.75,
        histnorm='probability density'
    ), row=1, col=1)
    
    # KDE curve
    kde_x = np.linspace(returns.min(), returns.max(), 100)
    kde_y = stats.gaussian_kde(returns)(kde_x)
    fig.add_trace(go.Scatter(
        x=kde_x, 
        y=kde_y,
        name='KDE',
        line=dict(color='#E74C3C', width=2)
    ), row=1, col=1)
    
    # Q-Q Plot
    qq = stats.probplot(returns, dist="norm")
    fig.add_trace(go.Scatter(
        x=qq[0][0], 
        y=qq[0][1],
        mode='markers',
        name='Returns',
        marker=dict(color='#3498DB')
    ), row=2, col=1)
    
    # Theoretical quantiles line
    fig.add_trace(go.Scatter(
        x=qq[0][0], 
        y=qq[1][0] * qq[0][0] + qq[1][1],
        name='Normal Distribution',
        line=dict(color='#E74C3C', width=2)
    ), row=2, col=1)
    
    fig.update_layout(
        title='Returns Distribution Analysis',
        height=700,
        showlegend=False,
        template='plotly_dark'
    )
    return fig

def plot_rolling_metrics(data):
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=('Rolling Volatility (21-day)', 
                                      'Rolling Sharpe Ratio (63-day)',
                                      'Drawdown Analysis'),
                       vertical_spacing=0.1)
    
    # Rolling volatility
    rolling_vol = data['Daily_Return'].rolling(21).std() * np.sqrt(21)
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol,
        name='Volatility',
        line=dict(color='#3498DB')
    ), row=1, col=1)
    
    # Rolling Sharpe
    rolling_sharpe = (data['Daily_Return'].rolling(63).mean() * 63 - RISK_FREE_RATE) / \
                    (data['Daily_Return'].rolling(63).std() * np.sqrt(63))
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0)
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe,
        name='Sharpe Ratio',
        line=dict(color='#2ECC71')
    ), row=2, col=1)
    
    # Drawdown analysis
    cumulative = (1 + data['Daily_Return']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.3)',
        line=dict(color='#E74C3C')
    ), row=3, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        template='plotly_dark'
    )
    
    return fig

def display_metrics(data, metrics):
    st.subheader("Key Values:")

    def colorize(val, good='high', threshold=0):
        if val is None:
            return "black"
        if good == 'high':
            return "#2ecc71" if val > threshold else "#e74c3c"
        else:
            return "#2ecc71" if val < threshold else "#e74c3c"

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div style="font-size:1.2em;">
        <b>Cumulative Return</b><br>
        <span style="color:{colorize(metrics.get('Cumulative_Return', 0))}; font-size:2em;">
            {metrics.get('Cumulative_Return', 0)*100:.2f}%</span>
        <br><span style="font-size:0.9em; color:#888;">Total return over the selected period.</span>
    </div>
    <hr style="margin:0.5em 0;">
    <div style="font-size:1.2em;">
        <b>Annualized Return</b><br>
        <span style="color:{colorize(metrics.get('Annualized_Return', 0))}; font-size:2em;">
            {metrics.get('Annualized_Return', 0)*100:.2f}%</span>
        <br><span style="font-size:0.9em; color:#888;">Yearly return, adjusted for period length.</span>
    </div>
    <hr style="margin:0.5em 0;">
    <div style="font-size:1.2em;">
        <b>Annualized Volatility</b><br>
        <span style="color:{colorize(metrics.get('Annualized_Volatility', 0), good='low', threshold=0.3)}; font-size:2em;">
            {metrics.get('Annualized_Volatility', 0)*100:.2f}%</span>
        <br><span style="font-size:0.9em; color:#888;">Annualized risk (std. dev. of returns).</span>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style="font-size:1.2em;">
        <b>Sharpe Ratio</b><br>
        <span style="color:{colorize(metrics.get('Sharpe_Ratio', 0), threshold=1)}; font-size:2em;">
            {metrics.get('Sharpe_Ratio', 0):.2f}</span>
        <br><span style="font-size:0.9em; color:#888;">Risk-adjusted return. Above 1 is good.</span>
    </div>
    <hr style="margin:0.5em 0;">
    <div style="font-size:1.2em;">
        <b>Max Drawdown</b><br>
        <span style="color:{colorize(metrics.get('Max_Drawdown', 0), good='low', threshold=0.2)}; font-size:2em;">
            {metrics.get('Max_Drawdown', 0)*100:.2f}%</span>
        <br><span style="font-size:0.9em; color:#888;">Worst peak-to-trough loss.</span>
    </div>
    <hr style="margin:0.5em 0;">
    <div style="font-size:1.2em;">
        <b>Value at Risk (95%)</b><br>
        <span style="color:{colorize(metrics.get('VaR_95', 0), good='low', threshold=0)}; font-size:2em;">
            {metrics.get('VaR_95', 0)*100:.2f}%</span>
        <br><span style="font-size:0.9em; color:#888;">Worst daily loss (95% confidence).</span>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div style="font-size:1.2em;">
        <b>Conditional VaR</b><br>
        <span style="color:{colorize(metrics.get('CVaR_95', 0), good='low', threshold=0)}; font-size:2em;">
            {metrics.get('CVaR_95', 0)*100:.2f}%</span>
        <br><span style="font-size:0.9em; color:#888;">Average loss on worst 5% of days.</span>
    </div>
    <hr style="margin:0.5em 0;">
    <div style="font-size:1.2em;">
        <b>ADF Statistic</b><br>
        <span style="color:#2980b9; font-size:2em;">
            {metrics.get('ADF_Statistic', 0):.2f}</span>
        <br><span style="font-size:0.9em; color:#888;">Stationarity test statistic.</span>
    </div>
    <hr style="margin:0.5em 0;">
    <div style="font-size:1.2em;">
        <b>ADF p-value</b><br>
        <span style="color:{colorize(metrics.get('ADF_pvalue', 0), good='low', threshold=0.05)}; font-size:2em;">
            {metrics.get('ADF_pvalue', 0):.4f}</span>
        <br><span style="font-size:0.9em; color:#888;">Below 0.05: likely stationary.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""---""")

    st.markdown("""
    <div style="background-color:#f8f9fa; border-radius:8px; padding:1em 1.5em; margin-top:1em; color:#222;">
    <b>Quick Insights:</b><br>
    <ul style="margin:0 0 0 1em; padding:0;">
      <li><b>Cumulative Return</b> shows total growth over your selected period.</li>
      <li><b>Sharpe Ratio</b> above 1 is generally considered good risk-adjusted performance.</li>
      <li><b>Max Drawdown</b> highlights the worst-case loss scenario.</li>
      <li><b>ADF p-value</b> below 0.05 means the price series is likely stationary (good for modeling).</li>
      <li><b>Value at Risk</b> and <b>Conditional VaR</b> help you understand potential downside risk.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("Parameters")

    ticker_options = {
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Amazon (AMZN)": "AMZN",
        "Alphabet (GOOGL)": "GOOGL",
        "Meta (META)": "META",
        "NVIDIA (NVDA)": "NVDA",
        "Tesla (TSLA)": "TSLA",
        "Netflix (NFLX)": "NFLX",
        "JPMorgan Chase (JPM)": "JPM",
        "Johnson & Johnson (JNJ)": "JNJ",
        "Visa (V)": "V",
        "Walmart (WMT)": "WMT",
        "Procter & Gamble (PG)": "PG",
        "Exxon Mobil (XOM)": "XOM",
        "UnitedHealth (UNH)": "UNH",
        "Coca-Cola (KO)": "KO",
        "PepsiCo (PEP)": "PEP",
        "Bank of America (BAC)": "BAC",
        "Intel (INTC)": "INTC",
        "Disney (DIS)": "DIS"
    }
    company_names = list(ticker_options.keys())
    selected_company = st.selectbox("Stock Ticker", company_names, index=0)
    ticker = ticker_options[selected_company]
    
    end_date = st.date_input("End Date", pd.Timestamp.today())
    start_date = st.date_input("Start Date", end_date - timedelta(days=365*3), 
                             max_value=end_date - timedelta(days=1))
    
    analysis_options = st.multiselect(
        "Advanced Analysis",
        options=["Technical Indicators", "Risk Metrics", "Statistical Tests", "Benchmark Comparison"],
        default=["Technical Indicators", "Risk Metrics"]
    )
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

st.title("Street Quant")
st.markdown("""
    *For the crowd that wants to find alpha in places Bloomberg won’t show you*
""")

if ticker:
    data, benchmark, error = fetch_data(ticker, start_date, end_date)
    
    if error:
        st.error(error)
        st.stop()
    
    if data is None or data.empty:
        st.warning("No data available for the selected ticker and date range.")
        st.stop()
    
    metrics = calculate_metrics(data, benchmark)
    
    display_metrics(data, metrics)
    
    st.subheader("Price Analysis")
    st.plotly_chart(plot_interactive_chart(data), use_container_width=True)

    if "Risk Metrics" in analysis_options:
        st.subheader("Risk Analysis")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(plot_rolling_metrics(data), use_container_width=True)
        
        with col2:
            st.markdown("### Risk Assessment")
            st.markdown(f"""
                - **Value at Risk (95%):** {float(metrics.get('VaR_95', 0))*100:.2f}% daily loss threshold
                - **Conditional VaR:** {float(metrics.get('CVaR_95', 0))*100:.2f}% average loss beyond VaR
                - **Max Drawdown:** {float(metrics.get('Max_Drawdown', 0))*100:.2f}% worst peak to trough decline
                - **Recovery Period:** {len(data[data['Close'] < data['Close'].cummax()].index)} days below peak
            """)
            
            if benchmark is not None and not benchmark.empty and 'Daily_Return' in benchmark.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(float(metrics.get('Annualized_Volatility', 0)), float(metrics.get('Annualized_Return', 0)),
                          color='#3498DB', s=100, label=ticker)
                benchmark_vol = float(benchmark['Daily_Return'].std()*np.sqrt(TRADING_DAYS))
                benchmark_ret = float(benchmark['Daily_Return'].mean()*TRADING_DAYS)
                ax.scatter(benchmark_vol, benchmark_ret,
                          color='#E74C3C', s=100, label='SPY')
                ax.set_xlabel('Volatility (Annualized)')
                ax.set_ylabel('Return (Annualized)')
                ax.set_title('Risk-Return Profile')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)

    if "Statistical Tests" in analysis_options:
        st.subheader("Statistical Analysis")
        
        tab1, tab2 = st.tabs(["Returns Distribution", "Stationarity Tests"])
        
        with tab1:
            st.plotly_chart(plot_returns_distribution(data), use_container_width=True)

            returns = data['Daily_Return'].dropna()
            if len(returns) > 0:
                k_stat, k_pval = stats.kurtosistest(returns)
                s_stat, s_pval = stats.skewtest(returns)
                
                st.markdown(f"""
                    #### Normality Tests:
                    - **Skewness:** {float(returns.skew()):.2f} (p-value: {float(s_pval):.4f})
                    - **Kurtosis:** {float(returns.kurtosis()):.2f} (p-value: {float(k_pval):.4f})
                    - **Jarque-Bera:** p-value {float(stats.jarque_bera(returns)[1]):.4f}
                """)
        
        with tab2:
            st.markdown("### Stationarity Analysis")
            
            # ADF Test
            st.markdown(f"""
                #### Augmented Dickey-Fuller Test:
                - **Test Statistic:** {float(metrics.get('ADF_Statistic', 0)):.2f}
                - **p-value:** {float(metrics.get('ADF_pvalue', 0)):.4f}
                - **Critical Values:"""
            )
            
            adf_crit = adfuller(data['Close'].dropna())[4]
            crit_df = pd.DataFrame(adf_crit.items(), columns=['Confidence Level', 'Critical Value'])
            st.dataframe(crit_df.style.format({'Critical Value': '{:.2f}'}))
            
            st.markdown("""
                *H0: The time series is non-stationary.*  
                *Small p-values (<0.05) suggest rejecting the null hypothesis.*
            """)