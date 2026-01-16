"""
Alpha Research Engine - Streamlit Dashboard
Main application entry point with modern UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DashboardConfig, DEFAULT_UNIVERSE, IDX_STOCKS, US_TECH_STOCKS,
    FactorConfig, PortfolioConfig
)
from data.data_loader import DataLoader
from factors.factor_engine import FactorEngine
from evaluation.ic_calculator import ICCalculator
from evaluation.factor_decay import FactorDecayAnalyzer
from evaluation.turnover import TurnoverCalculator
from backtest.long_short import LongShortPortfolio
from backtest.backtester import Backtester

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=DashboardConfig.PAGE_TITLE,
    page_icon=DashboardConfig.PAGE_ICON,
    layout=DashboardConfig.LAYOUT,
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5a 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Positive/Negative indicators */
    .positive {
        color: #00d4aa !important;
    }
    
    .negative {
        color: #ff6b6b !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%);
    }
    
    /* Data table styling */
    .dataframe {
        background: #1e1e3f !important;
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1e1e3f;
        border-radius: 10px;
        padding: 10px 20px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def load_data(tickers, start_date, end_date):
    """Load and cache stock data."""
    loader = DataLoader(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    loader.fetch_all()
    return loader

def create_equity_chart(cumulative_returns):
    """Create equity curve chart."""
    fig = go.Figure()
    
    for col in cumulative_returns.columns:
        color_map = {
            "Long": "#00d4aa",
            "Short": "#ff6b6b",
            "Long_Short": "#667eea"
        }
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[col],
            name=col,
            line=dict(color=color_map.get(col, "#667eea"), width=2),
            fill='tozeroy' if col == "Long_Short" else None,
            fillcolor='rgba(102, 126, 234, 0.1)' if col == "Long_Short" else None
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_ic_chart(ic_series):
    """Create IC time series chart."""
    fig = go.Figure()
    
    # Add IC line
    fig.add_trace(go.Scatter(
        x=ic_series.index,
        y=ic_series.values,
        name="Daily IC",
        line=dict(color="#667eea", width=1),
        opacity=0.6
    ))
    
    # Add rolling mean
    rolling_ic = ic_series.rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=rolling_ic.index,
        y=rolling_ic.values,
        name="20-day Rolling IC",
        line=dict(color="#00d4aa", width=2)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.05, line_dash="dot", line_color="green", annotation_text="Strong Signal")
    fig.add_hline(y=-0.05, line_dash="dot", line_color="red")
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Information Coefficient Over Time",
        xaxis_title="Date",
        yaxis_title="IC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_decay_chart(decay_curve):
    """Create factor decay chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=decay_curve.index,
        y=decay_curve["Mean_IC"],
        marker_color="#667eea",
        name="Mean IC"
    ))
    
    # Add error bars for IC std
    fig.add_trace(go.Scatter(
        x=decay_curve.index,
        y=decay_curve["Mean_IC"],
        error_y=dict(type='data', array=decay_curve["Std_IC"], visible=True),
        mode='markers',
        marker=dict(color="#667eea", size=8),
        name="IC ± Std"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Factor Decay Analysis",
        xaxis_title="Forward Lag (Days)",
        yaxis_title="Mean IC",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_ranking_table(top_stocks_df):
    """Create styled ranking table."""
    styled_df = top_stocks_df.copy()
    return styled_df

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Alpha Research Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Quantitative Factor Analysis & Long-Short Simulation Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Stock Universe Selection
        universe_option = st.selectbox(
            "Stock Universe",
            ["Indonesian Stocks (IDX)", "US Tech Stocks", "Custom"]
        )
        
        if universe_option == "Indonesian Stocks (IDX)":
            selected_tickers = IDX_STOCKS
        elif universe_option == "US Tech Stocks":
            selected_tickers = US_TECH_STOCKS
        else:
            custom_tickers = st.text_area(
                "Enter tickers (comma-separated)",
                "AAPL, MSFT, GOOGL, AMZN, NVDA"
            )
            selected_tickers = [t.strip() for t in custom_tickers.split(",")]
        
        st.markdown("---")
        
        # Date Range
        st.markdown("### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365*2)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now()
            )
        
        st.markdown("---")
        
        # Factor Weights
        st.markdown("### ⚖️ Factor Weights")
        momentum_weight = st.slider("Momentum", 0.0, 1.0, 0.4, 0.1)
        volatility_weight = st.slider("Low Volatility", 0.0, 1.0, 0.3, 0.1)
        liquidity_weight = st.slider("Liquidity", 0.0, 1.0, 0.3, 0.1)
        
        st.markdown("---")
        
        # Portfolio Settings
        st.markdown("### 📊 Portfolio Settings")
        top_pct = st.slider("Long (Top %)", 5, 30, 10, 5) / 100
        bottom_pct = st.slider("Short (Bottom %)", 5, 30, 10, 5) / 100
        
        st.markdown("---")
        
        # Run Analysis Button
        run_analysis = st.button("🚀 Run Analysis", use_container_width=True)
    
    # Main Content
    if run_analysis or 'data_loaded' in st.session_state:
        with st.spinner("🔄 Loading data and calculating factors..."):
            try:
                # Load data
                loader = load_data(
                    selected_tickers,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.min.time())
                )
                
                prices = loader.get_price_matrix()
                volumes = loader.get_volume_matrix()
                
                if prices.empty:
                    st.error("❌ No data loaded. Please check your tickers and date range.")
                    return
                
                # Calculate factors
                engine = FactorEngine(
                    momentum_weight=momentum_weight,
                    volatility_weight=volatility_weight,
                    liquidity_weight=liquidity_weight
                )
                engine.calculate_all_factors(prices, volumes)
                composite = engine.calculate_composite_score(prices, volumes)
                
                # Store in session state
                st.session_state['data_loaded'] = True
                st.session_state['prices'] = prices
                st.session_state['volumes'] = volumes
                st.session_state['engine'] = engine
                st.session_state['composite'] = composite
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                return
        
        # Get data from session state
        prices = st.session_state['prices']
        volumes = st.session_state['volumes']
        engine = st.session_state['engine']
        composite = st.session_state['composite']
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🔬 Factor Analysis", 
            "💹 Backtest Results",
            "📥 Reports"
        ])
        
        # =================================================================
        # TAB 1: OVERVIEW
        # =================================================================
        with tab1:
            st.markdown("### 🏆 Top Ranked Stocks")
            
            # Get top stocks
            top_stocks = engine.get_top_stocks(prices, volumes, n=10)
            bottom_stocks = engine.get_bottom_stocks(prices, volumes, n=10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🟢 Long Candidates (Top 10)")
                st.dataframe(
                    top_stocks.style.format({
                        'Composite_Score': '{:.3f}',
                        'Momentum': '{:.3f}',
                        'Volatility': '{:.3f}',
                        'Liquidity': '{:.3f}'
                    }).background_gradient(cmap='Greens', subset=['Composite_Score']),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 🔴 Short Candidates (Bottom 10)")
                st.dataframe(
                    bottom_stocks.style.format({
                        'Composite_Score': '{:.3f}'
                    }).background_gradient(cmap='Reds_r', subset=['Composite_Score']),
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Key Metrics
            st.markdown("### 📈 Key Metrics")
            
            # Run backtest for metrics
            portfolio = LongShortPortfolio(top_pct=top_pct, bottom_pct=bottom_pct)
            returns = portfolio.calculate_returns(composite, prices)
            cumulative = portfolio.calculate_cumulative_returns(composite, prices)
            summary = portfolio.get_portfolio_summary(composite, prices)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return",
                    summary["Total Return"],
                    delta="Long-Short"
                )
            
            with col2:
                st.metric(
                    "Sharpe Ratio",
                    summary["Sharpe Ratio"]
                )
            
            with col3:
                st.metric(
                    "Max Drawdown",
                    summary["Max Drawdown"]
                )
            
            with col4:
                st.metric(
                    "Win Rate",
                    summary["Win Rate"]
                )
            
            st.markdown("---")
            
            # Equity Curve
            st.markdown("### 💰 Equity Curve")
            equity_chart = create_equity_chart(cumulative)
            st.plotly_chart(equity_chart, use_container_width=True)
        
        # =================================================================
        # TAB 2: FACTOR ANALYSIS
        # =================================================================
        with tab2:
            st.markdown("### 📊 Information Coefficient Analysis")
            
            # Calculate IC
            ic_calc = ICCalculator()
            ic_series = ic_calc.calculate_ic_series(composite, prices, forward_period=5)
            ic_summary = ic_calc.calculate_ic_summary(composite, prices)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                ic_chart = create_ic_chart(ic_series)
                st.plotly_chart(ic_chart, use_container_width=True)
            
            with col2:
                st.markdown("#### IC Statistics")
                st.dataframe(
                    ic_summary.style.format("{:.4f}"),
                    use_container_width=True
                )
                
                mean_ic = ic_series.mean()
                interpretation = ic_calc.get_ic_interpretation(mean_ic)
                
                if mean_ic > 0.05:
                    st.success(f"✅ {interpretation}")
                elif mean_ic > 0.02:
                    st.info(f"ℹ️ {interpretation}")
                else:
                    st.warning(f"⚠️ {interpretation}")
            
            st.markdown("---")
            
            # Factor Decay
            st.markdown("### ⏱️ Factor Decay Analysis")
            
            decay_analyzer = FactorDecayAnalyzer(max_lag=15)
            decay_curve = decay_analyzer.calculate_decay_curve(composite, prices)
            decay_analysis = decay_analyzer.analyze_decay(composite, prices)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                decay_chart = create_decay_chart(decay_curve)
                st.plotly_chart(decay_chart, use_container_width=True)
            
            with col2:
                st.markdown("#### Decay Metrics")
                st.metric("Signal Half-Life", f"{decay_analysis['half_life']} days")
                st.metric("Optimal Holding", f"{decay_analysis['optimal_holding_period']} days")
                st.metric("Initial IC", f"{decay_analysis['initial_ic']:.4f}")
                st.metric("Decay Ratio", f"{decay_analysis['decay_ratio']:.2%}")
            
            st.markdown("---")
            
            # Turnover Analysis
            st.markdown("### 🔄 Turnover Analysis")
            
            turnover_calc = TurnoverCalculator()
            long_weights, short_weights = turnover_calc.calculate_weights(
                engine.get_rankings(prices, volumes),
                top_n=int(len(prices.columns) * top_pct),
                bottom_n=int(len(prices.columns) * bottom_pct)
            )
            turnover_summary = turnover_calc.get_turnover_summary(long_weights, "Composite Factor")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Monthly Turnover", f"{turnover_summary['Monthly_Turnover']:.1%}")
            with col2:
                st.metric("Avg Holding Period", f"{turnover_summary['Avg_Holding_Period_Days']:.1f} days")
            with col3:
                st.metric("Annual Cost Drag", f"{turnover_summary['Annual_Cost_Drag']:.2%}")
            with col4:
                if turnover_summary['Assessment'] == "ACCEPTABLE":
                    st.success("✅ ACCEPTABLE")
                else:
                    st.warning("⚠️ HIGH")
        
        # =================================================================
        # TAB 3: BACKTEST RESULTS
        # =================================================================
        with tab3:
            st.markdown("### 📈 Detailed Backtest Results")
            
            backtester = Backtester()
            backtest_results = backtester.run_backtest(returns["Long_Short"])
            
            # Performance Metrics Table
            st.markdown("#### Performance Metrics")
            metrics_formatted = backtester.format_metrics(backtest_results["metrics"])
            metrics_df = pd.DataFrame([metrics_formatted]).T
            metrics_df.columns = ["Value"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(metrics_df.iloc[:5], use_container_width=True)
            
            with col2:
                st.dataframe(metrics_df.iloc[5:], use_container_width=True)
            
            st.markdown("---")
            
            # Monthly Returns Heatmap
            st.markdown("#### Monthly Returns")
            monthly_returns = backtester.get_monthly_returns(returns["Long_Short"])
            
            # Create heatmap
            fig = px.imshow(
                monthly_returns.values,
                x=monthly_returns.columns,
                y=monthly_returns.index,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                labels=dict(x="Month", y="Year", color="Return")
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Drawdown Chart
            st.markdown("#### Drawdown Analysis")
            drawdown_data = backtest_results["drawdown"]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown_data.index,
                y=drawdown_data["Drawdown"] * 100,
                fill='tozeroy',
                fillcolor='rgba(255, 107, 107, 0.3)',
                line=dict(color="#ff6b6b", width=1),
                name="Drawdown"
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # =================================================================
        # TAB 4: REPORTS
        # =================================================================
        with tab4:
            st.markdown("### 📥 Download Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Top Buy Recommendations")
                st.markdown("Download the latest stock rankings for tomorrow morning.")
                
                # Prepare report data
                report_date = prices.index[-1].strftime("%Y-%m-%d")
                top_stocks_report = engine.get_top_stocks(prices, volumes, n=10)
                top_stocks_report = top_stocks_report.reset_index()
                top_stocks_report.insert(0, "Report_Date", report_date)
                
                # CSV download
                csv_buffer = io.StringIO()
                top_stocks_report.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"top_stocks_{report_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.dataframe(top_stocks_report, use_container_width=True)
            
            with col2:
                st.markdown("#### 📉 Short Candidates")
                st.markdown("Bottom-ranked stocks for short positions.")
                
                bottom_stocks_report = engine.get_bottom_stocks(prices, volumes, n=10)
                bottom_stocks_report = bottom_stocks_report.reset_index()
                bottom_stocks_report.insert(0, "Report_Date", report_date)
                
                csv_buffer2 = io.StringIO()
                bottom_stocks_report.to_csv(csv_buffer2, index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_buffer2.getvalue(),
                    file_name=f"short_candidates_{report_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.dataframe(bottom_stocks_report, use_container_width=True)
            
            st.markdown("---")
            
            # Full Backtest Report
            st.markdown("#### 📋 Full Backtest Report")
            
            # Compile full report
            full_report = pd.DataFrame({
                "Metric": list(backtest_results["metrics"].keys()),
                "Value": list(backtester.format_metrics(backtest_results["metrics"]).values())
            })
            
            csv_full = io.StringIO()
            full_report.to_csv(csv_full, index=False)
            
            st.download_button(
                label="📥 Download Full Report",
                data=csv_full.getvalue(),
                file_name=f"backtest_report_{report_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2 style="color: #667eea;">Welcome to Alpha Research Engine</h2>
            <p style="color: #888; font-size: 1.1rem; margin-top: 1rem;">
                Configure your parameters in the sidebar and click <strong>Run Analysis</strong> to begin.
            </p>
            <br>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
                <div style="text-align: center;">
                    <h1 style="font-size: 3rem;">📊</h1>
                    <p style="color: #667eea;">Factor Analysis</p>
                </div>
                <div style="text-align: center;">
                    <h1 style="font-size: 3rem;">📈</h1>
                    <p style="color: #667eea;">Long-Short Simulation</p>
                </div>
                <div style="text-align: center;">
                    <h1 style="font-size: 3rem;">📥</h1>
                    <p style="color: #667eea;">Exportable Reports</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
