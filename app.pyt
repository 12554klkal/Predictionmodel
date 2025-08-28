import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime
from pmdarima import auto_arima
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
import plotly.express as px

# --- Custom CSS for Duolingo-like Design ---
st.markdown("""
<style>
    /* Global Background and Text */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f7f7f7; /* Light background */
        color: #333;
    }

    /* Streamlit Main Container */
    .stApp {
        background-color: #f7f7f7; /* Light background */
    }

    /* Sidebar Styling */
    .css-1d391kg { /* Target for sidebar */
        background-color: #e5e5e5; /* Slightly darker grey for sidebar */
        border-right: 2px solid #ccc;
        border-radius: 10px;
        padding: 20px;
    }

    /* Header Styling */
    h1 {
        color: #1cb0f6; /* Duolingo blue */
        font-weight: 800;
        text-align: center;
        border-bottom: 3px solid #1cb0f6;
        padding-bottom: 10px;
        margin-bottom: 30px;
        font-size: 2.5em;
    }
    h2 {
        color: #4c4c4c; /* Darker grey */
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
        font-size: 1.8em;
    }
    h3 {
        color: #6a6a6a; /* Medium grey */
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 1.4em;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #58cc02; /* Duolingo green */
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 15px; /* Rounded corners */
        padding: 10px 20px;
        font-size: 1.1em;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px #58a700; /* Darker green shadow */
    }
    .stButton > button:hover {
        background-color: #79d700; /* Lighter green on hover */
        box-shadow: 0 2px #58a700;
        transform: translateY(2px);
    }
    .stButton > button:active {
        background-color: #4aa000;
        box-shadow: 0 0 #58a700;
        transform: translateY(4px);
    }

    /* Input Fields */
    .stTextInput > div > div > input, .stDateInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #ccc;
        padding: 10px;
        font-size: 1em;
    }
    .stTextInput > div > div > input:focus, .stDateInput > div > div > input:focus {
        border-color: #1cb0f6; /* Duolingo blue focus */
        box-shadow: 0 0 0 0.1rem rgba(28, 176, 246, 0.25);
        outline: none;
    }

    /* Metrics and Results Display */
    .stAlert {
        border-radius: 10px;
        padding: 15px;
        background-color: #dff0d8; /* Light green for success */
        border: 1px solid #d6e9c6;
        color: #3c763d;
    }

    .stExpander {
        border-radius: 15px;
        border: 2px solid #ddd;
        padding: 10px;
        margin-top: 20px;
        background-color: #fff;
    }

    /* Container for sections, mimicking Duolingo cards */
    .duo-card {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #eee;
    }

    .duo-header {
        color: #777;
        font-size: 0.9em;
        margin-bottom: 10px;
        text-transform: uppercase;
        font-weight: 600;
    }

    .duo-message {
        background-color: #d1effe; /* Light blue for messages */
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #a3e0ff;
        margin-bottom: 20px;
        color: #007bff;
        font-weight: 500;
    }

    /* Styling for the plot container */
    .stPlot {
        border-radius: 15px;
        overflow: hidden; /* Ensures plot edges are rounded */
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    /* Progress bar style - not directly used but illustrative */
    .progress-bar-container {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 20px;
    }
    .progress-bar {
        height: 20px;
        width: 0%; /* Will be dynamically set */
        background-color: #58cc02; /* Duolingo green */
        border-radius: 10px;
        text-align: center;
        color: white;
        line-height: 20px;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="DuoStock Predictor 🦉")

# --- App Title ---
st.title("DuoStock Predictor 🦉")
st.markdown('<div class="duo-message">Welcome, eager learner! Let\'s predict some stock prices using the power of Machine Learning. Your financial journey starts now! ✨</div>', unsafe_allow_html=True)

# --- Sidebar for Input ---
st.sidebar.header("Your Mission Controls 🕹️")

# Stock Ticker Input
ticker_symbol = st.sidebar.text_input("Stock Ticker (e.g., AAPL, GOOGL)", "GOOGL").upper()

# Date Range Selection
today = datetime.date.today()
start_date = st.sidebar.date_input("Start Date", today - datetime.timedelta(days=365*3)) # 3 years ago
end_date = st.sidebar.date_input("End Date", today)

# Model Selection
st.sidebar.subheader("Choose Your Strategy 🧠")
prediction_model = st.sidebar.selectbox(
    "Select Prediction Model",
    ("Averaging (Simple & Moving)", "Linear Regression", "ARIMA", "Prophet", "Strategy Combiner (MA Crossover)"),
    help="Averaging uses historical averages. Linear Regression models the relationship between past and current prices. ARIMA is a statistical time series model. Prophet is designed for business forecasting. Strategy Combiner demonstrates combining signals."
)

# Averaging specific parameter
if prediction_model == "Averaging (Simple & Moving)":
    ma_window = st.sidebar.slider(
        "Moving Average Window (Days)",
        min_value=5, max_value=60, value=20, step=1,
        help="Number of past days to consider for moving average."
    )

# Linear Regression specific parameter
if prediction_model == "Linear Regression":
    lookback_days = st.sidebar.slider(
        "Lookback Days for LR Model",
        min_value=5, max_value=30, value=10, step=1,
        help="Number of previous closing prices to use as features for Linear Regression."
    )
    test_size = st.sidebar.slider(
        "Test Data Size (%)",
        min_value=10, max_value=50, value=20, step=5,
        help="Percentage of data to reserve for testing the Linear Regression model."
    ) / 100.0

# ARIMA specific parameters
if prediction_model == "ARIMA":
    st.sidebar.markdown("---")
    st.sidebar.subheader("ARIMA Parameters")
    p = st.sidebar.slider("AR Order (p)", 0, 5, 5, help="Number of lag observations.")
    d = st.sidebar.slider("I Order (d)", 0, 2, 1, help="Number of times the raw observations are differenced.")
    q = st.sidebar.slider("MA Order (q)", 0, 5, 0, help="Size of the moving average window.")
    seasonal = st.sidebar.checkbox("Include Seasonal Component", value=False)
    if seasonal:
        P = st.sidebar.slider("Seasonal AR Order (P)", 0, 2, 1)
        D = st.sidebar.slider("Seasonal I Order (D)", 0, 1, 1)
        Q = st.sidebar.slider("Seasonal MA Order (Q)", 0, 2, 0)
        m = st.sidebar.slider("Seasonal Period (m)", 4, 30, 7, help="Frequency of the time series (e.g., 7 for weekly).")
    else:
        P, D, Q, m = 0, 0, 0, 0 # Set to non-seasonal defaults if not selected

# Strategy Combiner specific parameters
if prediction_model == "Strategy Combiner (MA Crossover)":
    short_window = st.sidebar.slider(
        "Short MA Window (Days)",
        min_value=5, max_value=30, value=10, step=1,
        help="Window for the shorter moving average."
    )
    long_window = st.sidebar.slider(
        "Long MA Window (Days)",
        min_value=20, max_value=100, value=50, step=5,
        help="Window for the longer moving average."
    )
    if short_window >= long_window:
        st.sidebar.warning("Short MA window must be smaller than Long MA window!")


# --- Fetch Data ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            return None
        # Drop rows with any NaN values, especially important for 'Close' price
        data.dropna(subset=['Close'], inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}. Please check the ticker symbol and date range.")
        return None

stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)

if stock_data is None or stock_data.empty:
    st.warning("Uh oh! 😱 No data to analyze. Please try a different stock ticker or adjust the date range.")
else:
    st.markdown(f'<div class="duo-card"><h3>📈 Learning about {ticker_symbol}!</h3>', unsafe_allow_html=True)
    st.write(f"**Data Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    st.write("Here's a sneak peek at the historical data:")
    st.dataframe(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail())
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Averaging Model ---
    if prediction_model == "Averaging (Simple & Moving)":
        st.markdown('<div class="duo-card"><h3>🎯 Averaging Strategies!</h3>', unsafe_allow_html=True)
        # Simple Averaging
        simple_avg = stock_data['Close'].mean()
        st.markdown(f'<p class="duo-message">The **simple average** of all historical closing prices is: **${simple_avg:.2f}**</p>', unsafe_allow_html=True)

        # Moving Average
        if len(stock_data) >= ma_window:
            stock_data['Moving_Avg'] = stock_data['Close'].rolling(window=ma_window).mean()
            current_ma = stock_data['Moving_Avg'].iloc[-1]
            st.markdown(f'<p class="duo-message">The **{ma_window}-day moving average** of recent closing prices is: **${current_ma:.2f}**</p>', unsafe_allow_html=True)

            # Predict next day's close based on moving average
            next_day_prediction_ma = current_ma
            st.markdown(f"<p class='duo-header'>**Predicted Close for Tomorrow (Moving Average):**</p> <h2 style='color: #ff9100;'>${next_day_prediction_ma:.2f} 🚀</h2>", unsafe_allow_html=True)

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data.index, stock_data['Close'], label='Actual Close Price', color='#58cc02') # Duolingo green
            ax.plot(stock_data.index, stock_data['Moving_Avg'], label=f'{ma_window}-Day Moving Average', color='#1cb0f6') # Duolingo blue
            ax.axhline(next_day_prediction_ma, color='#ff9100', linestyle='--', label='Tomorrow\'s Prediction (MA)') # Duolingo orange
            ax.scatter(stock_data.index[-1] + datetime.timedelta(days=1), next_day_prediction_ma, color='#ff9100', marker='^', s=200, zorder=5) # Next day prediction marker

            ax.set_title(f'{ticker_symbol} Stock Price with {ma_window}-Day Moving Average', fontsize=16, color='#4c4c4c')
            ax.set_xlabel('Date', fontsize=12, color='#6a6a6a')
            ax.set_ylabel('Close Price ($)', fontsize=12, color='#6a6a6a')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', rotation=45)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.2f'))
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"Not enough data points ({len(stock_data)}) to calculate a {ma_window}-day moving average. Try a smaller window or longer date range.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Linear Regression Model ---
    elif prediction_model == "Linear Regression":
        st.markdown('<div class="duo-card"><h3>🧠 Linear Regression Learning!</h3>', unsafe_allow_html=True)

        if len(stock_data) >= lookback_days + 2: # Need at least lookback_days + 1 for feature, and one for target
            # Create features (X) and target (y)
            X = []
            y = []
            for i in range(lookback_days, len(stock_data)):
                X.append(stock_data['Close'].iloc[i-lookback_days : i].values)
                y.append(stock_data['Close'].iloc[i])

            X = np.array(X)
            y = np.array(y)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Train the Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Evaluate the model
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.write(f"**Model Performance on Test Data:**")
            st.markdown(f'<p class="duo-message">RMSE (Root Mean Squared Error): **${rmse:.2f}** (Lower is better 🏅)</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="duo-message">R² Score (Coefficient of Determination): **{r2:.2f}** (Closer to 1 is better 🏆)</p>', unsafe_allow_html=True)

            # Predict the next day's closing price
            last_lookback_prices = stock_data['Close'].iloc[-lookback_days:].values.reshape(1, -1)
            next_day_prediction_lr = model.predict(last_lookback_prices)[0]

            st.markdown(f"<p class='duo-header'>**Predicted Close for Tomorrow (Linear Regression):**</p> <h2 style='color: #ff9100;'>${next_day_prediction_lr:.2f} 🌟</h2>", unsafe_allow_html=True)

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data.index, stock_data['Close'], label='Actual Close Price', color='#58cc02', alpha=0.7) # Duolingo green

            # Plot historical predictions (optional, for visual comparison on training data)
            # This is a bit tricky to plot directly on the time series, so we'll plot test predictions
            test_indices = stock_data.index[len(stock_data) - len(y_test):]
            ax.plot(test_indices, y_pred, label='Predicted Close (Test Set)', color='#ff4b4b', linestyle='--') # Red for prediction

            ax.axhline(next_day_prediction_lr, color='#ff9100', linestyle=':', label='Tomorrow\'s Prediction (LR)') # Duolingo orange
            ax.scatter(stock_data.index[-1] + datetime.timedelta(days=1), next_day_prediction_lr, color='#ff9100', marker='^', s=200, zorder=5) # Next day prediction marker

            ax.set_title(f'{ticker_symbol} Stock Price with Linear Regression Prediction', fontsize=16, color='#4c4c4c')
            ax.set_xlabel('Date', fontsize=12, color='#6a6a6a')
            ax.set_ylabel('Close Price ($)', fontsize=12, color='#6a6a6a')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', rotation=45)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.2f'))
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"Not enough data points ({len(stock_data)}) to train Linear Regression with {lookback_days} lookback days. Try a smaller lookback or longer date range. You need at least {lookback_days + 2} data points.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ARIMA Model ---
    elif prediction_model == "ARIMA":
        st.markdown('<div class="duo-card"><h3>📈 ARIMA Model Training!</h3>', unsafe_allow_html=True)
        if len(stock_data) < 30: # ARIMA needs a reasonable amount of data
            st.warning("ARIMA model requires more historical data. Please select a longer date range (at least 30 days recommended).")
        else:
            try:
                # Use 'Close' price for ARIMA
                train_data = stock_data['Close'].iloc[:-10] # Use most data for training, leave last 10 for testing/validation
                test_data = stock_data['Close'].iloc[-10:]

                # Fit ARIMA model
                # The auto_arima function finds the best ARIMA parameters for us if we don't specify them.
                # However, since the user asked for customizable, we'll use the sidebar sliders for (p,d,q)(P,D,Q,m)
                if seasonal:
                    model = auto_arima(train_data, start_p=p, start_q=q,
                                       max_p=p, max_q=q, # Enforce user selected p,q
                                       d=d, # Enforce user selected d
                                       seasonal=True, m=m,
                                       start_P=P, start_Q=Q,
                                       max_P=P, max_Q=Q, # Enforce user selected P,Q
                                       D=D, # Enforce user selected D
                                       trace=False,
                                       error_action='ignore',
                                       suppress_warnings=True,
                                       stepwise=True)
                else:
                    model = auto_arima(train_data, start_p=p, start_q=q,
                                       max_p=p, max_q=q, # Enforce user selected p,q
                                       d=d, # Enforce user selected d
                                       seasonal=False,
                                       trace=False,
                                       error_action='ignore',
                                       suppress_warnings=True,
                                       stepwise=True)

                st.write(f"**Best ARIMA Parameters:** {model.order} {'x' + str(model.seasonal_order) + ' ' + str(m) if seasonal else ''}")

                # Forecast for the next day
                forecast_steps = len(test_data) + 1 # Forecast for test data + next day
                forecast = model.predict(n_periods=forecast_steps)
                forecast_series = pd.Series(forecast, index=stock_data.index[-len(test_data)-1:].append(pd.Index([stock_data.index[-1] + datetime.timedelta(days=1)])))
                next_day_prediction_arima = forecast[-1]

                # Evaluate using RMSE for the test set
                rmse_arima = np.sqrt(mean_squared_error(test_data, forecast[:-1]))
                st.markdown(f'<p class="duo-message">RMSE on test data: **${rmse_arima:.2f}** (Lower is better 🏅)</p>', unsafe_allow_html=True)

                st.markdown(f"<p class='duo-header'>**Predicted Close for Tomorrow (ARIMA):**</p> <h2 style='color: #ff9100;'>${next_day_prediction_arima:.2f} 🔮</h2>", unsafe_allow_html=True)

                # Plotting
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(stock_data.index, stock_data['Close'], label='Actual Close Price', color='#58cc02', alpha=0.7)
                ax.plot(forecast_series.index[:-1], forecast_series.iloc[:-1], label='ARIMA Forecast (Test Set)', color='#ff4b4b', linestyle='--')
                ax.axhline(next_day_prediction_arima, color='#ff9100', linestyle=':', label='Tomorrow\'s Prediction (ARIMA)')
                ax.scatter(forecast_series.index[-1], next_day_prediction_arima, color='#ff9100', marker='^', s=200, zorder=5)

                ax.set_title(f'{ticker_symbol} Stock Price with ARIMA Prediction', fontsize=16, color='#4c4c4c')
                ax.set_xlabel('Date', fontsize=12, color='#6a6a6a')
                ax.set_ylabel('Close Price ($)', fontsize=12, color='#6a6a6a')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.tick_params(axis='x', rotation=45)
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.2f'))
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error training ARIMA model: {e}. Try adjusting parameters or selecting a longer date range.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Prophet Model ---
    elif prediction_model == "Prophet":
        st.markdown('<div class="duo-card"><h3>🔮 Prophet Model Insights!</h3>', unsafe_allow_html=True)
        if len(stock_data) < 2 * 365: # Prophet benefits from at least a year or two of data
            st.warning("Prophet model works best with more historical data, ideally at least two years to capture seasonality. Please select a longer date range.")
        else:
            try:
                # Prepare data for Prophet: 'ds' for date, 'y' for value
                prophet_df = stock_data['Close'].reset_index()
                prophet_df = prophet_df.rename(columns={'Date': 'ds', 'Close': 'y'})

                # Initialize and fit Prophet model
                m = Prophet(daily_seasonality=True)
                m.fit(prophet_df)

                # Create a future DataFrame for predictions
                future = m.make_future_dataframe(periods=1, include_history=True) # Predict one day into the future

                # Make predictions
                forecast = m.predict(future)

                # Get next day's prediction
                next_day_prediction_prophet = forecast['yhat'].iloc[-1]

                st.markdown(f"<p class='duo-header'>**Predicted Close for Tomorrow (Prophet):**</p> <h2 style='color: #ff9100;'>${next_day_prediction_prophet:.2f} 🌟</h2>", unsafe_allow_html=True)

                # Plotting with Plotly for interactivity
                fig_prophet = plot_plotly(m, forecast)
                fig_prophet.update_layout(title=f'{ticker_symbol} Stock Price Forecast by Prophet',
                                          xaxis_title='Date',
                                          yaxis_title='Close Price ($)',
                                          font=dict(family="Inter", size=12, color="#4c4c4c"))
                fig_prophet.add_trace(go.Scatter(x=[forecast['ds'].iloc[-1]], y=[next_day_prediction_prophet],
                                                 mode='markers', marker=dict(size=15, symbol='triangle-up', color='#ff9100'),
                                                 name='Tomorrow\'s Prediction'))
                st.plotly_chart(fig_prophet)

                # Plot components
                st.subheader("Prophet Trend Components")
                st.markdown('<p class="duo-message">Prophet can break down the forecast into trend, weekly, and yearly seasonality. This helps us understand what drives the predictions!</p>', unsafe_allow_html=True)
                fig_components = plot_components_plotly(m, forecast)
                st.plotly_chart(fig_components)


            except Exception as e:
                st.error(f"Error training Prophet model: {e}. Prophet often requires more data and can be sensitive to data irregularities.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Strategy Combiner (Moving Average Crossover) ---
    elif prediction_model == "Strategy Combiner (MA Crossover)":
        st.markdown('<div class="duo-card"><h3>🤝 Strategy Combiner: MA Crossover!</h3>', unsafe_allow_html=True)

        if len(stock_data) < long_window:
            st.warning(f"Not enough data points ({len(stock_data)}) to calculate a {long_window}-day moving average. Please select a longer date range.")
        elif short_window >= long_window:
            st.warning("Please ensure the Short MA Window is smaller than the Long MA Window in the sidebar.")
        else:
            stock_data['Short_MA'] = stock_data['Close'].rolling(window=short_window).mean()
            stock_data['Long_MA'] = stock_data['Close'].rolling(window=long_window).mean()

            # Generate signals
            stock_data['Signal'] = 0.0 # 0 for no signal
            stock_data['Signal'][short_window:] = np.where(stock_data['Short_MA'][short_window:] > stock_data['Long_MA'][short_window:], 1.0, 0.0) # 1 for buy, 0 for sell/hold
            stock_data['Position'] = stock_data['Signal'].diff() # When signal changes from 0 to 1 (buy) or 1 to 0 (sell)

            st.markdown(f'<p class="duo-message">We\'re combining two moving averages: a **{short_window}-day** and a **{long_window}-day**! When the short average crosses above the long average, it\'s a **BUY signal**. When it crosses below, it\'s a **SELL signal**. </p>', unsafe_allow_html=True)

            # Plotting with signals
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', line=dict(color='#58cc02')))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Short_MA'], mode='lines', name=f'Short MA ({short_window} days)', line=dict(color='#1cb0f6')))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Long_MA'], mode='lines', name=f'Long MA ({long_window} days)', line=dict(color='#ff9100')))

            # Add buy signals
            buy_signals = stock_data[stock_data['Position'] == 1.0]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(x=buy_signals.index, y=stock_data['Close'][buy_signals.index],
                                         mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                                         name='Buy Signal'))

            # Add sell signals
            sell_signals = stock_data[stock_data['Position'] == -1.0]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(x=sell_signals.index, y=stock_data['Close'][sell_signals.index],
                                         mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                                         name='Sell Signal'))

            fig.update_layout(title=f'{ticker_symbol} Moving Average Crossover Strategy',
                              xaxis_title='Date',
                              yaxis_title='Price ($)',
                              font=dict(family="Inter", size=12, color="#4c4c4c"))
            st.plotly_chart(fig)

            # Display current recommendation
            last_signal = stock_data['Signal'].iloc[-1]
            if last_signal == 1.0:
                st.markdown('<p class="duo-message" style="background-color: #dff0d8; color: #3c763d;">**Current Recommendation: BUY!** The short-term trend is looking positive. 🚀</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="duo-message" style="background-color: #ffe0b2; color: #cc7000;">**Current Recommendation: SELL/HOLD!** The long-term trend might be stronger. 📉</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="duo-message" style="margin-top: 30px;">Great job! You\'ve completed this lesson on stock prediction. Keep practicing! 💪</div>', unsafe_allow_html=True)

# --- How to use section (for GitHub/Streamlit deployment) ---
st.sidebar.markdown("---")
st.sidebar.subheader("How to Use & Deploy 🚀")
st.sidebar.markdown(
    """
    1.  **Save this code** as `main.py` (or any `.py` file).
    2.  **Create a `requirements.txt`** file in the same directory with these contents:
        ```
        streamlit
        numpy
        pandas
        scipy
        scikit-learn
        matplotlib
        yfinance==0.2.28 # Pinning to a stable version
        Cython # Required for pmdarima build
        statsmodels # Dependency for pmdarima
        pmdarima
        prophet
        plotly
        ```
    3.  **Run locally:** `streamlit run main.py` in your terminal.
    4.  **Deploy to Streamlit Cloud:**
        * Push your `main.py` and `requirements.txt` to a **public** GitHub repository.
        * Go to [Streamlit Cloud](https://share.streamlit.io/).
        * Click "New app" and point it to your GitHub repo and `main.py` file.
        * Your app will be live! 🎉
    """
)
