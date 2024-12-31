import yfinance as yf
import pandas as pd
import talib
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import requests
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Configuration
NEWS_API_KEY = 'dd03ee8347e24e7591fbe1fcda19f095'  # Replace with your NewsAPI key
TWITTER_BEARER_TOKEN = 'YOUR_TWITTER_BEARER_TOKEN'  # Replace with your Twitter API token
CRYPTO_SYMBOL = 'BTCUSD'
EXCHANGE = 'BINANCE'
INTERVALS = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

# Normalize TradingView Symbol to Yahoo Finance format (e.g., BTCUSD -> BTC-USD)
def normalize_symbol_for_yahoo(symbol):
    return symbol.replace("USD", "-USD")

# Fetch historical data from Yahoo Finance
def fetch_historical_data(symbol, period="1y", interval="1d"):
    try:
        yahoo_symbol = normalize_symbol_for_yahoo(symbol)
        data = yf.download(yahoo_symbol, period=period, interval=interval)
        if data.empty:
            print(f"No data fetched for {symbol}.")
            return None
        return data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

# Fetch news sentiment
def fetch_news_sentiment(symbol):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json()
            if news_data['articles']:
                text = news_data['articles'][0]['title']
                sentiment_analyzer = SentimentIntensityAnalyzer()
                sentiment_score = sentiment_analyzer.polarity_scores(text)['compound']
                return sentiment_score
            else:
                return 0  # No articles found
        else:
            return 0  # Default to neutral sentiment if error
    except Exception as e:
        print(f"Error fetching news sentiment: {e}")
        return 0  # Default to neutral sentiment if error

# Function to calculate technical indicators and generate signals
def calculate_signals(data, timeframe):
    if data is not None:
        # Calculate Moving Average (SMA) for Buy/Sell Signal
        data['SMA50'] = talib.SMA(data['Close'], timeperiod=50)
        data['SMA200'] = talib.SMA(data['Close'], timeperiod=200)

        # Calculate Exponential Moving Average (EMA)
        data['EMA12'] = talib.EMA(data['Close'], timeperiod=12)
        data['EMA26'] = talib.EMA(data['Close'], timeperiod=26)

        # Calculate RSI for Overbought/Oversold
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)

        # Calculate Bollinger Bands
        data['BB_middle'], data['BB_upper'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # Calculate MACD (Moving Average Convergence Divergence)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # Calculate Average Directional Index (ADX)
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

        # Calculate Stochastic Oscillator
        data['SlowK'], data['SlowD'] = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

        # Calculate ATR
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

        # Parabolic SAR
        data['SAR'] = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

        # Commodity Channel Index (CCI)
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)

        # Williams %R
        data['Williams_R'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)

        # Money Flow Index (MFI)
        data['MFI'] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)

        # Generate Signals
        data['Signal_SMA'] = np.where(data['SMA50'] > data['SMA200'], 'BUY', 'SELL')
        data['Signal_RSI'] = np.where(data['RSI'] > 70, 'SELL', np.where(data['RSI'] < 30, 'BUY', 'HOLD'))
        data['Signal_MACD'] = np.where(data['MACD_hist'] > 0, 'BUY', 'SELL')
        data['Signal_ADX'] = np.where(data['ADX'] > 25, 'BUY', 'HOLD')
        data['Signal_Bollinger'] = np.where(data['Close'] > data['BB_upper'], 'SELL',
                                            np.where(data['Close'] < data['BB_lower'], 'BUY', 'HOLD'))

        # Combine all signals into a final signal
        conditions = [
            (data['Signal_SMA'] == 'BUY') & (data['Signal_RSI'] == 'BUY') & (data['Signal_MACD'] == 'BUY'),
            (data['Signal_SMA'] == 'SELL') & (data['Signal_RSI'] == 'SELL') & (data['Signal_MACD'] == 'SELL'),
        ]
        choices = ['BUY', 'SELL']
        data['Final_Signal'] = np.select(conditions, choices, default='HOLD')

        return data
    return None



# Function to calculate short-term prediction (based on Linear Regression)
from keras.layers import Input

def short_term_prediction(data):
    if data is not None:
        # Prepare data for LSTM
        data['Date'] = data.index
        data['Date'] = data['Date'].map(pd.Timestamp.timestamp)
        prices = data['Close'].values.reshape(-1, 1)

        # Scale data to range [0, 1] using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences for LSTM
        look_back = 10
        X, y = [], []
        for i in range(look_back, len(scaled_prices)):
            X.append(scaled_prices[i - look_back:i, 0])
            y.append(scaled_prices[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Define LSTM model
        model = Sequential()
        model.add(Input(shape=(X.shape[1], 1)))  # Replace input_shape with Input()
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        # Compile and train the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, batch_size=1, epochs=10, verbose=0)
        
        # Predict next value
        last_sequence = scaled_prices[-look_back:]
        last_sequence = last_sequence.reshape((1, look_back, 1))
        prediction = model.predict(last_sequence)
        
        # Inverse scale the predicted value
        predicted_price = scaler.inverse_transform(prediction)
        return predicted_price[0][0]
    return None


def long_term_prediction(data):
    if data is not None:
        # Prepare data for Linear Regression
        data['Date'] = data.index
        data['Date'] = data['Date'].map(pd.Timestamp.timestamp)
        X = data[['Date']].values
        y = data['Close'].values

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict 30 days into the future
        future_date = pd.Timestamp.now() + pd.DateOffset(days=30)
        future_timestamp = future_date.timestamp()
        future_prediction = model.predict([[future_timestamp]])
        return future_prediction[0]
    return None


# Consensus Algorithm: Combine multiple signals to generate final decision
def consensus_algorithm(data):
    # Aggregate signals from data
    signals = data[['Signal_SMA', 'Signal_RSI', 'Signal_MACD', 'Signal_ADX', 'Signal_Bollinger']]
    signal_counts = signals.apply(lambda x: x.value_counts(), axis=1).fillna(0)

    # Final consensus signal logic
    data['Consensus_Signal'] = np.where(signal_counts['BUY'] > signal_counts['SELL'], 'BUY',
                                         np.where(signal_counts['SELL'] > signal_counts['BUY'], 'SELL', 'HOLD'))
    return data

    # Adjust based on sentiment (positive sentiment = Buy, negative = Sell)
    sentiment_signal = 'BUY' if sentiment > 0 else 'SELL'
    
    # Final decision based on consensus
    if signal_count['BUY'] > signal_count['SELL']:
        return 'BUY'
    elif signal_count['SELL'] > signal_count['BUY']:
        return 'SELL'
    else:
        return sentiment_signal  # Use sentiment if no consensus

# Main function for Streamlit app
def main():
    st.title("Crypto Signal and Prediction Dashboard")

    symbol = st.text_input("Enter Crypto Symbol", CRYPTO_SYMBOL)
    period = st.selectbox("Select Period", ["1d", "5d", "1mo", "1y"])
    interval = st.selectbox("Select Interval", ["5m", "15m", "30m", "1h", "4h", "1d"])

    # Fetch historical data
    data = fetch_historical_data(symbol, period, interval)
    if data is not None:
        st.write(f"Fetched data for {symbol}:")
        st.write(data.tail())

        # Calculate technical indicators and signals
        data_with_signals = calculate_signals(data, interval)
        if data_with_signals is not None:
            st.write("\nLatest Technical Indicators and Signals:")
            st.write(data_with_signals[['Close', 'SMA50', 'SMA200', 'EMA12', 'EMA26', 'RSI', 'Signal_SMA', 'Signal_RSI', 'Final_Signal', 'MACD', 'ADX', 'SlowK', 'SlowD', 'ATR', 'SAR', 'CCI', 'Williams_R', 'MFI']].tail())

            # Generate Signal Table for different timeframes
            signal_counts = data_with_signals['Final_Signal'].value_counts()
            st.subheader("Signal Table:")
            st.write(f"Buy Signal: {signal_counts.get('BUY', 0)}")
            st.write(f"Sell Signal: {signal_counts.get('SELL', 0)}")
            st.write(f"Hold Signal: {signal_counts.get('HOLD', 0)}")

            # Short-Term Prediction
            short_term_pred = short_term_prediction(data)
            st.write(f"Short-Term Prediction (next price): {short_term_pred}")

            # Long-Term Prediction (30 days ahead)
            long_term_pred = long_term_prediction(data)
            st.write(f"Long-Term Prediction (30 days ahead): {long_term_pred}")

            # Fetch News Sentiment
            sentiment_score = fetch_news_sentiment(symbol)
            st.write(f"News Sentiment: {sentiment_score}")

            # Consensus Algorithm: Combine all signals
            signals = data_with_signals['Final_Signal'].tail().tolist()
            data_with_signals = consensus_algorithm(data_with_signals)
            final_signal = data_with_signals['Consensus_Signal'].iloc[-1]  # Get the most recent consensus signal

            st.write(f"Final Consensus Signal: {final_signal}")

            # Display prediction table
            st.subheader("Predictions Table:")
            predictions_data = pd.DataFrame({
                "Prediction Type": ["Short-Term Prediction", "Long-Term Prediction", "Consensus Signal"],
                "Predicted Value": [short_term_pred, long_term_pred, final_signal]
            })
            st.write(predictions_data)
            
            # Plot the data: Add a simple plot of the closing price with SMA, EMA, Bollinger Bands, and MACD
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data_with_signals['Close'], label='Close Price')
            ax.plot(data_with_signals['SMA50'], label='50-period SMA')
            ax.plot(data_with_signals['SMA200'], label='200-period SMA')
            ax.plot(data_with_signals['EMA12'], label='12-period EMA', linestyle='--')
            ax.plot(data_with_signals['EMA26'], label='26-period EMA', linestyle='--')
            ax.plot(data_with_signals['BB_upper'], label='Upper Bollinger Band', linestyle='--')
            ax.plot(data_with_signals['BB_lower'], label='Lower Bollinger Band', linestyle='--')
            ax.set_title(f'{symbol} Price with Technical Indicators')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("Error calculating signals.")
    else:
        st.write("Error fetching historical data.")

if __name__ == '__main__':
    main()
