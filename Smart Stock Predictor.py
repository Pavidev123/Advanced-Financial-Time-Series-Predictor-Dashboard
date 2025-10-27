import streamlit as st
import yfinance as yf
from xgboost import XGBClassifier
import pandas as pd

st.title("📈 AI Stock Predictor (Demo)")

# Select stock
stock = st.selectbox("Choose a stock:", ["AAPL", "AMZN", "GOOGL", "MSFT", "NFLX"])
data = yf.download(stock, period="2y")
data['Return'] = data['Close'].pct_change()

# Simple demo model
data = data.dropna()
X = data[['Return']]
y = (data['Return'].shift(-1) > 0).astype(int)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X[:-1], y[:-1])

# Prediction
latest = model.predict(X.tail(1))
signal = "📈 Buy" if latest[0] == 1 else "📉 Sell"

st.metric("Prediction", signal)
st.line_chart(data['Close'])
