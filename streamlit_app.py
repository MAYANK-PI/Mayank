import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Nifty50 Stock Price Forecasting")

# Load cleaned data
df = pd.read_csv(parse_dates=["Date"], index_col="Date")

st.subheader("Historical Data")
st.line_chart(df["Close"])

# Load model forecasts
results = pd.read_csv("results.csv", parse_dates=["Date"], index_col="Date")

st.subheader("Forecast Comparison")
st.line_chart(results)

st.write("### Model Evaluation Metrics")
metrics = pd.read_csv("model_performance.csv")
st.dataframe(metrics)

st.success("Best Model: " + metrics.loc[metrics['RMSE'].idxmin(),'Model'])

