# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

st.set_page_config(layout="wide", page_title="Nifty50 Forecasts")

st.title("📈 Nifty50 — Forecast Viewer")

# helper to read forecast files flexibly
def read_forecast_file(path):
    if not os.path.exists(path):
        return None
    try:
        # try read with date index
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # if single column, return that series
        if df.shape[1] == 1:
            return df.iloc[:,0].rename(os.path.splitext(os.path.basename(path))[0])
        # if multiple columns, try to find numeric forecast column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return df[c].rename(os.path.splitext(os.path.basename(path))[0])
    except Exception:
        # fallback: try reading ds/yhat style (prophet)
        try:
            df2 = pd.read_csv(path)
            if 'ds' in df2.columns and 'yhat' in df2.columns:
                s = pd.Series(df2['yhat'].values, index=pd.to_datetime(df2['ds']))
                return s.rename(os.path.splitext(os.path.basename(path))[0])
        except Exception:
            return None
    return None

# load main data
def load_main():
    for fname in ["nifty50_clean.csv", "nifty50.csv", "NIFTY50.csv"]:
        if os.path.exists(fname):
            df0 = pd.read_csv(fname, parse_dates=['Date'], index_col='Date')
            return df0
    return None

df = load_main()
if df is None:
    st.error("Couldn't find 'nifty50_clean.csv' or 'nifty50.csv' in this folder. Upload one below.")
    uploaded = st.file_uploader("Upload CSV (must contain a 'Date' column and 'Close' column)", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=['Date'], index_col='Date')
        st.success("File uploaded.")
if df is None:
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

# plot historical close
st.subheader("Historical Closing Price")
st.line_chart(df['Close'])

# load forecasts available
forecasts = {}
for fname in ["arima_forecast.csv","prophet_forecast.csv","sarima_forecast.csv","lstm_forecast.csv"]:
    s = read_forecast_file(fname)
    if s is not None:
        forecasts[fname.split('_')[0].upper()] = s

if not forecasts:
    st.info("No forecast files found (arima_forecast.csv, prophet_forecast.csv, sarima_forecast.csv, lstm_forecast.csv). Place them in this folder or generate them and re-run.")
else:
    st.subheader("Available forecasts")
    st.write(list(forecasts.keys()))

    # select models to display
    selected = st.multiselect("Select models to show", list(forecasts.keys()), default=list(forecasts.keys()))

    # build results DF aligned to actual
    results = pd.DataFrame({'Actual': df['Close']})
    for name, series in forecasts.items():
        results[name] = series

    # intersection to evaluate on common dates
    # drop rows where actual is NaN
    results = results.dropna(subset=['Actual'])
    common_idx = results.index
    # drop model columns' NaNs if user wants strict intersection:
    if st.checkbox("Require forecasts to be present on evaluation dates (intersect indices)", value=True):
        for name in list(forecasts.keys()):
            common_idx = common_idx.intersection(results[name].dropna().index)
        results = results.loc[common_idx]

    # show metrics
    st.subheader("Evaluation on selected / common dates")
    metrics = []
    for name in selected:
        if name in results.columns and results[name].notna().any():
            y_true = results['Actual']
            y_pred = results[name]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics.append({'Model':name, 'MAE':mae, 'RMSE':rmse, 'MAPE':mape})
    if metrics:
        metrics_df = pd.DataFrame(metrics).set_index('Model')
        st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE":"{:.2f}", "MAPE":"{:.2f}%"}))
    else:
        st.write("No overlapping forecast data to compute metrics.")

    # plot actual + forecasts
    st.subheader("Plot: Actual vs Forecasts")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(results.index, results['Actual'], label='Actual', color='black')
    for name in selected:
        if name in results.columns:
            ax.plot(results.index, results[name], linestyle='--', label=name)
    ax.legend()
    ax.set_title("Actual vs Forecasts")
    st.pyplot(fig)

    # allow download combined CSV
    if st.button("Download combined results as CSV"):
        out_csv = "combined_results.csv"
        results.to_csv(out_csv)
        with open(out_csv, "rb") as f:
            st.download_button("Click to download", f, file_name=out_csv)

st.info("Run this app with: streamlit run app.py")
