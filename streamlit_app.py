import streamlit as st
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Nifty50 Forecast Dashboard", layout="wide")

# =============================
# 1️⃣ App Title
# =============================
st.title("📈 Nifty50 Forecast Dashboard")
st.markdown("Compare forecasts from ARIMA, SARIMA, Prophet, and LSTM models.")

# =============================
# 2️⃣ Upload Forecast CSV
# =============================
uploaded_file = st.file_uploader("Upload your forecast CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # =============================
    # 3️⃣ Sidebar Filters
    # =============================
    st.sidebar.header("Filter Options")
    start_date = st.sidebar.date_input("Start Date", df["Date"].min())
    end_date = st.sidebar.date_input("End Date", df["Date"].max())
    models = st.sidebar.multiselect(
        "Select Models", 
        options=["ARIMA", "SARIMA", "Prophet", "LSTM"], 
        default=["ARIMA", "SARIMA", "Prophet", "LSTM"]
    )

    filtered_df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

    # =============================
    # 4️⃣ Actual vs Predicted Chart
    # =============================
    st.subheader("Actual vs Predicted Prices")
    fig = px.line(filtered_df, x="Date", y="Actual", color_discrete_sequence=["black"], labels={"Actual": "Price"})
    for model in models:
        if model in filtered_df.columns:
            fig.add_scatter(x=filtered_df["Date"], y=filtered_df[model], mode="lines", name=model)
    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # 5️⃣ Error Metrics Table
    # =============================
    st.subheader("Forecast Errors")
    error_dict = {}
    for model in models:
        filtered_df[f"{model}_Error"] = filtered_df["Actual"] - filtered_df[model]
        mse = (filtered_df[f"{model}_Error"]**2).mean()
        rmse = mse ** 0.5
        mae = filtered_df[f"{model}_Error"].abs().mean()
        error_dict[model] = {"MSE": round(mse,2), "RMSE": round(rmse,2), "MAE": round(mae,2)}

    errors_df = pd.DataFrame(error_dict).T
    st.dataframe(errors_df)

    # =============================
    # 6️⃣ Error Plot
    # =============================
    st.subheader("Forecast Errors Over Time")
    fig2, ax = plt.subplots(figsize=(12,4))
    for model in models:
        sns.lineplot(data=filtered_df, x="Date", y=f"{model}_Error", label=f"{model} Error", ax=ax)
    ax.set_ylabel("Error")
    ax.set_title("Forecast Errors")
    st.pyplot(fig2)

else:
    st.info("Please upload a forecast CSV file with columns: Date, Actual, ARIMA, SARIMA, Prophet, LSTM")


st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
