import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Tesla Stock Price Prediction")

st.title("📈 Tesla Stock Price Prediction")
st.write("LSTM-based Time Series Forecasting Application")

# Load model and scaler
model = load_model("lstm_tesla_model.h5")
scaler = joblib.load("scaler.pkl")

WINDOW_SIZE = 60

st.info(
    "This application predicts future Tesla stock prices using a trained LSTM model. "
    "Predictions are for educational purposes only."
)

# User input
days = st.slider("Select number of days to predict", 1, 10, 5)

if st.button("Predict"):
    st.success("Model loaded successfully!")

    # Dummy input sequence (deployment demo purpose)
    last_sequence = np.zeros((WINDOW_SIZE, 1))

    predictions = []

    for _ in range(days):
        x_input = last_sequence.reshape(1, WINDOW_SIZE, 1)
        pred = model.predict(x_input, verbose=0)[0][0]
        predictions.append(pred)
        last_sequence = np.append(last_sequence[1:], pred)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    st.subheader("Predicted Prices")
    for i, price in enumerate(predictions, start=1):
        st.write(f"Day {i}: ₹ {price[0]:.2f}")

st.caption("⚠ This is an academic project, not financial advice.")
