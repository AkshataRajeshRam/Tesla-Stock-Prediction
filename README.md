# 📈 Tesla Stock Price Prediction

This project implements a **Tesla Stock Price Prediction system** using **Deep Learning–based time series models**, specifically **SimpleRNN and LSTM**.  
The trained model is deployed as an interactive **Streamlit web application** that allows users to predict future Tesla stock prices for the next 1–10 days.


---

<img width="787" height="638" alt="Screenshot 2026-01-12 183232" src="https://github.com/user-attachments/assets/16c1a1fb-01ca-489f-a364-bb17c54b2fc3" /> 
<img width="1000" height="779" alt="Screenshot 2026-01-12 183202" src="https://github.com/user-attachments/assets/a8315eb9-d861-41f2-bdb3-8c893ad5a907" />
<img width="1033" height="858" alt="Screenshot 2026-01-12 183221" src="https://github.com/user-attachments/assets/baa288cd-c2ac-4277-b1ad-2cf7a527e6f8" />


## 🚀 Project Overview

Stock prices are sequential and highly volatile, making **Recurrent Neural Networks (RNNs)** well-suited for forecasting tasks.  
In this project, historical Tesla stock data is used to train and compare:

- **SimpleRNN**
- **LSTM (Long Short-Term Memory)**
- **Tuned LSTM (best-performing model)**

The final LSTM model is deployed using **Streamlit Cloud**.

---

## 📊 Dataset

- **Source**: Historical Tesla stock price data
- **Features Available**:
  - Date
  - Open
  - High
  - Low
  - Close
  - Adjusted Close
  - Volume
- **Target Variable**: `Adjusted Close Price`

> Adjusted Close is used because it accounts for stock splits and dividends and is commonly used in financial analysis.

---

## ⚙️ Data Preprocessing

- Converted date column to datetime format
- Handled missing values using forward/backward filling
- Applied **Min-Max Scaling** for normalization
- Created **sliding window sequences** (past 60 days → next day prediction)
- Used a **time-aware train-test split (80/20)** to avoid data leakage

---

## 🧠 Models Used

### 1. SimpleRNN
- Captures short-term dependencies
- Faster but less accurate for long sequences

### 2. LSTM
- Handles long-term dependencies using memory cells
- Achieved lower Mean Squared Error (MSE)

### 3. Tuned LSTM
- Hyperparameters tuned manually:
  - Number of units
  - Dropout rate
  - Batch size
- Selected as the **final model**

---

## 📈 Model Evaluation

- **Evaluation Metric**: Mean Squared Error (MSE)
- Compared:
  - Actual prices
  - SimpleRNN predictions
  - LSTM predictions
  - Tuned LSTM predictions

The LSTM-based models consistently outperformed SimpleRNN.

---

## 🔮 Multi-Day Forecasting

The final LSTM model predicts stock prices for:
- **1 Day**
- **5 Days**
- **10 Days**

Predictions are generated using an **autoregressive forecasting approach**, where each predicted value is used as input for the next prediction.

---

## 🌐 Streamlit Deployment

The trained model is deployed as an interactive web application using **Streamlit**.

### Features:
- Slider to select number of future days (1–10)
- Button to trigger prediction
- Displays predicted stock prices dynamically

> ⚠️ This application is for **educational purposes only** and should not be used as financial advice.

---

## 📁 Project Structure

tesla-stock-price-prediction/
│
├── app.py # Streamlit application
├── requirements.txt # Project dependencies
├── lstm_tesla_model.keras # Trained LSTM model
├── scaler.pkl # MinMaxScaler object
├── README.md # Project documentation


---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 📌 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

