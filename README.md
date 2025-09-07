# Large-Scale Time Series Forecasting with Transformers

📅 **Timeline:** Jun '25  
📌 **Type:** Self Project  

---

## 📖 Overview
This project focuses on **scalable multivariate time series forecasting** using **attention-based Transformer architectures**. By applying **Temporal Fusion Transformers (TFT)** and **Informer**, the pipeline forecasts over **50+ financial and macroeconomic time series** simultaneously, improving both **accuracy** and **interpretability** compared to classical models.

---

## 🚀 Key Features
- **Transformer Models:** Implemented **TFT** and **Informer** for long-sequence forecasting.
- **Baseline Comparisons:** Benchmarked against **ARIMA, VAR, and LSTM**, achieving:
  - **~27% lower MAPE**
  - **~20% lower RMSE**
- **Feature Engineering:** Incorporated:
  - Volatility measures
  - Regime shift indicators
  - Fourier seasonal encodings
  - Lagged correlation features
- **Non-Stationarity Handling:** Designed training procedures to stabilize learning under complex, non-stationary dynamics.
- **Model Explainability:**
  - Visualized **attention weights**
  - Applied **SHAP values** to highlight key temporal drivers.

---

## 📊 Results
- Significant accuracy improvements over classical baselines.
- **Explainability layer** provided interpretable insights for decision-making.
- Demonstrated **scalability** to high-dimensional financial & macroeconomic datasets.

---

## 🛠️ Tech Stack
- **Python**
- **PyTorch**
- **PyTorch Forecasting**
- **scikit-learn**
- **statsmodels**
- **SHAP**

---

## 📂 Project Structure
