# 📈 Financial Trading & Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Dash](https://img.shields.io/badge/Dash-Framework-green)
![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Plotly](https://img.shields.io/badge/Viz-Plotly-lightblue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

**Real-time financial analytics app** that tracks and forecasts prices for Crypto, Stocks, and Forex using Machine Learning.  
Built with **Python**, **Dash**, **Random Forest Regressor**, and **Polygon/CoinGecko APIs**.

---

## 📸 Demo (Dashboard Views)
| Stock Market Analysis | Crypto Forecasting |
|-----------------------|--------------------|
| ![Stock View](assets/stock_dashboard.png) | ![Crypto View](assets/crypto_dashboard.png) |

> *Interactive Candlestick charts with 6-day future price predictions using Random Forest.*

---

## ✨ Features
- 🌍 **Multi-Asset Support:** Tracks Cryptocurrencies (Bitcoin, ETH), Stocks (AAPL, TSLA), and Forex (EUR/USD).
- 🤖 **ML Forecasting:** Uses **Random Forest Regressor** to predict prices for the next 6 days.
- 📊 **Interactive Visualization:** Plotly-based candlestick charts with volume bars and trend lines.
- 📉 **Performance Metrics:** Real-time calculation of **Accuracy**, **MAE**, and **RMSE** errors.
- ⚡ **Recursive Prediction:** Implements multi-step forecasting loop for trend trajectory.
- 🔄 **Live Data:** Fetches real-time OHLC data via REST APIs (Polygon.io, CoinGecko).

---

## 🧰 Tech Stack
| Category | Technology |
|-----------|-------------|
| 💻 Language | Python 3.12 |
| 🌐 Framework | Dash (by Plotly) |
| 🧠 ML Engine | Scikit-Learn (Random Forest) |
| 🎨 Visualization | Plotly Graph Objects |
| 🧮 Libraries | Pandas, NumPy, Requests |
| 🔗 APIs | Polygon.io, CoinGecko, ExchangeRate-API |

---

### 1️⃣ Clone the repository  
```bash
git clone [https://github.com/shk-javed/financial-trading-dashboard.git](https://github.com/shk-javed/financial-trading-dashboard.git)
cd financial-trading-dashboard



### Create & activate virtual environment
```bash

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate



####Install dependencies
pip install -r requirements.txt


###Configure API Keys & Run
Open app.py and update your API keys, then run:
python app.py



Open Dashboard: https://www.google.com/search?q=http://127.0.0.1:8050


### 1️⃣ Clone the repository  
```bash
