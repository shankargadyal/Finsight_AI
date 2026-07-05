# 📈 FinSight AI

**Intelligent Stock Market Prediction & Financial Analysis Platform**

FinSight AI is a full-stack web application that combines Machine Learning, Time-Series Forecasting, Sentiment Analysis, and an AI-powered assistant into a single financial intelligence platform — helping investors make faster, data-driven decisions without needing deep technical expertise.

> Built as an MSc Data Science project (Semester 2) — Shankar Gadyal | SCA25MSD064

---

## 🚀 Features

### 🤖 Machine Learning Models
- **Linear Regression** — baseline trend prediction
- **ARIMA** — time-series forecasting with seasonality
- **LSTM (Deep Learning)** — sequential pattern recognition for long-term forecasts
- **Ensemble Engine** — combines all three models for robust, reliable output

### 📰 AI News Summarizer
- Fetches live financial news via News API
- Summarizes articles and generates sentiment scores (−1 to +1)
- Classifies market mood: **Bullish / Bearish / Neutral**

### 🤖 AI Financial Assistant
- Explains predictions and recommendations in plain English
- Answers stock-related questions in a conversational Q&A format
- Powered by the Anthropic API with rule-based fallbacks

### 🛡️ Risk Analyzer
- Assigns **Low / Medium / High** risk ratings per stock
- Visual SVG gauge for instant risk visibility
- Considers volatility, trend strength, and sentiment together

### 📊 Prediction Confidence Meter
- Displays a confidence percentage for every forecast
- Communicates model reliability and uncertainty to users

### ⭐ Watchlist System
- Save and monitor favourite stocks
- Instant price and signal refresh

### 🔐 User Authentication
- Secure registration and login with **bcrypt** password hashing
- Session management with **SQLite** user database
- Personalized dashboard per user

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5 |
| Backend | Python 3, Flask |
| Machine Learning | Scikit-Learn, TensorFlow / Keras |
| Deep Learning | LSTM (via Keras) |
| NLP / Sentiment | VADER Sentiment Analysis |
| Data Source | Yahoo Finance API (`yfinance`), News API |
| AI Assistant | Anthropic Claude API |
| Database | SQLite |
| Auth | bcrypt |

---

## 📁 Project Structure

```
finsight-ai/
├── app.py                  # Flask app entry point
├── requirements.txt
├── models/
│   ├── linear_regression.py
│   ├── arima_model.py
│   ├── lstm_model.py
│   └── ensemble.py
├── features/
│   ├── risk_analyzer.py
│   ├── sentiment.py
│   ├── news_summarizer.py
│   └── confidence.py
├── auth/
│   └── user_auth.py
├── static/
│   ├── css/
│   ├── js/
│   └── assets/
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── login.html
│   ├── register.html
│   └── assistant.html
└── database/
    └── users.db
```

---

## ⚡ Getting Started

### Prerequisites

- Python 3.9+
- pip
- An [Anthropic API key](https://console.anthropic.com/) (for the AI assistant)
- A [News API key](https://newsapi.org/) (for the news summarizer)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/finsight-ai.git
cd finsight-ai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export ANTHROPIC_API_KEY=your_key_here
export NEWS_API_KEY=your_key_here

# 5. Run the app
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## 📦 Requirements

```
flask
scikit-learn
tensorflow
keras
yfinance
vaderSentiment
newsapi-python
anthropic
bcrypt
pandas
numpy
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔄 How It Works

```
Yahoo Finance API
      ↓
 Data Collection & Preprocessing
      ↓
 Feature Engineering
      ↓
 ┌──────────┬──────────┬──────────┐
 │    LR    │  ARIMA   │   LSTM   │
 └──────────┴──────────┴──────────┘
      ↓
 Ensemble Prediction Engine
      ↓
 Risk Analyzer + News Sentiment NLP
      ↓
 AI Assistant (Anthropic API)
      ↓
 Interactive Dashboard
```

---

## 📊 Model Comparison

| Model | Type | Best For |
|---|---|---|
| Linear Regression | Statistical | Trend baselines, fast inference |
| ARIMA | Time-Series | Short-term seasonal forecasting |
| LSTM | Deep Learning | Long-term sequential patterns |
| Ensemble | Hybrid | Final production predictions |

---

## 🔮 Future Scope

- [ ] Transformer-based forecasting (Temporal Fusion Transformer)
- [ ] RAG-powered Financial Assistant with document retrieval
- [ ] Real-time market data streaming (WebSocket)
- [ ] Portfolio optimization engine
- [ ] Reinforcement learning trading agents
- [ ] Voice-enabled financial assistant

---

## 👤 Author

**Shankar Gadyal**
MSc Data Science | SCA25MSD064

---

## 📄 License

This project is for academic purposes. All rights reserved.
