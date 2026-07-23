# =============================================================================
#  assistant.py  —  FinSight AI v2 · AI Financial Assistant
# =============================================================================
#  Powers the in-app chatbot. Uses Google Gemini when the API key is present; 
#  falls back to a rule-based response engine.
# =============================================================================

import os, re, json
from datetime import datetime
from google import genai
from google.genai import types
from net_timeout import call_with_timeout, NetworkTimeoutError

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """You are FinSight AI's financial assistant — a knowledgeable, concise,
and professional financial analyst chatbot embedded in a stock forecasting platform.

Your capabilities:
- Explain stock predictions, ML model outputs, and ensemble forecasting
- Interpret Buy / Hold / Sell recommendations
- Explain risk levels and confidence scores
- Answer general stock market questions
- Clarify technical indicators (RSI, MACD, Bollinger Bands, SMA)
- Explain ARIMA, LSTM, and Linear Regression in plain English

Tone: Professional but approachable. Use data when available.
Always include a disclaimer that this is educational, not financial advice.
Keep responses under 200 words unless the question demands more depth.
Format key points as short bullet points when helpful."""


# =============================================================================
#  Main entry point
# =============================================================================

def chat(messages: list[dict], context: dict | None = None) -> str:
    """
    Send a conversation to the AI assistant.

    Parameters
    ----------
    messages : list of {"role": "user"|"assistant", "content": str}
    context  : optional dict with current ticker data to inject into system prompt

    Returns
    -------
    str — assistant reply
    """
    enriched_system = _build_system_prompt(context)

    if GEMINI_API_KEY:
        return _chat_gemini(messages, enriched_system)
    else:
        return _chat_rule_based(messages[-1]["content"] if messages else "", context)


# =============================================================================
#  Gemini-powered chat
# =============================================================================

def _build_system_prompt(context: dict | None) -> str:
    base = SYSTEM_PROMPT
    if not context:
        return base

    ctx_lines = [f"\n\n## Current Analysis Context ({context.get('symbol','')})"]
    if context.get("last_close"):
        ctx_lines.append(f"- Current Price: ${context['last_close']:.2f}")
    if context.get("recommendation"):
        ctx_lines.append(f"- Recommendation: {context['recommendation']}")
    if context.get("confidence"):
        ctx_lines.append(f"- Confidence: {context['confidence']}%")
    if context.get("trend"):
        ctx_lines.append(f"- Trend: {context['trend'].upper()}")
    if context.get("risk_level"):
        ctx_lines.append(f"- Risk Level: {context['risk_level']}")
    if context.get("risk_score"):
        ctx_lines.append(f"- Risk Score: {context['risk_score']}/100")
    if context.get("sentiment_label"):
        ctx_lines.append(f"- News Sentiment: {context['sentiment_label']}")
    if context.get("ensemble_d7"):
        ctx_lines.append(f"- 7-Day Ensemble Target: ${context['ensemble_d7']:.2f}")

    return base + "\n".join(ctx_lines)


def _chat_gemini(messages: list[dict], system: str) -> str:
    try:
        # Initialize the modern standard GenAI client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Format history to meet Gemini's expected types.Content schema
        # Maps 'assistant' role to 'model' as required by Google
        gemini_contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )

        # Configure the request (system instruction, token limits, etc.)
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=600,
            temperature=0.3, # Keeps financial analysis grounded and deterministic
        )

        # Call Gemini 2.5 Flash for rapid, accurate chatbot responses.
        # Hard 8s timeout — see net_timeout.py for why this matters: the SDK's
        # own timeout can't be relied on to stop a stalled socket.
        response = call_with_timeout(
            client.models.generate_content,
            timeout=8,
            model='gemini-flash-latest',
            contents=gemini_contents,
            config=config,
        )
        
        if response.text:
            return response.text
        else:
            raise ValueError("Empty response received from Gemini.")

    except Exception as e:
        print(f"[assistant] Gemini API error: {e}")
        last_msg = messages[-1]["content"] if messages else ""
        return _chat_rule_based(last_msg, None)


# =============================================================================
#  Rule-based fallback
# =============================================================================

_RULES = [
    # (regex pattern, response template)
    (r"\b(buy|sell|hold|recommend)\b",
     "**Recommendation Logic:** FinSight combines three ML models (Linear Regression, ARIMA, LSTM) "
     "into a weighted ensemble forecast. A **BUY** signal is issued when the ensemble predicts upward movement, "
     "sentiment is positive, and confidence ≥ 60%. **SELL** requires downward trend + negative sentiment. "
     "**HOLD** is the cautious default when signals are mixed.\n\n"
     "*Note: This is for educational purposes only. Not financial advice.*"),

    (r"\b(risk|volatile|volatility|drawdown)\b",
     "**Risk Analysis:** FinSight calculates risk from four factors:\n"
     "- **Annualised Volatility** (std of log returns × √252) — 35%\n"
     "- **Max Drawdown** (largest peak-to-trough decline) — 25%\n"
     "- **Daily Price Range** (average absolute move) — 20%\n"
     "- **Downside Deviation** (std of negative days only) — 20%\n\n"
     "Scores below 35 = Low Risk 🟢, 35–65 = Medium Risk 🟡, above 65 = High Risk 🔴."),

    (r"\b(confidence|certain|accuracy|reliable)\b",
     "**Confidence Score:** Calculated from model agreement, prediction stability, "
     "and historical accuracy (RMSE). Three models agreeing on the same direction raises confidence. "
     "High RMSE lowers it. The score ranges from 40% (low agreement) to 95% (strong consensus)."),

    (r"\b(lstm|deep learning|neural|rnn)\b",
     "**LSTM Model:** Long Short-Term Memory is a type of recurrent neural network ideal for "
     "time series. FinSight uses a 60-day sliding window → LSTM(64) → Dropout → LSTM(32) → Dense(1). "
     "It learns temporal patterns that simple regression misses. First run trains the model (~30 sec); "
     "subsequent calls use a cached `.keras` file."),

    (r"\b(arima|time series|statsmodel)\b",
     "**ARIMA (5,1,0):** AutoRegressive Integrated Moving Average. "
     "p=5 uses the last 5 lagged differences, d=1 applies first-order differencing for stationarity, "
     "q=0 omits the MA term for speed and stability. Uses last 200 trading days. "
     "Provides confidence intervals alongside point forecasts."),

    (r"\b(linear|regression|lr)\b",
     "**Linear Regression Model:** Uses scikit-learn with polynomial degree-2 expansion. "
     "Features: day index + 7/14/21-day moving averages. Fastest model (~0.1 sec). "
     "Best for identifying trend direction in stable markets."),

    (r"\b(sentiment|news|vader)\b",
     "**Sentiment Analysis:** FinSight uses VADER (Valence Aware Dictionary and sEntiment Reasoner) "
     "augmented with a custom financial lexicon (bullish/bearish terms). News is fetched from NewsAPI, "
     "yfinance, or simulated headlines. Compound scores range from -1 (very negative) to +1 (very positive). "
     "Sentiment feeds directly into the final recommendation logic."),

    (r"\b(rsi|macd|bollinger|sma|ema|indicator)\b",
     "**Technical Indicators available in FinSight:**\n"
     "- **RSI (14):** Overbought > 70, Oversold < 30\n"
     "- **MACD:** Trend momentum (EMA12 − EMA26); signal line crossover\n"
     "- **Bollinger Bands:** 20-day SMA ± 2σ; price at upper band = potentially overbought\n"
     "- **SMA 20/50:** Simple moving averages; golden cross (SMA20 > SMA50) = bullish signal"),

    (r"\b(ensemble|combine|weight)\b",
     "**Ensemble Model:** Weighted average of all three forecasts:\n"
     "- LSTM → **50%** (most accurate, learns complex patterns)\n"
     "- ARIMA → **30%** (statistically grounded, provides CIs)\n"
     "- Linear Regression → **20%** (fast trend signal)\n\n"
     "If a model errors, its weight is redistributed to the remaining models."),

    (r"\b(portfolio|diversif|invest)\b",
     "**Portfolio Perspective:** FinSight analyses individual stocks in isolation. "
     "For portfolio management, consider diversification across sectors, correlation analysis, "
     "and your personal risk tolerance alongside these predictions. "
     "Always consult a licensed financial advisor for actual investment decisions."),
]

_DEFAULT_RESPONSE = (
    "I'm FinSight's AI assistant. I can help you understand:\n"
    "- 📊 ML model predictions (LSTM, ARIMA, Linear Regression)\n"
    "- ⚡ Buy/Hold/Sell recommendations\n"
    "- 🔴 Risk scores and volatility analysis\n"
    "- 📡 News sentiment analysis\n"
    "- 📈 Technical indicators (RSI, MACD, Bollinger Bands)\n\n"
    "Ask me anything about your current stock analysis!"
)


def _chat_rule_based(user_msg: str, context: dict | None) -> str:
    msg_lower = user_msg.lower()

    # Context-aware response for specific ticker questions
    if context and context.get("symbol"):
        sym = context["symbol"]
        rec = context.get("recommendation", "HOLD")
        risk = context.get("risk_level", "Medium Risk")
        conf = context.get("confidence", 0)
        sent = context.get("sentiment_label", "neutral")
        target = context.get("ensemble_d7")

        if re.search(r"\b(why|reason|explain|how)\b.*\b(buy|sell|hold|recommend)\b", msg_lower) or \
           re.search(r"\b(buy|sell|hold|recommend)\b.*\b(why|reason|explain)\b", msg_lower):
            trend_word = "upward" if "BUY" in rec else "downward" if "SELL" in rec else "mixed"
            return (
                f"**{sym} is rated {rec}** for the following reasons:\n\n"
                f"- 📈 **Ensemble trend:** {trend_word} movement predicted over 7 days\n"
                f"- 📡 **News sentiment:** {sent.capitalize()} ({context.get('sentiment_score', 0):.2f} score)\n"
                f"- ⚡ **Confidence:** {conf:.0f}% — model agreement is {'strong' if conf >= 70 else 'moderate'}\n"
                f"- 🔴 **Risk:** {risk}\n"
                + (f"- 🎯 **Target:** ${target:.2f} (7-day ensemble)\n" if target else "") +
                "\n*This is educational analysis. Not financial advice.*"
            )

    for pattern, response in _RULES:
        if re.search(pattern, msg_lower, re.I):
            return response

    return _DEFAULT_RESPONSE
