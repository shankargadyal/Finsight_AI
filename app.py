# =============================================================================
#  app.py  —  FinSight AI v2 · Flask Production Backend
# =============================================================================
#  New in v2:
#    • User Authentication  (register / login / logout / profile)
#    • Risk Analyzer        (/api/risk/<ticker>)
#    • AI News Summarizer   (bundled into /api/analyze)
#    • AI Financial Chat    (/api/chat)
#    • Watchlist CRUD       (/api/watchlist/*)
#    • User Dashboard       (/api/dashboard)
#    • Future Scope page    (/future)
#
#  Run:
#    python app.py
# =============================================================================

import os, traceback
from datetime import datetime, timedelta

from flask      import Flask, render_template, jsonify, request, session
from flask_cors import CORS
from dotenv     import load_dotenv

from utils     import (fetch_ohlcv, get_company_info,
                        add_technical_indicators,
                        future_business_dates, safe_float, serialise_row)
from models    import EnsembleModel
from sentiment import analyse_ticker
from risk      import calculate_risk
from news_summarizer import summarise_news
from assistant import chat as assistant_chat
from database  import (init_db, get_watchlist, add_to_watchlist, remove_from_watchlist,
                        log_search, get_recent_searches,
                        save_prediction, get_prediction_history,
                        save_chat_message, get_chat_history)
from auth      import register_auth_routes, login_required, optional_auth

load_dotenv()

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32))
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# ── Init DB + auth routes ──────────────────────────────────────────────────────
init_db()
register_auth_routes(app)


# =============================================================================
#  HELPERS
# =============================================================================

def _make_recommendation(trend: str, sentiment: str, confidence: float) -> str:
    if confidence >= 80:
        if trend == "up"   and sentiment == "positive": return "STRONG BUY"
        if trend == "down" and sentiment == "negative": return "STRONG SELL"
    if confidence >= 60:
        return "BUY" if trend == "up" else "SELL"
    return "HOLD"


def _confidence_range(last_close: float, ensemble_preds: list, rmse_avg: float) -> dict:
    """Calculate expected price range based on ensemble variance + RMSE."""
    if not ensemble_preds:
        return {"low": None, "high": None, "target": None}
    target = ensemble_preds[-1]
    margin = max(rmse_avg, abs(target - last_close) * 0.15, last_close * 0.02)
    return {
        "target": round(target, 2),
        "low":    round(target - margin, 2),
        "high":   round(target + margin, 2),
    }


# =============================================================================
#  PAGES
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/future")
def future_scope():
    return render_template("future.html")


# =============================================================================
#  API — HEALTH
# =============================================================================

@app.route("/api/health")
def health():
    return jsonify({
        "status":  "ok",
        "message": "FinSight AI v2 is running ✅",
        "time":    datetime.utcnow().isoformat(),
    })


# =============================================================================
#  API — STOCK DATA
# =============================================================================

@app.route("/api/stock/<ticker>")
def stock(ticker: str):
    symbol = ticker.upper().strip()
    period = request.args.get("period", "1y")
    try:
        df = fetch_ohlcv(symbol, period)
        if df.empty:
            return jsonify({"error": f"No data found for '{symbol}'."}), 404
        df   = add_technical_indicators(df)
        rows = [serialise_row(row) for _, row in df.iterrows()]
        last = df.iloc[-1]; prev = df.iloc[-2] if len(df) >= 2 else last
        chg  = float(last["Close"] - prev["Close"])
        chgp = (chg / float(prev["Close"]) * 100) if float(prev["Close"]) else 0
        return jsonify({
            "symbol": symbol, "period": period,
            "info":   get_company_info(symbol),
            "data":   rows,
            "summary": {
                "current_price": safe_float(last["Close"]),
                "change":        round(chg, 2),
                "change_pct":    round(chgp, 2),
                "sma20":         safe_float(last.get("SMA20")),
                "sma50":         safe_float(last.get("SMA50")),
                "rsi14":         safe_float(last.get("RSI14")),
                "macd":          safe_float(last.get("MACD")),
                "bb_upper":      safe_float(last.get("BB_Upper")),
                "bb_lower":      safe_float(last.get("BB_Lower")),
                "volume":        int(last["Volume"]) if last.get("Volume") else None,
            },
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# =============================================================================
#  API — RISK
# =============================================================================

@app.route("/api/risk/<ticker>")
def risk_endpoint(ticker: str):
    symbol = ticker.upper().strip()
    try:
        df = fetch_ohlcv(symbol, period="1y")
        if df.empty:
            return jsonify({"error": f"No data for '{symbol}'"}), 404
        risk = calculate_risk(df["Close"].tolist())
        return jsonify({"symbol": symbol, **risk})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# =============================================================================
#  API — FULL ANALYSIS
# =============================================================================

@app.route("/api/analyze/<ticker>")
@optional_auth
def analyze(ticker: str):
    symbol = ticker.upper().strip()
    print(f"\n{'='*55}\n  /api/analyze/{symbol}\n{'='*55}")
    try:
        # ── 1. Price data ──────────────────────────────────────────────────────
        df = fetch_ohlcv(symbol, period="2y")
        if df.empty or len(df) < 80:
            return jsonify({"error": f"Not enough data for '{symbol}'."}), 404
        df = add_technical_indicators(df)
        prices     = df["Close"].tolist()
        last_date  = df["Date"].iloc[-1].strftime("%Y-%m-%d")
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_close

        # ── 2. ML models ───────────────────────────────────────────────────────
        result     = EnsembleModel().predict(symbol, prices, future_days=7)
        fut_dates  = future_business_dates(last_date, 7)

        # ── 3. Sentiment ───────────────────────────────────────────────────────
        sent       = analyse_ticker(symbol, max_articles=10)

        # ── 4. Risk ────────────────────────────────────────────────────────────
        risk       = calculate_risk(prices)

        # ── 5. News summary ────────────────────────────────────────────────────
        news_sum   = summarise_news(symbol, sent.get("articles", []), sent)

        # ── 6. Recommendation ──────────────────────────────────────────────────
        trend = result["trend"]
        conf  = result["confidence"]
        rec   = _make_recommendation(trend, sent["overall_label"], conf)

        # ── 7. Confidence range ────────────────────────────────────────────────
        avg_rmse = sum(
            r.get("rmse") or 0
            for r in [result["linear_reg"], result["lstm"]]
        ) / 2
        price_range = _confidence_range(last_close, result["ensemble"], avg_rmse)

        # ── 8. Historical chart (last 90 days) ─────────────────────────────────
        hist = [
            {
                "date":     r["Date"].strftime("%Y-%m-%d"),
                "close":    safe_float(r["Close"]),
                "sma20":    safe_float(r.get("SMA20")),
                "sma50":    safe_float(r.get("SMA50")),
                "rsi":      safe_float(r.get("RSI14")),
                "bb_upper": safe_float(r.get("BB_Upper")),
                "bb_lower": safe_float(r.get("BB_Lower")),
            }
            for _, r in df.tail(90).iterrows()
        ]

        # ── 9. Log search + prediction ─────────────────────────────────────────
        uid = getattr(request, "current_user_id", None)
        if uid:
            log_search(uid, symbol)
            save_prediction(uid, {
                "symbol":          symbol,
                "last_close":      last_close,
                "ensemble_d7":     result["ensemble"][-1] if result["ensemble"] else None,
                "recommendation":  rec,
                "confidence":      conf,
                "risk_score":      risk["score"],
                "risk_level":      risk["level"],
                "sentiment_label": sent["overall_label"],
            })

        return jsonify({
            "symbol":     symbol,
            "last_close": round(last_close, 2),
            "change":     round(last_close - prev_close, 2),
            "change_pct": round((last_close - prev_close) / prev_close * 100, 2) if prev_close else 0,
            "info":       get_company_info(symbol),
            "historical": hist,
            "future_dates": fut_dates,
            "linear_reg": result["linear_reg"],
            "arima":      result["arima"],
            "lstm":       result["lstm"],
            "ensemble":   result["ensemble"],
            "trend":      trend,
            "confidence": conf,
            "price_range": price_range,
            "recommendation": rec,
            "sentiment": {
                "score":         sent["overall_score"],
                "label":         sent["overall_label"],
                "positive_pct":  sent["positive_pct"],
                "negative_pct":  sent["negative_pct"],
                "neutral_pct":   sent["neutral_pct"],
                "article_count": sent["article_count"],
            },
            "articles": [
                {"title": a["title"], "source": a.get("source",""),
                 "compound": a["compound"], "label": a["label"]}
                for a in sent["articles"]
            ],
            "risk":        risk,
            "news_summary": news_sum,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# =============================================================================
#  API — CHAT ASSISTANT
# =============================================================================

@app.route("/api/chat", methods=["POST"])
@optional_auth
def chat_endpoint():
    body       = request.get_json(silent=True) or {}
    user_msg   = (body.get("message") or "").strip()
    session_id = body.get("session_id") or "anon"
    context    = body.get("context")          # optional ticker context

    if not user_msg:
        return jsonify({"error": "Message required"}), 400

    uid = getattr(request, "current_user_id", None)

    # Load history
    history = get_chat_history(session_id, limit=20)
    history.append({"role": "user", "content": user_msg})

    # Get reply
    reply = assistant_chat(history, context)

    # Persist
    save_chat_message(uid, session_id, "user", user_msg)
    save_chat_message(uid, session_id, "assistant", reply)

    return jsonify({"reply": reply, "session_id": session_id})


@app.route("/api/chat/history")
@optional_auth
def chat_history_endpoint():
    session_id = request.args.get("session_id", "anon")
    history    = get_chat_history(session_id)
    return jsonify({"history": history})


# =============================================================================
#  API — WATCHLIST
# =============================================================================

@app.route("/api/watchlist", methods=["GET"])
@login_required
def get_watchlist_endpoint():
    wl = get_watchlist(request.current_user_id)
    return jsonify({"watchlist": wl})


@app.route("/api/watchlist/<symbol>", methods=["POST"])
@login_required
def add_watchlist(symbol: str):
    add_to_watchlist(request.current_user_id, symbol.upper())
    return jsonify({"message": f"{symbol.upper()} added to watchlist"})


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
@login_required
def remove_watchlist(symbol: str):
    remove_from_watchlist(request.current_user_id, symbol.upper())
    return jsonify({"message": f"{symbol.upper()} removed from watchlist"})


# =============================================================================
#  API — USER DASHBOARD
# =============================================================================

@app.route("/api/dashboard")
@login_required
def dashboard():
    uid = request.current_user_id
    searches    = get_recent_searches(uid, limit=8)
    watchlist   = get_watchlist(uid)
    predictions = get_prediction_history(uid, limit=10)

    # Portfolio risk summary
    if predictions:
        avg_risk = sum(p["risk_score"] or 50 for p in predictions) / len(predictions)
        risk_label = "Low" if avg_risk < 35 else "High" if avg_risk > 65 else "Medium"
    else:
        avg_risk, risk_label = 0, "N/A"

    return jsonify({
        "recent_searches":    searches,
        "watchlist":          watchlist,
        "prediction_history": predictions,
        "portfolio_risk": {
            "avg_score": round(avg_risk, 1),
            "label":     risk_label,
        },
    })


# =============================================================================
#  ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# =============================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 60000))
    print(f"\n{'='*52}")
    print(f"  FinSight AI v2 — Production Backend")
    print(f"  http://localhost:{port}")
    print(f"  Future Scope: http://localhost:{port}/future")
    print(f"{'='*52}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
