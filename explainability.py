"""
explainability.py — Lightweight, exact SHAP-equivalent explanations for FinSight AI.

Design constraints (Render Free tier: ~512MB RAM, fractional shared CPU):
  - Does NOT use shap.KernelExplainer or shap.DeepExplainer. Those explain a
    model by re-running it hundreds of times on perturbed inputs — exactly
    the pattern that caused OOM kills / worker timeouts on this deployment.
  - Does NOT import the `shap` package at all. For a plain linear model,
    shap.LinearExplainer's output is just a closed-form calculation:
        shap_value_i = coef_i * standardized_feature_i
        base_value   = model.intercept_
    (since StandardScaler-transformed features have ~0 mean, so the
    "expected"/background prediction collapses to the intercept). This
    file computes exactly that, avoiding `shap`'s numba/llvmlite dependency
    chain, which adds meaningful install size and import-time RAM — not
    worth it for a calculation this simple.
  - Trains a small, separate LinearRegression on technical indicators →
    next-day return, purely for explanation purposes. It does NOT feed the
    ensemble's price forecast — it answers a narrower, honest question:
    "of these technical indicators, which ones are pushing tomorrow's
    return up or down, right now?"

Reuses the OHLCV + technical-indicator DataFrame the caller already fetched
(via utils.fetch_ohlcv + utils.add_technical_indicators) — no extra network
calls, no extra heavy dependencies beyond scikit-learn (already required).
"""
import numpy as np
import pandas as pd

FEATURES = [
    "SMA20", "SMA50", "EMA12", "EMA26",
    "RSI14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower",
]

# Short, honest, generic templates — not claims of certainty.
_PLAIN_ENGLISH = {
    "RSI14":       lambda v, up: f"RSI14 is {'elevated' if v > 60 else 'low' if v < 40 else 'neutral'} "
                                  f"({v:.1f}), {'pushing the outlook up' if up else 'pushing the outlook down'}.",
    "MACD":        lambda v, up: f"MACD is {'above' if v > 0 else 'below'} zero ({v:.2f}), "
                                  f"{'supporting' if up else 'weighing on'} the short-term trend.",
    "MACD_Signal": lambda v, up: f"MACD signal line is {'supporting' if up else 'weighing on'} momentum.",
    "MACD_Hist":   lambda v, up: f"MACD histogram is {'positive' if v > 0 else 'negative'}, "
                                  f"{'supporting' if up else 'weighing on'} near-term momentum.",
    "SMA20":       lambda v, up: f"Price relative to the 20-day average is {'supporting' if up else 'weighing on'} the outlook.",
    "SMA50":       lambda v, up: f"Price relative to the 50-day average is {'supporting' if up else 'weighing on'} the outlook.",
    "EMA12":       lambda v, up: f"Short-term trend (EMA12) is {'supporting' if up else 'weighing on'} the outlook.",
    "EMA26":       lambda v, up: f"Longer-term trend (EMA26) is {'supporting' if up else 'weighing on'} the outlook.",
    "BB_Upper":    lambda v, up: f"Price relative to the upper Bollinger Band is {'supporting' if up else 'weighing on'} the outlook.",
    "BB_Lower":    lambda v, up: f"Price relative to the lower Bollinger Band is {'supporting' if up else 'weighing on'} the outlook.",
}


class ExplainabilityUnavailableError(Exception):
    pass


def explain_next_day_return(df_with_indicators: pd.DataFrame, top_n: int = 4) -> dict:
    """
    df_with_indicators: output of utils.add_technical_indicators(fetch_ohlcv(...))
    Returns top positive/negative contributing indicators for tomorrow's
    predicted return, with exact SHAP values and plain-English notes.
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ExplainabilityUnavailableError(f"Missing dependency: {e}")

    df = df_with_indicators.copy()
    df["NextReturn"] = df["Close"].pct_change().shift(-1)
    df = df.dropna(subset=FEATURES + ["NextReturn"])

    if len(df) < 40:
        raise ExplainabilityUnavailableError("Not enough history to fit an explainability model.")

    X_hist = df[FEATURES].iloc[:-1]       # exclude the last row (today) — it's our query point
    y_hist = df["NextReturn"].iloc[:-1]
    x_today = df[FEATURES].iloc[[-1]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hist)
    x_today_scaled = scaler.transform(x_today)

    model = LinearRegression()
    model.fit(X_scaled, y_hist)

    predicted_return = float(model.predict(x_today_scaled)[0])

    # Exact linear SHAP, computed directly:
    #   shap_i     = coef_i * x_scaled_i   (StandardScaler → training mean ≈ 0,
    #                                        so this equals coef_i * (x_i - mean_i))
    #   base_value = intercept_            (the model's prediction at the mean feature vector)
    base_value = float(model.intercept_)
    shap_values = model.coef_ * x_today_scaled[0]
    assert abs((base_value + shap_values.sum()) - predicted_return) < 1e-6, \
        "sanity check: base_value + shap contributions must equal the prediction"

    contributions = sorted(
        zip(FEATURES, shap_values, x_today.iloc[0].tolist()),
        key=lambda t: abs(t[1]),
        reverse=True,
    )

    positives = [(f, sv, v) for f, sv, v in contributions if sv > 0][:top_n]
    negatives = [(f, sv, v) for f, sv, v in contributions if sv < 0][:top_n]

    def _fmt(items):
        out = []
        for feat, sv, val in items:
            note = _PLAIN_ENGLISH.get(feat, lambda v, up: f"{feat} is {'supporting' if up else 'weighing on'} the outlook.")
            out.append({
                "feature": feat,
                "value": round(float(val), 3),
                "shap_value": round(float(sv), 5),
                "note": note(val, sv > 0),
            })
        return out

    return {
        "predicted_next_day_return_pct": round(predicted_return * 100, 3),
        "base_value_pct": round(base_value * 100, 3),
        "top_positive_features": _fmt(positives),
        "top_negative_features": _fmt(negatives),
        "disclaimer": ("This explains a small technical-indicator model trained just for this purpose — "
                        "it does not explain the ARIMA/LSTM components of the main forecast."),
    }
