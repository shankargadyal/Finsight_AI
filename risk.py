# =============================================================================
#  risk.py  —  FinSight AI v2 · Risk Analysis Engine
# =============================================================================
import numpy as np
import pandas as pd


# =============================================================================
#  Core calculation
# =============================================================================

def calculate_risk(close_prices: list) -> dict:
    """
    Calculate a comprehensive risk score (0–100) from historical price data.

    Factors
    -------
    1. Annualised Volatility (historical std of log returns) — weight 35%
    2. Max Drawdown from peak — weight 25%
    3. Average True Range ratio (ATR / price) — weight 20%
    4. Downside Deviation — weight 20%

    Returns
    -------
    dict with: score, level, volatility, max_drawdown, avg_daily_return,
               std_daily_return, downside_deviation, explanation, factors
    """
    prices = np.array(close_prices, dtype=float)
    if len(prices) < 30:
        return _empty_risk()

    # ── Daily log returns ──────────────────────────────────────────────────────
    log_returns   = np.diff(np.log(prices))
    daily_returns = np.diff(prices) / prices[:-1]

    # ── 1. Annualised volatility (σ × √252) ────────────────────────────────────
    ann_vol = float(np.std(log_returns) * np.sqrt(252))

    # ── 2. Maximum drawdown ────────────────────────────────────────────────────
    rolling_max = np.maximum.accumulate(prices)
    drawdowns   = (prices - rolling_max) / rolling_max
    max_dd      = float(abs(drawdowns.min()))

    # ── 3. ATR-based volatility (approximate) ─────────────────────────────────
    price_range_pct = float(np.mean(np.abs(np.diff(prices)) / prices[:-1]))

    # ── 4. Downside deviation ─────────────────────────────────────────────────
    neg_returns     = daily_returns[daily_returns < 0]
    downside_dev    = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.0

    # ── Scoring (each mapped to 0–100 then weighted) ──────────────────────────
    # Annualised vol thresholds: < 15% = low, 15–35% = medium, > 35% = high
    vol_score = _pct_score(ann_vol, low=0.15, high=0.50)

    # Max drawdown thresholds: < 10% = low, 10–30% = medium, > 30% = high
    dd_score = _pct_score(max_dd, low=0.10, high=0.40)

    # Price range: < 1% = low, 1–3% = medium, > 3% = high
    atr_score = _pct_score(price_range_pct, low=0.01, high=0.04)

    # Downside deviation: < 0.8% = low, 0.8–2% = medium, > 2% = high
    ds_score = _pct_score(downside_dev, low=0.008, high=0.025)

    composite = (
        vol_score * 0.35 +
        dd_score  * 0.25 +
        atr_score * 0.20 +
        ds_score  * 0.20
    )
    score = round(float(composite), 1)

    # ── Category ──────────────────────────────────────────────────────────────
    if score < 35:
        level = "Low Risk"
        color = "#00FF9D"
        icon  = "🟢"
        explanation = (
            f"This asset shows low historical volatility ({ann_vol*100:.1f}% annualised) "
            f"and a modest max drawdown of {max_dd*100:.1f}%. "
            "Suitable for conservative investors seeking stability."
        )
    elif score < 65:
        level = "Medium Risk"
        color = "#FFB800"
        icon  = "🟡"
        explanation = (
            f"Moderate volatility ({ann_vol*100:.1f}% annualised) with a "
            f"max drawdown of {max_dd*100:.1f}%. "
            "Balanced risk profile — appropriate for growth-oriented investors."
        )
    else:
        level = "High Risk"
        color = "#FF3860"
        icon  = "🔴"
        explanation = (
            f"Elevated volatility ({ann_vol*100:.1f}% annualised) and a "
            f"significant max drawdown of {max_dd*100:.1f}%. "
            "Only suitable for risk-tolerant investors with a long time horizon."
        )

    # ── Factor breakdown ──────────────────────────────────────────────────────
    factors = [
        {
            "name":        "Annualised Volatility",
            "value":       f"{ann_vol*100:.2f}%",
            "score":       round(vol_score, 1),
            "weight":      "35%",
            "description": "Yearly price fluctuation from log returns",
        },
        {
            "name":        "Maximum Drawdown",
            "value":       f"{max_dd*100:.2f}%",
            "score":       round(dd_score, 1),
            "weight":      "25%",
            "description": "Largest peak-to-trough decline",
        },
        {
            "name":        "Daily Price Range",
            "value":       f"{price_range_pct*100:.2f}%",
            "score":       round(atr_score, 1),
            "weight":      "20%",
            "description": "Average absolute daily price move",
        },
        {
            "name":        "Downside Deviation",
            "value":       f"{downside_dev*100:.2f}%",
            "score":       round(ds_score, 1),
            "weight":      "20%",
            "description": "Std deviation of negative-day returns",
        },
    ]

    # Recent vs long-term vol comparison
    recent_vol  = float(np.std(log_returns[-30:]) * np.sqrt(252)) if len(log_returns) >= 30 else ann_vol
    historic_vol = ann_vol
    vol_trend = "increasing" if recent_vol > historic_vol * 1.1 else \
                "decreasing" if recent_vol < historic_vol * 0.9 else "stable"

    return {
        "score":              score,
        "level":              level,
        "color":              color,
        "icon":               icon,
        "explanation":        explanation,
        "factors":            factors,
        "volatility":         round(ann_vol * 100, 2),
        "max_drawdown":       round(max_dd  * 100, 2),
        "avg_daily_return":   round(float(np.mean(daily_returns)) * 100, 4),
        "std_daily_return":   round(float(np.std(daily_returns))  * 100, 4),
        "downside_deviation": round(downside_dev * 100, 4),
        "volatility_trend":   vol_trend,
        "recent_vol":         round(recent_vol * 100, 2),
    }


def _pct_score(value: float, low: float, high: float) -> float:
    """Map a metric to 0–100 linearly between low (→0) and high (→100)."""
    return float(np.clip((value - low) / (high - low) * 100, 0, 100))


def _empty_risk() -> dict:
    return {
        "score": 50, "level": "Medium Risk", "color": "#FFB800", "icon": "🟡",
        "explanation": "Insufficient data to calculate risk.",
        "factors": [], "volatility": 0, "max_drawdown": 0,
        "avg_daily_return": 0, "std_daily_return": 0,
        "downside_deviation": 0, "volatility_trend": "stable",
        "recent_vol": 0,
    }
