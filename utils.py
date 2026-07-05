# =============================================================================
#  utils.py  —  FinSight AI · Shared Utility Functions
# =============================================================================
#  Responsibilities:
#    • fetch_ohlcv()            — download price data from Yahoo Finance
#    • get_company_info()       — fetch ticker metadata
#    • add_technical_indicators() — SMA, EMA, RSI, MACD, Bollinger Bands
#    • future_business_dates()  — generate forecast date labels
#    • safe_float()             — NaN-safe value converter
#    • simulate_ohlcv()         — deterministic fallback when offline
# =============================================================================

import os, math, warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd

# ── Realistic offline fallback parameters ────────────────────────────────────
_STOCK_PARAMS = {
    "AAPL": {"name":"Apple Inc.",       "sector":"Technology",    "base":213,  "vol":.012, "trend":.00025},
    "TSLA": {"name":"Tesla Inc.",        "sector":"Auto / EV",     "base":177,  "vol":.028, "trend":.00030},
    "MSFT": {"name":"Microsoft Corp.",   "sector":"Technology",    "base":422,  "vol":.011, "trend":.00028},
    "NVDA": {"name":"NVIDIA Corp.",      "sector":"Semiconductors","base":131,  "vol":.025, "trend":.00060},
    "AMZN": {"name":"Amazon.com Inc.",   "sector":"E-Commerce",    "base":197,  "vol":.014, "trend":.00022},
    "META": {"name":"Meta Platforms",    "sector":"Social Media",  "base":583,  "vol":.017, "trend":.00035},
    "GOOGL":{"name":"Alphabet Inc.",     "sector":"Technology",    "base":178,  "vol":.013, "trend":.00020},
    "JPM":  {"name":"JPMorgan Chase",    "sector":"Financials",    "base":244,  "vol":.013, "trend":.00020},
    "NFLX": {"name":"Netflix Inc.",      "sector":"Streaming",     "base":1100, "vol":.020, "trend":.00030},
    "BTC-USD":{"name":"Bitcoin USD",     "sector":"Crypto",        "base":97000,"vol":.040, "trend":.00080},
    "ETH-USD":{"name":"Ethereum USD",    "sector":"Crypto",        "base":3400, "vol":.045, "trend":.00070},
    "SPY":  {"name":"S&P 500 ETF",       "sector":"ETF",           "base":591,  "vol":.010, "trend":.00022},
}


# =============================================================================
#  PRICE DATA
# =============================================================================

def fetch_ohlcv(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Download OHLCV price data for a ticker.

    Tries Yahoo Finance first (requires internet).
    Falls back to a deterministic simulated series if yfinance fails
    so the app always returns something useful.

    Parameters
    ----------
    symbol : str    e.g. "AAPL"
    period : str    yfinance period string — "1mo" "3mo" "6mo" "1y" "2y"

    Returns
    -------
    pd.DataFrame  with columns: Date, Open, High, Low, Close, Volume
    """
    try:
        import yfinance as yf
        df = yf.Ticker(symbol.upper()).history(period=period)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            keep = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]
            df   = df[keep].dropna(subset=["Close"])
            if len(df) > 20:
                print(f"[utils] ✅ yfinance: {symbol} {len(df)} rows")
                return df
    except Exception as e:
        print(f"[utils] yfinance error ({e.__class__.__name__}), using simulation")

    return simulate_ohlcv(symbol, period)


def simulate_ohlcv(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Generate realistic OHLCV data via Geometric Brownian Motion.
    Deterministic: same symbol always produces the same series.
    """
    days_map = {"1mo":21, "3mo":63, "6mo":126, "1y":252, "2y":504, "5y":1260}
    n        = days_map.get(period, 252)
    cfg      = _STOCK_PARAMS.get(symbol.upper(), {"base":150, "vol":.015, "trend":.0002})
    rng      = np.random.default_rng(abs(hash(symbol)) % (2**31))

    price = cfg["base"] * (0.88 + rng.random() * .12)
    rows, today = [], datetime.today()

    for i in range(n, -1, -1):
        dt = today - timedelta(days=i)
        if dt.weekday() >= 5:
            continue
        price = max(price * (1 + cfg["trend"] + rng.normal(0, cfg["vol"])), 0.01)
        o = price * (1 + rng.uniform(-.004, .004))
        h = max(o, price) * (1 + rng.uniform(0, .008))
        l = min(o, price) * (1 - rng.uniform(0, .008))
        rows.append({
            "Date":   dt,
            "Open":   round(o, 2),
            "High":   round(h, 2),
            "Low":    round(l, 2),
            "Close":  round(price, 2),
            "Volume": int(cfg["base"] * 500_000 * rng.uniform(.4, 1.8)),
        })

    print(f"[utils] 📊 simulated {symbol}: {len(rows)} rows")
    return pd.DataFrame(rows)


def get_company_info(symbol: str) -> dict:
    """Return company metadata from yfinance or local lookup."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol.upper()).info
        if info.get("longName"):
            return {
                "name":        info.get("longName", symbol.upper()),
                "sector":      info.get("sector", "N/A"),
                "industry":    info.get("industry", "N/A"),
                "currency":    info.get("currency", "USD"),
                "exchange":    info.get("exchange", "N/A"),
                "market_cap":  info.get("marketCap"),
                "pe_ratio":    info.get("trailingPE"),
                "week52_high": info.get("fiftyTwoWeekHigh"),
                "week52_low":  info.get("fiftyTwoWeekLow"),
                "avg_volume":  info.get("averageVolume"),
                "description": (info.get("longBusinessSummary","")[:280] + "…")
                               if info.get("longBusinessSummary") else "",
            }
    except:
        pass
    cfg = _STOCK_PARAMS.get(symbol.upper(), {})
    return {
        "name": cfg.get("name", symbol.upper()),
        "sector": cfg.get("sector", "N/A"),
        "industry": "N/A", "currency": "USD", "exchange": "N/A",
        "market_cap": None, "pe_ratio": None,
        "week52_high": None, "week52_low": None,
        "avg_volume": None, "description": "",
    }


# =============================================================================
#  TECHNICAL INDICATORS
# =============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append common technical indicators as new columns.

    Added columns:
        SMA20, SMA50         — Simple Moving Averages
        EMA12, EMA26         — Exponential Moving Averages
        RSI14                — Relative Strength Index
        MACD, MACD_Signal    — Moving Average Convergence Divergence
        BB_Upper, BB_Lower   — Bollinger Bands (20-day, 2σ)
    """
    df  = df.copy()
    c   = df["Close"]

    # ── Simple Moving Averages ────────────────────────────────────────────────
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()

    # ── Exponential Moving Averages ───────────────────────────────────────────
    df["EMA12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA26"] = c.ewm(span=26, adjust=False).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ── MACD ──────────────────────────────────────────────────────────────────
    df["MACD"]        = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    rolling_mean = c.rolling(20).mean()
    rolling_std  = c.rolling(20).std()
    df["BB_Upper"] = rolling_mean + 2 * rolling_std
    df["BB_Lower"] = rolling_mean - 2 * rolling_std

    return df


# =============================================================================
#  DATE HELPERS
# =============================================================================

def future_business_dates(last_date: str, n: int = 7) -> list[str]:
    """
    Return n business-day (Mon–Fri) date strings after last_date.
    Format: "YYYY-MM-DD"
    """
    dates, d = [], datetime.strptime(last_date, "%Y-%m-%d")
    while len(dates) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
    return dates


# =============================================================================
#  MISC
# =============================================================================

def safe_float(val) -> float | None:
    """Convert to float, returning None instead of NaN/inf."""
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def serialise_row(row: pd.Series) -> dict:
    """Convert one DataFrame row into a JSON-safe dict."""
    return {
        "date":   row["Date"].strftime("%Y-%m-%d"),
        "open":   safe_float(row.get("Open")),
        "high":   safe_float(row.get("High")),
        "low":    safe_float(row.get("Low")),
        "close":  safe_float(row.get("Close")),
        "volume": int(row["Volume"]) if row.get("Volume") else None,
        "sma20":  safe_float(row.get("SMA20")),
        "sma50":  safe_float(row.get("SMA50")),
        "rsi":    safe_float(row.get("RSI14")),
        "macd":   safe_float(row.get("MACD")),
        "bb_upper": safe_float(row.get("BB_Upper")),
        "bb_lower": safe_float(row.get("BB_Lower")),
    }
