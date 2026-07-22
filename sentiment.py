# =============================================================================
#  sentiment.py  —  FinSight AI · News Sentiment Analysis
# =============================================================================
#  Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) extended
#  with a custom financial lexicon to score news headlines.
#
#  Data sources (tried in order):
#    1. NewsAPI          (if NEWS_API_KEY set in .env)
#    2. yfinance .news   (built-in, no key needed)
#    3. Simulated news   (offline fallback)
#
#  Public API:
#    analyse_ticker(symbol)  →  full sentiment result dict
# =============================================================================

import os
from datetime import datetime

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

# ── Build the analyser once at import time ────────────────────────────────────
_analyser = SentimentIntensityAnalyzer()

# ── Custom financial lexicon additions ───────────────────────────────────────
# Higher absolute value = stronger opinion signal.
_FINANCE_LEXICON = {
    # Bullish
    "bullish": 2.5,  "upgrade": 1.5,    "outperform": 2.0,  "overweight": 1.5,
    "beat":    1.5,  "rally":   2.0,     "surge":      2.0,  "soar":       2.5,
    "breakout":1.8,  "record":  1.2,     "profit":     1.5,  "growth":     1.2,
    "buyback": 1.2,  "dividend":1.0,     "expand":     1.0,  "guidance":   0.5,
    "exceed":  1.5,  "strong":  1.5,     "robust":     1.4,  "momentum":   1.2,

    # Bearish
    "bearish": -2.5, "downgrade": -1.5,  "underperform":-2.0,"underweight":-1.5,
    "miss":    -1.5, "crash":    -2.5,   "plunge":     -2.5, "tumble":     -2.0,
    "slump":   -2.0, "layoff":   -2.0,   "loss":       -1.8, "debt":       -0.8,
    "lawsuit": -1.5, "probe":    -1.8,   "fraud":      -3.0, "recall":     -1.8,
    "volatile":-0.5, "recession":-2.2,   "shortfall":  -1.5, "decline":    -1.2,

    # Neutral / context-sensitive
    "merger":  0.3,  "ipo":     0.5,     "split":      0.2,  "report":     0.0,
    "earnings":0.0,  "quarter": 0.0,     "forecast":   0.0,  "analyst":    0.0,
}
_analyser.lexicon.update(_FINANCE_LEXICON)

# ── Simulated headlines (offline fallback) ────────────────────────────────────
_HEADLINES = {
    "positive": [
        "{s} beats earnings estimates; raises full-year guidance",
        "{s} announces $10B stock buyback programme",
        "Analysts upgrade {s} to Strong Buy ahead of product launch",
        "{s} reports record quarterly revenue driven by AI division",
        "{s} expands into new markets; shares hit 52-week high",
        "Institutional investors increase {s} holdings significantly",
        "{s} secures landmark partnership deal worth billions",
    ],
    "negative": [
        "{s} misses revenue expectations amid slowing consumer demand",
        "Regulatory probe into {s} business practices widens",
        "{s} announces layoffs affecting 5% of global workforce",
        "Supply chain disruptions weigh on {s} quarterly outlook",
        "{s} faces increased competition; margin pressure expected",
        "Short sellers increase bets against {s} as growth slows",
        "{s} warns of weaker-than-expected Q4 results",
    ],
    "neutral": [
        "{s} management presents five-year strategic roadmap",
        "Analyst maintains Hold rating on {s} with unchanged target",
        "{s} CFO speaks at Goldman Sachs investor conference",
        "{s} to report quarterly earnings next week",
        "{s} files routine 10-K disclosure with the SEC",
        "Market awaits {s} product announcement next quarter",
    ],
}
_SOURCES = [
    "Reuters", "Bloomberg", "CNBC", "MarketWatch",
    "Barron's", "The WSJ", "Seeking Alpha", "Motley Fool",
]


# =============================================================================
#  NEWS FETCHING
# =============================================================================

def fetch_news(symbol: str, max_articles: int = 10) -> list[dict]:
    """
    Fetch recent news articles for a ticker symbol.

    Tries (in order):
      1. NewsAPI.org — if NEWS_API_KEY is set in environment / .env
      2. yfinance built-in .news property
      3. Simulated plausible headlines (always works, no internet needed)

    Each returned article dict has:
        title, description, source, published_at
    """
    api_key = os.getenv("NEWS_API_KEY", "").strip()

    if api_key:
        articles = _from_newsapi(symbol, api_key, max_articles)
        if articles:
            return articles

    articles = _from_yfinance(symbol, max_articles)
    if articles:
        return articles

    return _simulate_news(symbol, max_articles)


def _from_newsapi(symbol: str, api_key: str, n: int) -> list[dict]:
    """Query newsapi.org for the latest articles about a ticker."""
    try:
        import requests
        from utils import get_company_info
        name  = get_company_info(symbol).get("name", symbol)
        query = f"{name} OR {symbol.upper()} stock"
        resp  = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "language": "en", "sortBy": "publishedAt",
                    "pageSize": n, "apiKey": api_key},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") != "ok":
            return []
        return [
            {
                "title":        a.get("title", ""),
                "description":  a.get("description") or a.get("title", ""),
                "source":       a.get("source", {}).get("name", "NewsAPI"),
                "published_at": a.get("publishedAt", ""),
            }
            for a in data.get("articles", [])
            if a.get("title")
        ]
    except Exception as e:
        print(f"[sentiment] NewsAPI error: {e}")
        return []


def _from_yfinance(symbol: str, n: int) -> list[dict]:
    """Use yfinance's built-in .news property."""
    try:
        import yfinance as yf
        raw = yf.Ticker(symbol.upper()).news or []
        result = []
        for item in raw[:n]:
            ct    = item.get("content", item)
            title = ct.get("title") or item.get("title", "")
            desc  = ct.get("summary") or item.get("summary") or title
            src   = (ct.get("provider", {}).get("displayName")
                     or item.get("publisher", "Yahoo Finance"))
            pub   = ct.get("pubDate") or item.get("providerPublishTime", "")
            if title:
                result.append({
                    "title": title, "description": desc,
                    "source": src, "published_at": str(pub),
                })
        if result:
            print(f"[sentiment] ✅ yfinance news: {len(result)} articles")
        return result
    except Exception as e:
        print(f"[sentiment] yfinance news error: {e}")
        return []


def _simulate_news(symbol: str, n: int = 8) -> list[dict]:
    """Generate deterministic realistic headlines (no internet needed)."""
    rng = np.random.default_rng(
        abs(hash(symbol + str(datetime.today().date()))) % (2**31)
    )
    pool = (
        [(t, "positive") for t in _HEADLINES["positive"]] * 3 +
        [(t, "neutral")  for t in _HEADLINES["neutral"]]  * 2 +
        [(t, "negative") for t in _HEADLINES["negative"]] * 1
    )
    idxs = rng.permutation(len(pool))[:n]
    return [
        {
            "title":        pool[i][0].replace("{s}", symbol.upper()),
            "description":  pool[i][0].replace("{s}", symbol.upper()),
            "source":       _SOURCES[j % len(_SOURCES)],
            "published_at": datetime.today().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for j, i in enumerate(idxs)
    ]


# =============================================================================
#  SCORING
# =============================================================================

def score_text(text: str) -> dict:
    """
    Run VADER on a single piece of text.

    Returns
    -------
    dict: positive, negative, neutral, compound, label
    """
    scores = _analyser.polarity_scores(str(text))
    c      = scores["compound"]
    label  = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
    return {
        "positive": round(scores["pos"], 3),
        "negative": round(scores["neg"], 3),
        "neutral":  round(scores["neu"], 3),
        "compound": round(c, 3),
        "label":    label,
    }


def score_articles(articles: list[dict]) -> list[dict]:
    """Score every article and attach label + compound to each dict."""
    scored = []
    for art in articles:
        text   = f"{art.get('title', '')}. {art.get('description', '')}".strip(". ")
        result = score_text(text)
        scored.append({**art, **result})
    return scored


def aggregate_scores(scored_articles: list[dict]) -> dict:
    """
    Average VADER compound scores across all articles.

    Returns
    -------
    dict:
        overall_score   float    (-1 to +1)
        overall_label   str      positive | negative | neutral
        positive_pct    float    (0–100)
        negative_pct    float    (0–100)
        neutral_pct     float    (0–100)
        article_count   int
    """
    if not scored_articles:
        return {
            "overall_score":  0.0,
            "overall_label":  "neutral",
            "positive_pct":   0.0,
            "negative_pct":   0.0,
            "neutral_pct":    100.0,
            "article_count":  0,
        }

    compounds = [a["compound"] for a in scored_articles]
    n         = len(compounds)
    avg       = round(float(np.mean(compounds)), 3)
    pos       = sum(1 for c in compounds if c >= 0.05)
    neg       = sum(1 for c in compounds if c <= -0.05)
    neu       = n - pos - neg
    label     = "positive" if avg >= 0.05 else "negative" if avg <= -0.05 else "neutral"

    return {
        "overall_score":  avg,
        "overall_label":  label,
        "positive_pct":   round(pos / n * 100, 1),
        "negative_pct":   round(neg / n * 100, 1),
        "neutral_pct":    round(neu / n * 100, 1),
        "article_count":  n,
    }


# =============================================================================
#  PUBLIC ENTRY POINT
# =============================================================================

def analyse_ticker(symbol: str, max_articles: int = 10) -> dict:
    """
    Full sentiment pipeline for a ticker.

    Returns
    -------
    dict:
        overall_score   float
        overall_label   str
        positive_pct    float
        negative_pct    float
        neutral_pct     float
        article_count   int
        articles        list of scored article dicts
    """
    print(f"[sentiment] Analysing {symbol} …")
    articles = fetch_news(symbol, max_articles)
    scored   = score_articles(articles)
    summary  = aggregate_scores(scored)

    print(f"[sentiment] Score={summary['overall_score']}  "
          f"Label={summary['overall_label']}  "
          f"n={summary['article_count']}")

    return {**summary, "articles": scored}
