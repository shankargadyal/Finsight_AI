# =============================================================================
#  news_summarizer.py  —  FinSight AI v2 · AI News Summarizer
# =============================================================================
#  Uses the Google Gemini API to generate intelligent, structured summaries
#  of raw news articles fetched for a ticker using Pydantic structured schemas.
#
#  Falls back gracefully if no API key is present.
# =============================================================================

import os, json, re
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Define the expected JSON output schema using Pydantic
class FinancialSummarySchema(BaseModel):
    summary: str = Field(description="2-3 sentence synthesis of market narrative")
    key_themes: list[str] = Field(description="List of 3 primary market themes found in the context")
    impact_level: str = Field(description="Must be strictly one of: High Positive|Moderate Positive|Neutral|Moderate Negative|High Negative")
    bullets: list[str] = Field(description="List of 3 key insight bullet points")
    analyst_note: str = Field(description="1-sentence analyst recommendation context")


def summarise_news(symbol: str, articles: list[dict], sentiment_summary: dict) -> dict:
    """
    Generate an AI-powered news summary for a ticker.

    Parameters
    ----------
    symbol            : str   — ticker symbol
    articles          : list  — scored article dicts from sentiment.py
    sentiment_summary : dict  — aggregate sentiment from sentiment.py

    Returns
    -------
    dict with: summary, key_themes, impact_level, impact_color,
               bullets, confidence, generated_at
    """
    if not articles:
        return _empty_summary(symbol)

    # Build prompt context
    headlines = "\n".join(
        f"- [{a.get('label','neutral').upper()}] {a.get('title','')}"
        for a in articles[:10]
    )

    score       = sentiment_summary.get("overall_score", 0)
    label       = sentiment_summary.get("overall_label", "neutral")
    pos_pct     = sentiment_summary.get("positive_pct", 0)
    neg_pct     = sentiment_summary.get("negative_pct", 0)
    art_count   = sentiment_summary.get("article_count", 0)

    if GEMINI_API_KEY:
        return _summarise_with_gemini(symbol, headlines, score, label,
                                      pos_pct, neg_pct, art_count)
    else:
        return _summarise_heuristic(symbol, articles, score, label,
                                     pos_pct, neg_pct, art_count)


# =============================================================================
#  Gemini-powered summary
# =============================================================================

def _summarise_with_gemini(symbol, headlines, score, label,
                           pos_pct, neg_pct, art_count) -> dict:
    try:
        # Initialize standard GenAI client
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""You are a senior financial analyst. Analyse these recent news headlines for {symbol}:

{headlines}

Sentiment data: overall score={score:.3f}, label={label},
positive={pos_pct:.0f}%, negative={neg_pct:.0f}%, articles={art_count}"""

        # Request structured JSON matching our Pydantic schema
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are an expert financial market summarizer. Analyze data objectively.",
                response_mime_type="application/json",
                response_schema=FinancialSummarySchema,
                max_output_tokens=600,
                temperature=0.2
            ),
        )

        # Parse verified JSON data directly from the response
        data = json.loads(response.text)

        impact = data.get("impact_level", "Neutral")
        return {
            "summary":       data.get("summary", ""),
            "key_themes":    data.get("key_themes", []),
            "impact_level":  impact,
            "impact_color":  _impact_color(impact),
            "bullets":       data.get("bullets", []),
            "analyst_note":  data.get("analyst_note", ""),
            "sentiment_score": round(score, 3),
            "sentiment_label": label,
            "article_count": art_count,
            "ai_powered":    True,
            "generated_at":  datetime.utcnow().isoformat(),
        }
    except Exception as e:
        print(f"[news_summarizer] Gemini API error: {e}")
        return _summarise_heuristic(symbol, [], score, label,
                                     pos_pct, neg_pct, art_count)


# =============================================================================
#  Heuristic fallback (no API key)
# =============================================================================

def _summarise_heuristic(symbol, articles, score, label,
                          pos_pct, neg_pct, art_count) -> dict:
    """Rule-based summary when no LLM is available."""

    # Extract prominent words from titles
    themes = []
    title_words = " ".join(a.get("title", "") for a in articles).lower()
    if any(w in title_words for w in ["earnings", "revenue", "profit", "quarterly"]):
        themes.append("Earnings & Revenue")
    if any(w in title_words for w in ["upgrade", "downgrade", "analyst", "rating"]):
        themes.append("Analyst Actions")
    if any(w in title_words for w in ["product", "launch", "deal", "partner"]):
        themes.append("Business Development")
    if any(w in title_words for w in ["regulation", "probe", "lawsuit", "sec"]):
        themes.append("Regulatory Risk")
    if any(w in title_words for w in ["ai", "technology", "cloud", "innovation"]):
        themes.append("Technology & Innovation")
    if not themes:
        themes = ["Market Activity", "General News"]

    if score >= 0.4:
        impact = "High Positive"
        summary = (
            f"Recent news flow for {symbol} is strongly positive, with multiple reports "
            f"highlighting bullish catalysts. {pos_pct:.0f}% of coverage carries a positive tone, "
            "suggesting strong investor confidence and favourable near-term sentiment."
        )
        bullets = [
            f"Positive sentiment dominates at {pos_pct:.0f}% of articles",
            "Analyst community appears broadly supportive",
            "Market narrative aligns with upward price potential",
        ]
    elif score >= 0.10:
        impact = "Moderate Positive"
        summary = (
            f"News sentiment for {symbol} is moderately positive. "
            f"With {pos_pct:.0f}% positive coverage and a compound score of {score:.2f}, "
            "the market tone is constructive, though mixed signals warrant careful monitoring."
        )
        bullets = [
            f"Majority of {art_count} articles lean positive",
            "Some cautionary tones noted in broader coverage",
            "Near-term catalyst watch recommended",
        ]
    elif score >= -0.10:
        impact = "Neutral"
        summary = (
            f"News sentiment for {symbol} is broadly neutral (score: {score:.2f}). "
            "Coverage is balanced between positive and negative narratives, "
            "suggesting the market is in a wait-and-see posture ahead of new catalysts."
        )
        bullets = [
            "Balanced coverage with no dominant narrative",
            f"{pos_pct:.0f}% positive vs {neg_pct:.0f}% negative articles",
            "Recommend monitoring upcoming earnings or announcements",
        ]
    elif score >= -0.4:
        impact = "Moderate Negative"
        summary = (
            f"Recent news for {symbol} skews negative (score: {score:.2f}). "
            f"{neg_pct:.0f}% of articles carry bearish tones, indicating headwinds "
            "that may pressure near-term price action."
        )
        bullets = [
            f"Negative sentiment at {neg_pct:.0f}% of coverage",
            "Potential downside catalysts being reported",
            "Consider risk management strategies",
        ]
    else:
        impact = "High Negative"
        summary = (
            f"News sentiment for {symbol} is strongly negative (score: {score:.2f}). "
            "Predominantly bearish coverage suggests significant headwinds and "
            "heightened investor concern over near-term prospects."
        )
        bullets = [
            f"Strongly bearish coverage: {neg_pct:.0f}% negative articles",
            "Multiple risk factors highlighted in recent reporting",
            "High caution advised; monitor for stabilising signals",
        ]

    return {
        "summary":         summary,
        "key_themes":      themes[:4],
        "impact_level":    impact,
        "impact_color":    _impact_color(impact),
        "bullets":         bullets,
        "analyst_note":    "AI summarizer offline — rule-based analysis applied.",
        "sentiment_score": round(score, 3),
        "sentiment_label": label,
        "article_count":   art_count,
        "ai_powered":      False,
        "generated_at":    datetime.utcnow().isoformat(),
    }


def _impact_color(impact: str) -> str:
    return {
        "High Positive":     "#00FF9D",
        "Moderate Positive": "#7AFF7A",
        "Neutral":           "#FFB800",
        "Moderate Negative": "#FF9060",
        "High Negative":     "#FF3860",
    }.get(impact, "#7A9CC5")


def _empty_summary(symbol: str) -> dict:
    return {
        "summary":       f"No recent news articles found for {symbol}.",
        "key_themes":    [],
        "impact_level":  "Neutral",
        "impact_color":  "#FFB800",
        "bullets":       [],
        "analyst_note":  "",
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "article_count": 0,
        "ai_powered":    False,
        "generated_at":  datetime.utcnow().isoformat(),
    }