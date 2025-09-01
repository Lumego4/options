import math
import pandas as pd
from typing import Optional, Dict, Any
from data_helpers import _to_float

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def technical_reco(ticker: str) -> str:
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d")
        if hist is None or hist.empty:
            return "hold"
        close = hist["Close"].dropna()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        rsi14 = rsi(close, 14)
        curr = close.iloc[-1]
        s50 = float(sma50.iloc[-1])
        s200 = float(sma200.iloc[-1])
        r = float(rsi14.iloc[-1]) if not math.isnan(rsi14.iloc[-1]) else 50.0

        score = 0.0
        if curr > s50 > s200: score += 0.6
        elif s50 > s200: score += 0.3
        if 40 <= r <= 60: score += 0.2
        elif r < 35: score += 0.3
        elif r > 70: score -= 0.3

        return "buy" if score >= 0.7 else ("hold" if score >= 0.45 else "sell")
    except Exception:
        return "hold"

def fundamental_reco(info: Dict[str, Any], price: Optional[float]) -> str:
    ps = _to_float(info.get("priceToSalesTrailing12Months"))
    rev_g = _to_float(info.get("revenueGrowth"))
    earn_g = _to_float(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))
    margin = _to_float(info.get("profitMargins"))
    roe = _to_float(info.get("returnOnEquity"))
    score = 0.0
    if ps is not None and ps > 0:
        score += min(1.0, 1.0 / (1.0 + ps)) * 0.4
    else:
        score += 0.15
    if rev_g is not None:
        score += max(0.0, math.tanh(rev_g / 0.15)) * 0.2
    if earn_g is not None:
        score += max(0.0, math.tanh(earn_g / 0.2)) * 0.15
    if margin is not None:
        score += max(0.0, math.tanh(margin / 0.15)) * 0.15
    if roe is not None:
        score += max(0.0, math.tanh(roe / 0.15)) * 0.1
    return "buy" if score >= 0.65 else ("hold" if score >= 0.45 else "sell")

def compute_custom_optimizer(ps: Optional[float],
                             premium_ratio_pct: Optional[float],
                             norm_buffer_pct: Optional[float],
                             iv: Optional[float],
                             analyst_mean: Optional[float]) -> Optional[float]:
    if premium_ratio_pct is None or norm_buffer_pct is None:
        return None
    pr = math.tanh((premium_ratio_pct / 10.0))
    nb = math.tanh((max(0.0, norm_buffer_pct) / 5.0))
    ps_term = (1.0 / (1.0 + ps)) if ps and ps > 0 else 0.5
    rec_term = 0.5
    if analyst_mean is not None and analyst_mean > 0:
        rec_term = (5.0 - analyst_mean) / 4.0
    iv_pen = 0.0
    if iv is not None and iv > 0:
        iv_pen = max(0.0, math.tanh(max(0.0, iv - 0.60) / 0.20))
    raw = (0.35 * pr + 0.25 * nb + 0.25 * ps_term + 0.15 * rec_term - 0.10 * iv_pen)
    return round(max(0.0, min(1.0, raw)) * 100.0, 2)
