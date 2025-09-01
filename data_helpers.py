from typing import Optional, List, Dict, Any
from datetime import date, datetime
import pandas as pd
import yfinance as yf
import math
import streamlit as st

def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

def _to_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return int(x)
    except Exception:
        return None

def normalize_date(d: date | str) -> str:
    if isinstance(d, date):
        return d.strftime("%Y-%m-%d")
    try:
        return pd.to_datetime(d).strftime("%Y-%m-%d")
    except Exception:
        return str(d)

@st.cache_data(ttl=300)
def get_risk_free_rate(default: float = 0.05) -> float:
    try:
        tr = yf.Ticker("^IRX")
        y = tr.fast_info.last_price
        if y and y > 0:
            return float(y) / 100.0
    except Exception:
        pass
    return default

@st.cache_data(ttl=300)
def get_underlying_price(ticker: str) -> Optional[float]:
    t = yf.Ticker(ticker)
    px = None
    try:
        px = t.fast_info.last_price
    except Exception:
        pass
    if not px or px <= 0:
        try:
            px = t.info.get("regularMarketPrice")
        except Exception:
            px = None
    if not px or px <= 0:
        try:
            hist = t.history(period="1d", interval="1m")
            if hist is not None and not hist.empty:
                px = float(hist["Close"].iloc[-1])
        except Exception:
            px = None
    return float(px) if px and px > 0 else None

@st.cache_data(ttl=900)
def get_info(ticker: str) -> Dict[str, Any]:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=600)
def get_expirations(ticker: str) -> List[str]:
    try:
        exps = yf.Ticker(ticker).options or []
        return list(exps)
    except Exception:
        return []

def closest_expiration(expirations: List[str], target_iso: str) -> Optional[str]:
    try:
        target = pd.to_datetime(target_iso).date()
        if not expirations:
            return None
        best = None
        for s in expirations:
            try:
                d = pd.to_datetime(s).date()
                dist = abs((d - target).days)
                if best is None or dist < best[0]:
                    best = (dist, s)
            except Exception:
                continue
        return best[1] if best else None
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_chain(ticker: str, expiration: str) -> Dict[str, pd.DataFrame]:
    ch = yf.Ticker(ticker).option_chain(expiration)
    calls = ch.calls.copy()
    puts = ch.puts.copy()
    for df in (calls, puts):
        for col in ["impliedVolatility", "strike", "bid", "ask", "lastPrice", "volume"]:
            if col not in df.columns:
                df[col] = None
    return {"calls": calls, "puts": puts}

@st.cache_data(ttl=900)
def get_hist_close(ticker: str, period="60d", interval="1d") -> pd.Series:
    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        return hist["Close"].dropna() if hist is not None and not hist.empty else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

def get_dividend_yield(info: Dict[str, Any]) -> Optional[float]:
    return _to_float(info.get("dividendYield"))

def ts_to_iso(ts: Any) -> Optional[str]:
    try:
        if ts is None: return None
        if isinstance(ts, (int, float)) and ts > 0:
            return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
        s = str(ts)
        if len(s) >= 10:
            return s[:10]
        return None
    except Exception:
        return None

@st.cache_data(ttl=1800)
def get_last_earnings_date(ticker: str) -> Optional[str]:
    t = yf.Ticker(ticker)
    try:
        df = t.get_earnings_dates(limit=8)
        if isinstance(df, pd.DataFrame) and not df.empty:
            dates = pd.to_datetime(df.index)
            today = pd.Timestamp.today().normalize()
            past = dates[dates <= today]
            if len(past) > 0:
                return past.sort_values()[-1].strftime("%Y-%m-%d")
    except Exception:
        pass
    try:
        cal = t.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for col in cal.columns:
                if "Earnings" in col and "Date" in col:
                    val = cal[col].iloc[0]
                    if pd.notna(val):
                        return pd.to_datetime(val).strftime("%Y-%m-%d")
    except Exception:
        pass
    return None
