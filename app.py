# app.py
# Streamlit UI for the 31-delta put picker with enriched metrics & formatting.

from __future__ import annotations
import math
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import pandas as pd
import streamlit as st
import yfinance as yf


# --------------------------- Page config ---------------------------

st.set_page_config(
    page_title="31-Delta Put Scanner",
    page_icon="📈",
    layout="wide",
)


# --------------------------- Math / Greeks ---------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def put_delta_black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Black-Scholes-Merton delta for a European put with continuous dividend yield q.
    Delta_put = -exp(-qT) * N(-d1)
    """
    if sigma is None or sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return -math.exp(-q * T) * _norm_cdf(-d1)

def mid_price(bid: Optional[float], ask: Optional[float], last: Optional[float]) -> Optional[float]:
    try:
        if bid and ask and bid > 0 and ask > 0:
            return (float(bid) + float(ask)) / 2.0
        if last and last > 0:
            return float(last)
    except Exception:
        pass
    return None


# --------------------------- Helpers ---------------------------

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

@st.cache_data(ttl=300)
def get_risk_free_rate(default: float = 0.05) -> float:
    """Proxy risk-free from Yahoo ^IRX (13w T-bill) as a percent -> decimal."""
    try:
        tr = yf.Ticker("^IRX")
        y = tr.fast_info.last_price
        if y and y > 0:
            return float(y) / 100.0
    except Exception:
        pass
    return default

def normalize_date(d: date | str) -> str:
    if isinstance(d, date):
        return d.strftime("%Y-%m-%d")
    try:
        return pd.to_datetime(d).strftime("%Y-%m-%d")
    except Exception:
        return str(d)

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

@st.cache_data(ttl=600)
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
def get_chain_puts(ticker: str, expiration: str) -> pd.DataFrame:
    ch = yf.Ticker(ticker).option_chain(expiration)
    puts = ch.puts.copy()
    # Ensure required cols exist
    for col in ["impliedVolatility", "strike", "bid", "ask", "lastPrice", "volume"]:
        if col not in puts.columns:
            puts[col] = None
    return puts

@st.cache_data(ttl=900)
def get_hist_close(ticker: str, period="60d", interval="1d") -> pd.Series:
    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        return hist["Close"].dropna() if hist is not None and not hist.empty else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

def get_dividend_yield(info: Dict[str, Any]) -> Optional[float]:
    y = info.get("dividendYield")
    return _to_float(y)

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
    # Preferred: get_earnings_dates (newer yfinance)
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
    # Fallback: calendar
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


# --------------------------- Technicals & Fundamentals ---------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def technical_reco(ticker: str) -> str:
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


# --------------------------- Custom Optimizer ---------------------------

def compute_custom_optimizer(ps: Optional[float],
                             premium_ratio_pct: Optional[float],
                             norm_buffer_pct: Optional[float],
                             iv: Optional[float],
                             analyst_mean: Optional[float]) -> Optional[float]:
    """
    Composite 0..100 (higher is better).
    Rewards: premium_ratio %, normalized buffer %, reasonable P/S, supportive analyst mean.
    Penalizes: very high IV.
    """
    if premium_ratio_pct is None or norm_buffer_pct is None:
        return None
    pr = math.tanh((premium_ratio_pct / 10.0))         # saturate ~10%
    nb = math.tanh((max(0.0, norm_buffer_pct) / 5.0))  # saturate ~5%
    ps_term = (1.0 / (1.0 + ps)) if ps and ps > 0 else 0.5
    rec_term = 0.5
    if analyst_mean is not None and analyst_mean > 0:
        rec_term = (5.0 - analyst_mean) / 4.0  # 1..5 -> 1..0
    iv_pen = 0.0
    if iv is not None and iv > 0:
        iv_pen = max(0.0, math.tanh(max(0.0, iv - 0.60) / 0.20))
    raw = (0.35 * pr + 0.25 * nb + 0.25 * ps_term + 0.15 * rec_term - 0.10 * iv_pen)
    return round(max(0.0, min(1.0, raw)) * 100.0, 2)


# --------------------------- Core per-ticker pipeline ---------------------------

def pick_row_for_ticker(ticker: str, requested_date_iso: str, target_abs_delta: float, rf_rate: float) -> Dict[str, Any]:
    info = get_info(ticker)
    S = get_underlying_price(ticker)
    expirations = get_expirations(ticker)
    resolved_exp = closest_expiration(expirations, requested_date_iso)

    # Trailing EPS and P/E rule
    trailing_eps = _to_float(info.get("trailingEps"))
    if S is not None and trailing_eps is not None and trailing_eps > 0:
        pe_val = round(S / trailing_eps, 1)  # tenths
    else:
        pe_val = 0.0  # as requested

    # P/S (hundredths)
    ps_raw = _to_float(info.get("priceToSalesTrailing12Months"))
    ps_val = round(ps_raw, 2) if ps_raw is not None else None

    # EPS (TTM) rounded (for display)
    eps_ttm = round(trailing_eps, 2) if trailing_eps is not None else None

    # Dividends
    div_yield = get_dividend_yield(info)
    ex_div = ts_to_iso(info.get("exDividendDate"))
    pay_div = ts_to_iso(info.get("dividendDate"))

    # Analysts
    analyst_key = info.get("recommendationKey")
    analyst_mean = _to_float(info.get("recommendationMean"))

    # Last earnings date
    last_earn = get_last_earnings_date(ticker)

    # Time to expiry (years)
    if resolved_exp:
        try:
            T = max((pd.to_datetime(resolved_exp) - pd.Timestamp.today()).days / 365.0, 1e-6)
        except Exception:
            T = 7/365.0
    else:
        T = 7/365.0

    # Option chain selection
    strike = premium = delta = iv = opt_vol = None
    if resolved_exp and S:
        try:
            puts = get_chain_puts(ticker, resolved_exp)
            # Fallback HV if IV missing
            hv = None
            clos = get_hist_close(ticker, "60d", "1d")
            if not clos.empty:
                hv = float(clos.pct_change().dropna().std() * (252 ** 0.5))
            deltas = []
            for _, row in puts.iterrows():
                K = _to_float(row.get("strike")) or 0.0
                iv_i = _to_float(row.get("impliedVolatility"))
                if not iv_i or iv_i <= 0:
                    iv_i = hv if (hv and hv > 0) else 0.5
                dlt = put_delta_black_scholes(S, K, T, rf_rate, iv_i, _to_float(info.get("dividendYield")) or 0.0)
                deltas.append(dlt)
            puts = puts.assign(delta=pd.Series(deltas).values)

            subset = puts[puts["delta"].abs() <= target_abs_delta].copy()
            if subset.empty:
                puts["abs_delta"] = puts["delta"].abs()
                cand = puts.sort_values(["abs_delta", "strike"], ascending=[True, True]).iloc[0]
            else:
                subset["abs_delta"] = subset["delta"].abs()
                cand = subset.sort_values(["abs_delta", "strike"], ascending=[False, True]).iloc[0]

            bid = _to_float(cand.get("bid"))
            ask = _to_float(cand.get("ask"))
            last = _to_float(cand.get("lastPrice"))
            premium = mid_price(bid, ask, last)
            strike = _to_float(cand.get("strike"))
            delta = _to_float(cand.get("delta"))
            iv = _to_float(cand.get("impliedVolatility"))
            opt_vol = _to_int(cand.get("volume"))
        except Exception:
            pass

    # ---- Your rounding/format rules ----
    # premium_mid -> hundredths
    premium_mid = round(premium, 2) if premium is not None else None

    # Premium Ratio = (premium/strike)*100 -> hundredths
    premium_ratio_pct = None
    if premium is not None and strike:
        premium_ratio_pct = round((premium / strike) * 100.0, 2)

    # Buffer Distance = S + premium - strike -> tenths
    buffer_distance = None
    if S is not None and premium is not None and strike is not None:
        buffer_distance = round(S + premium - strike, 1)

    # Normalized Buffer Distance = buffer/strike * 100 -> hundredths
    normalized_buffer_pct = None
    if buffer_distance is not None and strike:
        normalized_buffer_pct = round((buffer_distance / strike) * 100.0, 2)

    # delta -> positive, thousandths
    delta_pos = round(abs(delta), 3) if delta is not None else None

    # impliedVolatility (decimal) -> thousandths
    iv_round = round(iv, 3) if iv is not None else None

    # Rec. Fundamental / Technical
    rec_fund = fundamental_reco(info, S)
    rec_tech = technical_reco(ticker)

    # Custom optimizer
    custom_opt = compute_custom_optimizer(
        ps=ps_raw,
        premium_ratio_pct=premium_ratio_pct,
        norm_buffer_pct=normalized_buffer_pct,
        iv=iv,
        analyst_mean=analyst_mean
    )

    return {
        "ticker": ticker,
        "underlying_price": round(S, 2) if S is not None else None,
        "P/E": pe_val,
        "P/S": ps_val,
        "strike": round(strike, 2) if strike is not None else None,
        "premium_mid": premium_mid,
        "Premium Ratio": premium_ratio_pct,
        "delta": delta_pos,
        "option_volume": opt_vol,
        "Buffer Distance": buffer_distance,
        "Normalized Buffer Distance": normalized_buffer_pct,
        "impliedVolatility": iv_round,
        "custom-optimizer": custom_opt,
        "EPS (TTM)": eps_ttm,
        "last earnings date": last_earn,
        "Recommendation (buy/hold/sell)": (str(analyst_key) if analyst_key else None),
        "Rec. Fundamental": rec_fund,
        "Rec. Technical": rec_tech,
        "dividend_yield": round(div_yield, 4) if div_yield is not None else None,
        "Dividend ex date": ts_to_iso(info.get("exDividendDate")),
        "dividend pay out date": ts_to_iso(info.get("dividendDate")),
    }


# --------------------------- UI ---------------------------

st.title("📈 31-Delta Put Scanner (Streamlit)")
st.caption("Upload tickers or paste them below, pick a date, and scan. Free data via yfinance; quotes may be delayed.")

with st.sidebar:
    st.header("Settings")
    target_date = st.date_input("Target expiration date", value=date.today())
    target_delta = st.slider("Target |Δ| (put)", min_value=0.15, max_value=0.50, value=0.31, step=0.01)
    run_btn = st.button("Run Scan 🚀", type="primary")
    st.divider()
    st.markdown("**Input tickers**")
    file = st.file_uploader("Upload CSV (tickers comma/space/line separated)", type=["csv"], accept_multiple_files=False)
    default_list = "TSLA, CVS, PLTR, GOOG"
    pasted = st.text_area("…or paste tickers", value=default_list, height=100)

def parse_tickers_from_text(text: str) -> List[str]:
    toks: List[str] = []
    for piece in text.replace(";", ",").replace("\n", ",").split(","):
        for q in piece.strip().split():
            if q.strip():
                toks.append(q.strip().upper())
    # de-duplicate preserving order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def parse_uploaded_csv(file) -> List[str]:
    content = file.read().decode("utf-8", errors="ignore")
    return parse_tickers_from_text(content)

# Collect tickers
tickers: List[str] = []
if file is not None:
    tickers = parse_uploaded_csv(file)
else:
    tickers = parse_tickers_from_text(pasted)

requested_date_iso = normalize_date(target_date)

# --------------------------- Run ---------------------------

if run_btn:
    if not tickers:
        st.error("No tickers provided.")
        st.stop()

    st.info(f"Scanning **{len(tickers)}** tickers for requested date **{requested_date_iso}** with target |Δ| ≤ **{target_delta:.2f}**.")
    rf = get_risk_free_rate()
    progress = st.progress(0)
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for i, tk in enumerate(tickers, 1):
        try:
            row = pick_row_for_ticker(tk, requested_date_iso, target_delta, rf)
            rows.append(row)
        except Exception as e:
            errors.append(f"{tk}: {e}")
        progress.progress(i / len(tickers))

    if errors:
        with st.expander("⚠️ Warnings / Errors"):
            for msg in errors:
                st.write("- " + msg)

    if not rows:
        st.warning("No results.")
        st.stop()

    df = pd.DataFrame(rows)

    # Display table
    st.subheader("Results")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download CSV (output-{requested_date_iso}.csv)",
        data=csv_bytes,
        file_name=f"output-{requested_date_iso}.csv",
        mime="text/csv",
        type="primary"
    )

    st.success("Done ✔️")


# --------------------------- Footer ---------------------------

st.caption("Not investment advice. Data from Yahoo via yfinance; may be delayed / incomplete.")
