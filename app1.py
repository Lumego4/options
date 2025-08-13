# app.py
# Streamlit UI for 31-delta PUTs or nearest-lower-strike CALLs, with Big Dave / Luke views,
# and support for group CSVs (e.g., spy-100).

from __future__ import annotations
import math
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import pandas as pd
import streamlit as st
import yfinance as yf


# --------------------------- Page config ---------------------------

st.set_page_config(
    page_title="Options Scanner (Puts & Calls)",
    page_icon="📈",
    layout="wide",
)


# --------------------------- Math / Greeks ---------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def call_delta_black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Δ_call = exp(-qT) * N(d1)"""
    if sigma is None or sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = bs_d1(S, K, T, r, sigma, q)
    return math.exp(-q * T) * _norm_cdf(d1)

def put_delta_black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Δ_put = -exp(-qT) * N(-d1)"""
    if sigma is None or sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = bs_d1(S, K, T, r, sigma, q)
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


# --------------------------- Helpers / Caching ---------------------------

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
    """Proxy risk-free from Yahoo ^IRX (13w T-bill) as percent -> decimal."""
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


# --------------------------- Selection Logic ---------------------------

def select_put_row(puts: pd.DataFrame, S: float, T: float, rf: float, q: float, target_abs_delta: float) -> Optional[pd.Series]:
    """Pick |Δ| ≤ target closest to target; else smallest |Δ|."""
    # Fallback HV if IV missing
    iv_series = puts["impliedVolatility"]
    hv = None
    if iv_series.isna().all():
        clos = get_hist_close(ticker, "60d", "1d")  # uses external 'ticker' if present in context
        if not clos.empty:
            hv = float(clos.pct_change().dropna().std() * (252 ** 0.5))
    deltas = []
    for _, row in puts.iterrows():
        K = _to_float(row.get("strike")) or 0.0
        iv_i = _to_float(row.get("impliedVolatility"))
        if not iv_i or iv_i <= 0:
            iv_i = hv if (hv and hv > 0) else 0.5
        dlt = put_delta_black_scholes(S, K, T, rf, iv_i, q)
        deltas.append(dlt)
    puts = puts.assign(delta=pd.Series(deltas).values)
    subset = puts[puts["delta"].abs() <= target_abs_delta].copy()
    if subset.empty:
        puts["abs_delta"] = puts["delta"].abs()
        return puts.sort_values(["abs_delta", "strike"], ascending=[True, True]).iloc[0]
    subset["abs_delta"] = subset["delta"].abs()
    return subset.sort_values(["abs_delta", "strike"], ascending=[False, True]).iloc[0]

def select_call_row_nearest_lower(calls: pd.DataFrame, S: float) -> Optional[pd.Series]:
    """Pick the call with strike ≤ S that is closest to S (i.e., the 'immediately below spot' strike)."""
    df = calls.dropna(subset=["strike"]).copy()
    df["strike"] = df["strike"].astype(float)
    df_le = df[df["strike"] <= S]
    if df_le.empty:
        # if no strike below spot, pick the lowest strike available
        return df.sort_values("strike").iloc[0]
    return df_le.sort_values("strike", ascending=True).iloc[-1]  # max strike ≤ S


# --------------------------- Row builder ---------------------------

def build_row(ticker: str,
              requested_date_iso: str,
              scan_type: str,           # "puts" or "calls"
              target_abs_delta: float,
              rf_rate: float) -> Dict[str, Any]:

    info = get_info(ticker)
    S = get_underlying_price(ticker)
    expirations = get_expirations(ticker)
    resolved_exp = closest_expiration(expirations, requested_date_iso)

    # Trailing EPS and P/E rule
    trailing_eps = _to_float(info.get("trailingEps"))
    if S is not None and trailing_eps is not None and trailing_eps > 0:
        pe_val = round(S / trailing_eps, 1)  # tenths
    else:
        pe_val = 0.0

    # P/S (hundredths)
    ps_raw = _to_float(info.get("priceToSalesTrailing12Months"))
    ps_val = round(ps_raw, 2) if ps_raw is not None else None

    # Market cap (billions, 2dp)
    mcap_raw = _to_float(info.get("marketCap"))
    market_cap = f"${mcap_raw/1e9:.2f}B" if mcap_raw is not None and mcap_raw > 0 else None

    # EPS (TTM) rounded for display
    eps_ttm = round(trailing_eps, 2) if trailing_eps is not None else None

    # Dividends
    div_yield = get_dividend_yield(info)
    ex_div = ts_to_iso(info.get("exDividendDate"))
    pay_div = ts_to_iso(info.get("dividendDate"))

    # Analyst
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

    strike = premium = delta = iv = opt_vol = None

    if resolved_exp and S:
        try:
            ch = get_chain(ticker, resolved_exp)
            if scan_type == "puts":
                puts = ch["puts"]
                # compute deltas using IV/HV
                # small local q (dividend yield) for BSM
                q = _to_float(info.get("dividendYield")) or 0.0
                # compute once a historical vol in case needed
                hv = None
                if puts["impliedVolatility"].isna().all():
                    clos = get_hist_close(ticker, "60d", "1d")
                    if not clos.empty:
                        hv = float(clos.pct_change().dropna().std() * (252 ** 0.5))
                deltas = []
                for _, row in puts.iterrows():
                    K = _to_float(row.get("strike")) or 0.0
                    iv_i = _to_float(row.get("impliedVolatility"))
                    if not iv_i or iv_i <= 0:
                        iv_i = hv if (hv and hv > 0) else 0.5
                    dlt = put_delta_black_scholes(S, K, T, rf_rate, iv_i, q)
                    deltas.append(dlt)
                puts = puts.assign(delta=pd.Series(deltas).values)
                subset = puts[puts["delta"].abs() <= target_abs_delta].copy()
                if subset.empty:
                    puts["abs_delta"] = puts["delta"].abs()
                    cand = puts.sort_values(["abs_delta", "strike"], ascending=[True, True]).iloc[0]
                else:
                    subset["abs_delta"] = subset["delta"].abs()
                    cand = subset.sort_values(["abs_delta", "strike"], ascending=[False, True]).iloc[0]
            else:
                calls = ch["calls"]
                # select nearest-lower strike call
                cand = select_call_row_nearest_lower(calls, S)
                # compute call delta for consistency
                q = _to_float(info.get("dividendYield")) or 0.0
                iv_tmp = _to_float(cand.get("impliedVolatility"))
                if not iv_tmp or iv_tmp <= 0:
                    clos = get_hist_close(ticker, "60d", "1d")
                    hv = float(clos.pct_change().dropna().std() * (252 ** 0.5)) if not clos.empty else 0.5
                    iv_tmp = hv if (hv and hv > 0) else 0.5
                delta = call_delta_black_scholes(S, _to_float(cand.get("strike")) or 0.0, T, rf_rate, iv_tmp, q)

            bid = _to_float(cand.get("bid"))
            ask = _to_float(cand.get("ask"))
            last = _to_float(cand.get("lastPrice"))
            premium = mid_price(bid, ask, last)
            strike = _to_float(cand.get("strike"))
            if scan_type == "puts":
                delta = _to_float(cand.get("delta"))
            iv = _to_float(cand.get("impliedVolatility"))
            opt_vol = _to_int(cand.get("volume"))
        except Exception:
            pass

    # ---- Common metrics + your rounding rules ----
    premium_mid = round(premium, 2) if premium is not None else None
    premium_ratio_pct = round((premium / strike) * 100.0, 2) if (premium is not None and strike) else None
    buffer_distance = round(S + premium - strike, 1) if (S is not None and premium is not None and strike is not None) else None
    normalized_buffer_pct = round((buffer_distance / strike) * 100.0, 2) if (buffer_distance is not None and strike) else None
    delta_pos = round(abs(delta), 3) if delta is not None else None
    iv_round = round(iv, 3) if iv is not None else None

    # Heuristic recs (unchanged)
    rec_fund = fundamental_reco(info, S)
    rec_tech = technical_reco(ticker)

    # Custom optimizer (same definition)
    custom_opt = compute_custom_optimizer(
        ps=ps_raw,
        premium_ratio_pct=premium_ratio_pct,
        norm_buffer_pct=normalized_buffer_pct,
        iv=iv,
        analyst_mean=analyst_mean
    )

    return {
        "ticker": ticker,
        "market_cap": market_cap,  # included in BOTH views
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
        "Dividend ex date": ex_div,
        "dividend pay out date": pay_div,
    }


# --------------------------- Technicals & Fundamentals (same as earlier) ---------------------------

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


# --------------------------- UI ---------------------------

st.title("📈 Options Scanner — Puts & Calls")
st.caption("Free data via yfinance; quotes may be delayed.")

with st.sidebar:
    st.header("Scan Settings")
    scan_type_label = st.radio("Scan type", ["Puts (≤31Δ)", "Calls (nearest lower strike)"])
    scan_type = "puts" if "Puts" in scan_type_label else "calls"
    target_date = st.date_input("Target expiration date", value=date.today())
    target_delta = st.slider("Target |Δ| for PUTs", min_value=0.15, max_value=0.50, value=0.31, step=0.01,
                             help="Used only for PUT selection. Calls use nearest-lower strike to spot.")
    view_mode = st.radio("View mode", ["Big Dave", "Luke"],
                         help="Big Dave includes dividend info; Luke hides it. Both include Market Cap.")
    st.divider()
    st.markdown("**Ticker source**")
    ticker_source = st.radio("Choose source", ["Custom list", "Group CSV (e.g., spy-100)"])
    custom_file = st.file_uploader("Upload CSV for custom tickers", type=["csv"], accept_multiple_files=False)
    default_list = "TSLA, CVS, PLTR, GOOG"
    pasted = st.text_area("…or paste tickers", value=default_list, height=100,
                          help="Comma/space/line separated")
    group_file = st.file_uploader("Upload group CSV (e.g., spy-100.csv)", type=["csv"], accept_multiple_files=False,
                                  help="Used when 'Group CSV' is selected above")
    run_btn = st.button("Run Scan 🚀", type="primary")

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

# Decide tickers
tickers: List[str] = []
if ticker_source == "Group CSV (e.g., spy-100)":
    if group_file is not None:
        tickers = parse_uploaded_csv(group_file)
else:
    if custom_file is not None:
        tickers = parse_uploaded_csv(custom_file)
    else:
        tickers = parse_tickers_from_text(pasted)

requested_date_iso = normalize_date(target_date)


# --------------------------- Run ---------------------------

if run_btn:
    if not tickers:
        st.error("No tickers provided for the selected source.")
        st.stop()

    st.info(f"Scanning **{len(tickers)}** tickers for date **{requested_date_iso}** "
            f"using **{scan_type_label}**. View: **{view_mode}**.")
    rf = get_risk_free_rate()
    progress = st.progress(0)
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for i, tk in enumerate(tickers, 1):
        try:
            row = build_row(
                ticker=tk,
                requested_date_iso=requested_date_iso,
                scan_type=scan_type,
                target_abs_delta=target_delta,
                rf_rate=rf
            )
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

    df_full = pd.DataFrame(rows)

    # View-specific columns
    base_cols = [
        "ticker", "market_cap", "underlying_price", "P/E", "P/S",
        "strike", "premium_mid", "Premium Ratio", "delta", "option_volume",
        "Buffer Distance", "Normalized Buffer Distance", "impliedVolatility",
        "custom-optimizer", "EPS (TTM)", "last earnings date",
        "Recommendation (buy/hold/sell)", "Rec. Fundamental", "Rec. Technical",
    ]
    div_cols = ["dividend_yield", "Dividend ex date", "dividend pay out date"]

    if view_mode == "Big Dave":
        view_cols = base_cols + div_cols
    else:
        view_cols = base_cols

    # Ensure columns exist
    for c in view_cols:
        if c not in df_full.columns:
            df_full[c] = None

    display_df = df_full[view_cols]

    # Display table
    st.subheader("Results")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download CSV (output-{requested_date_iso}-{scan_type}.csv)",
        data=csv_bytes,
        file_name=f"output-{requested_date_iso}-{scan_type}.csv",
        mime="text/csv",
        type="primary"
    )

    st.success("Done ✔️")


# --------------------------- Footer ---------------------------

st.caption("Not investment advice. Data from Yahoo via yfinance; may be delayed / incomplete.")
