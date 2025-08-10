#!/usr/bin/env python3
"""
Batch 31-delta put picker with enriched fundamentals/technicals + strict formatting.

Input:
  - input.csv  -> list of tickers (comma/space/line separated)

Usage:
  python batch_31_delta_puts_plus.py 2025-08-16
  python batch_31_delta_puts_plus.py "Aug 16 2025"

Output:
  - output-[requested-date].csv  -> row per ticker with the exact columns & rounding rules you specified.

Dependencies:
  pip install yfinance pandas
"""

from __future__ import annotations
import sys
import csv
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    print("This script requires: pip install yfinance pandas")
    sys.exit(1)


# --------------------------- Math / Greeks ---------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def put_delta_black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Black-Scholes-Merton delta for a European put with continuous dividend yield q.
    Delta_put = -exp(-qT) * N(-d1).
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


# --------------------------- CSV & helpers ---------------------------

def load_tickers_from_csv(path: str = "input.csv") -> List[str]:
    tickers: List[str] = []
    try:
        with open(path, newline="") as f:
            sniff = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            reader = csv.reader(f, dialect=sniff)
            for row in reader:
                parts = []
                for cell in row:
                    for p in str(cell).replace(";", ",").split(","):
                        for q in p.strip().split():
                            if q.strip():
                                parts.append(q.strip().upper())
                tickers.extend(parts)
    except Exception:
        with open(path, "r") as f:
            raw = f.read()
        for tok in raw.replace(";", ",").replace("\n", ",").split(","):
            for q in tok.strip().split():
                if q.strip():
                    tickers.append(q.strip().upper())

    seen = set()
    ordered: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    if not ordered:
        print("ERROR: No tickers found in input.csv")
        sys.exit(3)
    return ordered

def normalize_date(date_like: str) -> str:
    try:
        dt = pd.to_datetime(date_like, utc=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        print("ERROR: Could not parse the date. Try 2025-08-16 or Aug 16 2025.")
        sys.exit(2)

def closest_expiration_for(t: yf.Ticker, target_date_iso: str) -> Optional[str]:
    try:
        target = datetime.strptime(target_date_iso, "%Y-%m-%d").date()
        expirations = t.options or []
        if not expirations:
            return None
        best = None
        for s in expirations:
            try:
                d = datetime.strptime(s, "%Y-%m-%d").date()
                dist = abs((d - target).days)
                if best is None or dist < best[0]:
                    best = (dist, s)
            except Exception:
                continue
        return best[1] if best else None
    except Exception:
        return None

def safe_info(t: yf.Ticker) -> Dict[str, Any]:
    try:
        return t.info or {}
    except Exception:
        return {}

def get_underlying_price(t: yf.Ticker) -> Optional[float]:
    px = None
    try: px = t.fast_info.last_price
    except Exception: pass
    if not px or px <= 0:
        try: px = t.info.get("regularMarketPrice")
        except Exception: px = None
    if not px or px <= 0:
        try:
            hist = t.history(period="1d", interval="1m")
            if hist is not None and not hist.empty:
                px = float(hist["Close"].iloc[-1])
        except Exception:
            px = None
    return float(px) if px and px > 0 else None

def get_risk_free_rate(default: float = 0.05) -> float:
    try:
        tr = yf.Ticker("^IRX")
        y = tr.fast_info.last_price
        if y and y > 0:
            return float(y) / 100.0
    except Exception:
        pass
    return default

def get_dividend_yield(info: Dict[str, Any]) -> Optional[float]:
    y = info.get("dividendYield")
    try:
        if y is None: return None
        return float(y)
    except Exception:
        return None

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

def get_last_earnings_date(t: yf.Ticker) -> Optional[str]:
    """
    Most recent earnings date (<= today). If unavailable, None.
    """
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

    # Fallback: older 'calendar'
    try:
        cal = t.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            # Calendar may have 'Earnings Date'
            for col in cal.columns:
                if "Earnings" in col and "Date" in col:
                    val = cal[col].iloc[0]
                    if pd.notna(val):
                        dt = pd.to_datetime(val)
                        if dt:
                            return dt.strftime("%Y-%m-%d")
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

def technical_reco(t: yf.Ticker) -> str:
    """
    Lightweight rule-of-thumb technical recommendation using SMA & RSI.
    """
    try:
        hist = t.history(period="1y", interval="1d")
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
    """
    Heuristic based on P/S, growth, margins, profitability.
    """
    def _f(x):
        try:
            if x is None: return None
            if isinstance(x, float) and math.isnan(x): return None
            return float(x)
        except Exception:
            return None

    ps = _f(info.get("priceToSalesTrailing12Months"))
    rev_g = _f(info.get("revenueGrowth"))
    earn_g = _f(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))
    margin = _f(info.get("profitMargins"))
    roe = _f(info.get("returnOnEquity"))
    fcf_margin = _f(info.get("operatingCashflow"))  # may be None

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
    Composite 0..100 score (higher is better).
      Inputs:
        - ps: Price/Sales (lower better)
        - premium_ratio_pct: premium/strike * 100
        - norm_buffer_pct: (buffer/strike) * 100
        - iv: implied volatility (decimal)
        - analyst_mean: 1 (Strong Buy) .. 5 (Sell)
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
        iv_pen = max(0.0, math.tanh(max(0.0, iv - 0.60) / 0.20))  # soft penalty above ~60% IV

    raw = (0.35 * pr + 0.25 * nb + 0.25 * ps_term + 0.15 * rec_term - 0.10 * iv_pen)
    score = max(0.0, min(1.0, raw)) * 100.0
    return round(score, 2)


# --------------------------- Core per-ticker logic ---------------------------

@dataclass
class RowOut:
    ticker: str
    underlying_price: Optional[float]
    pe: Optional[float]
    ps: Optional[float]
    strike: Optional[float]
    premium_mid: Optional[float]
    premium_ratio_pct: Optional[float]
    delta_pos: Optional[float]
    option_volume: Optional[int]
    buffer_distance: Optional[float]
    normalized_buffer_pct: Optional[float]
    impliedVolatility: Optional[float]
    custom_optimizer: Optional[float]
    eps_ttm: Optional[float]
    last_earnings_date: Optional[str]
    analyst_recommendation: Optional[str]
    rec_fundamental: Optional[str]
    rec_technical: Optional[str]
    dividend_yield: Optional[float]
    dividend_ex_date: Optional[str]
    dividend_pay_out_date: Optional[str]

def pick_row_for_ticker(ticker: str,
                        requested_date_iso: str,
                        rf_rate: float,
                        target_abs_delta: float = 0.31) -> RowOut:
    t = yf.Ticker(ticker)
    info = safe_info(t)

    # Price & expiration
    S = get_underlying_price(t)
    resolved_exp = closest_expiration_for(t, requested_date_iso)

    # Fundamentals
    trailing_eps = None
    try: trailing_eps = float(info.get("trailingEps")) if info.get("trailingEps") is not None else None
    except Exception: trailing_eps = None

    # P/E: 0 if negative earnings; otherwise round to tenths
    if S is not None and trailing_eps is not None and trailing_eps > 0:
        pe_calc = S / trailing_eps
        pe_val = round(pe_calc, 1)
    else:
        pe_val = 0.0

    # P/S: hundredths place
    try:
        ps_raw = float(info.get("priceToSalesTrailing12Months")) if info.get("priceToSalesTrailing12Months") is not None else None
        ps_val = round(ps_raw, 2) if ps_raw is not None else None
    except Exception:
        ps_val = None

    # EPS (TTM): round to hundredths for readability
    eps_ttm = round(trailing_eps, 2) if (trailing_eps is not None and not math.isnan(trailing_eps)) else None

    # Dividends
    div_yield = get_dividend_yield(info)
    ex_div = ts_to_iso(info.get("exDividendDate"))
    pay_div = ts_to_iso(info.get("dividendDate"))

    # Analyst recommendation
    analyst_key = info.get("recommendationKey")  # 'buy', 'hold', 'sell', etc.
    analyst_mean = None
    try:
        am = info.get("recommendationMean")
        analyst_mean = float(am) if am is not None else None
    except Exception:
        analyst_mean = None

    # Last earnings date
    last_earn = get_last_earnings_date(t)

    # Time to expiry (years)
    if resolved_exp:
        try:
            T = max((pd.to_datetime(resolved_exp) - pd.Timestamp.today()).days / 365.0, 1e-6)
        except Exception:
            T = 7/365.0
    else:
        T = 7/365.0

    # Option chain & ~31-delta put
    strike = premium = delta = iv = opt_vol = None
    try:
        if resolved_exp is None or S is None:
            raise RuntimeError("Missing expiration or price.")
        chain = t.option_chain(resolved_exp)
        puts = chain.puts.copy()
        for col in ["impliedVolatility", "strike", "bid", "ask", "lastPrice", "volume"]:
            if col not in puts.columns:
                puts[col] = None

        # Single fallback HV
        hv = None
        try:
            hist = t.history(period="60d", interval="1d")["Close"].pct_change().dropna()
            hv = float(hist.std() * (252 ** 0.5)) if not hist.empty else None
        except Exception:
            hv = None

        deltas = []
        for _, row in puts.iterrows():
            K = float(row["strike"])
            iv_i = float(row["impliedVolatility"]) if row.get("impliedVolatility") not in (None, "") else None
            if not iv_i or iv_i <= 0:
                iv_i = hv if (hv and hv > 0) else 0.5
            dlt = put_delta_black_scholes(S, K, T, rf_rate, iv_i, float(info.get("dividendYield") or 0.0))
            deltas.append(dlt)
        puts = puts.assign(delta=pd.Series(deltas).values)

        subset = puts[puts["delta"].abs() <= target_abs_delta].copy()
        if subset.empty:
            puts["abs_delta"] = puts["delta"].abs()
            cand = puts.sort_values(["abs_delta", "strike"], ascending=[True, True]).iloc[0]
        else:
            subset["abs_delta"] = subset["delta"].abs()
            cand = subset.sort_values(["abs_delta", "strike"], ascending=[False, True]).iloc[0]

        bid = float(cand.get("bid")) if cand.get("bid") not in (None, "") else None
        ask = float(cand.get("ask")) if cand.get("ask") not in (None, "") else None
        last = float(cand.get("lastPrice")) if cand.get("lastPrice") not in (None, "") else None

        premium = mid_price(bid, ask, last)
        strike = float(cand.get("strike"))
        delta = float(cand.get("delta"))
        iv = float(cand.get("impliedVolatility")) if cand.get("impliedVolatility") not in (None, "") else None
        opt_vol = int(cand.get("volume")) if cand.get("volume") not in (None, "") else None
    except Exception:
        pass

    # ---- Custom metrics with your rounding rules ----
    # Premium Ratio = (premium_mid / strike) * 100, round to hundredths
    premium_ratio_pct = None
    if premium is not None and strike:
        premium_ratio_pct = round((premium / strike) * 100.0, 2)

    # Buffer Distance = underlying + premium - strike, round to tenths
    buffer_distance = None
    if S is not None and premium is not None and strike is not None:
        buffer_distance = round(S + premium - strike, 1)

    # Normalized Buffer Distance = (buffer / strike) * 100, round to hundredths
    normalized_buffer_pct = None
    if buffer_distance is not None and strike:
        normalized_buffer_pct = round((buffer_distance / strike) * 100.0, 2)

    # premium_mid -> hundredths
    premium_mid_rounded = round(premium, 2) if premium is not None else None

    # delta -> positive, thousandths
    delta_pos = round(abs(delta), 3) if delta is not None else None

    # impliedVolatility -> thousandths (still in decimal form)
    iv_round = round(iv, 3) if iv is not None else None

    # Rec. Fundamental / Technical
    rec_fund = fundamental_reco(info, S)
    rec_tech = technical_reco(t)

    # Custom optimizer (uses P/S (raw), premium ratio pct, normalized buffer pct, IV (decimal), analyst mean)
    custom_opt = compute_custom_optimizer(
        ps=ps_val if ps_val is not None else (float(info.get("priceToSalesTrailing12Months")) if info.get("priceToSalesTrailing12Months") else None),
        premium_ratio_pct=premium_ratio_pct,
        norm_buffer_pct=normalized_buffer_pct,
        iv=iv,
        analyst_mean=analyst_mean
    )

    return RowOut(
        ticker=ticker,
        underlying_price=round(S, 2) if S is not None else None,  # 2dp for readability
        pe=pe_val,
        ps=ps_val,
        strike=round(strike, 2) if strike is not None else None,
        premium_mid=premium_mid_rounded,
        premium_ratio_pct=premium_ratio_pct,
        delta_pos=delta_pos,
        option_volume=opt_vol,
        buffer_distance=buffer_distance,
        normalized_buffer_pct=normalized_buffer_pct,
        impliedVolatility=iv_round,
        custom_optimizer=custom_opt,
        eps_ttm=eps_ttm,
        last_earnings_date=last_earn,
        analyst_recommendation=(str(analyst_key) if analyst_key else None),
        rec_fundamental=rec_fund,
        rec_technical=rec_tech,
        dividend_yield=(round(float(div_yield), 4) if div_yield is not None else None),
        dividend_ex_date=ex_div,
        dividend_pay_out_date=pay_div
    )


# --------------------------- CLI ---------------------------

def parse_args() -> str:
    if len(sys.argv) < 2:
        print("Usage: python batch_31_delta_puts_plus.py <DATE>\n"
              "Example: python batch_31_delta_puts_plus.py 2025-08-16")
        sys.exit(2)
    return " ".join(sys.argv[1:])

def main():
    user_date = parse_args()
    requested_date_iso = normalize_date(user_date)
    tickers = load_tickers_from_csv("input.csv")
    rf_rate = get_risk_free_rate()

    rows: List[Dict[str, Any]] = []
    print(f"Processing {len(tickers)} tickers for requested date {requested_date_iso}...")
    for tk in tickers:
        try:
            r = pick_row_for_ticker(tk, requested_date_iso, rf_rate)
            rows.append(asdict(r))
            print(f"  -> {tk} done.")
        except Exception as e:
            print(f"  -> {tk} ERROR: {e}")
            rows.append(asdict(RowOut(
                ticker=tk, underlying_price=None, pe=None, ps=None, strike=None, premium_mid=None,
                premium_ratio_pct=None, delta_pos=None, option_volume=None, buffer_distance=None,
                normalized_buffer_pct=None, impliedVolatility=None, custom_optimizer=None, eps_ttm=None,
                last_earnings_date=None, analyst_recommendation=None, rec_fundamental=None, rec_technical=None,
                dividend_yield=None, dividend_ex_date=None, dividend_pay_out_date=None
            )))

    df = pd.DataFrame(rows)

    # Exact column names & order you requested:
    col_map = {
        "ticker": "ticker",
        "underlying_price": "underlying_price",
        "pe": "P/E",
        "ps": "P/S",
        "strike": "strike",
        "premium_mid": "premium_mid",
        "premium_ratio_pct": "Premium Ratio",
        "delta_pos": "delta",
        "option_volume": "option_volume",
        "buffer_distance": "Buffer Distance",
        "normalized_buffer_pct": "Normalized Buffer Distance",
        "impliedVolatility": "impliedVolatility",
        "custom_optimizer": "custom-optimizer",
        "eps_ttm": "EPS (TTM)",
        "last_earnings_date": "last earnings date",
        "analyst_recommendation": "Recommendation (buy/hold/sell)",
        "rec_fundamental": "Rec. Fundamental",
        "rec_technical": "Rec. Technical",
        "dividend_yield": "dividend_yield",
        "dividend_ex_date": "Dividend ex date",
        "dividend_pay_out_date": "dividend pay out date",
    }

    # Ensure all columns exist and order them
    for k in col_map.keys():
        if k not in df.columns:
            df[k] = None
    df = df[list(col_map.keys())]
    df.columns = [col_map[k] for k in col_map.keys()]

    out_path = f"output-{requested_date_iso}.csv"
    df.to_csv(out_path, index=False)
    print("\n✅ Finished.")
    print(f"Saved results to: {out_path}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()
