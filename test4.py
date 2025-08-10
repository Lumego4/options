#!/usr/bin/env python3
"""
Batch 31-delta put picker with fundamentals/technicals/enrichment.

Input:
  - input.csv  -> list of tickers (comma/space/line separated)

Usage:
  python batch_31_delta_puts_plus.py 2025-08-16
  python batch_31_delta_puts_plus.py "Aug 16 2025"

Output:
  - output-[requested-date].csv  -> row per ticker with requested columns

Dependencies:
  pip install yfinance pandas
"""

from __future__ import annotations
import sys
import csv
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
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


# --------------------------- CSV I/O ---------------------------

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

    # de-duplicate preserving order
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


# --------------------------- yfinance helpers ---------------------------

def normalize_date(date_like: str) -> str:
    try:
        dt = pd.to_datetime(date_like, utc=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        print("ERROR: Could not parse the date. Try formats like 2025-08-16 or Aug 16 2025.")
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
        # yfinance returns epoch seconds for exDividendDate/dividendDate
        if isinstance(ts, (int, float)) and ts > 0:
            return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
        # Sometimes it's already a string
        s = str(ts)
        if len(s) >= 10:
            return s[:10]
        return None
    except Exception:
        return None

def get_next_earnings_date(t: yf.Ticker) -> Optional[str]:
    """Best-effort: try to get a future earnings date; else add 3 months to the most recent one."""
    try:
        # Preferred: get_earnings_dates (yfinance >= 0.2.x)
        df = t.get_earnings_dates(limit=8)
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Index often holds the dates
            # Pick the first future date; else last past date + ~3 months
            today = pd.Timestamp.today().normalize()
            # Index may be DatetimeIndex; if not, try to parse
            idx = df.index
            dates = pd.to_datetime(idx)
            future = dates[dates >= today]
            if len(future) > 0:
                return future[0].strftime("%Y-%m-%d")
            # No future date; use last date + 3 months
            last = dates.sort_values()[-1]
            return (last + pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    except Exception:
        pass

    # Fallback: t.calendar (older yfinance)
    try:
        cal = t.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            # Try next earnings date directly
            for col in cal.columns:
                if "Earnings" in col and "Date" in col:
                    val = cal[col].iloc[0]
                    if pd.notna(val):
                        dt = pd.to_datetime(val)
                        if dt:
                            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    # Last resort: None
    return None

def get_next_quarter_eps_estimate(t: yf.Ticker) -> Optional[float]:
    """
    Best-effort for 'next quarter EPS estimate'.
    We try multiple yfinance endpoints and fall back to None if unavailable.
    """
    # Try analysis table (older yfinance)
    try:
        analysis = t.analysis
        if isinstance(analysis, pd.DataFrame) and not analysis.empty:
            # Common layout: rows like 'Earnings Estimate' with columns 'Current Qtr', 'Next Qtr'
            if "Earnings Estimate" in analysis.index:
                row = analysis.loc["Earnings Estimate"]
                # Prefer 'Next Qtr' avg estimate if available
                for col in ["Next Qtr", "NextQuarter", "Quarter(+1)", "1Q Ahead", "Next Qtr."]:
                    if col in row.index:
                        cell = row[col]
                        # When row is a Series of (Avg, Low, High, etc.), try 'Avg'
                        if isinstance(cell, pd.Series) and "Avg" in cell.index:
                            v = cell["Avg"]
                            return float(v) if pd.notna(v) else None
                        # Sometimes 'Avg' is directly in the multiindex columns
                # Some versions keep columns as MultiIndex
                try:
                    return float(analysis.loc[("Earnings Estimate", "Avg"), "Next Qtr"])
                except Exception:
                    pass
    except Exception:
        pass

    # Try earnings trend (newer yfinance)
    try:
        trend = t.get_earnings_trend()
        if isinstance(trend, pd.DataFrame) and not trend.empty:
            # Look for a period that clearly references next quarter
            # Typical periods: '0q', '+1q', etc. We'll try a few options.
            for key in ["+1q", "nextQ", "1q", "0q"]:
                sel = trend[trend.index.astype(str).str.contains(key, case=False, regex=False)]
                if not sel.empty:
                    # Many builds expose 'epsTrend' or 'epsEstimate'
                    for col in ["epsEstimate", "eps_estimate", "epsAvg", "eps_avg"]:
                        if col in sel.columns and pd.notna(sel[col].iloc[0]):
                            return float(sel[col].iloc[0])
    except Exception:
        pass

    # If everything fails:
    return None


# --------------------------- Technicals ---------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def technical_summary_and_reco(t: yf.Ticker) -> (str, Dict[str, float]):
    """
    Compute a lightweight technical view: 20/50/200 DMA, RSI(14), volume trend.
    Return recommendation ('buy'/'hold'/'sell') and metrics.
    """
    try:
        hist = t.history(period="1y", interval="1d")
        if hist is None or hist.empty:
            return "hold", {}
        close = hist["Close"].dropna()
        vol = hist["Volume"].dropna() if "Volume" in hist else pd.Series(dtype=float)
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        rsi14 = rsi(close, 14)
        curr = close.iloc[-1]
        s20, s50, s200 = float(sma20.iloc[-1]), float(sma50.iloc[-1]), float(sma200.iloc[-1])
        r = float(rsi14.iloc[-1]) if not math.isnan(rsi14.iloc[-1]) else None

        vol20 = float(vol.rolling(20).mean().iloc[-1]) if not vol.empty else None
        vol5 = float(vol.rolling(5).mean().iloc[-1]) if not vol.empty else None
        vol_up = (vol5 is not None and vol20 is not None and vol5 > 1.1 * vol20)

        # Scoring
        score = 0.0
        if curr > s50 > s200: score += 0.5
        elif s50 > s200: score += 0.3
        if r is not None:
            if 40 <= r <= 60: score += 0.2
            elif r < 35: score += 0.3
            elif r > 70: score -= 0.3
        dist50 = abs(curr - s50) / s50 if s50 else 0.0
        if dist50 <= 0.03: score += 0.1
        if vol_up: score += 0.1

        reco = "buy" if score >= 0.7 else ("hold" if score >= 0.45 else "sell")
        metrics = {
            "price": float(curr),
            "sma20": s20, "sma50": s50, "sma200": s200,
            "rsi14": r if r is not None else float("nan"),
            "vol5_gt_20": 1.0 if vol_up else 0.0,
        }
        return reco, metrics
    except Exception:
        return "hold", {}


# --------------------------- Fundamentals-based reco ---------------------------

def fundamental_reco(info: Dict[str, Any]) -> str:
    """
    Heuristic scoring using valuation + growth + profitability.
    Returns 'buy' / 'hold' / 'sell'.
    """
    pe = _to_float(info.get("trailingPE"))
    ps = _to_float(info.get("priceToSalesTrailing12Months"))
    rev_g = _to_float(info.get("revenueGrowth"))           # YoY %
    earn_g = _to_float(info.get("earningsGrowth")) or _to_float(info.get("earningsQuarterlyGrowth"))
    margin = _to_float(info.get("profitMargins"))

    score = 0.0

    # P/S (lower better) -> map via 1/(1+ps)
    if ps is not None and ps > 0:
        score += min(1.0, 1.0 / (1.0 + ps)) * 0.4
    else:
        score += 0.2 * 0.4  # neutral

    # P/E sweet spot ~10-25, penalty if negative or very high
    if pe is not None and pe > 0:
        val = 1.0 - math.tanh(max(0.0, (pe - 25.0)) / 40.0)  # ~1 near 25, decays if >>25
        score += max(0.0, val) * 0.2
    else:
        score += 0.05  # small credit if NA/negative

    # Growth
    if rev_g is not None:
        score += max(0.0, math.tanh(rev_g / 0.15)) * 0.2
    if earn_g is not None:
        score += max(0.0, math.tanh(earn_g / 0.2)) * 0.1

    # Profitability
    if margin is not None:
        score += max(0.0, math.tanh(margin / 0.15)) * 0.1

    return "buy" if score >= 0.65 else ("hold" if score >= 0.45 else "sell")


# --------------------------- Utility ---------------------------

def _to_float(x) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, float) and math.isnan(x): return None
        return float(x)
    except Exception:
        return None

def _to_int(x) -> Optional[int]:
    try:
        if x is None: return None
        if isinstance(x, float) and math.isnan(x): return None
        return int(x)
    except Exception:
        return None


# --------------------------- Core per-ticker logic ---------------------------

@dataclass
class RowOut:
    ticker: str
    underlying_price: Optional[float]
    pe: Optional[float]
    ps: Optional[float]
    strike: Optional[float]
    premium_mid: Optional[float]
    premium_ratio: Optional[float]
    delta: Optional[float]
    option_volume: Optional[int]
    buffer_distance: Optional[float]
    normalized_buffer_distance: Optional[float]
    impliedVolatility: Optional[float]
    custom_optimizer: Optional[float]
    eps_ttm: Optional[float]
    next_quarter_eps_estimate: Optional[float]
    estimated_next_earnings_date: Optional[str]
    analyst_recommendation: Optional[str]
    rec_fundamental: Optional[str]
    rec_technical: Optional[str]
    dividend_yield: Optional[float]
    dividend_ex_date: Optional[str]
    dividend_pay_out_date: Optional[str]

def compute_custom_optimizer(ps: Optional[float],
                             premium_ratio: Optional[float],
                             norm_buffer: Optional[float],
                             iv: Optional[float],
                             days_to_earnings: Optional[int],
                             analyst_mean: Optional[float]) -> Optional[float]:
    """
    Composite 0..100 score (higher is better):
      + premium_ratio (saturating)
      + normalized buffer distance (saturating)
      + valuation via 1/(1+P/S)
      + analyst mean (1=StrongBuy..5=Sell) -> invert
      - small penalty for very high IV
      - penalty if earnings very near (<7 days)
    """
    if premium_ratio is None or norm_buffer is None:
        return None

    # Saturating transforms (tanh) around typical scales:
    pr = math.tanh(premium_ratio / 0.10)        # 0.10 ~ 10% of strike
    nb = math.tanh(max(0.0, norm_buffer) / 0.05)  # 5% buffer saturates
    ps_term = (1.0 / (1.0 + ps)) if ps and ps > 0 else 0.5

    rec_term = None
    if analyst_mean is not None and analyst_mean > 0:
        # 1=Strong Buy, 5=Sell -> map to 0..1 (higher better)
        rec_term = (5.0 - analyst_mean) / 4.0
    else:
        rec_term = 0.5

    iv_pen = 0.0
    if iv is not None and iv > 0:
        iv_pen = max(0.0, math.tanh(max(0.0, iv - 0.60) / 0.20))  # soft penalty above ~60% IV

    earn_pen = 0.0
    if days_to_earnings is not None:
        if days_to_earnings <= 3: earn_pen = 0.6
        elif days_to_earnings <= 7: earn_pen = 0.3
        elif days_to_earnings <= 14: earn_pen = 0.15

    raw = (0.35 * pr + 0.25 * nb + 0.20 * ps_term + 0.15 * rec_term
           - 0.05 * iv_pen - earn_pen)
    score = max(0.0, min(1.0, raw)) * 100.0
    return round(score, 2)

def pick_31_delta_put_for_ticker(ticker: str,
                                 requested_date_iso: str,
                                 rf_rate: float,
                                 target_abs_delta: float = 0.31) -> RowOut:
    t = yf.Ticker(ticker)
    info = safe_info(t)

    # Underlying and expiration
    S = get_underlying_price(t)
    resolved_exp = closest_expiration_for(t, requested_date_iso)

    # Dividend info
    div_yield = get_dividend_yield(info)
    ex_div = ts_to_iso(info.get("exDividendDate"))
    pay_div = ts_to_iso(info.get("dividendDate"))

    # Fundamentals
    pe = _to_float(info.get("trailingPE"))
    ps = _to_float(info.get("priceToSalesTrailing12Months"))
    eps_ttm = _to_float(info.get("trailingEps"))

    # Analyst recommendation (buy/hold/sell string + mean)
    analyst_key = info.get("recommendationKey")  # 'buy', 'hold', 'sell', etc.
    analyst_mean = _to_float(info.get("recommendationMean"))

    # Next quarter EPS estimate (best effort)
    next_q_eps = get_next_quarter_eps_estimate(t)

    # Earnings date (estimate if needed)
    next_earn = get_next_earnings_date(t)
    days_to_earnings = None
    try:
        if next_earn:
            dte = (pd.to_datetime(next_earn) - pd.Timestamp.today().normalize()).days
            days_to_earnings = int(dte)
    except Exception:
        pass

    # Time to expiry
    if resolved_exp:
        try:
            T = max((pd.to_datetime(resolved_exp) - pd.Timestamp.today()).days / 365.0, 1e-6)
        except Exception:
            T = 7.0/365.0  # small fallback
    else:
        T = 7.0/365.0

    # Option chain and 31-delta put
    strike = premium = delta = iv = opt_vol = None
    try:
        if resolved_exp is None:
            raise RuntimeError("No expirations available.")
        chain = t.option_chain(resolved_exp)
        puts = chain.puts.copy()
        # Ensure required columns
        for col in ["impliedVolatility", "strike", "bid", "ask", "lastPrice", "volume", "contractSymbol"]:
            if col not in puts.columns:
                puts[col] = None

        # Single fallback HV for this ticker if IV missing
        hv = None
        try:
            hist = t.history(period="60d", interval="1d")["Close"].pct_change().dropna()
            hv = float(hist.std() * (252 ** 0.5)) if not hist.empty else None
        except Exception:
            hv = None

        # Compute deltas
        deltas = []
        for _, row in puts.iterrows():
            K = _to_float(row.get("strike"))
            iv_i = _to_float(row.get("impliedVolatility")) or (hv if hv and hv > 0 else 0.5)
            dlt = put_delta_black_scholes(S or 0.0, K or 0.0, T, rf_rate, iv_i, _to_float(info.get("dividendYield")) or 0.0)
            deltas.append(dlt)
        puts = puts.assign(delta=pd.Series(deltas).values)

        # Select |delta| <= 0.31 closest to 0.31; else smallest |delta|
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

    # Custom metrics
    premium_ratio = (premium / strike) if (premium and strike and strike > 0) else None
    buffer_distance = (S + premium - strike) if (S is not None and premium is not None and strike is not None) else None
    normalized_buffer = (buffer_distance / strike) if (buffer_distance is not None and strike) else None

    # Technical & Fundamental recommendations
    rec_fund = fundamental_reco(info) if info else "hold"
    rec_tech, _metrics = technical_summary_and_reco(t)

    # Custom optimizer score
    cust = compute_custom_optimizer(ps, premium_ratio, normalized_buffer, iv, days_to_earnings, analyst_mean)

    return RowOut(
        ticker=ticker,
        underlying_price=S,
        pe=pe,
        ps=ps,
        strike=strike,
        premium_mid=premium,
        premium_ratio=premium_ratio,
        delta=delta,
        option_volume=opt_vol,
        buffer_distance=buffer_distance,
        normalized_buffer_distance=normalized_buffer,
        impliedVolatility=iv,
        custom_optimizer=cust,
        eps_ttm=eps_ttm,
        next_quarter_eps_estimate=next_q_eps,
        estimated_next_earnings_date=next_earn,
        analyst_recommendation=(str(analyst_key) if analyst_key else None),
        rec_fundamental=rec_fund,
        rec_technical=rec_tech,
        dividend_yield=div_yield,
        dividend_ex_date=ex_div,
        dividend_pay_out_date=pay_div,
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
            r = pick_31_delta_put_for_ticker(tk, requested_date_iso, rf_rate)
            rows.append(asdict(r))
            print(f"  -> {tk} done.")
        except Exception as e:
            print(f"  -> {tk} ERROR: {e}")
            rows.append(asdict(RowOut(
                ticker=tk,
                underlying_price=None, pe=None, ps=None, strike=None, premium_mid=None,
                premium_ratio=None, delta=None, option_volume=None, buffer_distance=None,
                normalized_buffer_distance=None, impliedVolatility=None, custom_optimizer=None,
                eps_ttm=None, next_quarter_eps_estimate=None, estimated_next_earnings_date=None,
                analyst_recommendation=None, rec_fundamental=None, rec_technical=None,
                dividend_yield=None, dividend_ex_date=None, dividend_pay_out_date=None
            )))

    df = pd.DataFrame(rows)

    # Order & rename columns exactly as requested
    col_map = {
        "ticker": "ticker",
        "underlying_price": "underlying_price",
        "pe": "P/E",
        "ps": "P/S",
        "strike": "strike",
        "premium_mid": "premium_mid",
        "premium_ratio": "Premium Ratio",
        "delta": "delta",
        "option_volume": "option_volume",
        "buffer_distance": "Buffer Distance",
        "normalized_buffer_distance": "Normalized Buffer Distance",
        "impliedVolatility": "impliedVolatility",
        "custom_optimizer": "custom-optimizer",
        "eps_ttm": "EPS (TTM)",
        "next_quarter_eps_estimate": "next quarter EPS estimate",
        "estimated_next_earnings_date": "estimated next earnings date",
        "analyst_recommendation": "Recommendation (buy/hold/sell)",
        "rec_fundamental": "Rec. Fundamental",
        "rec_technical": "Rec. Technical",
        "dividend_yield": "dividend_yield",
        "dividend_ex_date": "Dividend ex date",
        "dividend_pay_out_date": "dividend pay out date",
    }

    # ensure all columns exist
    for k in col_map.keys():
        if k not in df.columns:
            df[k] = None

    ordered_cols = [col_map[k] for k in col_map.keys()]
    df = df[list(col_map.keys())]
    df.columns = ordered_cols

    out_path = f"output-{requested_date_iso}.csv"
    df.to_csv(out_path, index=False)
    print("\n✅ Finished.")
    print(f"Saved results to: {out_path}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()
