#!/usr/bin/env python3
"""
Pick the ~31-delta put for a given ticker and date using free Yahoo data (yfinance).

Examples:
    python pick_31_delta_put.py TSLA 2025-08-16
    python pick_31_delta_put.py AAPL "Aug 16 2025"
"""

from __future__ import annotations
import sys
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("This script requires: pip install yfinance pandas")
    sys.exit(1)


# --------------------------- Math / Greeks ---------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function (no scipy required)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def put_delta_black_scholes(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """
    Black-Scholes-Merton delta for a European put with continuous dividend yield q.
    Delta_put = -exp(-qT) * N(-d1).
    """
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return -math.exp(-q * T) * _norm_cdf(-d1)


def mid_price(bid: Optional[float], ask: Optional[float], last: Optional[float]) -> Optional[float]:
    """Mid from bid/ask; fallback to last."""
    try:
        if bid and ask and bid > 0 and ask > 0:
            return (float(bid) + float(ask)) / 2.0
        if last and last > 0:
            return float(last)
    except Exception:
        pass
    return None


# --------------------------- Data Models ---------------------------

@dataclass
class ChosenPut:
    contractSymbol: str
    expiration: str
    strike: float
    bid: Optional[float]
    ask: Optional[float]
    lastPrice: Optional[float]
    volume: Optional[int]
    openInterest: Optional[int]
    inTheMoney: Optional[bool]
    impliedVolatility: Optional[float]
    currency: Optional[str]
    delta: Optional[float]
    premium_mid: Optional[float]


# --------------------------- Helpers ---------------------------

def parse_args() -> Tuple[str, str]:
    if len(sys.argv) < 3:
        print("Usage: python pick_31_delta_put.py <TICKER> <DATE>\n"
              "Example: python pick_31_delta_put.py TSLA 2025-08-16")
        sys.exit(2)
    return sys.argv[1].upper(), " ".join(sys.argv[2:])  # accept spaces like "Aug 16 2025"


def to_yyyy_mm_dd(date_like: str) -> str:
    """Accept many human formats and normalize to YYYY-MM-DD (uses pandas parser)."""
    try:
        dt = pd.to_datetime(date_like, utc=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        print("ERROR: Could not parse the date. Try formats like 2025-08-16 or Aug 16 2025.")
        sys.exit(2)


def closest_expiration_for(t: yf.Ticker, target_date: str) -> str:
    """Pick the listed expiration nearest to the user's requested date."""
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    expirations = t.options or []
    if not expirations:
        raise RuntimeError("No listed options expirations for this ticker.")
    candidates = []
    for s in expirations:
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
            candidates.append((abs((d - target).days), d, s))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("Could not interpret Yahoo expirations.")
    candidates.sort()
    return candidates[0][2]


def get_underlying_price(t: yf.Ticker) -> float:
    """Best-effort real-time-ish price."""
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
        hist = t.history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            px = float(hist["Close"].iloc[-1])
    if not px or px <= 0:
        raise RuntimeError("Unable to obtain underlying price.")
    return float(px)


def get_risk_free_rate(default: float = 0.05) -> float:
    """
    Grab a simple risk-free proxy from Yahoo:
    - ^IRX (13-week T-bill) quotes a percentage (e.g., 5.20 -> 0.052)
    Fallback to `default` if unavailable.
    """
    try:
        tr = yf.Ticker("^IRX")
        y = tr.fast_info.last_price
        if y and y > 0:
            return float(y) / 100.0
    except Exception:
        pass
    return default


def get_dividend_yield_approx(t: yf.Ticker) -> float:
    """
    Lightweight approximate dividend yield (annual) as decimal.
    Many tickers (e.g., TSLA) pay 0; we safely return 0 if unknown.
    """
    try:
        info = t.info or {}
        y = info.get("dividendYield")
        if y is not None and y >= 0:
            return float(y)
    except Exception:
        pass
    return 0.0


def pick_31_delta_put(
    ticker: str, user_date: str, max_abs_delta: float = 0.31
) -> Tuple[ChosenPut, Dict[str, Any]]:
    """
    Returns the chosen put (closest to target abs(delta) from below) and a context dict.
    """
    t = yf.Ticker(ticker)
    target_date = to_yyyy_mm_dd(user_date)
    expiration = closest_expiration_for(t, target_date)

    S = get_underlying_price(t)
    r = get_risk_free_rate()
    q = get_dividend_yield_approx(t)

    # Time to expiry in years (ACT/365)
    expiry_dt = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=None)
    now = datetime.now().replace(tzinfo=None)
    T = max((expiry_dt - now).total_seconds() / (365.0 * 24 * 3600), 1e-6)

    chain = t.option_chain(expiration)
    puts = chain.puts.copy()
    if puts is None or puts.empty:
        raise RuntimeError(f"No puts found for {ticker} @ {expiration}")

    # Ensure columns exist
    for col in ["impliedVolatility", "strike", "bid", "ask", "lastPrice",
                "volume", "openInterest", "inTheMoney", "contractSymbol", "currency"]:
        if col not in puts.columns:
            puts[col] = None

    # Compute put delta for each row using its IV
    deltas = []
    for _, row in puts.iterrows():
        K = float(row["strike"])
        iv = float(row["impliedVolatility"]) if pd.notna(row["impliedVolatility"]) else float("nan")
        if not (iv and iv > 0):
            # fallback IV: use 20-day historical vol if needed
            try:
                hist = yf.Ticker(ticker).history(period="60d", interval="1d")["Close"].pct_change().dropna()
                hv = float(hist.std() * (252 ** 0.5))
            except Exception:
                hv = float("nan")
            iv = hv if hv and hv > 0 else 0.5  # final fallback 50%
        dlt = put_delta_black_scholes(S, K, T, r, iv, q)
        deltas.append(dlt)

    puts = puts.assign(delta=pd.Series(deltas).values)
    # Filter with |delta| <= 0.31 (and delta is negative for puts)
    subset = puts[puts["delta"].abs() <= max_abs_delta].copy()
    if subset.empty:
        # If nothing ≤ 0.31, take the smallest |delta| available (deep OTM)
        subset = puts.copy()
        subset["abs_delta"] = subset["delta"].abs()
        candidate = subset.sort_values(["abs_delta", "strike"], ascending=[True, True]).iloc[0]
    else:
        subset["abs_delta"] = subset["delta"].abs()
        # Choose the one with abs(delta) closest to target from below (i.e., max abs_delta)
        candidate = subset.sort_values(["abs_delta", "strike"], ascending=[False, True]).iloc[0]

    chosen = ChosenPut(
        contractSymbol=str(candidate.get("contractSymbol")),
        expiration=expiration,
        strike=float(candidate.get("strike")),
        bid=to_float_or_none(candidate.get("bid")),
        ask=to_float_or_none(candidate.get("ask")),
        lastPrice=to_float_or_none(candidate.get("lastPrice")),
        volume=to_int_or_none(candidate.get("volume")),
        openInterest=to_int_or_none(candidate.get("openInterest")),
        inTheMoney=to_bool_or_none(candidate.get("inTheMoney")),
        impliedVolatility=to_float_or_none(candidate.get("impliedVolatility")),
        currency=(candidate.get("currency") or "USD"),
        delta=to_float_or_none(candidate.get("delta")),
        premium_mid=mid_price(
            to_float_or_none(candidate.get("bid")),
            to_float_or_none(candidate.get("ask")),
            to_float_or_none(candidate.get("lastPrice")),
        ),
    )

    ctx = {
        "ticker": ticker,
        "requested_date": target_date,
        "resolved_expiration": expiration,
        "now_local": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "underlying_price": S,
        "risk_free_rate_used": r,
        "dividend_yield_used": q,
        "time_to_expiry_years": T,
        "target_abs_delta": max_abs_delta,
    }
    return chosen, ctx


# --------------------------- Type helpers ---------------------------

def to_float_or_none(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def to_int_or_none(x) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return int(x)
    except Exception:
        return None


def to_bool_or_none(x) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    # pandas may store as numpy bool or 0/1
    try:
        return bool(x)
    except Exception:
        return None


# --------------------------- CLI ---------------------------

def main():
    ticker, user_date = parse_args()
    chosen, ctx = pick_31_delta_put(ticker, user_date, max_abs_delta=0.31)

    print("=" * 80)
    print(f"{ticker}  |  Requested: {ctx['requested_date']}  ->  Resolved Expiry: {ctx['resolved_expiration']}")
    print(f"Now: {ctx['now_local']}   Underlying: ${ctx['underlying_price']:.2f}")
    print(f"Assumptions -> r: {ctx['risk_free_rate_used']:.4f}  q: {ctx['dividend_yield_used']:.4f}  "
          f"T: {ctx['time_to_expiry_years']:.6f} yr  target |Δ| ≤ {ctx['target_abs_delta']:.2f}")
    print("=" * 80)

    data = asdict(chosen)
    # Pretty print all option information we have:
    for k, v in data.items():
        if isinstance(v, float):
            if "delta" in k:
                print(f"{k:>18}: {v: .4f}")
            elif "impliedVolatility" in k:
                print(f"{k:>18}: {v:.4%}")
            elif "premium_mid" in k or "strike" in k or "Price" in k or "bid" in k or "ask" in k:
                print(f"{k:>18}: ${v:,.4f}")
            else:
                print(f"{k:>18}: {v}")
        else:
            print(f"{k:>18}: {v}")
    print("=" * 80)

    # Minimal explicit output you asked for:
    print("\nResult ➜  PUT with |delta| ≤ 0.31 (closest to 0.31):")
    print(f"  {data['contractSymbol']}  |  strike ${data['strike']:,.2f}  |  mid premium "
          f"{('$' + format(data['premium_mid'], ',.2f')) if data['premium_mid'] else 'N/A'}  |  delta {data['delta']:.4f}")


if __name__ == "__main__":
    main()
