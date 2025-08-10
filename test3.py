#!/usr/bin/env python3
"""
Batch 31-delta put picker from a CSV list of tickers.

- Reads tickers from input.csv (robust to comma/space/line separated)
- User supplies a target date (any common format; "2025-08-16" is fine)
- For each ticker:
    * resolves the closest listed expiration to the target date
    * computes put delta (Black-Scholes) using each row's IV (fallbacks included)
    * selects the put with |delta| <= 0.31 that's closest to 0.31; if none, takes smallest |delta|
    * records full option info + context fields
- Writes results to: output-[requested-date].csv

Usage:
    python batch_31_delta_puts.py 2025-08-16
    python batch_31_delta_puts.py "Aug 16 2025"

Dependencies:
    pip install yfinance pandas
"""

from __future__ import annotations
import sys
import math
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("This script requires: pip install yfinance pandas")
    sys.exit(1)


# --------------------------- Math / Greeks ---------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def put_delta_black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
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
    ticker: str
    contractSymbol: Optional[str]
    resolved_expiration: Optional[str]
    strike: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    lastPrice: Optional[float]
    premium_mid: Optional[float]
    volume: Optional[int]
    openInterest: Optional[int]
    inTheMoney: Optional[bool]
    impliedVolatility: Optional[float]
    currency: Optional[str]
    delta: Optional[float]
    underlying_price: Optional[float]
    risk_free_rate_used: Optional[float]
    dividend_yield_used: Optional[float]
    time_to_expiry_years: Optional[float]
    now_local: Optional[str]
    requested_date: Optional[str]
    error: Optional[str] = None


# --------------------------- Helpers ---------------------------

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
    try:
        return bool(x)
    except Exception:
        return None


def normalize_date(date_like: str) -> str:
    """Accept many human formats and normalize to YYYY-MM-DD (uses pandas parser)."""
    try:
        dt = pd.to_datetime(date_like, utc=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        print("ERROR: Could not parse the date. Try formats like 2025-08-16 or Aug 16 2025.")
        sys.exit(2)


def load_tickers_from_csv(path: str = "input.csv") -> List[str]:
    """
    Robustly load tickers from input.csv.
    Accepts:
      - single column (one ticker per row)
      - comma/space separated on one or multiple lines
    """
    tickers: List[str] = []
    try:
        with open(path, newline="") as f:
            sniff = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            reader = csv.reader(f, dialect=sniff)
            for row in reader:
                parts = []
                for cell in row:
                    # split by comma/space and clean
                    for p in str(cell).replace(";", ",").split(","):
                        for q in p.strip().split():
                            if q.strip():
                                parts.append(q.strip().upper())
                tickers.extend(parts)
    except Exception:
        # Fallback: treat as plain text, split by commas/whitespace
        with open(path, "r") as f:
            raw = f.read()
        for tok in raw.replace(";", ",").replace("\n", ",").split(","):
            for q in tok.strip().split():
                if q.strip():
                    tickers.append(q.strip().upper())

    # de-duplicate while preserving order
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


def closest_expiration_for(t: yf.Ticker, target_date: str) -> str:
    """Pick the listed expiration nearest to the user's requested date."""
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    expirations = t.options or []
    if not expirations:
        raise RuntimeError("No listed options expirations for this ticker.")
    best = None
    for s in expirations:
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
            dist = abs((d - target).days)
            if best is None or dist < best[0]:
                best = (dist, s)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Could not interpret Yahoo expirations.")
    return best[1]


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
    Proxy risk-free from Yahoo: ^IRX (13-week T-bill), quoted as percent (e.g., 5.20 -> 0.052).
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
    """Approximate dividend yield (annual) as decimal; 0 if unknown."""
    try:
        info = t.info or {}
        y = info.get("dividendYield")
        if y is not None and y >= 0:
            return float(y)
    except Exception:
        pass
    return 0.0


def pick_31_delta_put_for_ticker(
    ticker: str, requested_date_iso: str, max_abs_delta: float, rf_rate: float
) -> ChosenPut:
    """Compute the chosen ~31-delta put for a single ticker and return all fields."""
    now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    try:
        t = yf.Ticker(ticker)
        resolved_exp = closest_expiration_for(t, requested_date_iso)
        S = get_underlying_price(t)
        q = get_dividend_yield_approx(t)

        # Time to expiry (ACT/365)
        expiry_dt = datetime.strptime(resolved_exp, "%Y-%m-%d")
        now_naive = datetime.now()
        T = max((expiry_dt - now_naive).total_seconds() / (365.0 * 24 * 3600), 1e-6)

        chain = t.option_chain(resolved_exp)
        puts = chain.puts.copy()
        if puts is None or puts.empty:
            raise RuntimeError("No puts returned for this expiration.")

        # Ensure columns exist
        for col in [
            "impliedVolatility", "strike", "bid", "ask", "lastPrice", "volume",
            "openInterest", "inTheMoney", "contractSymbol", "currency"
        ]:
            if col not in puts.columns:
                puts[col] = None

        # Pre-compute fallback HV once per ticker if needed
        hv = None
        try:
            hist = t.history(period="60d", interval="1d")["Close"].pct_change().dropna()
            hv = float(hist.std() * (252 ** 0.5)) if not hist.empty else None
        except Exception:
            hv = None

        deltas = []
        for _, row in puts.iterrows():
            K = float(row["strike"])
            iv = to_float_or_none(row.get("impliedVolatility"))
            if not (iv and iv > 0):
                iv = hv if (hv and hv > 0) else 0.5  # last resort 50%
            dlt = put_delta_black_scholes(S, K, T, rf_rate, iv, q)
            deltas.append(dlt)
        puts = puts.assign(delta=pd.Series(deltas).values)

        # Filter with |delta| <= max_abs_delta
        subset = puts[puts["delta"].abs() <= max_abs_delta].copy()
        if subset.empty:
            # If nothing ≤ target, choose smallest |delta| (deepest OTM)
            puts["abs_delta"] = puts["delta"].abs()
            candidate = puts.sort_values(["abs_delta", "strike"], ascending=[True, True]).iloc[0]
        else:
            subset["abs_delta"] = subset["delta"].abs()
            candidate = subset.sort_values(["abs_delta", "strike"], ascending=[False, True]).iloc[0]

        bid = to_float_or_none(candidate.get("bid"))
        ask = to_float_or_none(candidate.get("ask"))
        last = to_float_or_none(candidate.get("lastPrice"))
        premium = mid_price(bid, ask, last)

        return ChosenPut(
            ticker=ticker,
            contractSymbol=str(candidate.get("contractSymbol")),
            resolved_expiration=resolved_exp,
            strike=to_float_or_none(candidate.get("strike")),
            bid=bid,
            ask=ask,
            lastPrice=last,
            premium_mid=premium,
            volume=to_int_or_none(candidate.get("volume")),
            openInterest=to_int_or_none(candidate.get("openInterest")),
            inTheMoney=to_bool_or_none(candidate.get("inTheMoney")),
            impliedVolatility=to_float_or_none(candidate.get("impliedVolatility")),
            currency=(candidate.get("currency") or "USD"),
            delta=to_float_or_none(candidate.get("delta")),
            underlying_price=S,
            risk_free_rate_used=rf_rate,
            dividend_yield_used=q,
            time_to_expiry_years=T,
            now_local=now_local,
            requested_date=requested_date_iso,
            error=None,
        )
    except Exception as e:
        # Return an error row with at least ticker + requested_date
        return ChosenPut(
            ticker=ticker,
            contractSymbol=None,
            resolved_expiration=None,
            strike=None,
            bid=None,
            ask=None,
            lastPrice=None,
            premium_mid=None,
            volume=None,
            openInterest=None,
            inTheMoney=None,
            impliedVolatility=None,
            currency=None,
            delta=None,
            underlying_price=None,
            risk_free_rate_used=None,
            dividend_yield_used=None,
            time_to_expiry_years=None,
            now_local=now_local,
            requested_date=requested_date_iso,
            error=str(e),
        )


# --------------------------- CLI ---------------------------

def parse_args() -> str:
    if len(sys.argv) < 2:
        print("Usage: python batch_31_delta_puts.py <DATE>\n"
              "Example: python batch_31_delta_puts.py 2025-08-16")
        sys.exit(2)
    return " ".join(sys.argv[1:])


def main():
    user_date = parse_args()
    requested_date_iso = normalize_date(user_date)
    tickers = load_tickers_from_csv("input.csv")

    # Risk-free rate once (applies to all); cheaper on network and consistent
    rf_rate = get_risk_free_rate()
    target_abs_delta = 0.31

    rows: List[Dict[str, Any]] = []
    print(f"Processing {len(tickers)} tickers for requested date {requested_date_iso}...")
    for tk in tickers:
        print(f"  -> {tk} ...", end="", flush=True)
        result = pick_31_delta_put_for_ticker(
            ticker=tk,
            requested_date_iso=requested_date_iso,
            max_abs_delta=target_abs_delta,
            rf_rate=rf_rate,
        )
        rows.append(asdict(result))
        print(" done." if not result.error else f" ERROR: {result.error}")

    # Create dataframe with a consistent, human-friendly column order
    COL_ORDER = [
        "ticker", "requested_date", "resolved_expiration", "now_local",
        "contractSymbol", "currency",
        "strike", "bid", "ask", "lastPrice", "premium_mid",
        "impliedVolatility", "delta",
        "volume", "openInterest", "inTheMoney",
        "underlying_price", "risk_free_rate_used", "dividend_yield_used", "time_to_expiry_years",
        "error",
    ]
    df = pd.DataFrame(rows)
    # Ensure all columns exist
    for c in COL_ORDER:
        if c not in df.columns:
            df[c] = None
    df = df[COL_ORDER]

    out_path = f"output-{requested_date_iso}.csv"
    df.to_csv(out_path, index=False)
    print("\n✅ Finished.")
    print(f"Saved results to: {out_path}")
    print(f"Rows: {len(df)}  |  Errors: {df['error'].notna().sum()}")

if __name__ == "__main__":
    main()
