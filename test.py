#!/usr/bin/env python3
"""
Fetch live-ish option quotes via yfinance.

Test case: TSLA option expiring around 2025-08-16 (next Friday vs official Saturday date ambiguity).
We auto-resolve to the closest available Yahoo expiration (likely 2025-08-15).

Usage:
    python live_options.py              # uses default TSLA and 2025-08-16
    python live_options.py TSLA 2025-08-16
    python live_options.py AAPL 2025-09-20
"""

from __future__ import annotations
import sys
import math
from datetime import datetime, timezone
from typing import Tuple, Optional

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("This script requires yfinance and pandas:\n  pip install yfinance pandas")
    sys.exit(1)


def parse_args() -> Tuple[str, str]:
    ticker = "TSLA"
    target_date_str = "2025-08-16"
    if len(sys.argv) >= 2:
        ticker = sys.argv[1].upper()
    if len(sys.argv) >= 3:
        target_date_str = sys.argv[2]
    # Basic validation
    try:
        datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        print("ERROR: date must be in YYYY-MM-DD format, e.g. 2025-08-16")
        sys.exit(2)
    return ticker, target_date_str


def pick_closest_expiration(t: yf.Ticker, target_date_str: str) -> str:
    """Return expiration string from t.options that is closest to target_date."""
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    expirations = t.options  # list of 'YYYY-MM-DD' strings
    if not expirations:
        raise RuntimeError("No option expirations available for this ticker.")
    # Map to (date, str)
    dated = []
    for s in expirations:
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
            dated.append((abs((d - target_dt).days), d, s))
        except Exception:
            continue
    if not dated:
        raise RuntimeError("Could not parse option expirations returned by Yahoo.")
    dated.sort()
    closest = dated[0][2]
    return closest


def _mid(bid: Optional[float], ask: Optional[float], last: Optional[float]) -> Optional[float]:
    """Compute mid from bid/ask; if missing, fallback to last."""
    try:
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (float(bid) + float(ask)) / 2.0
        if last is not None and last > 0:
            return float(last)
    except Exception:
        pass
    return None


def _nearest_atm_strike(underlying_price: float, strikes) -> float:
    """Pick the strike closest to underlying price (ATM)."""
    return min(strikes, key=lambda s: abs(float(s) - underlying_price))


def fetch_atm_call_put(
    ticker: str, expiration: str
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Return (call_row, put_row, underlying_price)

    call_row/put_row are the ATM contracts.
    """
    t = yf.Ticker(ticker)
    underlying = t.fast_info.last_price or t.info.get("regularMarketPrice")
    if underlying is None or underlying <= 0:
        # Robust fallback via history()
        hist = t.history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            underlying = float(hist["Close"][-1])
        else:
            raise RuntimeError("Could not determine underlying price.")
    # Pull chain
    chain = t.option_chain(expiration)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    if calls.empty or puts.empty:
        raise RuntimeError(f"No options returned for expiration {expiration}.")

    # Find ATM strike
    strikes = calls["strike"].tolist()
    atm_strike = _nearest_atm_strike(underlying, strikes)

    # Get ATM rows
    call_row = calls.loc[calls["strike"] == atm_strike].iloc[0]
    put_row = puts.loc[puts["strike"] == atm_strike].iloc[0]
    return call_row, put_row, float(underlying)


def describe_contract(side: str, row: pd.Series) -> dict:
    """Extract and compute useful fields from an option row."""
    bid = row.get("bid", None)
    ask = row.get("ask", None)
    last = row.get("lastPrice", None) or row.get("last_price", None)
    strike = float(row["strike"])
    premium = _mid(bid, ask, last)
    volume = row.get("volume", None)
    open_interest = row.get("openInterest", None) or row.get("open_interest", None)
    symbol = row.get("contractSymbol", row.get("contract_symbol", ""))
    return {
        "side": side.upper(),
        "symbol": symbol,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "last": last,
        "premium_mid": premium,
        "volume": volume,
        "open_interest": open_interest,
    }


def main():
    ticker, target_date_str = parse_args()
    t = yf.Ticker(ticker)

    # Resolve the real expiration closest to the requested date (Fri vs Sat ambiguity)
    expiration = pick_closest_expiration(t, target_date_str)

    # Fetch ATM call/put
    call_row, put_row, underlying = fetch_atm_call_put(ticker, expiration)
    call_info = describe_contract("call", call_row)
    put_info = describe_contract("put", put_row)

    # Present
    print("=" * 72)
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{ticker} OPTIONS @ {now}")
    print(f"Resolved expiration: {expiration} (requested {target_date_str})")
    print(f"Underlying last: ${underlying:,.2f}")
    print("-" * 72)

    def fmt(info: dict) -> str:
        prem = info["premium_mid"]
        prem_s = f"${prem:,.2f}" if prem is not None else "N/A"
        bid = info["bid"]
        ask = info["ask"]
        bid_s = f"${bid:,.2f}" if bid is not None else "N/A"
        ask_s = f"${ask:,.2f}" if ask is not None else "N/A"
        last = info["last"]
        last_s = f"${last:,.2f}" if last is not None else "N/A"
        return (
            f"{info['side']}: {info['symbol']}\n"
            f"  Strike: ${info['strike']:,.2f}\n"
            f"  Bid/Ask: {bid_s} / {ask_s}\n"
            f"  Last: {last_s}\n"
            f"  Premium (mid): {prem_s}\n"
            f"  OI: {info['open_interest']}, Volume: {info['volume']}"
        )

    print(fmt(call_info))
    print("-" * 72)
    print(fmt(put_info))
    print("=" * 72)

    # Minimal explicit result asked for by the user:
    print("\nRequested output (example):")
    print(f"Call premium & strike: {call_info['premium_mid']}, {call_info['strike']}")
    print(f"Put  premium & strike: {put_info['premium_mid']}, {put_info['strike']}")


if __name__ == "__main__":
    main()
