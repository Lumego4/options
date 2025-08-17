import math
from typing import Optional

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def call_delta_black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if sigma is None or sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = bs_d1(S, K, T, r, sigma, q)
    return math.exp(-q * T) * _norm_cdf(d1)

def put_delta_black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
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
