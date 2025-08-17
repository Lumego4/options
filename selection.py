import pandas as pd
from typing import Optional
from data_helpers import get_hist_close, _to_float, _to_int
from greeks import put_delta_black_scholes, call_delta_black_scholes

def select_put_row(puts: pd.DataFrame, S: float, T: float, rf: float, q: float, target_abs_delta: float, ticker: str) -> Optional[pd.Series]:
    iv_series = puts["impliedVolatility"]
    hv = None
    if iv_series.isna().all():
        clos = get_hist_close(ticker, "60d", "1d")
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
    df = calls.dropna(subset=["strike"]).copy()
    df["strike"] = df["strike"].astype(float)
    df_le = df[df["strike"] <= S]
    if df_le.empty:
        return df.sort_values("strike").iloc[0]
    return df_le.sort_values("strike", ascending=True).iloc[-1]
