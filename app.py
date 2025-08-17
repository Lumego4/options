# app.py
# Streamlit UI for 31-delta PUTs or nearest-lower-strike CALLs, with Big Dave / Luke views,
# and support for group CSVs (e.g., spy-100).

from __future__ import annotations
import math
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from data_helpers import (
    normalize_date,
    get_risk_free_rate,
    get_underlying_price,
    get_info,
    get_expirations,
    get_chain,
    get_hist_close,
    get_dividend_yield,
    ts_to_iso,
    get_last_earnings_date,
    closest_expiration,
    _to_float,
    _to_int
)

import pandas as pd
from pathlib import Path
import streamlit as st
import yfinance as yf
from metrics import technical_reco
from selection import select_put_row, select_call_row_nearest_lower


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


# --------------------------- Helpers ---------------------------

def next_upcoming_friday(today: date) -> date:
    # Friday is 4; always move forward to a future Friday
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + pd.Timedelta(days=days_ahead)

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
            T = (pd.to_datetime(resolved_exp) - pd.Timestamp.today()).days / 365.0
        except Exception:
            T = 7/365.0
    else:
        T = 7/365.0

    strike = premium = delta = iv = opt_vol = None

    if resolved_exp and S:
        chain = get_chain(ticker, resolved_exp)
        calls, puts = chain["calls"], chain["puts"]
        q = get_dividend_yield(info) or 0.0
        if scan_type == "puts":
            put_row = None
            try:
                # fallback for missing function argument
                from selection import select_put_row
                put_row = select_put_row(puts, S, T, rf_rate, q, target_abs_delta, ticker)
            except Exception:
                put_row = None
            if put_row is not None:
                strike = _to_float(put_row.get("strike"))
                premium = mid_price(put_row.get("bid"), put_row.get("ask"), put_row.get("lastPrice"))
                delta = _to_float(put_row.get("delta"))
                iv = _to_float(put_row.get("impliedVolatility"))
                opt_vol = _to_int(put_row.get("volume"))
        else:
            call_row = None
            try:
                from selection import select_call_row_nearest_lower
                call_row = select_call_row_nearest_lower(calls, S)
            except Exception:
                call_row = None
            if call_row is not None:
                strike = _to_float(call_row.get("strike"))
                premium = mid_price(call_row.get("bid"), call_row.get("ask"), call_row.get("lastPrice"))
                delta = _to_float(call_row.get("delta"))
                iv = _to_float(call_row.get("impliedVolatility"))
                opt_vol = _to_int(call_row.get("volume"))

    # ---- Common metrics + your rounding rules ----
    premium_mid = round(premium, 2) if premium is not None else None
    premium_ratio_pct = None
    if premium is not None and strike:
        try:
            premium_ratio_pct = round((premium / strike) * 100.0, 2)
        except Exception:
            premium_ratio_pct = None
    buffer_distance = None
    if S is not None and premium is not None and strike is not None:
        try:
            buffer_distance = round(S + premium - strike, 1)
        except Exception:
            buffer_distance = None
    normalized_buffer_pct = None
    if buffer_distance is not None and strike:
        try:
            normalized_buffer_pct = round((buffer_distance / strike) * 100.0, 2)
        except Exception:
            normalized_buffer_pct = None
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
        "market_cap": market_cap,
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

from metrics import compute_custom_optimizer


# --------------------------- UI ---------------------------

st.title("📈 Options Scanner — Puts & Calls")
st.caption("Free data via yfinance; quotes may be delayed.")

# Session state for results and selections
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "results_ctx" not in st.session_state:
    st.session_state["results_ctx"] = {}
if "selected_row_ids" not in st.session_state:
    st.session_state["selected_row_ids"] = set()

with st.sidebar:
    # Run control at top
    run_btn_top = st.button("Run Scan 🚀", type="primary", use_container_width=True, key="run_top")

    st.header("Scan Settings")
    target_date = st.date_input(
        "Target expiration date",
        value=next_upcoming_friday(date.today()),
        help="Defaults to the upcoming Friday."
    )
    scan_type_label = st.radio("Scan type", ["Puts", "Calls"])
    scan_type = "puts" if "Puts" in scan_type_label else "calls"    

    # Show target delta only for PUTs
    if scan_type == "puts":
        target_delta = st.slider(
            "Target Delta for PUTs",
            min_value=0.15,
            max_value=0.50,
            value=0.31,
            step=0.01,
            help="Used only for PUT selection. Calls use nearest-lower strike to spot."
        )
    else:
        # keep a default in scope for logic; won't be used for calls
        target_delta = 0.31

    st.divider()
    st.subheader("Ticker source")

    # Preset CSVs from /ticker-lists
    preset_dir = Path("ticker-lists")
    preset_files = []
    try:
        if preset_dir.exists():
            preset_files = sorted([p.name for p in preset_dir.glob("*.csv")])
    except Exception:
        preset_files = []

    source_choice = st.radio(
        "Choose source",
        ["Paste tickers", "Preset list", "Upload CSV"],
        index=0
    )

    preset_selected = None
    csv_file = None
    default_list = "TSLA, CVS, PLTR, GOOG"
    pasted = default_list

    if source_choice == "Paste tickers":
        pasted = st.text_area("Paste tickers", value=default_list, height=100,
                              help="Comma/space/line separated")
    if source_choice == "Preset list":
        if preset_files:
            preset_selected = st.selectbox(
                "Preset list (from /ticker-lists)",
                options=["— select —"] + preset_files,
                index=0,
            )
        else:
            st.info("No preset CSVs found under /ticker-lists.")
            source_choice = "Paste tickers"
    if source_choice == "Upload CSV":
        csv_file = st.file_uploader("Upload CSV for tickers", type=["csv"], accept_multiple_files=False)
    

    # View mode at bottom
    st.divider()
    view_mode = st.radio("View mode", ["Big Dave", "Luke"],
                         help="Big Dave includes dividend info; Luke hides it. Both include Market Cap.")

    # Run control at bottom
    run_btn_bottom = st.button("Run Scan 🚀", type="primary", use_container_width=True, key="run_bottom")
    run_btn = run_btn_top or run_btn_bottom

    # Watchlist controls
    st.divider()
    st.subheader("Watchlist")
    WATCHLIST_DIR = Path("watchlists")
    try:
        WATCHLIST_DIR.mkdir(exist_ok=True)
    except Exception:
        pass
    try:
        existing_watchlists = sorted([p.stem for p in WATCHLIST_DIR.glob("*.csv")])
    except Exception:
        existing_watchlists = []

    wl_choice = st.selectbox(
        "Active watchlist",
        options=["— select —", "Create new…"] + existing_watchlists,
        index=0,
        key="wl_choice",
    )
    wl_name: Optional[str] = None
    if wl_choice == "Create new…":
        proposed = st.text_input("New watchlist name", placeholder="e.g., My-puts-Aug", key="wl_new_name")
        if proposed:
            wl_name = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in proposed).strip("-")
    elif wl_choice != "— select —":
        wl_name = wl_choice
    st.session_state["active_watchlist_name"] = wl_name
    st.checkbox("Show active watchlist below", value=False, key="show_watchlist")

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

# Decide tickers based on source priority: preset > upload > paste
tickers: List[str] = []
if source_choice == "Preset list" and preset_selected and preset_selected != "— select —":
    try:
        with open(preset_dir / preset_selected, "r", encoding="utf-8", errors="ignore") as f:
            tickers = parse_tickers_from_text(f.read())
    except Exception:
        tickers = []
elif source_choice == "Upload CSV" and csv_file is not None:
    tickers = parse_uploaded_csv(csv_file)
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

    # Persist results for reuse across reruns (sorting/selection)
    st.session_state["results_df"] = display_df.copy()
    st.session_state["results_ctx"] = {
        "requested_date_iso": requested_date_iso,
        "scan_type": scan_type,
    }
    st.session_state["results_csv_bytes"] = display_df.to_csv(index=False).encode("utf-8")

    st.success("Done ✔️")

# --------------------------- Results rendering + Watchlist ---------------------------
if st.session_state.get("results_df") is not None:
    display_df = st.session_state["results_df"].copy()
    ctx = st.session_state.get("results_ctx", {})
    requested_date_iso = ctx.get("requested_date_iso", "")
    scan_type = ctx.get("scan_type", "")

    # 1) Results (sortable, read-only)
    st.subheader("Results")
    st.data_editor(
        display_df,
        hide_index=True,
        use_container_width=True,
        disabled=True,
        key="results_table",
    )

    # Download latest results
    csv_bytes = st.session_state.get("results_csv_bytes", display_df.to_csv(index=False).encode("utf-8"))
    st.download_button(
        label=f"Download CSV (output-{requested_date_iso}-{scan_type}.csv)",
        data=csv_bytes,
        file_name=f"output-{requested_date_iso}-{scan_type}.csv",
        mime="text/csv",
        type="primary",
        key="download_results_btn",
    )

    # 2) Results with checkboxes (selection for watchlist)
    st.divider()
    st.subheader("Select rows to add to watchlist")

    def make_row_id(row: pd.Series) -> str:
        return "|".join([
            str(row.get("ticker", "")),
            str(row.get("strike", "")),
            scan_type,
            requested_date_iso,
        ])

    selectable = display_df.copy()
    selectable["row_id"] = selectable.apply(make_row_id, axis=1)
    # Use stable row_id as index so sorting won’t break selection
    selectable = selectable.set_index("row_id", drop=False)
    sel_ids = set(st.session_state.get("selected_row_ids", set()))
    selectable.insert(0, "Select", selectable.index.map(lambda rid: rid in sel_ids))

    edited = st.data_editor(
        selectable.drop(columns=["row_id"]),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Select": st.column_config.CheckboxColumn(help="Check rows to add to the active watchlist"),
        },
        disabled=[c for c in selectable.columns if c != "Select"],
        key="watchlist_selector",
    )

    # Persist selection using editor index (row_id)
    if isinstance(edited, pd.DataFrame) and "Select" in edited.columns:
        new_selected = set(edited.index[edited["Select"] == True].tolist())
        st.session_state["selected_row_ids"] = new_selected

    # Save to watchlist controls
    wl_dir = Path("watchlists"); wl_dir.mkdir(exist_ok=True)
    wl_name_input = st.text_input(
        "Watchlist name",
        value=st.session_state.get("active_watchlist_name") or "my-watchlist",
        key="wl_name_input",
    )
    add_btn = st.button("Add selected to watchlist", type="primary", key="add_selected_btn")
    if add_btn:
        wl_name_final = (wl_name_input or "").strip()
        if not wl_name_final:
            st.error("Please enter a watchlist name.")
        else:
            sel_ids = st.session_state.get("selected_row_ids", set())
            if not sel_ids:
                st.warning("No rows selected.")
            else:
                to_save = selectable.loc[list(sel_ids)].drop(columns=["Select", "row_id"]).reset_index(drop=True)
                to_save = to_save.copy()
                to_save["scan_type"] = scan_type
                to_save["target_exp"] = requested_date_iso
                wl_path = wl_dir / f"{wl_name_final}.csv"
                try:
                    if wl_path.exists():
                        existing = pd.read_csv(wl_path)
                    else:
                        existing = pd.DataFrame(columns=list(to_save.columns))
                except Exception:
                    existing = pd.DataFrame(columns=list(to_save.columns))
                combined = pd.concat([existing, to_save], ignore_index=True)
                keys = [k for k in ["ticker", "strike", "scan_type", "target_exp"] if k in combined.columns]
                if keys:
                    combined = combined.drop_duplicates(subset=keys, keep="first")
                try:
                    combined.to_csv(wl_path, index=False)
                    st.success(f"Added {len(to_save)} row(s) to watchlist '{wl_name_final}'.")
                    if not st.session_state.get("active_watchlist_name"):
                        st.session_state["active_watchlist_name"] = wl_name_final
                except Exception as e:
                    st.error(f"Failed to save watchlist: {e}")

    # 3) Active watchlist (viewer)
    wl_name = st.session_state.get("active_watchlist_name")
    if st.session_state.get("show_watchlist") and wl_name:
        wl_path = Path("watchlists") / f"{wl_name}.csv"
        st.divider()
        st.subheader(f"Active Watchlist — {wl_name}")
        if wl_path.exists():
            try:
                wl_df = pd.read_csv(wl_path)
                st.data_editor(
                    wl_df,
                    hide_index=True,
                    use_container_width=True,
                    disabled=True,
                    key="watchlist_view",
                )
                dl_bytes = wl_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download watchlist '{wl_name}'",
                    data=dl_bytes,
                    file_name=f"{wl_name}.csv",
                    mime="text/csv",
                    key="download_watchlist_btn",
                )
            except Exception as e:
                st.error(f"Failed to load watchlist '{wl_name}': {e}")
        else:
            st.info("This watchlist is empty. Add rows from results to create it.")


# --------------------------- Footer ---------------------------

st.caption("Not investment advice. Data from Yahoo via yfinance; may be delayed / incomplete.")
