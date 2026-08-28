import os
import time
import warnings
from datetime import datetime, time as dt_time

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests
import streamlit as st
import twstock
import yfinance as yf

# ============================================================
# 基本設定與頁面配置
# ============================================================

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="台股 V2.2.3 強勢突破全自動雷達",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 台股 V2.2.3 全自動選股雷達 (防死鎖完全版)")

st.caption(
    "V2.2.3 穩定版：修復上市櫃股本防鎖機制｜全台上市＋上櫃｜"
    "Yahoo歷史資料＋TWSE/TPEX官方最新行情｜"
    "已完成交易日｜週20MA｜前5日均量放量｜"
    "40日創高 OR W底突破｜產業集中"
)


# ============================================================
# 全局常數
# ============================================================

TW_TZ = "Asia/Taipei"
MIN_VOLUME_LOTS = 1000
DAILY_HISTORY_PERIOD = "1y"
FULL_HISTORY_PERIOD = "2y"
CHART_DAYS = 250
BATCH_SIZE = 80
REQUEST_TIMEOUT = 10
MIN_DAILY_ROWS = 60
MIN_FULL_ROWS = 100


# ============================================================
# TWSE / TPEX 官方行情 API 靜態設定
# ============================================================

TWSE_STOCK_DAY_ALL_URL = (
    "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
)

TPEX_MAINBOARD_QUOTE_URL = (
    "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
)

OFFICIAL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.tpex.org.tw/",
    "Origin": "https://www.tpex.org.tw"
}


# ============================================================
# Streamlit Cache - 公司股本資料 (防連線逾時死鎖)
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_capital_data_v2():
    capital_map = {}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.tpex.org.tw/",
        "Origin": "https://www.tpex.org.tw"
    }

    # 1. 抓取 TWSE 上市公司實收資本額
    try:
        url_twse = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        res = requests.get(url_twse, headers=headers, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                for row in data:
                    code = str(row.get("公司代號", row.get("CompanyCode", ""))).strip()
                    cap_str = str(row.get("實收資本額(元)", row.get("Capital", "0"))).replace(",", "").strip()
                    try:
                        cap = float(cap_str)
                        if cap > 0:
                            capital_map[code] = cap
                    except Exception:
                        pass
    except Exception:
        pass

    # 2. 抓取 TPEX 上櫃公司實收資本額 (多端點備援機制)
    tpex_urls = [
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
        "https://www.tpex.org.tw/openapi/v1/t187ap03_O"
    ]

    for url in tpex_urls:
        try:
            res = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    for row in data:
                        code = str(
                            row.get("SecuritiesCompanyCode", 
                            row.get("公司代號", 
                            row.get("StkNo", "")))
                        ).strip()

                        cap_str = str(
                            row.get("CapitalStock", 
                            row.get("PaidInCapital", 
                            row.get("實收資本額(元)", 
                            row.get("實收資本額", "0"))))
                        ).replace(",", "").strip()

                        try:
                            cap = float(cap_str)
                            if cap > 0:
                                # 部分 API 單位若為千元則擴增至完整金額
                                if cap < 100000000:
                                    cap = cap * 1000
                                capital_map[code] = cap
                        except Exception:
                            pass
                    if len(capital_map) > 1000:
                        break
        except Exception:
            continue

    return capital_map


# ============================================================
# 時間與日期解析工具
# ============================================================

def get_taiwan_now():
    return pd.Timestamp.now(tz=TW_TZ)


def is_market_closed_for_today():
    now = get_taiwan_now()
    if now.weekday() >= 5:
        return True
    return now.time() >= dt_time(13, 30)


def parse_official_date(date_raw):
    if date_raw is None or pd.isna(date_raw):
        return pd.NaT
    text = str(date_raw).strip()
    if not text:
        return pd.NaT

    if text.isdigit():
        if len(text) == 7:
            try:
                return pd.Timestamp(int(text[:3]) + 1911, int(text[3:5]), int(text[5:7]))
            except Exception:
                pass
        if len(text) == 8:
            try:
                return pd.Timestamp(int(text[:4]), int(text[4:6]), int(text[6:8]))
            except Exception:
                pass

    if "/" in text:
        parts = text.split("/")
        if len(parts) == 3:
            try:
                year = int(parts[0])
                if year < 1911:
                    year += 1911
                return pd.Timestamp(year, int(parts[1]), int(parts[2]))
            except Exception:
                pass

    try:
        res = pd.to_datetime(text, errors="coerce")
        return pd.NaT if pd.isna(res) else pd.Timestamp(res).normalize()
    except Exception:
        return pd.NaT


# ============================================================
# 官方最新行情整合
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_official_latest_quotes():
    quotes = {}

    # TWSE 上市行情
    try:
        res = requests.get(TWSE_STOCK_DAY_ALL_URL, headers=OFFICIAL_HEADERS, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                for row in data:
                    code = str(row.get("Code", "")).strip()
                    if not code:
                        continue
                    try:
                        date_raw = str(row.get("Date", "")).strip()
                        off_date = parse_official_date(date_raw)
                        close = float(str(row.get("ClosingPrice", "0")).replace(",", "") or 0)
                        vol = float(str(row.get("TradeVolume", "0")).replace(",", "") or 0)
                        if close > 0 and vol > 0 and not pd.isna(off_date):
                            quotes[code] = {
                                "market": "上市", "date": off_date, "date_raw": date_raw,
                                "Open": float(str(row.get("OpeningPrice", "0")).replace(",", "") or 0),
                                "High": float(str(row.get("HighestPrice", "0")).replace(",", "") or 0),
                                "Low": float(str(row.get("LowestPrice", "0")).replace(",", "") or 0),
                                "Close": close, "Volume": vol
                            }
                    except Exception:
                        continue
    except Exception:
        pass

    # TPEX 上櫃行情
    try:
        res = requests.get(TPEX_MAINBOARD_QUOTE_URL, headers=OFFICIAL_HEADERS, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                for row in data:
                    code = str(row.get("SecuritiesCompanyCode", "")).strip()
                    if not code:
                        continue
                    try:
                        date_raw = str(row.get("Date", row.get("日期", ""))).strip()
                        off_date = parse_official_date(date_raw)
                        close = float(str(row.get("Close", "0")).replace(",", "") or 0)
                        vol = float(str(row.get("TradingShares", "0")).replace(",", "") or 0)
                        if close > 0 and vol > 0 and not pd.isna(off_date):
                            quotes[code] = {
                                "market": "上櫃", "date": off_date, "date_raw": date_raw,
                                "Open": float(str(row.get("Open", "0")).replace(",", "") or 0),
                                "High": float(str(row.get("High", "0")).replace(",", "") or 0),
                                "Low": float(str(row.get("Low", "0")).replace(",", "") or 0),
                                "Close": close, "Volume": vol
                            }
                    except Exception:
                        continue
    except Exception:
        pass

    return quotes


# ============================================================
# Yahoo 數據清洗與對齊
# ============================================================

def flatten_yfinance_columns(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if "Close" in level0:
            df.columns = level0
        else:
            df.columns = df.columns.get_level_values(-1)
    return df


def apply_official_latest_quote(df, stock_id, official_quotes):
    if df is None or df.empty or stock_id not in official_quotes:
        return df

    quote = official_quotes[stock_id]
    try:
        df = df.copy()
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx.normalize()

        off_date = pd.Timestamp(quote["date"]).normalize()
        today = pd.Timestamp(get_taiwan_now().date()).normalize()

        if off_date > today:
            return df

        vals = {
            "Open": quote["Open"], "High": quote["High"],
            "Low": quote["Low"], "Close": quote["Close"], "Volume": quote["Volume"]
        }

        if off_date in df.index:
            for col, val in vals.items():
                if col in df.columns and val > 0:
                    df.loc[off_date, col] = val
        else:
            new_row = pd.DataFrame([vals], index=[off_date])
            df = pd.concat([df, new_row])

        return df.sort_index()
    except Exception:
        return df


def prepare_completed_daily_data(df_day):
    if df_day is None or df_day.empty:
        return pd.DataFrame()
    df_day = df_day.copy()
    idx = pd.to_datetime(df_day.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df_day.index = idx.normalize()

    now = get_taiwan_now()
    if df_day.index[-1].date() == now.date() and not is_market_closed_for_today():
        df_day = df_day.iloc[:-1].copy()

    return df_day


def build_completed_weekly_data(df_day):
    if df_day is None or df_day.empty:
        return pd.DataFrame()
    req = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df_day.columns for c in req):
        return pd.DataFrame()

    weekly = df_day[req].resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()

    now = get_taiwan_now()
    if now.weekday() < 5 and not weekly.empty and weekly.index[-1].date() >= now.date():
        weekly = weekly.iloc[:-1]

    return weekly


# ============================================================
# W底型態演算
# ============================================================

def calculate_pivot_lows(low_values, pivot_window=3):
    lows = np.asarray(low_values, dtype=float)
    pivots = []
    if len(lows) < pivot_window * 2 + 1:
        return pivots
    for i in range(pivot_window, len(lows) - pivot_window):
        if lows[i] <= np.min(lows[i - pivot_window:i]) and lows[i] <= np.min(lows[i + 1:i + pivot_window + 1]):
            pivots.append(i)
    return pivots


def detect_w_bottom(high_day, low_day, close_day, tolerance=0.06, lookback=60, pivot_window=3, min_gap=7, max_gap=35):
    res = {
        "is_w_bottom": False, "left_idx": None, "right_idx": None,
        "left_foot": None, "right_foot": None, "neck_high": None, "foot_diff_pct": None
    }
    if len(low_day) < lookback:
        return res

    highs = np.asarray(high_day[-lookback:], dtype=float)
    lows = np.asarray(low_day[-lookback:], dtype=float)
    closes = np.asarray(close_day[-lookback:], dtype=float)

    pivots = calculate_pivot_lows(lows, pivot_window)
    if len(pivots) < 2:
        return res

    candidates = []
    for l_idx in pivots:
        if l_idx >= lookback // 2:
            continue
        for r_idx in pivots:
            if r_idx <= l_idx or r_idx >= lookback - 5:
                continue
            gap = r_idx - l_idx
            if gap < min_gap or gap > max_gap:
                continue
            
            l_foot, r_foot = lows[l_idx], lows[r_idx]
            if l_foot <= 0:
                continue
            diff = abs(l_foot - r_foot) / ((l_foot + r_foot) / 2)
            if diff > tolerance:
                continue

            neck = np.max(highs[l_idx:r_idx + 1])
            if neck <= max(l_foot, r_foot) or closes[-1] <= neck:
                continue

            candidates.append({
                "left_idx": l_idx, "right_idx": r_idx,
                "left_foot": float(l_foot), "right_foot": float(r_foot),
                "neck_high": float(neck), "foot_diff_pct": float(diff * 100)
            })

    if not candidates:
        return res

    candidates.sort(key=lambda x: (-x["right_idx"], x["foot_diff_pct"]))
    best = candidates[0]
    best["is_w_bottom"] = True
    return best


# ============================================================
# 股票資料抓取與兩階段過濾器
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_all_tw_stocks_info():
    stocks_info = {}
    for code, info in twstock.codes.items():
        if info.type == "股票" and info.market in ["上市", "上櫃"]:
            suffix = ".TW" if info.market == "上市" else ".TWO"
            stocks_info[f"{code}{suffix}"] = {
                "code": code, "name": info.name,
                "group": info.group if info.group else "其他",
                "market": info.market
            }
    return stocks_info


def fast_filter_batch(batch_df, stocks_info, capital_map, min_capital, vol_multiplier, breakout_days, official_quotes):
    candidates, errors = [], []
    if batch_df is None or batch_df.empty or not isinstance(batch_df.columns, pd.MultiIndex):
        return candidates, errors

    level0 = batch_df.columns.get_level_values(0)
    if "Close" not in level0 or "High" not in level0 or "Volume" not in level0:
        return candidates, errors

    close_df, high_df, volume_df = batch_df["Close"], batch_df["High"], batch_df["Volume"]
    today = get_taiwan_now().date()
    market_closed = is_market_closed_for_today()

    for ticker in close_df.columns:
        if ticker not in stocks_info:
            continue
        try:
            code = stocks_info[ticker]["code"]
            
            # 寬鬆股本過濾條件：有抓到股本且低於設定值才過濾；抓不到股本時則安全放行
            capital = capital_map.get(code)
            if capital is not None and capital < min_capital:
                continue

            stock_df = pd.DataFrame({
                "Close": close_df[ticker],
                "High": high_df[ticker],
                "Volume": volume_df[ticker]
            }).dropna()

            if len(stock_df) < MIN_DAILY_ROWS:
                continue

            stock_df = apply_official_latest_quote(stock_df, code, official_quotes)
            if stock_df.index[-1].date() == today and not market_closed:
                stock_df = stock_df.iloc[:-1].copy()

            if len(stock_df) < MIN_DAILY_ROWS:
                continue

            close_s, high_s, vol_s = stock_df["Close"], stock_df["High"], stock_df["Volume"]

            latest_close = float(close_s.iloc[-1])
            latest_vol = float(vol_s.iloc[-1])
            latest_vol_lots = latest_vol / 1000.0

            if latest_vol_lots < MIN_VOLUME_LOTS:
                continue

            prev_5_vol = vol_s.iloc[-6:-1]
            avg_5_vol = prev_5_vol.mean()

            if not np.isfinite(avg_5_vol) or avg_5_vol <= 0:
                continue

            vol_ratio = latest_vol / avg_5_vol
            if vol_ratio < vol_multiplier:
                continue

            if len(high_s) <= breakout_days:
                continue

            prev_high = high_s.iloc[-(breakout_days + 1):-1].max()
            is_breakout = latest_close >= prev_high

            candidates.append({
                "ticker": ticker, "latest_close": latest_close,
                "latest_volume": latest_vol, "latest_volume_lots": latest_vol_lots,
                "avg_5_volume": avg_5_vol, "volume_ratio": vol_ratio,
                "previous_high": prev_high, "is_breakout": is_breakout,
                "capital": capital, "data_date": stock_df.index[-1].strftime("%Y-%m-%d")
            })
        except Exception as e:
            errors.append({"ticker": ticker, "error": repr(e)})

    return candidates, errors


def analyze_candidate_from_df(candidate, df_day, stocks_info, params, official_quotes):
    ticker = candidate["ticker"]
    try:
        df_day = flatten_yfinance_columns(df_day)
        if df_day.empty:
            return None

        code = stocks_info[ticker]["code"]
        df_day = apply_official_latest_quote(df_day, code, official_quotes)
        df_day = prepare_completed_daily_data(df_day)

        if df_day.empty or len(df_day) < MIN_FULL_ROWS:
            return None

        df_week = build_completed_weekly_data(df_day)
        if df_week.empty or len(df_week) < params["ma_week"]:
            return None

        close_day, high_day = df_day["Close"].to_numpy(float), df_day["High"].to_numpy(float)
        low_day, vol_day = df_day["Low"].to_numpy(float), df_day["Volume"].to_numpy(float)
        close_week = df_week["Close"].to_numpy(float)

        ma_week_val = pd.Series(close_week).rolling(params["ma_week"]).mean().iloc[-1]
        if not np.isfinite(ma_week_val) or close_week[-1] <= ma_week_val:
            return None

        latest_close, latest_vol = close_day[-1], vol_day[-1]
        latest_vol_lots = latest_vol / 1000.0
        avg_5_vol = np.mean(vol_day[-6:-1])

        if avg_5_vol <= 0 or (latest_vol / avg_5_vol) < params["vol_multiplier"]:
            return None

        breakout_days = params["breakout_days"]
        prev_high = np.max(high_day[-(breakout_days + 1):-1])
        is_breakout = latest_close >= prev_high

        w_info = detect_w_bottom(
            high_day=high_day, low_day=low_day, close_day=close_day,
            tolerance=params["w_tolerance"], lookback=params["w_lookback"],
            pivot_window=params["pivot_window"], min_gap=params["w_min_gap"], max_gap=params["w_max_gap"]
        )
        is_w_bottom = w_info["is_w_bottom"]

        if not (is_breakout or is_w_bottom):
            return None

        reasons = []
        if is_breakout:
            reasons.append(f"{breakout_days}日創高突破")
        if is_w_bottom:
            reasons.append("W底突破")

        signal_type = "雙重訊號" if (is_breakout and is_w_bottom) else ("區間創高" if is_breakout else "W底突破")

        return {
            "status": "match", "ticker": ticker, "code": code,
            "name": stocks_info[ticker]["name"], "group": stocks_info[ticker]["group"],
            "market": stocks_info[ticker]["market"], "capital": candidate.get("capital"),
            "data_date": df_day.index[-1].strftime("%Y-%m-%d"), "df_day": df_day,
            "close": round(latest_close, 2), "volume": int(latest_vol_lots),
            "volume_avg_5": round(avg_5_vol / 1000, 0), "volume_ratio": round(latest_vol / avg_5_vol, 2),
            "ma_week_val": round(float(ma_week_val), 2),
            "distance_to_week_ma_pct": round(float((latest_close - ma_week_val) / latest_close * 100), 2),
            "previous_high": round(float(prev_high), 2),
            "breakout_distance_pct": round(float((latest_close - prev_high) / prev_high * 100), 2),
            "is_breakout": bool(is_breakout), "is_w_bottom": bool(is_w_bottom),
            "signal_type": signal_type, "reasons": reasons, "w_info": w_info
        }
    except Exception:
        return None


# ============================================================
# 控制面板 UI 與邏輯啟動
# ============================================================

st.sidebar.header("🔍 V2.2.3 選股控制台")

min_capital_yi = st.sidebar.number_input("最低股本（億元）", min_value=0.0, max_value=5000.0, value=10.0, step=1.0)
min_capital = min_capital_yi * 100_000_000

vol_multiplier = st.sidebar.slider("放量倍數（對比前5日均量）", min_value=1.0, max_value=5.0, value=1.2, step=0.1)
breakout_days = st.sidebar.number_input("突破回看期間（交易日）", min_value=10, max_value=60, value=40, step=1)
ma_week = st.sidebar.number_input("長期趨勢均線（週MA）", min_value=10, max_value=40, value=20, step=1)

w_tolerance = st.sidebar.slider("W底左右腳容錯率", min_value=1.0, max_value=15.0, value=6.0, step=0.5) / 100.0
pivot_window = st.sidebar.number_input("W底 Pivot Low 判定寬度", min_value=2, max_value=6, value=3, step=1)
w_min_gap = st.sidebar.number_input("W底左右腳最小間隔", min_value=5, max_value=15, value=7, step=1)
w_max_gap = st.sidebar.number_input("W底左右腳最大間隔", min_value=20, max_value=45, value=35, step=1)

params = {
    "min_capital": min_capital, "vol_multiplier": vol_multiplier,
    "breakout_days": breakout_days, "ma_week": ma_week,
    "w_tolerance": w_tolerance, "w_lookback": 60,
    "pivot_window": pivot_window, "w_min_gap": w_min_gap, "w_max_gap": w_max_gap
}

if st.sidebar.button("🚀 開始 V2.2.3 雷達掃描", type="primary"):
    scan_start = time.time()
    stocks_info = get_all_tw_stocks_info()

    if not stocks_info:
        st.error("❌ 無法取得台股股票清單。")
        st.stop()

    st.info(f"股票清單載入成功，共 {len(stocks_info)} 支（包含上市與上櫃）")

    with st.spinner("正在取得 TWSE / TPEX 最新行情與股本..."):
        official_quotes = get_official_latest_quotes()
        capital_map = get_company_capital_data_v2()

    st.info(f"官方行情取得：{len(official_quotes)} 筆｜股本資料庫取得：{len(capital_map)} 筆")

    # 第一階段：批量快篩
    st.subheader("🔎 第一階段：快速篩選")
    progress = st.progress(0)
    fast_candidates, batch_errors = [], []
    tickers = list(stocks_info.keys())
    total_batches = int(np.ceil(len(tickers) / BATCH_SIZE))

    for b_idx, start in enumerate(range(0, len(tickers), BATCH_SIZE), start=1):
        batch_tickers = tickers[start:start + BATCH_SIZE]
        try:
            batch_df = yf.download(
                batch_tickers, period=DAILY_HISTORY_PERIOD, interval="1d",
                auto_adjust=True, progress=False, group_by="column", threads=True
            )
            cands, errs = fast_filter_batch(
                batch_df, stocks_info, capital_map, min_capital,
                vol_multiplier, breakout_days, official_quotes
            )
            fast_candidates.extend(cands)
            batch_errors.extend(errs)
        except Exception as e:
            batch_errors.append({"ticker": ",".join(batch_tickers), "error": repr(e)})

        progress.progress(b_idx / total_batches)

    st.success(f"第一階段完成：成功保留 {len(fast_candidates)} 支候選個股")

    if not fast_candidates:
        st.warning("⚠️ 第一階段未找到符合條件的個股，請嘗試降低「最低股本」或「放量倍數」。")
        st.stop()

    # 第二階段：型態精算
    st.subheader("📐 第二階段：完整型態分析")
    progress2 = st.progress(0)
    matches = []
    cand_tickers = [x["ticker"] for x in fast_candidates]
    cand_map = {x["ticker"]: x for x in fast_candidates}
    total_cand_batches = int(np.ceil(len(cand_tickers) / BATCH_SIZE))

    for b_idx, start in enumerate(range(0, len(cand_tickers), BATCH_SIZE), start=1):
        batch_tickers = cand_tickers[start:start + BATCH_SIZE]
        try:
            full_batch_df = yf.download(
                batch_tickers, period=FULL_HISTORY_PERIOD, interval="1d",
                auto_adjust=True, progress=False, group_by="column", threads=True
            )
            for ticker in batch_tickers:
                res = analyze_candidate_from_df(cand_map[ticker], full_batch_df, stocks_info, params, official_quotes)
                if res is not None:
                    matches.append(res)
        except Exception:
            pass
        progress2.progress(b_idx / total_cand_batches)

    matches.sort(key=lambda x: (x["group"], -x["volume_ratio"]))
    elapsed = time.time() - scan_start

    st.success(f"🎉 掃描完畢！耗時 {elapsed:.1f} 秒，最終符合條件個股共 {len(matches)} 支。")

    # 結果輸出
    if matches:
        st.subheader(f"📋 入選股票總覽（共 {len(matches)} 支）")
        summary_rows = []
        for m in matches:
            summary_rows.append({
                "產業": m["group"],
                "股票": f"{m['name']} ({m['code']})",
                "市場": m["market"],
                "股本(億)": f"{m['capital'] / 100_000_000:.1f}" if m["capital"] else "—",
                "收盤價": m["close"],
                "今日量(張)": f"{m['volume']:,}",
                "放量倍數": f"{m['volume_ratio']:.2f}x",
                "訊號": m["signal_type"],
                "資料日期": m["data_date"]
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ 目前無符合條件之股票。")
