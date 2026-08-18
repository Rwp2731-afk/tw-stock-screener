import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mplfinance as mpf
import twstock
import warnings
import time
from datetime import time as dt_time


# ============================================================
# 基本設定
# ============================================================

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="台股 V2.1 全市場資金雷達",
    layout="wide"
)

st.title("📈 台股 V2.1 全市場資金雷達")

st.caption(
    "全台上市＋上櫃｜股本＋成交量＋5日均量放量＋週20MA＋"
    "40日創高／W底突破｜產業資金集中分析"
)


# ============================================================
# 常數
# ============================================================

TW_TZ = "Asia/Taipei"

# 最低成交量：1000張
MIN_VOLUME_LOTS = 1000

# K線顯示天數
CHART_DAYS = 250

# 批量下載大小
BATCH_SIZE = 80

# 批次之間休息
BATCH_SLEEP = 0.4


# ============================================================
# Matplotlib 中文字型
# ============================================================

def setup_chinese_font():

    candidates = [
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Microsoft JhengHei",
        "PingFang TC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS"
    ]

    available = {
        f.name
        for f in fm.fontManager.ttflist
    }

    for font_name in candidates:

        if font_name in available:

            plt.rcParams["font.sans-serif"] = [
                font_name
            ]

            plt.rcParams["axes.unicode_minus"] = False

            return font_name

    plt.rcParams["axes.unicode_minus"] = False

    return None


CHINESE_FONT = setup_chinese_font()


# ============================================================
# 台灣時間
# ============================================================

def get_taiwan_now():

    return pd.Timestamp.now(
        tz=TW_TZ
    )


def is_market_closed_for_today():

    now = get_taiwan_now()

    # 星期六、日
    if now.weekday() >= 5:
        return True

    market_close = dt_time(
        13,
        30
    )

    return now.time() >= market_close


# ============================================================
# yfinance 欄位處理
# ============================================================

def normalize_single_stock_df(df):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        level0 = list(
            df.columns
            .get_level_values(0)
        )

        level1 = list(
            df.columns
            .get_level_values(1)
        )

        price_cols = {
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        }

        if any(
            x in price_cols
            for x in level0
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        elif any(
            x in price_cols
            for x in level1
        ):

            df.columns = (
                df.columns
                .get_level_values(1)
            )

        else:

            df.columns = (
                df.columns
                .get_level_values(0)
            )

    return df


# ============================================================
# 只保留完成交易日
#
# 盤中：
# Yahoo 若已經出現今天資料
# → 排除今天尚未完成的K棒
#
# 盤後：
# 今天已收盤
# → 今天K棒可以使用
# ============================================================

def prepare_completed_daily_data(df):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    df.index = pd.to_datetime(
        df.index
    )

    now = get_taiwan_now()

    today = now.date()

    last_date = (
        df.index[-1].date()
    )

    # --------------------------------------------------------
    # 盤中發現 Yahoo 已經有今天資料
    # → 移除今天未完成K棒
    # --------------------------------------------------------

    if (
        last_date == today
        and not is_market_closed_for_today()
    ):

        df = df.iloc[:-1].copy()

    return df


# ============================================================
# 建立完整週K
#
# 重要：
#
# 盤中：
# 本週尚未完成 → 不使用本週週K
#
# 週五13:30後：
# 本週完整 → 使用本週週K
#
# 六日：
# 上週五已完成 → 使用最新週K
# ============================================================

def build_completed_weekly_data(
    df_day
):

    if df_day is None or df_day.empty:
        return pd.DataFrame()

    required_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if not all(
        col in df_day.columns
        for col in required_cols
    ):

        return pd.DataFrame()

    weekly = (
        df_day[
            required_cols
        ]
        .resample("W-FRI")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
    )

    weekly = weekly.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    if weekly.empty:
        return weekly

    last_day = (
        pd.Timestamp(
            df_day.index[-1]
        ).normalize()
    )

    weekday = last_day.weekday()

    # --------------------------------------------------------
    # 如果最新完成交易日是週一～週四
    # → 本週尚未完成
    # → 排除本週
    # --------------------------------------------------------

    if weekday < 4:

        this_friday = (
            last_day
            + pd.Timedelta(
                days=(4 - weekday)
            )
        )

        weekly = weekly[
            weekly.index < this_friday
        ]

    # --------------------------------------------------------
    # 如果最新完成交易日是週五
    # → 代表週五已完成
    # → 本週週K可以使用
    # --------------------------------------------------------

    else:

        this_friday = (
            last_day
        )

        weekly = weekly[
            weekly.index <= this_friday
        ]

    return weekly


# ============================================================
# Pivot Low
# ============================================================

def calculate_pivot_lows(
    low_values,
    pivot_window=3
):

    lows = np.asarray(
        low_values,
        dtype=float
    )

    pivot_indices = []

    if len(lows) < (
        pivot_window * 2 + 1
    ):

        return pivot_indices

    for i in range(
        pivot_window,
        len(lows) - pivot_window
    ):

        left = lows[
            i - pivot_window:i
        ]

        right = lows[
            i + 1:
            i + pivot_window + 1
        ]

        if (
            lows[i] <= np.min(left)
            and
            lows[i] <= np.min(right)
        ):

            pivot_indices.append(i)

    return pivot_indices


# ============================================================
# W底辨識
# ============================================================

def detect_w_bottom(
    high_day,
    low_day,
    close_day,
    tolerance=0.06,
    lookback=60,
    pivot_window=3,
    min_gap=7,
    max_gap=35
):

    empty_result = {

        "is_w_bottom": False,

        "left_idx": None,

        "right_idx": None,

        "left_foot": None,

        "right_foot": None,

        "neck_high": None,

        "foot_diff_pct": None
    }

    if len(low_day) < lookback:
        return empty_result

    highs = np.asarray(
        high_day[-lookback:],
        dtype=float
    )

    lows = np.asarray(
        low_day[-lookback:],
        dtype=float
    )

    closes = np.asarray(
        close_day[-lookback:],
        dtype=float
    )

    pivot_lows = calculate_pivot_lows(
        lows,
        pivot_window=pivot_window
    )

    if len(pivot_lows) < 2:
        return empty_result

    latest_close = closes[-1]

    candidates = []

    for left_idx in pivot_lows:

        if left_idx >= lookback // 2:
            continue

        for right_idx in pivot_lows:

            if right_idx <= left_idx:
                continue

            # 右腳至少距離現在5天
            if right_idx >= lookback - 5:
                continue

            gap = (
                right_idx
                - left_idx
            )

            if (
                gap < min_gap
                or gap > max_gap
            ):
                continue

            left_foot = lows[
                left_idx
            ]

            right_foot = lows[
                right_idx
            ]

            if left_foot <= 0:
                continue

            avg_foot = (
                left_foot
                + right_foot
            ) / 2

            foot_diff_pct = (
                abs(
                    left_foot
                    - right_foot
                )
                / avg_foot
            )

            if foot_diff_pct > tolerance:
                continue

            between_highs = highs[
                left_idx:
                right_idx + 1
            ]

            if len(between_highs) == 0:
                continue

            neck_high = np.max(
                between_highs
            )

            if (
                neck_high
                <= max(
                    left_foot,
                    right_foot
                )
            ):

                continue

            # 最新完成交易日收盤突破頸線
            if latest_close <= neck_high:
                continue

            right_after = closes[
                right_idx:
            ]

            if len(right_after) < 2:
                continue

            right_rebound_high = np.max(
                right_after
            )

            if (
                right_rebound_high
                <= right_foot
            ):

                continue

            right_to_neck_pct = (
                (
                    neck_high
                    - right_foot
                )
                / right_foot
            ) * 100

            if right_to_neck_pct < 3:
                continue

            candidates.append({

                "left_idx":
                    left_idx,

                "right_idx":
                    right_idx,

                "left_foot":
                    float(
                        left_foot
                    ),

                "right_foot":
                    float(
                        right_foot
                    ),

                "neck_high":
                    float(
                        neck_high
                    ),

                "foot_diff_pct":
                    float(
                        foot_diff_pct * 100
                    )
            })

    if not candidates:
        return empty_result

    candidates.sort(
        key=lambda x: (
            -x["right_idx"],
            x["foot_diff_pct"]
        )
    )

    best = candidates[0]

    return {

        "is_w_bottom": True,

        "left_idx":
            best["left_idx"],

        "right_idx":
            best["right_idx"],

        "left_foot":
            round(
                best["left_foot"],
                2
            ),

        "right_foot":
            round(
                best["right_foot"],
                2
            ),

        "neck_high":
            round(
                best["neck_high"],
                2
            ),

        "foot_diff_pct":
            round(
                best["foot_diff_pct"],
                2
            )
    }


# ============================================================
# 全台股票清單
# ============================================================

@st.cache_data(ttl=86400)
def get_all_tw_stocks_info():

    stocks_info = {}

    for code, info in twstock.codes.items():

        if (
            info.type == "股票"
            and info.market in [
                "上市",
                "上櫃"
            ]
        ):

            suffix = (
                ".TW"
                if info.market == "上市"
                else ".TWO"
            )

            ticker = (
                f"{code}{suffix}"
            )

            stocks_info[ticker] = {

                "code": code,

                "name": info.name,

                "group":
                    info.group
                    if info.group
                    else "其他",

                "market":
                    info.market
            }

    return stocks_info


# ============================================================
# 批量下載市場資料
#
# 重點：
# threads=False
#
# 不使用暴力多執行緒
# ============================================================

def download_market_data(
    tickers,
    period="1y"
):

    all_data = {}

    total = len(tickers)

    batches = [
        tickers[i:i + BATCH_SIZE]
        for i in range(
            0,
            total,
            BATCH_SIZE
        )
    ]

    batch_progress = st.progress(
        0
    )

    batch_status = st.empty()

    for idx, batch in enumerate(
        batches,
        start=1
    ):

        try:

            data = yf.download(

                batch,

                period=period,

                interval="1d",

                # 不使用自動還原
                # 避免股價與市值估算不一致
                auto_adjust=False,

                progress=False,

                group_by="ticker",

                threads=False
            )

            if (
                data is not None
                and not data.empty
            ):

                # =================================================
                # MultiIndex
                # =================================================

                if isinstance(
                    data.columns,
                    pd.MultiIndex
                ):

                    level0 = list(
                        data.columns
                        .get_level_values(0)
                    )

                    level1 = list(
                        data.columns
                        .get_level_values(1)
                    )

                    price_cols = {
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Adj Close",
                        "Volume"
                    }

                    # ------------------------------------------------
                    # ticker 在第一層
                    # ------------------------------------------------

                    if any(
                        ticker in level0
                        for ticker in batch
                    ):

                        for ticker in batch:

                            if ticker not in level0:
                                continue

                            try:

                                stock_df = (
                                    data[ticker]
                                    .copy()
                                )

                                if (
                                    stock_df is not None
                                    and not stock_df.empty
                                ):

                                    stock_df = (
                                        normalize_single_stock_df(
                                            stock_df
                                        )
                                    )

                                    all_data[
                                        ticker
                                    ] = stock_df

                            except Exception:
                                continue

                    # ------------------------------------------------
                    # ticker 在第二層
                    # ------------------------------------------------

                    elif any(
                        ticker in level1
                        for ticker in batch
                    ):

                        for ticker in batch:

                            if ticker not in level1:
                                continue

                            try:

                                stock_df = (
                                    data.xs(
                                        ticker,
                                        axis=1,
                                        level=1
                                    )
                                    .copy()
                                )

                                if (
                                    stock_df is not None
                                    and not stock_df.empty
                                ):

                                    stock_df = (
                                        normalize_single_stock_df(
                                            stock_df
                                        )
                                    )

                                    all_data[
                                        ticker
                                    ] = stock_df

                            except Exception:
                                continue

                    # ------------------------------------------------
                    # 某些 Yahoo 回傳格式：
                    # 第一層是價格欄位
                    # ------------------------------------------------

                    elif any(
                        x in price_cols
                        for x in level0
                    ):

                        if len(batch) == 1:

                            stock_df = data.copy()

                            stock_df = (
                                normalize_single_stock_df(
                                    stock_df
                                )
                            )

                            all_data[
                                batch[0]
                            ] = stock_df

                else:

                    # 單一股票
                    if len(batch) == 1:

                        stock_df = data.copy()

                        stock_df = (
                            normalize_single_stock_df(
                                stock_df
                            )
                        )

                        if not stock_df.empty:

                            all_data[
                                batch[0]
                            ] = stock_df

        except Exception:
            pass

        batch_progress.progress(
            idx / len(batches)
        )

        batch_status.text(
            f"資料下載："
            f"{idx}/{len(batches)} 批"
            f"｜成功：{len(all_data)} 支"
        )

        if idx < len(batches):

            time.sleep(
                BATCH_SLEEP
            )

    batch_status.empty()
    batch_progress.empty()

    return all_data


# ============================================================
# 單股技術分析
#
# 注意：
# 這裡不抓股本
#
# 股本會在技術面初篩後再抓
# 避免1925支股票逐支查基本資料
# ============================================================

def analyze_stock(
    ticker,
    name,
    group,
    market,
    df_day,
    params
):

    try:

        df_day = (
            normalize_single_stock_df(
                df_day
            )
        )

        df_day = (
            prepare_completed_daily_data(
                df_day
            )
        )

        if df_day.empty:
            return None

        required_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            col in df_day.columns
            for col in required_cols
        ):

            return None

        df_day = (
            df_day[
                required_cols
            ]
            .dropna()
            .copy()
        )

        min_required = max(
            120,
            params["breakout_days"] + 10,
            params["w_lookback"] + 10
        )

        if len(df_day) < min_required:
            return None

        close_day = (
            df_day["Close"]
            .to_numpy(
                dtype=float
            )
        )

        high_day = (
            df_day["High"]
            .to_numpy(
                dtype=float
            )
        )

        low_day = (
            df_day["Low"]
            .to_numpy(
                dtype=float
            )
        )

        vol_day = (
            df_day["Volume"]
            .to_numpy(
                dtype=float
            )
        )

        latest_close = float(
            close_day[-1]
        )

        latest_volume = float(
            vol_day[-1]
        )

        latest_volume_lots = (
            latest_volume / 1000
        )

        # ====================================================
        # 1. 最低成交量
        # ====================================================

        if (
            latest_volume_lots
            < MIN_VOLUME_LOTS
        ):

            return None

        # ====================================================
        # 2. 5日均量
        # ====================================================

        if len(vol_day) < 5:
            return None

        ma5_volume = (
            pd.Series(vol_day)
            .rolling(5)
            .mean()
            .iloc[-1]
        )

        if (
            not np.isfinite(
                ma5_volume
            )
            or ma5_volume <= 0
        ):

            return None

        volume_ratio = (
            latest_volume
            / ma5_volume
        )

        if (
            volume_ratio
            < params["vol_multiplier"]
        ):

            return None

        # ====================================================
        # 3. 完整週K
        # ====================================================

        df_week = (
            build_completed_weekly_data(
                df_day
            )
        )

        if (
            df_week.empty
            or len(df_week)
            < params["ma_week"]
        ):

            return None

        close_week = (
            df_week["Close"]
            .to_numpy(
                dtype=float
            )
        )

        # ====================================================
        # 4. 週MA
        # ====================================================

        ma_week = (
            pd.Series(close_week)
            .rolling(
                params["ma_week"]
            )
            .mean()
            .iloc[-1]
        )

        if not np.isfinite(
            ma_week
        ):

            return None

        latest_week_close = (
            close_week[-1]
        )

        if (
            latest_week_close
            <= ma_week
        ):

            return None

        # ====================================================
        # 5A. 40日創高
        # ====================================================

        breakout_days = int(
            params["breakout_days"]
        )

        if (
            len(close_day)
            <= breakout_days
        ):

            return None

        previous_highs = (
            high_day[
                -(breakout_days + 1):-1
            ]
        )

        previous_high = float(
            np.max(
                previous_highs
            )
        )

        is_breakout = (
            latest_close
            >= previous_high
        )

        breakout_distance = (
            (
                latest_close
                - previous_high
            )
            / previous_high
        ) * 100

        # ====================================================
        # 5B. W底
        # ====================================================

        w_info = detect_w_bottom(

            high_day=high_day,

            low_day=low_day,

            close_day=close_day,

            tolerance=
                params["w_tolerance"],

            lookback=
                params["w_lookback"],

            pivot_window=
                params["pivot_window"],

            min_gap=
                params["w_min_gap"],

            max_gap=
                params["w_max_gap"]
        )

        is_w_bottom = bool(
            w_info[
                "is_w_bottom"
            ]
        )

        # ====================================================
        # 6. 型態擇一
        # ====================================================

        if not (
            is_breakout
            or is_w_bottom
        ):

            return None

        # ====================================================
        # 訊號
        # ====================================================

        if (
            is_breakout
            and is_w_bottom
        ):

            signal_type = (
                "雙重訊號"
            )

        elif is_breakout:

            signal_type = (
                "40日創高"
            )

        else:

            signal_type = (
                "W底突破"
            )

        reasons = []

        if is_breakout:

            reasons.append(
                f"{breakout_days}日創高突破"
            )

        if is_w_bottom:

            reasons.append(
                "W底突破"
            )

        # ====================================================
        # 距離週MA
        # ====================================================

        distance_to_ma = (
            (
                latest_close
                - ma_week
            )
            / latest_close
        ) * 100

        # ====================================================
        # 資料日期
        # ====================================================

        data_date = (
            df_day.index[-1]
            .strftime(
                "%Y-%m-%d"
            )
        )

        return {

            "status":
                "match",

            "ticker":
                ticker,

            "name":
                name,

            "group":
                group,

            "market":
                market,

            "data_date":
                data_date,

            "df_day":
                df_day,

            "close":
                round(
                    latest_close,
                    2
                ),

            "volume":
                int(
                    latest_volume_lots
                ),

            "volume_avg_5":
                round(
                    ma5_volume / 1000,
                    0
                ),

            "volume_ratio":
                round(
                    volume_ratio,
                    2
                ),

            "ma_week_val":
                round(
                    float(ma_week),
                    2
                ),

            "distance_to_week_ma_pct":
                round(
                    float(
                        distance_to_ma
                    ),
                    2
                ),

            "previous_high":
                round(
                    previous_high,
                    2
                ),

            "breakout_distance_pct":
                round(
                    float(
                        breakout_distance
                    ),
                    2
                ),

            "is_breakout":
                is_breakout,

            "is_w_bottom":
                is_w_bottom,

            "signal_type":
                signal_type,

            "reasons":
                reasons,

            "w_info":
                w_info
        }

    except Exception:

        return None


# ============================================================
# 股本
#
# 技術面初篩後才抓
#
# 市值 ÷ 股價 × 10元
# ≈ 估算股本
#
# 注意：
# 這是篩選用途的估算值，不是財報上的精確實收資本額。
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def get_estimated_capital(
    ticker
):

    try:

        stock = yf.Ticker(
            ticker
        )

        market_cap = np.nan

        # ----------------------------------------------------
        # 優先 fast_info
        # ----------------------------------------------------

        try:

            fast_info = (
                stock.fast_info
            )

            market_cap = (
                fast_info.get(
                    "market_cap",
                    np.nan
                )
            )

        except Exception:

            market_cap = np.nan

        # ----------------------------------------------------
        # fast_info 沒有 → info
        # ----------------------------------------------------

        if (
            not np.isfinite(
                market_cap
            )
        ):

            try:

                info = stock.info

                market_cap = (
                    info.get(
                        "marketCap",
                        np.nan
                    )
                )

            except Exception:

                market_cap = np.nan

        if (
            market_cap is None
            or not np.isfinite(
                market_cap
            )
            or market_cap <= 0
        ):

            return np.nan

        # ----------------------------------------------------
        # 取得目前價格
        # ----------------------------------------------------

        try:

            current_price = (
                stock.fast_info.get(
                    "last_price",
                    np.nan
                )
            )

        except Exception:

            current_price = np.nan

        if (
            current_price is None
            or not np.isfinite(
                current_price
            )
            or current_price <= 0
        ):

            return np.nan

        # ----------------------------------------------------
        # 市值 / 股價 = 股數
        #
        # 股本 ≈ 股數 × 10元
        # ----------------------------------------------------

        shares = (
            market_cap
            / current_price
        )

        capital = (
            shares * 10
        )

        return float(
            capital
        )

    except Exception:

        return np.nan


# ============================================================
# 股本篩選
# ============================================================

def apply_capital_filter(
    matches,
    min_capital
):

    if not matches:

        return []

    capital_progress = st.progress(
        0
    )

    capital_status = st.empty()

    capital_matches = []

    total = len(matches)

    for idx, m in enumerate(
        matches,
        start=1
    ):

        capital = (
            get_estimated_capital(
                m["ticker"]
            )
        )

        m["estimated_capital"] = (
            capital
        )

        if (
            np.isfinite(capital)
            and capital >= min_capital
        ):

            capital_matches.append(
                m
            )

        capital_progress.progress(
            idx / total
        )

        capital_status.text(
            f"股本資料："
            f"{idx}/{total}"
            f"｜符合股本："
            f"{len(capital_matches)} 支"
        )

    capital_progress.empty()
    capital_status.empty()

    return capital_matches


# ============================================================
# 股利
#
# 最後才抓
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def get_dividend_history(
    ticker
):

    try:

        stock = yf.Ticker(
            ticker
        )

        dividends = (
            stock.dividends
        )

        if (
            dividends is None
            or dividends.empty
        ):

            return pd.DataFrame(
                columns=[
                    "年份",
                    "現金股利"
                ]
            )

        dividends = dividends.copy()

        dividends.index = (
            pd.to_datetime(
                dividends.index
            )
        )

        div_df = pd.DataFrame({

            "Dividend":
                dividends
        })

        div_df["Year"] = (
            div_df.index.year
        )

        yearly = (
            div_df
            .groupby("Year")[
                "Dividend"
            ]
            .sum()
            .reset_index()
        )

        yearly = (
            yearly
            .sort_values(
                "Year"
            )
            .tail(10)
        )

        yearly.columns = [
            "年份",
            "現金股利"
        ]

        yearly[
            "現金股利"
        ] = (
            yearly[
                "現金股利"
            ].round(2)
        )

        return yearly

    except Exception:

        return pd.DataFrame(
            columns=[
                "年份",
                "現金股利"
            ]
        )


# ============================================================
# 股利圖
# ============================================================

def plot_dividend_bar_chart(
    div_df
):

    fig, ax = plt.subplots(
        figsize=(10, 3.0)
    )

    years = (
        div_df["年份"]
        .astype(str)
        .tolist()
    )

    dividends = (
        div_df["現金股利"]
        .tolist()
    )

    bars = ax.bar(
        years,
        dividends,
        color="teal",
        alpha=0.85,
        width=0.6
    )

    for bar in bars:

        height = (
            bar.get_height()
        )

        ax.annotate(

            f"{height:g}",

            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height
            ),

            xytext=(0, 3),

            textcoords="offset points",

            ha="center",

            va="bottom",

            fontsize=9
        )

    ax.set_title(
        "近十年現金股利",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylabel(
        "現金股利（元）"
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# 台股 K線圖
#
# 重點修正：
#
# 1. 紅漲綠跌
# 2. 不把長篇文字塞進 title
# 3. 中文圖例
# 4. 簡短標題
# ============================================================

def plot_stock_chart(
    m
):

    ticker = m["ticker"]

    name = m["name"]

    df_day = m["df_day"]

    ma_week_val = (
        m["ma_week_val"]
    )

    plot_df = (
        df_day
        .iloc[-CHART_DAYS:]
        .copy()
    )

    plot_df = (
        normalize_single_stock_df(
            plot_df
        )
    )

    if plot_df.empty:
        return

    # --------------------------------------------------------
    # 日20MA
    # --------------------------------------------------------

    ma20 = (
        plot_df["Close"]
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # 日100MA
    # --------------------------------------------------------

    ma100 = (
        plot_df["Close"]
        .rolling(100)
        .mean()
    )

    # --------------------------------------------------------
    # 台股：
    #
    # 上漲 = 紅色
    # 下跌 = 綠色
    # --------------------------------------------------------

    market_colors = (
        mpf.make_marketcolors(

            up="red",

            down="green",

            edge="inherit",

            wick="inherit",

            volume="inherit"
        )
    )

    taiwan_style = (
        mpf.make_mpf_style(

            base_mpf_style="yahoo",

            marketcolors=
                market_colors
        )
    )

    addplots = [

        # 日20MA
        mpf.make_addplot(
            ma20,
            color="dodgerblue",
            width=1.3
        ),

        # 日100MA
        mpf.make_addplot(
            ma100,
            color="purple",
            width=1.4
        ),

        # 週20MA
        mpf.make_addplot(

            [
                ma_week_val
            ] * len(plot_df),

            color="red",

            linestyle="dashed",

            width=1.1
        )
    ]

    # --------------------------------------------------------
    # W底頸線
    # --------------------------------------------------------

    if m["is_w_bottom"]:

        neck = (
            m["w_info"]
            .get(
                "neck_high"
            )
        )

        if neck is not None:

            addplots.append(

                mpf.make_addplot(

                    [
                        neck
                    ] * len(plot_df),

                    color="orange",

                    linestyle="dashdot",

                    width=1.1
                )
            )

    # --------------------------------------------------------
    # 簡短標題
    #
    # 不再把收盤、MA、距離、訊號全部塞到標題
    # 避免K線圖被壓縮
    # --------------------------------------------------------

    code = (
        ticker
        .split(".")[0]
    )

    title = (
        f"{code} {name}"
    )

    # --------------------------------------------------------
    # K線
    # --------------------------------------------------------

    fig, axes = mpf.plot(

        plot_df,

        type="candle",

        style=taiwan_style,

        addplot=addplots,

        title=title,

        ylabel="價格（元）",

        volume=True,

        ylabel_lower="成交量",

        figratio=(15, 8),

        figscale=1.35,

        returnfig=True,

        tight_layout=True
    )

    # --------------------------------------------------------
    # 中文圖例
    # --------------------------------------------------------

    if axes:

        ax = axes[0]

        from matplotlib.lines import Line2D

        legend_items = [

            Line2D(
                [0],
                [0],
                color="dodgerblue",
                lw=2,
                label="日20MA"
            ),

            Line2D(
                [0],
                [0],
                color="purple",
                lw=2,
                label="日100MA"
            ),

            Line2D(
                [0],
                [0],
                color="red",
                lw=2,
                linestyle="--",
                label="週20MA"
            )
        ]

        if m["is_w_bottom"]:

            legend_items.append(

                Line2D(
                    [0],
                    [0],
                    color="orange",
                    lw=2,
                    linestyle="-.",
                    label="W底頸線"
                )
            )

        ax.legend(
            handles=legend_items,
            loc="upper left",
            fontsize=9,
            frameon=True
        )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "🔍 V2.1 全市場選股控制台"
)

st.sidebar.info(
    "本版本固定掃描：\n\n"
    "🇹🇼 全台上市＋上櫃\n\n"
    "不再限制成交金額前150大"
)

st.sidebar.divider()

st.sidebar.subheader(
    "⚙️ 技術策略參數"
)


# ============================================================
# 股本
# ============================================================

min_capital_billion = (
    st.sidebar.number_input(

        "最低股本（億元）",

        min_value=0.0,

        max_value=1000.0,

        value=10.0,

        step=1.0
    )
)


# ============================================================
# 成交量
# ============================================================

vol_multiplier = (
    st.sidebar.slider(

        "放量倍數（最新完成交易日 vs 5日均量）",

        min_value=1.0,

        max_value=3.0,

        value=1.5,

        step=0.1
    )
)


# ============================================================
# 突破
# ============================================================

breakout_days = (
    st.sidebar.number_input(

        "突破回看期間（交易日）",

        min_value=10,

        max_value=60,

        value=40,

        step=1
    )
)


# ============================================================
# 週MA
# ============================================================

ma_week = (
    st.sidebar.number_input(

        "長期趨勢均線（週MA）",

        min_value=10,

        max_value=40,

        value=20,

        step=1
    )
)


# ============================================================
# W底
# ============================================================

st.sidebar.subheader(
    "🔵 W底參數"
)

w_tolerance = (
    st.sidebar.slider(

        "W底左右腳容錯率",

        min_value=1.0,

        max_value=15.0,

        value=6.0,

        step=0.5
    )
    / 100
)

pivot_window = (
    st.sidebar.number_input(

        "Pivot Low 判定寬度",

        min_value=2,

        max_value=6,

        value=3,

        step=1
    )
)

w_min_gap = (
    st.sidebar.number_input(

        "W底左右腳最小間隔",

        min_value=5,

        max_value=15,

        value=7,

        step=1
    )
)

w_max_gap = (
    st.sidebar.number_input(

        "W底左右腳最大間隔",

        min_value=20,

        max_value=45,

        value=35,

        step=1
    )

)

# 固定60日
w_lookback = 60


# ============================================================
# Parameters
# ============================================================

params = {

    "min_capital":
        min_capital_billion
        * 100_000_000,

    "vol_multiplier":
        vol_multiplier,

    "breakout_days":
        int(
            breakout_days
        ),

    "ma_week":
        int(
            ma_week
        ),

    "w_tolerance":
        w_tolerance,

    "pivot_window":
        int(
            pivot_window
        ),

    "w_min_gap":
        int(
            w_min_gap
        ),

    "w_max_gap":
        int(
            w_max_gap
        ),

    "w_lookback":
        w_lookback
}


st.sidebar.divider()

st.sidebar.markdown(
    """
### 📌 V2.1 選股邏輯

**第一層：快速技術篩選**

1. 最新完成交易日成交量 ≥ 1,000張
2. 最新完成交易日成交量 ≥ 5日均量 × 設定倍數
3. 股價 > 完整週20MA

**第二層：型態**

- 40日創高
- W底突破

兩者擇一即可。

兩者同時成立：

**🔥 雙重訊號**

**第三層：股本**

只對技術面入選股票抓市值，
估算股本後再套用最低股本。

**第四層：股利**

最後才抓股利資料。
"""
)


# ============================================================
# 開始掃描
# ============================================================

if st.sidebar.button(
    "🚀 開始 V2.1 全市場掃描",
    type="primary"
):

    overall_start = time.time()

    # ========================================================
    # 取得股票清單
    # ========================================================

    stocks_info = (
        get_all_tw_stocks_info()
    )

    all_tickers = list(
        stocks_info.keys()
    )

    st.info(
        f"🇹🇼 全台上市＋上櫃"
        f"｜目前股票清單："
        f"**{len(all_tickers)} 支**"
    )

    # ========================================================
    # 第一階段
    # 批量下載
    # ========================================================

    st.subheader(
        "📥 第一階段：批量取得市場資料"
    )

    start_time = time.time()

    market_data = (
        download_market_data(
            all_tickers,
            period="1y"
        )
    )

    download_elapsed = (
        time.time()
        - start_time
    )

    st.success(
        f"資料下載完成："
        f"**{len(market_data)} 支**"
        f"｜耗時 "
        f"{download_elapsed:.1f} 秒"
    )

    # ========================================================
    # 第二階段
    # 快速技術篩選
    # ========================================================

    st.subheader(
        "🔎 第二階段：快速技術篩選"
    )

    matches = []

    total = len(
        market_data
    )

    if total == 0:

        st.error(
            "⚠️ Yahoo Finance 沒有成功取得股票資料。"
            "這通常是資料來源暫時限制或批量下載格式問題，"
            "不是選股條件本身造成的。"
        )

        st.stop()

    progress = st.progress(
        0
    )

    status = st.empty()

    for idx, ticker in enumerate(
        market_data.keys(),
        start=1
    ):

        info = stocks_info.get(
            ticker
        )

        if info is None:
            continue

        df = market_data[
            ticker
        ]

        result = analyze_stock(

            ticker=ticker,

            name=info["name"],

            group=info["group"],

            market=info["market"],

            df_day=df,

            params=params
        )

        if result is not None:

            matches.append(
                result
            )

        if (
            idx % 50 == 0
            or idx == total
        ):

            progress.progress(
                idx / total
            )

            status.text(
                f"分析："
                f"{idx}/{total}"
                f"｜技術面符合："
                f"{len(matches)} 支"
            )

    progress.progress(
        1.0
    )

    status.text(
        "技術篩選完成！"
    )

    # ========================================================
    # 技術面沒有股票
    # ========================================================

    if not matches:

        elapsed = (
            time.time()
            - overall_start
        )

        st.warning(
            "ℹ️ 目前參數設定下，"
            "沒有股票通過技術面條件。"
        )

        st.info(
            f"掃描 {len(market_data)} 支"
            f"｜耗時 {elapsed:.1f} 秒"
        )

        st.stop()

    # ========================================================
    # 第三階段
    # 股本篩選
    # ========================================================

    st.subheader(
        "💰 第三階段：股本篩選"
    )

    capital_start = time.time()

    capital_matches = (
        apply_capital_filter(
            matches,
            params["min_capital"]
        )
    )

    capital_elapsed = (
        time.time()
        - capital_start
    )

    st.success(
        f"股本篩選完成："
        f"技術面 {len(matches)} 支"
        f" → 股本符合 "
        f"**{len(capital_matches)} 支**"
        f"｜耗時 "
        f"{capital_elapsed:.1f} 秒"
    )

    matches = capital_matches

    # ========================================================
    # 股本全部淘汰
    # ========================================================

    if not matches:

        elapsed = (
            time.time()
            - overall_start
        )

        st.warning(
            "ℹ️ 技術面有符合股票，"
            "但全部未達目前設定的最低股本。"
        )

        st.info(
            f"最低股本："
            f"{min_capital_billion:.1f} 億元"
            f"｜總耗時："
            f"{elapsed:.1f} 秒"
        )

        st.stop()

    # ========================================================
    # 排序
    # ========================================================

    matches.sort(

        key=lambda x: (

            (
                x["is_breakout"]
                + x["is_w_bottom"]
            ),

            x["volume_ratio"],

            x[
                "distance_to_week_ma_pct"
            ]
        ),

        reverse=True
    )

    # ========================================================
    # 第四階段
    # 最後才抓股利
    # ========================================================

    st.subheader(
        "💵 第四階段：補充股利資料"
    )

    dividend_progress = st.progress(
        0
    )

    dividend_status = st.empty()

    for idx, m in enumerate(
        matches,
        start=1
    ):

        m["div_history"] = (
            get_dividend_history(
                m["ticker"]
            )
        )

        dividend_progress.progress(
            idx / len(matches)
        )

        dividend_status.text(
            f"股利資料："
            f"{idx}/{len(matches)}"
        )

        time.sleep(
            0.08
        )

    dividend_progress.empty()
    dividend_status.empty()

    # ========================================================
    # 完成
    # ========================================================

    elapsed = (
        time.time()
        - overall_start
    )

    st.success(
        f"🎉 V2.1 掃描完成！"
        f"｜掃描 {len(market_data)} 支"
        f"｜技術面 {len(capital_matches)} 支"
        f"｜最終入選 {len(matches)} 支"
        f"｜耗時 {elapsed:.1f} 秒"
    )

    # ========================================================
    # 產業集中統計
    # ========================================================

    st.divider()

    st.subheader(
        "🔥 市場資金集中產業"
    )

    industry_df = (
        pd.DataFrame([

            {
                "產業":
                    m["group"],

                "入選家數":
                    1,

                "雙重訊號":
                    int(
                        m["signal_type"]
                        == "雙重訊號"
                    )
            }

            for m in matches
        ])
        .groupby(
            "產業",
            as_index=False
        )
        .sum()
        .sort_values(

            [
                "入選家數",
                "雙重訊號"
            ],

            ascending=False
        )
    )

    industry_df[
        "排名"
    ] = range(
        1,
        len(industry_df) + 1
    )

    industry_df = (
        industry_df[
            [
                "排名",
                "產業",
                "入選家數",
                "雙重訊號"
            ]
        ]
    )

    st.dataframe(
        industry_df,
        use_container_width=True,
        hide_index=True
    )

    if not industry_df.empty:

        top_industry = (
            industry_df.iloc[0]
        )

        st.info(
            f"🔥 目前入選最多的產業："
            f"**{top_industry['產業']}**"
            f"｜共 "
            f"**{top_industry['入選家數']}** 支"
        )

    # ========================================================
    # 總覽表
    # ========================================================

    st.divider()

    st.subheader(
        f"📊 入選股票總覽 "
        f"（共 {len(matches)} 支）"
    )

    summary_rows = []

    sorted_matches = sorted(

        matches,

        key=lambda x: (

            x["group"],

            -(
                x["is_breakout"]
                + x["is_w_bottom"]
            ),

            -x["volume_ratio"]
        )
    )

    for m in sorted_matches:

        capital_billion = (
            m["estimated_capital"]
            / 100_000_000
        )

        summary_rows.append({

            "產業":
                m["group"],

            "股票":
                f"{m['name']} "
                f"({m['ticker'].split('.')[0]})",

            "市場":
                m["market"],

            "股本":
                f"{capital_billion:.1f} 億",

            "收盤":
                m["close"],

            f"週{params['ma_week']}MA":
                m["ma_week_val"],

            "距週MA":
                f"{m['distance_to_week_ma_pct']:.2f}%",

            "最新成交量":
                f"{m['volume']:,} 張",

            "5日均量":
                f"{m['volume_avg_5']:,.0f} 張",

            "放量":
                f"{m['volume_ratio']:.2f}x",

            "40日創高":
                "✅"
                if m["is_breakout"]
                else "—",

            "W底突破":
                "✅"
                if m["is_w_bottom"]
                else "—",

            "訊號":
                m["signal_type"],

            "資料日期":
                m["data_date"]
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # 各產業詳細股票
    # ========================================================

    st.divider()

    st.subheader(
        "🏭 產業分組結果"
    )

    for industry in (
        industry_df["產業"]
    ):

        industry_matches = [

            m

            for m in sorted_matches

            if m["group"]
            == industry
        ]

        st.markdown(
            f"### 🏭 {industry}"
            f"　"
            f"({len(industry_matches)} 支)"
        )

        industry_rows = []

        for m in industry_matches:

            capital_billion = (
                m["estimated_capital"]
                / 100_000_000
            )

            industry_rows.append({

                "股票":
                    f"{m['name']} "
                    f"({m['ticker'].split('.')[0]})",

                "股本":
                    f"{capital_billion:.1f} 億",

                "收盤":
                    m["close"],

                "放量":
                    f"{m['volume_ratio']:.2f}x",

                "距週MA":
                    f"{m['distance_to_week_ma_pct']:.2f}%",

                "40日創高":
                    "✅"
                    if m["is_breakout"]
                    else "—",

                "W底突破":
                    "✅"
                    if m["is_w_bottom"]
                    else "—",

                "訊號":
                    m["signal_type"]
            })

        st.dataframe(

            pd.DataFrame(
                industry_rows
            ),

            use_container_width=True,

            hide_index=True
        )

    # ========================================================
    # 個股詳細分析
    # ========================================================

    st.divider()

    st.subheader(
        "📌 個股詳細分析"
    )

    for m in sorted_matches:

        st.markdown(
            f"## 📌 {m['name']} "
            f"({m['ticker'].split('.')[0]})"
        )

        capital_billion = (
            m["estimated_capital"]
            / 100_000_000
        )

        st.markdown(
            f"**{m['market']}｜"
            f"產業：{m['group']}｜"
            f"股本：約 "
            f"{capital_billion:.1f} 億元｜"
            f"資料日期：{m['data_date']}**"
        )

        # ====================================================
        # 第一排
        # ====================================================

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "收盤價",
                f"{m['close']:.2f} 元"
            )

        with c2:

            st.metric(
                f"週{params['ma_week']}MA",
                f"{m['ma_week_val']:.2f} 元"
            )

        with c3:

            st.metric(
                "距週MA",
                f"{m['distance_to_week_ma_pct']:.2f}%"
            )

        with c4:

            st.metric(
                "股本",
                f"{capital_billion:.1f} 億"
            )

        # ====================================================
        # 第二排
        # ====================================================

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "最新成交量",
                f"{m['volume']:,} 張"
            )

        with c2:

            st.metric(
                "5日均量",
                f"{m['volume_avg_5']:,.0f} 張"
            )

        with c3:

            st.metric(
                "實際放量",
                f"{m['volume_ratio']:.2f}x"
            )

        with c4:

            st.metric(
                f"{params['breakout_days']}日高",
                f"{m['previous_high']:.2f}"
            )

        # ====================================================
        # 第三排
        # ====================================================

        c1, c2 = (
            st.columns(2)
        )

        with c1:

            st.metric(
                "突破幅度",
                f"{m['breakout_distance_pct']:.2f}%"
            )

        with c2:

            st.metric(
                "訊號",
                m["signal_type"]
            )

        # ====================================================
        # 入選原因
        # ====================================================

        st.markdown(
            "### 🎯 入選原因"
        )

        for reason in m["reasons"]:

            st.success(
                f"✅ {reason}"
            )

        # ====================================================
        # W底
        # ====================================================

        if m["is_w_bottom"]:

            w = m["w_info"]

            st.markdown(
                "### 🔵 W底結構"
            )

            w1, w2, w3, w4 = (
                st.columns(4)
            )

            with w1:

                st.metric(
                    "左腳",
                    f"{w['left_foot']:.2f}"
                )

            with w2:

                st.metric(
                    "右腳",
                    f"{w['right_foot']:.2f}"
                )

            with w3:

                st.metric(
                    "頸線",
                    f"{w['neck_high']:.2f}"
                )

            with w4:

                st.metric(
                    "左右腳差異",
                    f"{w['foot_diff_pct']:.2f}%"
                )

        # ====================================================
        # 週MA風險
        # ====================================================

        distance = (
            m[
                "distance_to_week_ma_pct"
            ]
        )

        if distance < 3:

            risk_label = (
                "🔴 非常接近週MA"
            )

        elif distance < 7:

            risk_label = (
                "🟡 距週MA適中"
            )

        elif distance < 12:

            risk_label = (
                "🟢 距週MA較寬"
            )

        else:

            risk_label = (
                "🟠 距週MA過遠，注意追高"
            )

        st.markdown(
            f"""
**🛡️ 週{params['ma_week']}MA：**
{m['ma_week_val']:.2f} 元

**📏 距離週MA：**
{distance:.2f}%

**風險位置：**
{risk_label}

> 週MA僅作為技術面停損參考，
> 不代表實際最大損失。
"""
        )

        # ====================================================
        # 股利
        # ====================================================

        if (
            not m[
                "div_history"
            ].empty
        ):

            st.markdown(
                "### 💰 近十年現金股利"
            )

            plot_dividend_bar_chart(
                m["div_history"]
            )

        else:

            st.info(
                "沒有可取得的近期股利資料。"
            )

        # ====================================================
        # K線
        # ====================================================

        st.markdown(
            "### 📈 技術K線"
        )

        plot_stock_chart(
            m
        )

        st.divider()
