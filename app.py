import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import twstock
import warnings
import time
from datetime import datetime, time as dt_time

# ============================================================
# 基本設定
# ============================================================

warnings.filterwarnings("default")

st.set_page_config(
    page_title="台股 V2 強勢突破全自動雷達",
    layout="wide"
)

st.title("📈 台股 V2 全自動選股雷達")
st.caption(
    "V2 正確版：已完成交易日＋已完成週K＋週20MA趨勢＋成交量＋突破/W底 "
    "＋完整入選原因與風險資訊"
)

# ============================================================
# 常數
# ============================================================

TW_TZ = "Asia/Taipei"
MIN_VOLUME_LOTS = 1000
DAILY_LOOKBACK = 2
CHART_DAYS = 250

# ============================================================
# 工具函數
# ============================================================

def flatten_yfinance_columns(df):
    """
    統一處理 yfinance 可能產生的 MultiIndex。
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # 優先尋找標準 OHLCV 欄位
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        elif "Close" in df.columns.get_level_values(-1):
            df.columns = df.columns.get_level_values(-1)
        else:
            df.columns = df.columns.get_level_values(0)

    return df


def get_taiwan_now():
    """
    取得台灣目前時間。
    """
    return pd.Timestamp.now(tz=TW_TZ)


def is_market_closed_for_today():
    """
    台股現貨正常交易日大約 13:30 收盤。
    這裡只用於判斷最後一筆日K是否可能仍是未完成的今日K。
    """
    now = get_taiwan_now()

    # 週六、週日
    if now.weekday() >= 5:
        return True

    market_close = dt_time(13, 30)

    return now.time() >= market_close


def prepare_completed_daily_data(df_day):
    """
    只保留已完成的日K。

    如果今天尚未收盤，而且 Yahoo 已經提供今天的資料，
    就把今天這根未完成K排除。
    """
    if df_day is None or df_day.empty:
        return pd.DataFrame()

    df_day = df_day.copy()
    df_day.index = pd.to_datetime(df_day.index)

    # 如果今天尚未收盤，且最後一筆就是今天
    now = get_taiwan_now()
    today = now.date()

    last_date = df_day.index[-1].date()

    if last_date == today and not is_market_closed_for_today():
        df_day = df_day.iloc[:-1].copy()

    return df_day


def build_completed_weekly_data(df_day):
    """
    從已完成日K重新建立週K。

    使用 W-FRI：
    每週五作為一週結束。

    由於來源日K已經排除尚未完成的今天，
    因此不會把本週尚未結束的週K拿來做週MA判斷。
    """
    if df_day is None or df_day.empty:
        return pd.DataFrame()

    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in df_day.columns for col in required_cols):
        return pd.DataFrame()

    weekly = df_day[required_cols].resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    weekly = weekly.dropna(subset=["Open", "High", "Low", "Close"])

    return weekly


def calculate_pivot_lows(low_values, pivot_window=3):
    """
    找出真正具有局部低點特徵的 Pivot Low。

    Pivot Low 定義：
    該點的 Low 必須 <= 左右 pivot_window 根K線的 Low。
    """
    lows = np.asarray(low_values, dtype=float)

    pivot_indices = []

    if len(lows) < pivot_window * 2 + 1:
        return pivot_indices

    for i in range(
        pivot_window,
        len(lows) - pivot_window
    ):
        left = lows[i - pivot_window:i]
        right = lows[i + 1:i + pivot_window + 1]

        if lows[i] <= np.min(left) and lows[i] <= np.min(right):
            pivot_indices.append(i)

    return pivot_indices


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
    """
    V2 W底辨識。

    條件：

    1. 最近 lookback 天內找 Pivot Low
    2. 左右腳必須分別位於前後區域
    3. 左右腳間距合理
    4. 左右腳價格差在 tolerance 內
    5. 左右腳中間必須形成明顯頸線
    6. 今日收盤必須突破頸線
    7. 右腳之後必須有一定反彈
    """

    if len(low_day) < lookback:
        return {
            "is_w_bottom": False,
            "left_idx": None,
            "right_idx": None,
            "left_foot": None,
            "right_foot": None,
            "neck_high": None
        }

    highs = np.asarray(high_day[-lookback:], dtype=float)
    lows = np.asarray(low_day[-lookback:], dtype=float)
    closes = np.asarray(close_day[-lookback:], dtype=float)

    pivot_lows = calculate_pivot_lows(
        lows,
        pivot_window=pivot_window
    )

    if len(pivot_lows) < 2:
        return {
            "is_w_bottom": False,
            "left_idx": None,
            "right_idx": None,
            "left_foot": None,
            "right_foot": None,
            "neck_high": None
        }

    latest_close = closes[-1]

    candidates = []

    for left_idx in pivot_lows:

        # 左腳不能太靠近右邊
        if left_idx >= lookback // 2:
            continue

        for right_idx in pivot_lows:

            if right_idx <= left_idx:
                continue

            # 右腳不能太靠近今天
            if right_idx >= lookback - 5:
                continue

            gap = right_idx - left_idx

            if gap < min_gap or gap > max_gap:
                continue

            left_foot = lows[left_idx]
            right_foot = lows[right_idx]

            if left_foot <= 0:
                continue

            # 使用左右腳平均價格作為容錯率分母
            avg_foot = (left_foot + right_foot) / 2

            foot_diff_pct = abs(
                left_foot - right_foot
            ) / avg_foot

            if foot_diff_pct > tolerance:
                continue

            # 頸線：左右腳之間的最高價
            between_highs = highs[left_idx:right_idx + 1]

            if len(between_highs) == 0:
                continue

            neck_high = np.max(between_highs)

            # 頸線必須明顯高於兩腳
            if neck_high <= max(left_foot, right_foot):
                continue

            # 今日必須突破頸線
            if latest_close <= neck_high:
                continue

            # 右腳之後必須有反彈
            right_after = closes[right_idx:]

            if len(right_after) < 2:
                continue

            right_rebound_high = np.max(right_after)

            if right_rebound_high <= right_foot:
                continue

            # 計算右腳到頸線的反彈空間
            right_to_neck_pct = (
                (neck_high - right_foot)
                / right_foot
            ) * 100

            if right_to_neck_pct < 3:
                continue

            candidates.append({
                "left_idx": left_idx,
                "right_idx": right_idx,
                "left_foot": float(left_foot),
                "right_foot": float(right_foot),
                "neck_high": float(neck_high),
                "foot_diff_pct": float(foot_diff_pct * 100)
            })

    if not candidates:
        return {
            "is_w_bottom": False,
            "left_idx": None,
            "right_idx": None,
            "left_foot": None,
            "right_foot": None,
            "neck_high": None
        }

    # 優先選擇右腳最靠近目前、且左右腳差異最小的型態
    candidates.sort(
        key=lambda x: (
            -x["right_idx"],
            x["foot_diff_pct"]
        )
    )

    best = candidates[0]

    return {
        "is_w_bottom": True,
        "left_idx": best["left_idx"],
        "right_idx": best["right_idx"],
        "left_foot": round(best["left_foot"], 2),
        "right_foot": round(best["right_foot"], 2),
        "neck_high": round(best["neck_high"], 2),
        "foot_diff_pct": round(best["foot_diff_pct"], 2)
    }


# ============================================================
# 股票清單
# ============================================================

@st.cache_data(ttl=86400)
def get_all_tw_stocks_info():

    stocks_info = {}

    for code, info in twstock.codes.items():

        if (
            info.type == "股票"
            and info.market in ["上市", "上櫃"]
        ):

            suffix = (
                ".TW"
                if info.market == "上市"
                else ".TWO"
            )

            ticker = f"{code}{suffix}"

            stocks_info[ticker] = {
                "code": code,
                "name": info.name,
                "group": (
                    info.group
                    if info.group
                    else "其他"
                ),
                "market": info.market
            }

    return stocks_info


# ============================================================
# K線圖
# ============================================================

def plot_stock_chart(
    ticker,
    df_day,
    ma_week_val,
    breakout_days,
    is_breakout,
    w_info
):

    plot_df = df_day.iloc[-CHART_DAYS:].copy()

    if plot_df.empty:
        return

    plot_df = flatten_yfinance_columns(plot_df)

    ma20 = plot_df["Close"].rolling(20).mean()
    ma100 = plot_df["Close"].rolling(100).mean()

    addplots = [
        mpf.make_addplot(
            ma20,
            color="dodgerblue",
            width=1.5
        ),

        mpf.make_addplot(
            ma100,
            color="purple",
            width=1.8
        ),

        # 最新週20MA停損參考線
        mpf.make_addplot(
            [ma_week_val] * len(plot_df),
            color="red",
            linestyle="dashed",
            width=1.2
        )
    ]

    # 如果是 W底，把頸線畫出來
    if w_info.get("is_w_bottom"):

        neck_high = w_info.get("neck_high")

        if neck_high is not None:

            addplots.append(
                mpf.make_addplot(
                    [neck_high] * len(plot_df),
                    color="orange",
                    linestyle="dashdot",
                    width=1.2
                )
            )

    title_parts = [
        f"{ticker} - V2 Trend Radar",
        f"Weekly MA20 Stop: {ma_week_val:.2f}"
    ]

    if is_breakout:
        title_parts.append(
            f"{breakout_days}D Breakout"
        )

    if w_info.get("is_w_bottom"):
        title_parts.append(
            f"W-Bottom Neckline: {w_info['neck_high']:.2f}"
        )

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style="yahoo",
        addplot=addplots,
        title="\n" + " | ".join(title_parts),
        ylabel="Price (TWD)",
        volume=True,
        ylabel_lower="Volume",
        figratio=(16, 9),
        figscale=1.1,
        returnfig=True
    )

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 股利圖
# ============================================================

def plot_dividend_bar_chart(div_df):

    fig, ax = plt.subplots(
        figsize=(10, 3.5)
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

        height = bar.get_height()

        ax.annotate(
            f"{height}",
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
        "Recent 10-Year Cash Dividend",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Dividend (TWD)"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 股利資料
# ============================================================

def get_dividend_history(stock_obj):

    try:

        dividends = stock_obj.dividends

        if dividends is None or dividends.empty:

            return pd.DataFrame(
                columns=[
                    "年份",
                    "現金股利"
                ]
            )

        dividends = dividends.copy()

        dividends.index = pd.to_datetime(
            dividends.index
        )

        div_df = pd.DataFrame({
            "Dividend": dividends
        })

        div_df["Year"] = (
            div_df.index.year
        )

        yearly_div = (
            div_df
            .groupby("Year")["Dividend"]
            .sum()
            .reset_index()
        )

        yearly_div = (
            yearly_div
            .sort_values(
                by="Year",
                ascending=True
            )
            .tail(10)
        )

        yearly_div.columns = [
            "年份",
            "現金股利"
        ]

        yearly_div["現金股利"] = (
            yearly_div["現金股利"]
            .round(2)
        )

        return yearly_div

    except Exception:
        return pd.DataFrame(
            columns=[
                "年份",
                "現金股利"
            ]
        )


# ============================================================
# 核心選股策略
# ============================================================

def run_strategy(
    ticker,
    name,
    group,
    market,
    params
):

    try:

        stock_obj = yf.Ticker(ticker)

        # ----------------------------------------------------
        # 只抓 2 年日線
        # 不再另外抓一次週線
        # ----------------------------------------------------

        df_day = stock_obj.history(
            period="2y",
            interval="1d",
            auto_adjust=True
        )

        if df_day is None or df_day.empty:
            return {
                "status": "error",
                "ticker": ticker,
                "error": "Yahoo 無日線資料"
            }

        df_day = flatten_yfinance_columns(
            df_day
        )

        # ----------------------------------------------------
        # 只使用已完成交易日
        # ----------------------------------------------------

        df_day = prepare_completed_daily_data(
            df_day
        )

        if df_day.empty:
            return {
                "status": "error",
                "ticker": ticker,
                "error": "沒有已完成交易日資料"
            }

        # ----------------------------------------------------
        # 基本欄位確認
        # ----------------------------------------------------

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
            return {
                "status": "error",
                "ticker": ticker,
                "error": "缺少 OHLCV 欄位"
            }

        # ----------------------------------------------------
        # 資料清理
        # ----------------------------------------------------

        df_day = df_day[
            required_cols
        ].copy()

        df_day = df_day.dropna(
            subset=[
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        )

        if len(df_day) < max(
            120,
            params["breakout_days"] + 10,
            params["w_lookback"] + 10
        ):
            return {
                "status": "error",
                "ticker": ticker,
                "error": "有效日線資料不足"
            }

        # ====================================================
        # 產生週K
        # ====================================================

        df_week = build_completed_weekly_data(
            df_day
        )

        if (
            df_week.empty
            or len(df_week) < params["ma_week"]
        ):
            return {
                "status": "error",
                "ticker": ticker,
                "error": "完整週K資料不足"
            }

        # ====================================================
        # 轉成 numpy
        # ====================================================

        close_day = (
            df_day["Close"]
            .to_numpy(dtype=float)
        )

        high_day = (
            df_day["High"]
            .to_numpy(dtype=float)
        )

        low_day = (
            df_day["Low"]
            .to_numpy(dtype=float)
        )

        vol_day = (
            df_day["Volume"]
            .to_numpy(dtype=float)
        )

        close_week = (
            df_week["Close"]
            .to_numpy(dtype=float)
        )

        # ====================================================
        # 1. 週20MA 長期趨勢
        # ====================================================

        ma_week_series = (
            pd.Series(close_week)
            .rolling(
                params["ma_week"]
            )
            .mean()
        )

        ma_week_val = (
            ma_week_series.iloc[-1]
        )

        latest_week_close = (
            close_week[-1]
        )

        if not np.isfinite(
            ma_week_val
        ):
            return {
                "status": "error",
                "ticker": ticker,
                "error": "週MA計算失敗"
            }

        trend_pass = (
            latest_week_close
            > ma_week_val
        )

        if not trend_pass:
            return None

        # ====================================================
        # 最新完成交易日
        # ====================================================

        latest_close = float(
            close_day[-1]
        )

        latest_volume_shares = float(
            vol_day[-1]
        )

        latest_vol_lots = (
            latest_volume_shares / 1000
        )

        data_date = (
            df_day.index[-1]
            .strftime("%Y-%m-%d")
        )

        # ====================================================
        # 2. 最低成交量
        # ====================================================

        if (
            latest_vol_lots
            < MIN_VOLUME_LOTS
        ):
            return None

        # ====================================================
        # 3. 20日均量
        # ====================================================

        ma_vol_val = (
            pd.Series(vol_day)
            .rolling(20)
            .mean()
            .iloc[-1]
        )

        if (
            not np.isfinite(ma_vol_val)
            or ma_vol_val <= 0
        ):
            return {
                "status": "error",
                "ticker": ticker,
                "error": "20日均量計算失敗"
            }

        actual_volume_ratio = (
            latest_volume_shares
            / ma_vol_val
        )

        volume_pass = (
            actual_volume_ratio
            >= params["vol_multiplier"]
        )

        if not volume_pass:
            return None

        # ====================================================
        # 4A. 區間創高
        #
        # 「突破40日」真正比較前40個完整交易日
        # 不包含今天
        # ====================================================

        breakout_days = (
            params["breakout_days"]
        )

        if len(close_day) <= breakout_days:
            return {
                "status": "error",
                "ticker": ticker,
                "error": "突破回看資料不足"
            }

        previous_highs = (
            high_day[
                -(breakout_days + 1):-1
            ]
        )

        previous_high = (
            np.max(previous_highs)
        )

        is_breakout = (
            latest_close
            >= previous_high
        )

        breakout_distance_pct = (
            (
                latest_close
                - previous_high
            )
            / previous_high
            * 100
        )

        # ====================================================
        # 4B. V2 W底
        # ====================================================

        w_info = detect_w_bottom(
            high_day=high_day,
            low_day=low_day,
            close_day=close_day,
            tolerance=params["w_tolerance"],
            lookback=params["w_lookback"],
            pivot_window=params["pivot_window"],
            min_gap=params["w_min_gap"],
            max_gap=params["w_max_gap"]
        )

        is_w_bottom = (
            w_info["is_w_bottom"]
        )

        # ====================================================
        # 4C. 型態擇一
        # ====================================================

        if not (
            is_breakout
            or is_w_bottom
        ):
            return None

        # ====================================================
        # 入選原因
        # ====================================================

        reasons = []

        if is_breakout:
            reasons.append(
                f"{breakout_days}日創高突破"
            )

        if is_w_bottom:
            reasons.append(
                "W底突破"
            )

        if is_breakout and is_w_bottom:
            signal_type = "雙重訊號"
        elif is_breakout:
            signal_type = "區間創高"
        else:
            signal_type = "W底突破"

        # ====================================================
        # 距離週MA
        # ====================================================

        distance_to_week_ma_pct = (
            (
                latest_close
                - ma_week_val
            )
            / latest_close
            * 100
        )

        # ====================================================
        # 股利
        # ====================================================

        div_history_df = (
            get_dividend_history(
                stock_obj
            )
        )

        # ====================================================
        # 回傳完整結果
        # ====================================================

        return {
            "status": "match",

            "ticker": ticker,
            "name": name,
            "group": group,
            "market": market,

            "data_date": data_date,

            "df_day": df_day,

            "close": round(
                latest_close,
                2
            ),

            "volume": int(
                latest_vol_lots
            ),

            "volume_avg_20": round(
                ma_vol_val / 1000,
                0
            ),

            "volume_ratio": round(
                actual_volume_ratio,
                2
            ),

            "ma_week_val": round(
                float(ma_week_val),
                2
            ),

            "distance_to_week_ma_pct": round(
                float(
                    distance_to_week_ma_pct
                ),
                2
            ),

            "previous_high": round(
                float(previous_high),
                2
            ),

            "breakout_distance_pct": round(
                float(
                    breakout_distance_pct
                ),
                2
            ),

            "is_breakout": bool(
                is_breakout
            ),

            "is_w_bottom": bool(
                is_w_bottom
            ),

            "signal_type": signal_type,

            "reasons": reasons,

            "w_info": w_info,

            "div_history": div_history_df
        }

    except Exception as e:

        # 不再默默吞掉錯誤
        return {
            "status": "error",
            "ticker": ticker,
            "error": repr(e)
        }


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "🔍 V2 全自動選股控制台"
)

market_choice = st.sidebar.radio(
    "選擇掃描範圍",
    [
        "成交金額熱門前 150 大",
        "全台股（上市＋上櫃）"
    ]
)

st.sidebar.divider()

st.sidebar.subheader(
    "⚙️ 技術策略參數"
)

params = {

    # 成交量
    "vol_multiplier": st.sidebar.slider(
        "放量倍數（對比20日均量）",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1
    ),

    # W底左右腳
    "w_tolerance": st.sidebar.slider(
        "W底左右腳容錯率",
        min_value=1.0,
        max_value=15.0,
        value=6.0,
        step=0.5
    ) / 100.0,

    # 突破回看
    "breakout_days": st.sidebar.number_input(
        "突破回看期間（交易日）",
        min_value=10,
        max_value=60,
        value=40,
        step=1
    ),

    # 週MA
    "ma_week": st.sidebar.number_input(
        "長期趨勢均線（週MA）",
        min_value=10,
        max_value=40,
        value=20,
        step=1
    ),

    # W底
    "w_lookback": 60,

    "pivot_window": st.sidebar.number_input(
        "W底 Pivot Low 判定寬度",
        min_value=2,
        max_value=6,
        value=3,
        step=1
    ),

    "w_min_gap": st.sidebar.number_input(
        "W底左右腳最小間隔",
        min_value=5,
        max_value=15,
        value=7,
        step=1
    ),

    "w_max_gap": st.sidebar.number_input(
        "W底左右腳最大間隔",
        min_value=20,
        max_value=45,
        value=35,
        step=1
    )
}

st.sidebar.divider()

st.sidebar.info(
    "V2 已改為只使用已完成交易日與完整週K，"
    "避免盤中資料與未完成週K造成假訊號。"
)

# ============================================================
# 開始掃描
# ============================================================

if st.sidebar.button(
    "🚀 開始 V2 全自動雷達掃描",
    type="primary"
):

    stocks_info = (
        get_all_tw_stocks_info()
    )

    all_tickers = list(
        stocks_info.keys()
    )

    # ========================================================
    # 掃描範圍
    # ========================================================

    if (
        market_choice
        == "成交金額熱門前 150 大"
    ):

        st.info(
            "正在取得市場成交金額資料..."
        )

        try:

            download_df = yf.download(
                all_tickers,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=True
            )

            if (
                download_df is None
                or download_df.empty
            ):
                st.error(
                    "❌ 無法取得市場成交資料，"
                    "因此不會假裝使用前150名。"
                )

                st.stop()

            download_df = (
                flatten_yfinance_columns(
                    download_df
                )
            )

            if (
                "Close" not in download_df.columns
                or "Volume" not in download_df.columns
            ):
                st.error(
                    "❌ Yahoo 回傳資料缺少 Close / Volume。"
                )

                st.stop()

            close_sub = (
                download_df["Close"]
            )

            vol_sub = (
                download_df["Volume"]
            )

            latest_close_market = (
                close_sub.iloc[-1]
            )

            latest_vol_market = (
                vol_sub.iloc[-1]
            )

            turnover = (
                latest_close_market
                * latest_vol_market
            )

            turnover = (
                turnover
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
            )

            turnover = turnover[
                turnover > 0
            ]

            top_turnover_tickers = (
                turnover
                .sort_values(
                    ascending=False
                )
                .head(150)
                .index
                .tolist()
            )

            target_tickers = [
                t
                for t in top_turnover_tickers
                if t in stocks_info
            ]

            if not target_tickers:
                st.error(
                    "❌ 無法取得有效成交金額排名。"
                )

                st.stop()

        except Exception as e:

            st.error(
                f"❌ 成交金額前150取得失敗：{e}"
            )

            st.stop()

    else:

        target_tickers = all_tickers

    # ========================================================
    # 顯示掃描資訊
    # ========================================================

    st.info(
        f"準備掃描 {len(target_tickers)} 支股票"
    )

    st.write(
        f"""
        **目前策略：**
        
        - 趨勢：股價 > 週{params['ma_week']}MA
        - 成交量：≥ {MIN_VOLUME_LOTS:,} 張
        - 放量：≥ 20日均量 × {params['vol_multiplier']:.1f}
        - 型態：{params['breakout_days']}日創高 OR W底突破
        """
    )

    progress_bar = st.progress(0)

    status_text = st.empty()

    matches = []

    errors = []

    completed_count = 0

    total_count = len(
        target_tickers
    )

    start_time = time.time()

    # ========================================================
    # 逐檔掃描
    # ========================================================

    for ticker in target_tickers:

        info = stocks_info[ticker]

        res = run_strategy(
            ticker=ticker,
            name=info["name"],
            group=info["group"],
            market=info["market"],
            params=params
        )

        if res is not None:

            if res.get("status") == "match":

                matches.append(res)

            elif res.get("status") == "error":

                errors.append(res)

        completed_count += 1

        if (
            completed_count % 5 == 0
            or completed_count == total_count
        ):

            progress_bar.progress(
                completed_count
                / total_count
            )

            elapsed = (
                time.time()
                - start_time
            )

            status_text.text(
                f"已掃描："
                f"{completed_count}/"
                f"{total_count} | "
                f"找到：{len(matches)} 支 | "
                f"資料錯誤：{len(errors)}"
            )

        # 避免過度請求 Yahoo
        time.sleep(0.10)

    # ========================================================
    # 完成
    # ========================================================

    progress_bar.progress(1.0)

    elapsed = (
        time.time()
        - start_time
    )

    status_text.text(
        "掃描完畢！"
    )

    st.success(
        f"🎉 V2 掃描完成！"
        f" 共掃描 {total_count} 支，"
        f"符合 {len(matches)} 支，"
        f"資料錯誤 {len(errors)} 支，"
        f"耗時 {elapsed:.1f} 秒。"
    )

    # ========================================================
    # 結果排序
    # ========================================================

    if matches:

        # 雙重訊號優先
        matches.sort(
            key=lambda x: (
                x["is_breakout"]
                + x["is_w_bottom"],
                x["volume_ratio"],
                x["distance_to_week_ma_pct"]
            ),
            reverse=True
        )

        st.subheader(
            f"✅ 符合條件的強勢標的 "
            f"（共 {len(matches)} 支）"
        )

        # ====================================================
        # 總覽表
        # ====================================================

        summary_rows = []

        for m in matches:

            summary_rows.append({

                "股票": (
                    f"{m['name']} "
                    f"({m['ticker'].split('.')[0]})"
                ),

                "市場": m["market"],

                "收盤價": m["close"],

                "週20MA": m["ma_week_val"],

                "距週MA": (
                    f"{m['distance_to_week_ma_pct']:.2f}%"
                ),

                "今日成交量": (
                    f"{m['volume']:,}"
                ),

                "20日均量": (
                    f"{m['volume_avg_20']:,.0f}"
                ),

                "放量倍數": (
                    f"{m['volume_ratio']:.2f}x"
                ),

                "40日創高": (
                    "✅"
                    if m["is_breakout"]
                    else "—"
                ),

                "W底突破": (
                    "✅"
                    if m["is_w_bottom"]
                    else "—"
                ),

                "訊號": m["signal_type"],

                "資料日期": m["data_date"]
            })

        summary_df = pd.DataFrame(
            summary_rows
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ====================================================
        # 詳細結果
        # ====================================================

        for m in matches:

            st.markdown(
                f"### 📌 {m['name']} "
                f"({m['ticker'].split('.')[0]})"
            )

            st.markdown(
                f"**{m['market']}｜"
                f"產業分類：{m['group']}｜"
                f"資料日期：{m['data_date']}**"
            )

            # ------------------------------------------------
            # 第一排
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "收盤價",
                    f"{m['close']:.2f} 元"
                )

            with col2:

                st.metric(
                    f"週{params['ma_week']}MA",
                    f"{m['ma_week_val']:.2f} 元"
                )

            with col3:

                st.metric(
                    "距週MA",
                    f"{m['distance_to_week_ma_pct']:.2f}%"
                )

            with col4:

                st.metric(
                    "成交量",
                    f"{m['volume']:,} 張"
                )

            # ------------------------------------------------
            # 第二排
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "20日均量",
                    f"{m['volume_avg_20']:,.0f} 張"
                )

            with col2:

                st.metric(
                    "實際放量倍數",
                    f"{m['volume_ratio']:.2f}x"
                )

            with col3:

                st.metric(
                    f"{params['breakout_days']}日最高價",
                    f"{m['previous_high']:.2f}"
                )

            with col4:

                st.metric(
                    "突破幅度",
                    f"{m['breakout_distance_pct']:.2f}%"
                )

            # ------------------------------------------------
            # 入選原因
            # ------------------------------------------------

            st.markdown(
                "#### 🎯 入選原因"
            )

            for reason in m["reasons"]:

                st.success(
                    f"✅ {reason}"
                )

            # ------------------------------------------------
            # W底資訊
            # ------------------------------------------------

            if m["is_w_bottom"]:

                w = m["w_info"]

                st.markdown(
                    "#### 🔵 W底結構"
                )

                w_col1, w_col2, w_col3, w_col4 = (
                    st.columns(4)
                )

                with w_col1:

                    st.metric(
                        "左腳",
                        f"{w['left_foot']:.2f}"
                    )

                with w_col2:

                    st.metric(
                        "右腳",
                        f"{w['right_foot']:.2f}"
                    )

                with w_col3:

                    st.metric(
                        "頸線",
                        f"{w['neck_high']:.2f}"
                    )

                with w_col4:

                    st.metric(
                        "左右腳差異",
                        f"{w['foot_diff_pct']:.2f}%"
                    )

            # ------------------------------------------------
            # 停損說明
            # ------------------------------------------------

            distance = (
                m["distance_to_week_ma_pct"]
            )

            if distance < 3:

                risk_label = "⚠️ 非常接近週MA"

            elif distance < 7:

                risk_label = "🟡 距週MA適中"

            elif distance < 12:

                risk_label = "🟢 距週MA較寬"

            else:

                risk_label = "⚠️ 距週MA過遠，注意追高"

            st.markdown(
                f"""
                🛡️ **目前週{params['ma_week']}MA：**
                **{m['ma_week_val']:.2f} 元**
                
                📏 **目前價格距離週MA：**
                **{distance:.2f}%**
                
                {risk_label}
                
                > 注意：這只是技術面停損參考距離，
                > 並不代表實際最大損失。
                """
            )

            # ------------------------------------------------
            # 股利
            # ------------------------------------------------

            if not m["div_history"].empty:

                st.markdown(
                    "#### 📊 近十年現金股利"
                )

                plot_dividend_bar_chart(
                    m["div_history"]
                )

            else:

                st.info(
                    "該標的沒有可取得的近期股利資料。"
                )

            # ------------------------------------------------
            # K線
            # ------------------------------------------------

            st.markdown(
                "#### 📈 技術圖"
            )

            plot_stock_chart(
                ticker=m["ticker"],
                df_day=m["df_day"],
                ma_week_val=m["ma_week_val"],
                breakout_days=params["breakout_days"],
                is_breakout=m["is_breakout"],
                w_info=m["w_info"]
            )

            st.divider()

    else:

        st.warning(
            "ℹ️ 在目前參數下，"
            "沒有找到符合條件的股票。"
        )

    # ========================================================
    # 錯誤資訊
    # ========================================================

    if errors:

        with st.expander(
            f"⚠️ 資料取得錯誤 "
            f"（{len(errors)} 支）"
        ):

            error_df = pd.DataFrame([
                {
                    "股票": e["ticker"],
                    "錯誤": e["error"]
                }
                for e in errors
            ])

            st.dataframe(
                error_df,
                use_container_width=True,
                hide_index=True
            )
