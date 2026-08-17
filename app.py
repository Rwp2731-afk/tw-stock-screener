import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import twstock
import warnings
import time
import io

from datetime import datetime, time as dt_time

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# ============================================================
# 基本設定
# ============================================================

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="台股 V2.1 強勢突破全自動雷達",
    layout="wide"
)

st.title("📈 台股 V2.1 全自動選股雷達")

st.caption(
    "V2.1：全台上市＋上櫃｜批量下載安全加速｜"
    "週20MA＋5日均量放量＋40日創高/W底突破｜"
    "產業集中分析＋一鍵PDF"
)


# ============================================================
# 常數
# ============================================================

TW_TZ = "Asia/Taipei"

MIN_VOLUME_LOTS = 1000

DAILY_LOOKBACK_YEARS = "2y"

CHART_DAYS = 250

BATCH_SIZE = 100

DEFAULT_VOL_MULTIPLIER = 1.5


# ============================================================
# Streamlit Session State
# ============================================================

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []

if "scan_finished" not in st.session_state:
    st.session_state.scan_finished = False


# ============================================================
# 工具函數
# ============================================================

def flatten_single_dataframe(df):
    """
    處理單一股票 yfinance MultiIndex。
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):

        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)

        elif "Close" in df.columns.get_level_values(-1):
            df.columns = df.columns.get_level_values(-1)

        else:
            df.columns = df.columns.get_level_values(0)

    return df


def get_taiwan_now():

    return pd.Timestamp.now(tz=TW_TZ)


def is_market_closed_for_today():

    now = get_taiwan_now()

    if now.weekday() >= 5:
        return True

    market_close = dt_time(13, 30)

    return now.time() >= market_close


def prepare_completed_daily_data(df):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        return pd.DataFrame()

    now = get_taiwan_now()

    today = now.date()

    last_date = df.index[-1].date()

    # 如果今天還沒收盤
    # 且 Yahoo 已經給了今天資料
    # 則刪除今天未完成K
    if (
        last_date == today
        and not is_market_closed_for_today()
    ):

        df = df.iloc[:-1].copy()

    return df


# ============================================================
# 由日K建立完整週K
# ============================================================

def build_completed_weekly_data(df_day):

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
        c in df_day.columns
        for c in required_cols
    ):
        return pd.DataFrame()

    weekly = (
        df_day[required_cols]
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

    # 非星期五收盤前的資料不算完整週K
    # 因此如果最後一週日期還沒到星期五，
    # 排除最後一根。
    if not weekly.empty:

        last_week_date = weekly.index[-1]

        if (
            last_week_date.date()
            > df_day.index[-1].date()
        ):
            weekly = weekly.iloc[:-1]

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

    if len(lows) < pivot_window * 2 + 1:
        return pivot_indices

    for i in range(
        pivot_window,
        len(lows) - pivot_window
    ):

        left = lows[
            i - pivot_window:i
        ]

        right = lows[
            i + 1:i + pivot_window + 1
        ]

        if (
            lows[i] <= np.min(left)
            and lows[i] <= np.min(right)
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

            left_foot = lows[left_idx]

            right_foot = lows[right_idx]

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
                left_idx:right_idx + 1
            ]

            if len(between_highs) == 0:
                continue

            neck_high = np.max(
                between_highs
            )

            if neck_high <= max(
                left_foot,
                right_foot
            ):
                continue

            # 最新收盤必須突破頸線
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
                "left_idx": left_idx,
                "right_idx": right_idx,
                "left_foot": float(
                    left_foot
                ),
                "right_foot": float(
                    right_foot
                ),
                "neck_high": float(
                    neck_high
                ),
                "foot_diff_pct": float(
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
        "left_idx": best["left_idx"],
        "right_idx": best["right_idx"],
        "left_foot": round(
            best["left_foot"],
            2
        ),
        "right_foot": round(
            best["right_foot"],
            2
        ),
        "neck_high": round(
            best["neck_high"],
            2
        ),
        "foot_diff_pct": round(
            best["foot_diff_pct"],
            2
        )
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
            and info.market
            in ["上市", "上櫃"]
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

                "group": (
                    info.group
                    if info.group
                    else "其他"
                ),

                "market": info.market
            }

    return stocks_info


# ============================================================
# 批量下載
# ============================================================

def batch_download_market_data(
    tickers,
    progress_callback=None
):

    batches = []

    for i in range(
        0,
        len(tickers),
        BATCH_SIZE
    ):

        batches.append(
            tickers[
                i:i + BATCH_SIZE
            ]
        )

    all_data = {}

    total_batches = len(
        batches
    )

    for batch_index, batch in enumerate(
        batches,
        start=1
    ):

        try:

            data = yf.download(
                tickers=batch,
                period=DAILY_LOOKBACK_YEARS,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="ticker"
            )

            if (
                data is None
                or data.empty
            ):
                continue

            # 多股票下載通常為：
            # Ticker → OHLCV
            if isinstance(
                data.columns,
                pd.MultiIndex
            ):

                level0 = (
                    data.columns
                    .get_level_values(0)
                )

                # 正常格式
                for ticker in batch:

                    try:

                        if ticker in level0:

                            df = data[
                                ticker
                            ].copy()

                            if (
                                not df.empty
                            ):

                                all_data[
                                    ticker
                                ] = df

                    except Exception:
                        continue

            else:

                # 理論上批量不會進來
                # 但保留防呆
                if len(batch) == 1:

                    all_data[
                        batch[0]
                    ] = data.copy()

        except Exception:
            continue

        if progress_callback:

            progress_callback(
                batch_index,
                total_batches
            )

        # 小幅延遲
        # 不採暴力請求
        time.sleep(0.15)

    return all_data


# ============================================================
# 單檔技術分析
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

        df_day = flatten_single_dataframe(
            df_day
        )

        df_day = prepare_completed_daily_data(
            df_day
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
            c in df_day.columns
            for c in required_cols
        ):
            return None

        df_day = (
            df_day[required_cols]
            .copy()
            .dropna(
                subset=[
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]
            )
        )

        min_required = max(
            120,
            params["breakout_days"] + 10,
            params["w_lookback"] + 10
        )

        if len(df_day) < min_required:
            return None

        # ----------------------------------------------------
        # 建立週K
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # numpy
        # ----------------------------------------------------

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
        # 1. 週20MA
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
            return None

        if (
            latest_week_close
            <= ma_week_val
        ):
            return None

        # ====================================================
        # 最新完成日
        # ====================================================

        latest_close = float(
            close_day[-1]
        )

        latest_volume = float(
            vol_day[-1]
        )

        latest_vol_lots = (
            latest_volume / 1000
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
        # 3. 5日均量
        # ====================================================

        ma_vol_5 = (
            pd.Series(vol_day)
            .rolling(5)
            .mean()
            .iloc[-1]
        )

        if (
            not np.isfinite(ma_vol_5)
            or ma_vol_5 <= 0
        ):
            return None

        volume_ratio = (
            latest_volume
            / ma_vol_5
        )

        if (
            volume_ratio
            < params["vol_multiplier"]
        ):
            return None

        # ====================================================
        # 資料日期
        # ====================================================

        data_date = (
            df_day.index[-1]
            .strftime("%Y-%m-%d")
        )

        # ====================================================
        # 4A. 40日創高
        # ====================================================

        breakout_days = (
            params["breakout_days"]
        )

        previous_highs = (
            high_day[
                -(breakout_days + 1):-1
            ]
        )

        if len(previous_highs) < (
            breakout_days
        ):
            return None

        previous_high = np.max(
            previous_highs
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
        ) * 100

        # ====================================================
        # 4B. W底
        # ====================================================

        w_info = detect_w_bottom(

            high_day=high_day,

            low_day=low_day,

            close_day=close_day,

            tolerance=params[
                "w_tolerance"
            ],

            lookback=params[
                "w_lookback"
            ],

            pivot_window=params[
                "pivot_window"
            ],

            min_gap=params[
                "w_min_gap"
            ],

            max_gap=params[
                "w_max_gap"
            ]
        )

        is_w_bottom = (
            w_info["is_w_bottom"]
        )

        # ====================================================
        # 型態擇一
        # ====================================================

        if not (
            is_breakout
            or is_w_bottom
        ):
            return None

        # ====================================================
        # 訊號
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

        if (
            is_breakout
            and is_w_bottom
        ):

            signal_type = "雙重訊號"

        elif is_breakout:

            signal_type = "區間創高"

        else:

            signal_type = "W底突破"

        # ====================================================
        # 距週MA
        # ====================================================

        distance_to_week_ma_pct = (
            (
                latest_close
                - ma_week_val
            )
            / latest_close
        ) * 100

        # ====================================================
        # 回傳
        # ====================================================

        return {

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

            "volume_avg_5": round(
                ma_vol_5 / 1000,
                0
            ),

            "volume_ratio": round(
                volume_ratio,
                2
            ),

            "ma_week_val": round(
                float(ma_week_val),
                2
            ),

            "distance_to_week_ma_pct":
                round(
                    float(
                        distance_to_week_ma_pct
                    ),
                    2
                ),

            "previous_high": round(
                float(previous_high),
                2
            ),

            "breakout_distance_pct":
                round(
                    float(
                        breakout_distance_pct
                    ),
                    2
                ),

            "is_breakout":
                bool(is_breakout),

            "is_w_bottom":
                bool(is_w_bottom),

            "signal_type":
                signal_type,

            "reasons":
                reasons,

            "w_info":
                w_info,

            "div_history":
                pd.DataFrame()
        }

    except Exception:
        return None


# ============================================================
# 股利
# ============================================================

def get_dividend_history(
    ticker
):

    try:

        stock_obj = yf.Ticker(
            ticker
        )

        dividends = (
            stock_obj.dividends
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
            "Dividend": dividends
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

        yearly["現金股利"] = (
            yearly["現金股利"]
            .round(2)
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
# K線
# ============================================================

def plot_stock_chart(
    ticker,
    df_day,
    ma_week_val,
    breakout_days,
    is_breakout,
    w_info
):

    plot_df = (
        df_day
        .iloc[-CHART_DAYS:]
        .copy()
    )

    if plot_df.empty:
        return

    plot_df = flatten_single_dataframe(
        plot_df
    )

    ma20 = (
        plot_df["Close"]
        .rolling(20)
        .mean()
    )

    ma100 = (
        plot_df["Close"]
        .rolling(100)
        .mean()
    )

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

        mpf.make_addplot(
            [ma_week_val]
            * len(plot_df),
            color="red",
            linestyle="dashed",
            width=1.2
        )
    ]

    if w_info.get(
        "is_w_bottom"
    ):

        neck_high = (
            w_info.get(
                "neck_high"
            )
        )

        if neck_high is not None:

            addplots.append(
                mpf.make_addplot(
                    [neck_high]
                    * len(plot_df),
                    color="orange",
                    linestyle="dashdot",
                    width=1.2
                )
            )

    title_parts = [
        f"{ticker} - V2.1 Trend Radar",
        f"Weekly MA20 Stop: {ma_week_val:.2f}"
    ]

    if is_breakout:

        title_parts.append(
            f"{breakout_days}D Breakout"
        )

    if w_info.get(
        "is_w_bottom"
    ):

        title_parts.append(
            f"W-Bottom Neckline: "
            f"{w_info['neck_high']:.2f}"
        )

    fig, axes = mpf.plot(

        plot_df,

        type="candle",

        style="yahoo",

        addplot=addplots,

        title="\n"
        + " | ".join(
            title_parts
        ),

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
# 產業集中度
# ============================================================

def build_industry_summary(
    matches
):

    if not matches:
        return pd.DataFrame()

    rows = []

    for m in matches:

        rows.append({
            "產業": m["group"],
            "股票數": 1
        })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(
            "產業"
        )
        .size()
        .reset_index(
            name="入選股票數"
        )
        .sort_values(
            "入選股票數",
            ascending=False
        )
    )

    total = (
        summary["入選股票數"]
        .sum()
    )

    summary["集中比例"] = (
        summary["入選股票數"]
        / total
        * 100
    ).round(1)

    return summary


# ============================================================
# PDF
# ============================================================

def register_pdf_font():

    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msjh.ttc",
        "/System/Library/Fonts/PingFang.ttc"
    ]

    for path in font_candidates:

        try:

            pdfmetrics.registerFont(
                TTFont(
                    "TWFont",
                    path
                )
            )

            return "TWFont"

        except Exception:
            continue

    return "Helvetica"


def generate_pdf(
    matches,
    industry_summary,
    params,
    scan_time
):

    buffer = io.BytesIO()

    font_name = (
        register_pdf_font()
    )

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=25,
        leftMargin=25,
        topMargin=25,
        bottomMargin=25
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TWTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "TWHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
        leading=18,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "TWNormal",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=13
    )

    small_style = ParagraphStyle(
        "TWSmall",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=7,
        leading=10
    )

    story = []

    # --------------------------------------------------------
    # 標題
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "台股 V2.1 強勢突破全自動雷達",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"掃描時間：{scan_time}",
            normal_style
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------------
    # 策略
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "一、掃描條件",
            heading_style
        )
    )

    conditions = [
        [
            Paragraph("項目", small_style),
            Paragraph("條件", small_style)
        ],
        [
            Paragraph("掃描範圍", small_style),
            Paragraph(
                "全台上市＋上櫃",
                small_style
            )
        ],
        [
            Paragraph("長期趨勢", small_style),
            Paragraph(
                f"最新完整週K收盤價 > 週{params['ma_week']}MA",
                small_style
            )
        ],
        [
            Paragraph("最低成交量", small_style),
            Paragraph(
                f"≥ {MIN_VOLUME_LOTS:,} 張",
                small_style
            )
        ],
        [
            Paragraph("放量條件", small_style),
            Paragraph(
                f"今日成交量 ≥ 5日均量 × "
                f"{params['vol_multiplier']:.1f}",
                small_style
            )
        ],
        [
            Paragraph("突破條件", small_style),
            Paragraph(
                f"{params['breakout_days']}日創高 OR W底突破",
                small_style
            )
        ]
    ]

    table = Table(
        conditions,
        colWidths=[130, 400]
    )

    table.setStyle(
        TableStyle([
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                font_name
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # 入選數
    # --------------------------------------------------------

    story.append(
        Paragraph(
            f"本次共找到 {len(matches)} 支符合條件股票。",
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # 產業集中
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "二、產業集中度",
            heading_style
        )
    )

    industry_data = [
        [
            Paragraph("產業", small_style),
            Paragraph("入選數", small_style),
            Paragraph("集中比例", small_style)
        ]
    ]

    for _, row in industry_summary.iterrows():

        industry_data.append([
            Paragraph(
                str(row["產業"]),
                small_style
            ),

            Paragraph(
                str(row["入選股票數"]),
                small_style
            ),

            Paragraph(
                f"{row['集中比例']:.1f}%",
                small_style
            )
        ])

    industry_table = Table(
        industry_data,
        colWidths=[
            250,
            100,
            120
        ]
    )

    industry_table.setStyle(
        TableStyle([
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                font_name
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            )
        ])
    )

    story.append(
        industry_table
    )

    story.append(
        PageBreak()
    )

    # --------------------------------------------------------
    # 股票總覽
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "三、入選股票總覽",
            heading_style
        )
    )

    summary_data = [[

        Paragraph("產業", small_style),

        Paragraph("股票", small_style),

        Paragraph("收盤", small_style),

        Paragraph(
            f"週{params['ma_week']}MA",
            small_style
        ),

        Paragraph("距週MA", small_style),

        Paragraph("今日量", small_style),

        Paragraph("5日均量", small_style),

        Paragraph("放量", small_style),

        Paragraph("40日高", small_style),

        Paragraph("W底", small_style),

        Paragraph("訊號", small_style)
    ]]

    # 產業排序
    sorted_matches = sorted(
        matches,
        key=lambda x: (
            x["group"],
            x["signal_type"],
            -x["volume_ratio"]
        )
    )

    for m in sorted_matches:

        summary_data.append([

            Paragraph(
                str(m["group"]),
                small_style
            ),

            Paragraph(
                f"{m['name']} "
                f"({m['ticker'].split('.')[0]})",
                small_style
            ),

            Paragraph(
                f"{m['close']:.2f}",
                small_style
            ),

            Paragraph(
                f"{m['ma_week_val']:.2f}",
                small_style
            ),

            Paragraph(
                f"{m['distance_to_week_ma_pct']:.2f}%",
                small_style
            ),

            Paragraph(
                f"{m['volume']:,}",
                small_style
            ),

            Paragraph(
                f"{m['volume_avg_5']:,.0f}",
                small_style
            ),

            Paragraph(
                f"{m['volume_ratio']:.2f}x",
                small_style
            ),

            Paragraph(
                "✓"
                if m["is_breakout"]
                else "-",
                small_style
            ),

            Paragraph(
                "✓"
                if m["is_w_bottom"]
                else "-",
                small_style
            ),

            Paragraph(
                m["signal_type"],
                small_style
            )
        ])

    summary_table = Table(
        summary_data,
        repeatRows=1,
        colWidths=[
            100,
            110,
            55,
            60,
            60,
            55,
            55,
            50,
            45,
            40,
            65
        ]
    )

    summary_table.setStyle(
        TableStyle([
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                font_name
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            )
        ])
    )

    story.append(
        summary_table
    )

    story.append(
        PageBreak()
    )

    # --------------------------------------------------------
    # 詳細股票
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "四、入選股票詳細資料",
            heading_style
        )
    )

    for index, m in enumerate(
        sorted_matches,
        start=1
    ):

        story.append(
            Paragraph(
                f"{index}. {m['name']} "
                f"({m['ticker'].split('.')[0]})",
                heading_style
            )
        )

        detail_data = [

            [
                Paragraph("產業", small_style),
                Paragraph(
                    str(m["group"]),
                    small_style
                )
            ],

            [
                Paragraph("市場", small_style),
                Paragraph(
                    str(m["market"]),
                    small_style
                )
            ],

            [
                Paragraph("資料日期", small_style),
                Paragraph(
                    m["data_date"],
                    small_style
                )
            ],

            [
                Paragraph("收盤價", small_style),
                Paragraph(
                    f"{m['close']:.2f}",
                    small_style
                )
            ],

            [
                Paragraph(
                    f"週{params['ma_week']}MA",
                    small_style
                ),
                Paragraph(
                    f"{m['ma_week_val']:.2f}",
                    small_style
                )
            ],

            [
                Paragraph("距週MA", small_style),
                Paragraph(
                    f"{m['distance_to_week_ma_pct']:.2f}%",
                    small_style
                )
            ],

            [
                Paragraph("今日成交量", small_style),
                Paragraph(
                    f"{m['volume']:,} 張",
                    small_style
                )
            ],

            [
                Paragraph("5日均量", small_style),
                Paragraph(
                    f"{m['volume_avg_5']:,.0f} 張",
                    small_style
                )
            ],

            [
                Paragraph("放量倍數", small_style),
                Paragraph(
                    f"{m['volume_ratio']:.2f}x",
                    small_style
                )
            ],

            [
                Paragraph("訊號", small_style),
                Paragraph(
                    m["signal_type"],
                    small_style
                )
            ]
        ]

        detail_table = Table(
            detail_data,
            colWidths=[
                130,
                400
            ]
        )

        detail_table.setStyle(
            TableStyle([
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    font_name
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.grey
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                )
            ])
        )

        story.append(
            detail_table
        )

        story.append(
            Spacer(1, 10)
        )

        reasons_text = (
            "入選原因："
            + "、".join(
                m["reasons"]
            )
        )

        story.append(
            Paragraph(
                reasons_text,
                normal_style
            )
        )

        if m["is_w_bottom"]:

            w = m["w_info"]

            w_text = (
                f"W底：左腳 "
                f"{w['left_foot']:.2f}｜"
                f"右腳 "
                f"{w['right_foot']:.2f}｜"
                f"頸線 "
                f"{w['neck_high']:.2f}｜"
                f"左右腳差異 "
                f"{w['foot_diff_pct']:.2f}%"
            )

            story.append(
                Paragraph(
                    w_text,
                    normal_style
                )
            )

        story.append(
            Spacer(1, 12)
        )

        # 股利
        if (
            not m["div_history"].empty
        ):

            story.append(
                Paragraph(
                    "近十年現金股利：",
                    normal_style
                )
            )

            div_text = "｜".join(
                [
                    f"{int(row['年份'])}: "
                    f"{row['現金股利']:.2f}"
                    for _, row
                    in m[
                        "div_history"
                    ].iterrows()
                ]
            )

            story.append(
                Paragraph(
                    div_text,
                    small_style
                )
            )

        story.append(
            Spacer(1, 15)
        )

        if index != len(
            sorted_matches
        ):

            story.append(
                PageBreak()
            )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "🔍 V2.1 全自動選股控制台"
)

st.sidebar.info(
    "掃描範圍已固定為：\n\n"
    "🇹🇼 全台上市＋上櫃"
)

st.sidebar.divider()

st.sidebar.subheader(
    "⚙️ 技術策略參數"
)

params = {

    "vol_multiplier":
        st.sidebar.slider(
            "放量倍數（對比5日均量）",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1
        ),

    "w_tolerance":
        st.sidebar.slider(
            "W底左右腳容錯率",
            min_value=1.0,
            max_value=15.0,
            value=6.0,
            step=0.5
        ) / 100.0,

    "breakout_days":
        st.sidebar.number_input(
            "突破回看期間（交易日）",
            min_value=10,
            max_value=60,
            value=40,
            step=1
        ),

    "ma_week":
        st.sidebar.number_input(
            "長期趨勢均線（週MA）",
            min_value=10,
            max_value=40,
            value=20,
            step=1
        ),

    "w_lookback": 60,

    "pivot_window":
        st.sidebar.number_input(
            "W底 Pivot Low 判定寬度",
            min_value=2,
            max_value=6,
            value=3,
            step=1
        ),

    "w_min_gap":
        st.sidebar.number_input(
            "W底左右腳最小間隔",
            min_value=5,
            max_value=15,
            value=7,
            step=1
        ),

    "w_max_gap":
        st.sidebar.number_input(
            "W底左右腳最大間隔",
            min_value=20,
            max_value=45,
            value=35,
            step=1
        )
}

st.sidebar.divider()

st.sidebar.markdown(
    """
### V2.1 核心條件

**① 長期趨勢**
股價 > 週20MA

**② 流動性**
今日 ≥ 1,000 張

**③ 資金放量**
今日 ≥ 5日均量 × 1.5

**④ 型態**
40日創高 OR W底突破
"""
)

# ============================================================
# 開始掃描
# ============================================================

if st.sidebar.button(
    "🚀 開始 V2.1 全自動掃描",
    type="primary"
):

    # --------------------------------------------------------
    # 清除舊結果
    # --------------------------------------------------------

    st.session_state.scan_results = []

    st.session_state.scan_errors = []

    st.session_state.scan_finished = False

    # --------------------------------------------------------
    # 股票清單
    # --------------------------------------------------------

    stocks_info = (
        get_all_tw_stocks_info()
    )

    all_tickers = list(
        stocks_info.keys()
    )

    total_stocks = len(
        all_tickers
    )

    st.info(
        f"🇹🇼 準備掃描全台上市＋上櫃 "
        f"共 {total_stocks:,} 支股票"
    )

    # --------------------------------------------------------
    # 第一階段：批量下載
    # --------------------------------------------------------

    st.subheader(
        "① 批量下載市場日K資料"
    )

    download_progress = st.progress(
        0
    )

    download_status = st.empty()

    def download_progress_callback(
        current,
        total
    ):

        ratio = (
            current / total
        )

        download_progress.progress(
            ratio
        )

        download_status.text(
            f"批量下載進度："
            f"{current}/{total} 批"
        )

    start_time = time.time()

    market_data = (
        batch_download_market_data(
            all_tickers,
            download_progress_callback
        )
    )

    download_progress.progress(
        1.0
    )

    download_status.text(
        f"日K下載完成："
        f"{len(market_data):,} 支"
    )

    # --------------------------------------------------------
    # 第二階段：技術篩選
    # --------------------------------------------------------

    st.subheader(
        "② 技術條件快速篩選"
    )

    analysis_progress = st.progress(
        0
    )

    analysis_status = st.empty()

    matches = []

    errors = []

    total_analysis = len(
        market_data
    )

    for index, ticker in enumerate(
        market_data.keys(),
        start=1
    ):

        info = stocks_info.get(
            ticker
        )

        if info is None:
            continue

        result = analyze_stock(

            ticker=ticker,

            name=info["name"],

            group=info["group"],

            market=info["market"],

            df_day=market_data[ticker],

            params=params
        )

        if result is not None:

            matches.append(
                result
            )

        if (
            index % 20 == 0
            or index == total_analysis
        ):

            analysis_progress.progress(
                index
                / total_analysis
            )

            analysis_status.text(
                f"技術分析："
                f"{index:,}/"
                f"{total_analysis:,} | "
                f"目前入選："
                f"{len(matches)} 支"
            )

    analysis_progress.progress(
        1.0
    )

    # --------------------------------------------------------
    # 第三階段：只對入選股票抓股利
    # --------------------------------------------------------

    st.subheader(
        "③ 取得入選股票股利資料"
    )

    dividend_progress = st.progress(
        0
    )

    dividend_status = st.empty()

    dividend_total = len(
        matches
    )

    for index, m in enumerate(
        matches,
        start=1
    ):

        m["div_history"] = (
            get_dividend_history(
                m["ticker"]
            )
        )

        dividend_progress.progress(
            index
            / max(
                dividend_total,
                1
            )
        )

        dividend_status.text(
            f"股利資料："
            f"{index}/{dividend_total}"
        )

        # 股利只抓入選股票
        time.sleep(0.05)

    # --------------------------------------------------------
    # 排序
    # --------------------------------------------------------

    matches.sort(
        key=lambda x: (

            x["group"],

            -(
                int(
                    x["is_breakout"]
                )
                + int(
                    x["is_w_bottom"]
                )
            ),

            -x["volume_ratio"]
        )
    )

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    st.session_state.scan_results = (
        matches
    )

    st.session_state.scan_errors = (
        errors
    )

    st.session_state.scan_finished = True

    st.success(
        f"🎉 V2.1 掃描完成！ "
        f"共處理 {total_stocks:,} 支，"
        f"入選 {len(matches)} 支，"
        f"耗時 {elapsed:.1f} 秒。"
    )


# ============================================================
# 顯示結果
# ============================================================

if (
    st.session_state.scan_finished
):

    matches = (
        st.session_state.scan_results
    )

    errors = (
        st.session_state.scan_errors
    )

    if matches:

        # ====================================================
        # 產業集中
        # ====================================================

        industry_summary = (
            build_industry_summary(
                matches
            )
        )

        st.subheader(
            f"🏭 市場資金產業集中度"
        )

        industry_col1, industry_col2 = (
            st.columns(
                [2, 1]
            )
        )

        with industry_col1:

            st.dataframe(
                industry_summary,
                use_container_width=True,
                hide_index=True
            )

        with industry_col2:

            st.markdown(
                "### 🔎 觀察重點"
            )

            if not industry_summary.empty:

                top_industry = (
                    industry_summary.iloc[0]
                )

                st.metric(
                    "目前入選最多產業",
                    top_industry["產業"],
                    f"{int(top_industry['入選股票數'])} 支"
                )

                st.metric(
                    "產業集中比例",
                    f"{top_industry['集中比例']:.1f}%"
                )

        st.divider()

        # ====================================================
        # 總覽表
        # ====================================================

        st.subheader(
            f"📋 入選股票總覽 "
            f"（共 {len(matches)} 支）"
        )

        summary_rows = []

        for m in matches:

            summary_rows.append({

                "產業":
                    m["group"],

                "股票":
                    f"{m['name']} "
                    f"({m['ticker'].split('.')[0]})",

                "市場":
                    m["market"],

                "收盤價":
                    m["close"],

                "週20MA":
                    m["ma_week_val"],

                "距週MA":
                    f"{m['distance_to_week_ma_pct']:.2f}%",

                "今日成交量":
                    f"{m['volume']:,}",

                "5日均量":
                    f"{m['volume_avg_5']:,.0f}",

                "放量倍數":
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

        # ====================================================
        # PDF
        # ====================================================

        st.divider()

        st.subheader(
            "📄 報告匯出"
        )

        scan_time = (
            datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        try:

            pdf_bytes = generate_pdf(

                matches=matches,

                industry_summary=
                    industry_summary,

                params=params,

                scan_time=scan_time
            )

            st.download_button(

                label="📥 一鍵匯出 V2.1 PDF",

                data=pdf_bytes,

                file_name=(
                    "TW_V2.1_強勢突破雷達_"
                    + datetime.now().strftime(
                        "%Y%m%d_%H%M"
                    )
                    + ".pdf"
                ),

                mime="application/pdf",

                type="primary"
            )

        except Exception as e:

            st.error(
                f"PDF產生失敗：{e}"
            )

        st.divider()

        # ====================================================
        # 詳細資料
        # ====================================================

        st.subheader(
            "🔎 入選股票詳細分析"
        )

        for m in matches:

            st.markdown(
                f"## 📌 {m['name']} "
                f"({m['ticker'].split('.')[0]})"
            )

            st.markdown(
                f"**{m['market']}｜"
                f"產業：{m['group']}｜"
                f"資料日期：{m['data_date']}**"
            )

            # ------------------------------------------------
            # 第一排
            # ------------------------------------------------

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "收盤價",
                    f"{m['close']:.2f} 元"
                )

            with col2:

                st.metric(
                    "週20MA",
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

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "5日均量",
                    f"{m['volume_avg_5']:,.0f} 張"
                )

            with col2:

                st.metric(
                    "放量倍數",
                    f"{m['volume_ratio']:.2f}x"
                )

            with col3:

                st.metric(
                    "前40日最高價",
                    f"{m['previous_high']:.2f}"
                )

            with col4:

                st.metric(
                    "突破幅度",
                    f"{m['breakout_distance_pct']:.2f}%"
                )

            # ------------------------------------------------
            # 訊號
            # ------------------------------------------------

            st.markdown(
                "### 🎯 入選原因"
            )

            for reason in m[
                "reasons"
            ]:

                st.success(
                    f"✅ {reason}"
                )

            # ------------------------------------------------
            # W底
            # ------------------------------------------------

            if m[
                "is_w_bottom"
            ]:

                st.markdown(
                    "### 🔵 W底結構"
                )

                w = m[
                    "w_info"
                ]

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

            # ------------------------------------------------
            # 風險
            # ------------------------------------------------

            distance = (
                m[
                    "distance_to_week_ma_pct"
                ]
            )

            if distance < 3:

                risk_label = (
                    "⚠️ 非常接近週MA"
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
                    "⚠️ 距週MA過遠，注意追高"
                )

            st.info(
                f"🛡️ 週20MA："
                f"{m['ma_week_val']:.2f} 元\n\n"
                f"📏 距離週20MA："
                f"{distance:.2f}%\n\n"
                f"{risk_label}"
            )

            # ------------------------------------------------
            # 股利
            # ------------------------------------------------

            if not m[
                "div_history"
            ].empty:

                st.markdown(
                    "### 📊 近十年現金股利"
                )

                plot_dividend_bar_chart(
                    m[
                        "div_history"
                    ]
                )

            else:

                st.info(
                    "該標的沒有可取得的近期股利資料。"
                )

            # ------------------------------------------------
            # K線
            # ------------------------------------------------

            st.markdown(
                "### 📈 技術K線"
            )

            plot_stock_chart(

                ticker=m["ticker"],

                df_day=m["df_day"],

                ma_week_val=
                    m["ma_week_val"],

                breakout_days=
                    params["breakout_days"],

                is_breakout=
                    m["is_breakout"],

                w_info=
                    m["w_info"]
            )

            st.divider()

    else:

        st.warning(
            "ℹ️ 目前參數沒有找到符合條件的股票。"
        )

    # ========================================================
    # 錯誤
    # ========================================================

    if errors:

        with st.expander(
            f"⚠️ 資料錯誤 "
            f"（{len(errors)} 支）"
        ):

            error_df = pd.DataFrame(
                errors
            )

            st.dataframe(
                error_df,
                use_container_width=True,
                hide_index=True
            )
