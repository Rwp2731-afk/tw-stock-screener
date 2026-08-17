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
from matplotlib.backends.backend_pdf import PdfPages

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
    "V2.1 加速版：全台上市＋上櫃｜批量下載＋分層篩選｜"
    "5日均量放量｜週20MA趨勢｜40日創高／W底突破｜"
    "產業集中分析｜一鍵PDF匯出"
)

# ============================================================
# 常數
# ============================================================

TW_TZ = "Asia/Taipei"

MIN_VOLUME_LOTS = 1000

DAILY_HISTORY_PERIOD = "2y"

CHART_DAYS = 250

# 批量下載時使用
BATCH_SIZE = 80

# 每批之間稍微休息
BATCH_SLEEP = 0.3


# ============================================================
# yfinance 欄位處理
# ============================================================

def flatten_yfinance_columns(df):

    if df is None or df.empty:
        return df

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):

        # 多股票下載通常為：
        # Price / Ticker
        #
        # 例如：
        # Close / 2330.TW

        if "Close" in df.columns.get_level_values(0):

            df.columns = df.columns.get_level_values(0)

        elif "Close" in df.columns.get_level_values(-1):

            df.columns = df.columns.get_level_values(-1)

        else:

            df.columns = df.columns.get_level_values(0)

    return df


# ============================================================
# 台灣時間
# ============================================================

def get_taiwan_now():

    return pd.Timestamp.now(
        tz=TW_TZ
    )


# ============================================================
# 判斷台股今日是否已收盤
# ============================================================

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
# 只保留已完成日K
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

    last_date = df.index[-1].date()

    # 今天還沒收盤
    # 如果 Yahoo 已經有今天資料
    # 排除今天
    if (
        last_date == today
        and not is_market_closed_for_today()
    ):

        df = df.iloc[:-1].copy()

    return df


# ============================================================
# 建立完整週K
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

    # ========================================================
    # 非週五交易日也可能產生最後一根週K
    #
    # 例如週一、週二尚未到週五
    # 必須排除最後一根未完成週K
    # ========================================================

    now = get_taiwan_now()

    current_week_friday = (
        now.normalize()
        + pd.Timedelta(
            days=(4 - now.weekday()) % 7
        )
    )

    # 只保留真正已完成的週
    weekly = weekly[
        weekly.index < current_week_friday
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
            and lows[i] <= np.min(right)
        ):

            pivot_indices.append(i)

    return pivot_indices


# ============================================================
# W底
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

    if len(low_day) < lookback:

        return {
            "is_w_bottom": False,
            "left_idx": None,
            "right_idx": None,
            "left_foot": None,
            "right_foot": None,
            "neck_high": None,
            "foot_diff_pct": None
        }

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
        pivot_window
    )

    if len(pivot_lows) < 2:

        return {
            "is_w_bottom": False,
            "left_idx": None,
            "right_idx": None,
            "left_foot": None,
            "right_foot": None,
            "neck_high": None,
            "foot_diff_pct": None
        }

    latest_close = closes[-1]

    candidates = []

    for left_idx in pivot_lows:

        if left_idx >= (
            lookback // 2
        ):
            continue

        for right_idx in pivot_lows:

            if right_idx <= left_idx:
                continue

            if right_idx >= (
                lookback - 5
            ):
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

            if (
                foot_diff_pct
                > tolerance
            ):
                continue

            between_highs = highs[
                left_idx:
                right_idx + 1
            ]

            if len(
                between_highs
            ) == 0:
                continue

            neck_high = np.max(
                between_highs
            )

            if neck_high <= max(
                left_foot,
                right_foot
            ):
                continue

            # 最新收盤突破頸線
            if latest_close <= neck_high:
                continue

            # 右腳後必須有反彈
            right_after = closes[
                right_idx:
            ]

            if len(
                right_after
            ) < 2:
                continue

            right_rebound_high = np.max(
                right_after
            )

            if (
                right_rebound_high
                <= right_foot
            ):
                continue

            # 頸線距離右腳至少3%
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

        return {
            "is_w_bottom": False,
            "left_idx": None,
            "right_idx": None,
            "left_foot": None,
            "right_foot": None,
            "neck_high": None,
            "foot_diff_pct": None
        }

    candidates.sort(
        key=lambda x: (
            -x["right_idx"],
            x["foot_diff_pct"]
        )
    )

    best = candidates[0]

    return {

        "is_w_bottom": True,

        "left_idx": best[
            "left_idx"
        ],

        "right_idx": best[
            "right_idx"
        ],

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

                "group":
                    info.group
                    if info.group
                    else "其他",

                "market":
                    info.market
            }

    return stocks_info


# ============================================================
# 批量下載日K
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def batch_download_daily_data(
    tickers
):

    all_data = {}

    tickers = list(tickers)

    total = len(tickers)

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch = tickers[
            start:
            start + BATCH_SIZE
        ]

        try:

            data = yf.download(
                batch,
                period=DAILY_HISTORY_PERIOD,
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

            # 多股票
            if isinstance(
                data.columns,
                pd.MultiIndex
            ):

                level0 = (
                    data.columns
                    .get_level_values(0)
                )

                level1 = (
                    data.columns
                    .get_level_values(1)
                )

                # =================================================
                # 第一種格式：
                # Ticker / Price
                # =================================================

                if any(
                    ticker in level0
                    for ticker in batch
                ):

                    for ticker in batch:

                        try:

                            if ticker not in level0:
                                continue

                            df = data[
                                ticker
                            ].copy()

                            if not df.empty:
                                all_data[
                                    ticker
                                ] = df

                        except Exception:
                            continue

                # =================================================
                # 第二種格式：
                # Price / Ticker
                # =================================================

                elif "Close" in level0:

                    for ticker in batch:

                        try:

                            if ticker not in level1:
                                continue

                            df = (
                                data
                                .xs(
                                    ticker,
                                    axis=1,
                                    level=1
                                )
                                .copy()
                            )

                            if not df.empty:
                                all_data[
                                    ticker
                                ] = df

                        except Exception:
                            continue

            else:

                # 單股票
                if len(batch) == 1:

                    all_data[
                        batch[0]
                    ] = data.copy()

        except Exception:
            continue

        time.sleep(
            BATCH_SLEEP
        )

    return all_data


# ============================================================
# 快速第一層篩選
# ============================================================

def quick_filter(
    ticker,
    df_day,
    params
):

    if (
        df_day is None
        or df_day.empty
    ):
        return None

    df_day = flatten_yfinance_columns(
        df_day
    )

    df_day = prepare_completed_daily_data(
        df_day
    )

    if df_day.empty:
        return None

    required = [
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if not all(
        c in df_day.columns
        for c in required
    ):
        return None

    df_day = df_day[
        required
    ].dropna()

    if len(df_day) < 120:
        return None

    close = df_day[
        "Close"
    ].to_numpy(float)

    high = df_day[
        "High"
    ].to_numpy(float)

    low = df_day[
        "Low"
    ].to_numpy(float)

    volume = df_day[
        "Volume"
    ].to_numpy(float)

    # ========================================================
    # 今日資料
    # ========================================================

    latest_close = float(
        close[-1]
    )

    latest_volume = float(
        volume[-1]
    )

    latest_volume_lots = (
        latest_volume / 1000
    )

    # 第一層：
    # 最低成交量
    if (
        latest_volume_lots
        < MIN_VOLUME_LOTS
    ):
        return None

    # ========================================================
    # 5日均量
    # ========================================================

    if len(volume) < 6:
        return None

    avg5 = np.mean(
        volume[-6:-1]
    )

    if avg5 <= 0:
        return None

    volume_ratio = (
        latest_volume
        / avg5
    )

    # 今日量至少為5日均量設定倍數
    if (
        volume_ratio
        < params["vol_multiplier"]
    ):
        return None

    # ========================================================
    # 40日創高
    # ========================================================

    breakout_days = (
        params["breakout_days"]
    )

    if len(close) <= (
        breakout_days
    ):
        return None

    previous_high = np.max(
        high[
            -(breakout_days + 1):-1
        ]
    )

    is_breakout = (
        latest_close
        >= previous_high
    )

    # ========================================================
    # 週K
    # ========================================================

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
        .to_numpy(float)
    )

    ma_week = (
        pd.Series(
            close_week
        )
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

    # 趨勢必須成立
    if (
        latest_week_close
        <= ma_week
    ):
        return None

    # ========================================================
    # 第二層：
    # W底
    # ========================================================

    w_info = detect_w_bottom(

        high_day=high,

        low_day=low,

        close_day=close,

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

    # 型態擇一
    if not (
        is_breakout
        or is_w_bottom
    ):
        return None

    # ========================================================
    # 訊號
    # ========================================================

    if (
        is_breakout
        and is_w_bottom
    ):

        signal_type = "雙重訊號"

    elif is_breakout:

        signal_type = "區間創高"

    else:

        signal_type = "W底突破"

    reasons = []

    if is_breakout:

        reasons.append(
            f"{breakout_days}日創高突破"
        )

    if is_w_bottom:

        reasons.append(
            "W底突破"
        )

    # ========================================================
    # 距週MA
    # ========================================================

    distance_to_ma = (
        (
            latest_close
            - ma_week
        )
        / latest_close
    ) * 100

    # ========================================================
    # 資料日期
    # ========================================================

    data_date = (
        df_day.index[-1]
        .strftime(
            "%Y-%m-%d"
        )
    )

    return {

        "status": "match",

        "ticker": ticker,

        "df_day": df_day,

        "data_date": data_date,

        "close": round(
            latest_close,
            2
        ),

        "volume": int(
            latest_volume_lots
        ),

        "volume_avg_5": round(
            avg5 / 1000,
            0
        ),

        "volume_ratio": round(
            volume_ratio,
            2
        ),

        "ma_week_val": round(
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

        "previous_high": round(
            float(previous_high),
            2
        ),

        "breakout_distance_pct":
            round(
                (
                    (
                        latest_close
                        - previous_high
                    )
                    / previous_high
                ) * 100,
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
            w_info
    }


# ============================================================
# 最後才抓股利
# ============================================================

@st.cache_data(
    ttl=3600,
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

        yearly[
            "現金股利"
        ] = (
            yearly[
                "現金股利"
            ]
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

def create_dividend_figure(
    div_df
):

    fig, ax = plt.subplots(
        figsize=(10, 3.5)
    )

    if (
        div_df is None
        or div_df.empty
    ):

        ax.text(
            0.5,
            0.5,
            "No Dividend Data",
            ha="center",
            va="center"
        )

        ax.axis("off")

        return fig

    years = (
        div_df[
            "年份"
        ]
        .astype(str)
        .tolist()
    )

    dividends = (
        div_df[
            "現金股利"
        ]
        .tolist()
    )

    bars = ax.bar(
        years,
        dividends
    )

    for bar in bars:

        height = (
            bar.get_height()
        )

        ax.annotate(
            f"{height:.2f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height
            ),
            xytext=(
                0,
                3
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax.set_title(
        "Recent 10-Year Cash Dividend"
    )

    ax.set_ylabel(
        "Dividend (TWD)"
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ============================================================
# K線圖
# ============================================================

def create_stock_chart(
    m,
    params
):

    plot_df = (
        m["df_day"]
        .iloc[-CHART_DAYS:]
        .copy()
    )

    if plot_df.empty:
        return None

    plot_df = flatten_yfinance_columns(
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
            width=1.3
        ),

        mpf.make_addplot(
            ma100,
            width=1.6
        ),

        mpf.make_addplot(
            [
                m["ma_week_val"]
            ] * len(plot_df),
            linestyle="dashed",
            width=1.2
        )
    ]

    if m[
        "is_w_bottom"
    ]:

        neck = m[
            "w_info"
        ].get(
            "neck_high"
        )

        if neck is not None:

            addplots.append(
                mpf.make_addplot(
                    [
                        neck
                    ] * len(plot_df),
                    linestyle="dashdot",
                    width=1.2
                )
            )

    title = (
        f"{m['ticker']} "
        f"V2.1 Trend Radar"
    )

    if m[
        "is_breakout"
    ]:

        title += (
            f" | "
            f"{params['breakout_days']}D Breakout"
        )

    if m[
        "is_w_bottom"
    ]:

        title += (
            f" | W Bottom"
        )

    fig, axes = mpf.plot(

        plot_df,

        type="candle",

        style="yahoo",

        addplot=addplots,

        title="\n" + title,

        ylabel="Price",

        volume=True,

        ylabel_lower="Volume",

        figratio=(16, 9),

        figscale=1.0,

        returnfig=True
    )

    return fig


# ============================================================
# PDF產生器
# ============================================================

def create_pdf(
    matches,
    params,
    scan_time
):

    buffer = io.BytesIO()

    with PdfPages(
        buffer
    ) as pdf:

        # ====================================================
        # 第一頁：總覽
        # ====================================================

        fig = plt.figure(
            figsize=(11.69, 8.27)
        )

        fig.text(
            0.05,
            0.93,
            "台股 V2.1 全自動選股雷達",
            fontsize=20,
            fontweight="bold"
        )

        fig.text(
            0.05,
            0.88,
            f"掃描時間：{scan_time}",
            fontsize=10
        )

        fig.text(
            0.05,
            0.84,
            f"符合股票：{len(matches)} 支",
            fontsize=12
        )

        strategy_text = (
            f"條件：股價 > 週{params['ma_week']}MA；"
            f"成交量 ≥ {MIN_VOLUME_LOTS:,}張；"
            f"今日量 ≥ 5日均量 × "
            f"{params['vol_multiplier']:.1f}；"
            f"{params['breakout_days']}日創高 OR W底突破"
        )

        fig.text(
            0.05,
            0.79,
            strategy_text,
            fontsize=10
        )

        # ====================================================
        # 產業統計
        # ====================================================

        industry_counts = (
            pd.Series([
                m["group"]
                for m in matches
            ])
            .value_counts()
            .head(15)
        )

        ax = fig.add_axes(
            [0.10, 0.15, 0.80, 0.55]
        )

        industry_counts.sort_values().plot(
            kind="barh",
            ax=ax
        )

        ax.set_title(
            "入選股票產業分布"
        )

        ax.set_xlabel(
            "入選數量"
        )

        ax.set_ylabel(
            "產業"
        )

        plt.tight_layout()

        pdf.savefig(
            fig,
            bbox_inches="tight"
        )

        plt.close(fig)

        # ====================================================
        # 第二頁：股票總表
        # ====================================================

        rows = []

        for m in matches:

            rows.append({

                "股票":
                    f"{m['name']} "
                    f"({m['ticker'].split('.')[0]})",

                "市場":
                    m["market"],

                "產業":
                    m["group"],

                "收盤":
                    m["close"],

                "週MA":
                    m["ma_week_val"],

                "距MA":
                    f"{m['distance_to_week_ma_pct']:.2f}%",

                "今日量":
                    m["volume"],

                "5日均量":
                    m["volume_avg_5"],

                "量比":
                    f"{m['volume_ratio']:.2f}x",

                "40日高":
                    "是"
                    if m["is_breakout"]
                    else "—",

                "W底":
                    "是"
                    if m["is_w_bottom"]
                    else "—",

                "訊號":
                    m["signal_type"]
            })

        table_df = pd.DataFrame(
            rows
        )

        table_df = table_df.sort_values(
            by=[
                "產業",
                "量比"
            ],
            ascending=[
                True,
                False
            ]
        )

        # 分頁
        rows_per_page = 28

        for start in range(
            0,
            len(table_df),
            rows_per_page
        ):

            part = table_df.iloc[
                start:
                start + rows_per_page
            ]

            fig, ax = plt.subplots(
                figsize=(11.69, 8.27)
            )

            ax.axis("off")

            table = ax.table(
                cellText=part.values,
                colLabels=part.columns,
                loc="center",
                cellLoc="center"
            )

            table.auto_set_font_size(
                False
            )

            table.set_fontsize(
                7
            )

            table.scale(
                1,
                1.5
            )

            ax.set_title(
                f"入選股票總表 "
                f"({start + 1}-{min(start + rows_per_page, len(table_df))})",
                fontsize=14,
                pad=15
            )

            pdf.savefig(
                fig,
                bbox_inches="tight"
            )

            plt.close(fig)

        # ====================================================
        # 每檔股票詳細資料
        # ====================================================

        for m in matches:

            # ------------------------------------------------
            # 基本資訊頁
            # ------------------------------------------------

            fig = plt.figure(
                figsize=(11.69, 8.27)
            )

            fig.text(
                0.05,
                0.94,
                f"{m['name']} "
                f"({m['ticker'].split('.')[0]})",
                fontsize=18,
                fontweight="bold"
            )

            info_text = f"""
市場：{m['market']}
產業：{m['group']}
資料日期：{m['data_date']}

收盤價：{m['close']:.2f} 元
週{params['ma_week']}MA：{m['ma_week_val']:.2f} 元
距週MA：{m['distance_to_week_ma_pct']:.2f}%

今日成交量：{m['volume']:,} 張
5日均量：{m['volume_avg_5']:,.0f} 張
放量倍數：{m['volume_ratio']:.2f}x

{params['breakout_days']}日最高價：{m['previous_high']:.2f}
突破幅度：{m['breakout_distance_pct']:.2f}%

訊號：{m['signal_type']}

入選原因：
{'、'.join(m['reasons'])}
"""

            fig.text(
                0.08,
                0.78,
                info_text,
                fontsize=11,
                verticalalignment="top"
            )

            if m[
                "is_w_bottom"
            ]:

                w = m[
                    "w_info"
                ]

                w_text = f"""
W底結構：

左腳：{w['left_foot']:.2f}
右腳：{w['right_foot']:.2f}
頸線：{w['neck_high']:.2f}
左右腳差異：{w['foot_diff_pct']:.2f}%
"""

                fig.text(
                    0.55,
                    0.70,
                    w_text,
                    fontsize=11,
                    verticalalignment="top"
                )

            pdf.savefig(
                fig,
                bbox_inches="tight"
            )

            plt.close(fig)

            # ------------------------------------------------
            # K線
            # ------------------------------------------------

            try:

                chart_fig = create_stock_chart(
                    m,
                    params
                )

                if chart_fig is not None:

                    pdf.savefig(
                        chart_fig,
                        bbox_inches="tight"
                    )

                    plt.close(
                        chart_fig
                    )

            except Exception:
                pass

            # ------------------------------------------------
            # 股利
            # ------------------------------------------------

            try:

                div_df = (
                    get_dividend_history(
                        m["ticker"]
                    )
                )

                if (
                    div_df is not None
                    and not div_df.empty
                ):

                    div_fig = (
                        create_dividend_figure(
                            div_df
                        )
                    )

                    pdf.savefig(
                        div_fig,
                        bbox_inches="tight"
                    )

                    plt.close(
                        div_fig
                    )

            except Exception:
                pass

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "🔍 V2.1 全台股選股控制台"
)

st.sidebar.info(
    "本版本固定掃描全部上市＋上櫃股票，"
    "已移除成交金額熱門前150大選項。"
)

st.sidebar.divider()

st.sidebar.subheader(
    "⚙️ 技術策略參數"
)

params = {

    # ========================================================
    # 5日均量
    # ========================================================

    "vol_multiplier":
        st.sidebar.slider(
            "放量倍數（對比5日均量）",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1
        ),

    # ========================================================
    # W底
    # ========================================================

    "w_tolerance":
        st.sidebar.slider(
            "W底左右腳容錯率",
            min_value=1.0,
            max_value=15.0,
            value=6.0,
            step=0.5
        ) / 100.0,

    # ========================================================
    # 突破
    # ========================================================

    "breakout_days":
        st.sidebar.number_input(
            "突破回看期間（交易日）",
            min_value=10,
            max_value=60,
            value=40,
            step=1
        ),

    # ========================================================
    # 週MA
    # ========================================================

    "ma_week":
        st.sidebar.number_input(
            "長期趨勢均線（週MA）",
            min_value=10,
            max_value=40,
            value=20,
            step=1
        ),

    # ========================================================
    # W底參數
    # ========================================================

    "w_lookback":
        60,

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
### 🚀 V2.1 加速架構

① 批量下載日K  
↓  
② 已完成交易日  
↓  
③ 1,000張成交量過濾  
↓  
④ 5日均量放量過濾  
↓  
⑤ 週20MA趨勢  
↓  
⑥ 40日創高 / W底  
↓  
⑦ 最後才抓股利
"""
)


# ============================================================
# 開始掃描
# ============================================================

if st.sidebar.button(
    "🚀 開始全台股 V2.1 掃描",
    type="primary"
):

    scan_start = time.time()

    scan_time = (
        get_taiwan_now()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    # ========================================================
    # 股票清單
    # ========================================================

    with st.spinner(
        "正在取得全台上市＋上櫃股票清單..."
    ):

        stocks_info = (
            get_all_tw_stocks_info()
        )

    all_tickers = list(
        stocks_info.keys()
    )

    st.info(
        f"📊 本次將掃描 "
        f"**{len(all_tickers)}** 支上市＋上櫃股票"
    )

    # ========================================================
    # 批量下載
    # ========================================================

    st.subheader(
        "① 批量取得市場資料"
    )

    download_progress = st.progress(
        0
    )

    with st.spinner(
        "正在批量下載日K資料，請稍候..."
    ):

        daily_data = (
            batch_download_daily_data(
                all_tickers
            )
        )

    download_progress.progress(
        1.0
    )

    st.success(
        f"✅ 成功取得 "
        f"{len(daily_data)} 支股票資料"
    )

    # ========================================================
    # 技術篩選
    # ========================================================

    st.subheader(
        "② 技術條件分層篩選"
    )

    progress = st.progress(
        0
    )

    status = st.empty()

    matches = []

    total = len(
        daily_data
    )

    for idx, (
        ticker,
        df
    ) in enumerate(
        daily_data.items(),
        start=1
    ):

        result = quick_filter(
            ticker,
            df,
            params
        )

        if result is not None:

            info = stocks_info.get(
                ticker
            )

            if info is not None:

                result[
                    "name"
                ] = info[
                    "name"
                ]

                result[
                    "group"
                ] = info[
                    "group"
                ]

                result[
                    "market"
                ] = info[
                    "market"
                ]

                matches.append(
                    result
                )

        if (
            idx % 20 == 0
            or idx == total
        ):

            progress.progress(
                idx / total
            )

            status.text(
                f"技術分析："
                f"{idx}/{total} | "
                f"目前入選："
                f"{len(matches)} 支"
            )

    # ========================================================
    # 排序
    # ========================================================

    matches.sort(
        key=lambda x: (
            x["group"],
            -(
                int(x["is_breakout"])
                + int(x["is_w_bottom"])
            ),
            -x["volume_ratio"]
        )
    )

    elapsed = (
        time.time()
        - scan_start
    )

    st.success(
        f"🎉 技術掃描完成！"
        f" 共掃描 {total} 支，"
        f"符合條件 {len(matches)} 支，"
        f"耗時 {elapsed:.1f} 秒。"
    )

    # ========================================================
    # 結果
    # ========================================================

    if matches:

        st.subheader(
            f"✅ 入選股票 "
            f"（共 {len(matches)} 支）"
        )

        # ====================================================
        # 產業集中度
        # ====================================================

        industry_df = (
            pd.DataFrame({
                "產業": [
                    m["group"]
                    for m in matches
                ]
            })
            .value_counts()
            .reset_index(
                name="入選數量"
            )
        )

        industry_df[
            "占全部入選"
        ] = (
            industry_df[
                "入選數量"
            ]
            / len(matches)
            * 100
        ).round(1)

        st.markdown(
            "### 🏭 市場資金／強勢股產業集中度"
        )

        st.dataframe(
            industry_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # 總覽表
        # ====================================================

        st.markdown(
            "### 📋 入選股票總覽"
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

        # ====================================================
        # 依產業＋訊號＋量比排序
        # ====================================================

        summary_df = (
            summary_df
            .sort_values(
                by=[
                    "產業",
                    "訊號",
                    "放量倍數"
                ],
                ascending=[
                    True,
                    True,
                    False
                ]
            )
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

        st.caption(
            "PDF 使用 matplotlib 直接產生，"
            "不需要 reportlab。"
        )

        if st.button(
            "📄 產生完整 PDF 報告"
        ):

            with st.spinner(
                "正在取得入選股票股利資料並產生PDF..."
            ):

                # =================================================
                # 只有入選股票才抓股利
                # =================================================

                dividend_progress = st.progress(
                    0
                )

                for idx, m in enumerate(
                    matches,
                    start=1
                ):

                    m[
                        "div_history"
                    ] = get_dividend_history(
                        m["ticker"]
                    )

                    dividend_progress.progress(
                        idx / len(matches)
                    )

                pdf_bytes = create_pdf(
                    matches,
                    params,
                    scan_time
                )

            st.success(
                "✅ PDF 報告產生完成！"
            )

            filename = (
                f"台股V2.1選股報告_"
                f"{get_taiwan_now().strftime('%Y%m%d_%H%M')}.pdf"
            )

            st.download_button(
                label="⬇️ 一鍵下載 PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf"
            )

        # ====================================================
        # 詳細資料
        # ====================================================

        st.divider()

        st.subheader(
            "📊 入選股票詳細分析"
        )

        for m in matches:

            with st.expander(
                f"📌 {m['name']} "
                f"({m['ticker'].split('.')[0]})"
                f"｜{m['group']}"
                f"｜{m['signal_type']}"
            ):

                st.markdown(
                    f"""
                    **{m['market']}｜"
                    **產業：{m['group']}**
                    
                    資料日期：{m['data_date']}
                    """
                )

                # =================================================
                # 第一排
                # =================================================

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

                # =================================================
                # 第二排
                # =================================================

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
                        f"{params['breakout_days']}日最高",
                        f"{m['previous_high']:.2f}"
                    )

                with col4:

                    st.metric(
                        "突破幅度",
                        f"{m['breakout_distance_pct']:.2f}%"
                    )

                # =================================================
                # 入選原因
                # =================================================

                st.markdown(
                    "#### 🎯 入選原因"
                )

                for reason in m[
                    "reasons"
                ]:

                    st.success(
                        f"✅ {reason}"
                    )

                # =================================================
                # W底
                # =================================================

                if m[
                    "is_w_bottom"
                ]:

                    w = m[
                        "w_info"
                    ]

                    st.markdown(
                        "#### 🔵 W底結構"
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

                # =================================================
                # 停損距離
                # =================================================

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
                    f"📏 距離："
                    f"{distance:.2f}%\n\n"
                    f"{risk_label}"
                )

                # =================================================
                # 股利
                # =================================================

                if st.checkbox(
                    "📊 顯示近十年股利",
                    key=f"div_{m['ticker']}"
                ):

                    with st.spinner(
                        "取得股利資料..."
                    ):

                        div_df = (
                            get_dividend_history(
                                m["ticker"]
                            )
                        )

                    if (
                        div_df is not None
                        and not div_df.empty
                    ):

                        fig = (
                            create_dividend_figure(
                                div_df
                            )
                        )

                        st.pyplot(
                            fig
                        )

                        plt.close(
                            fig
                        )

                    else:

                        st.info(
                            "無可取得的近期股利資料。"
                        )

                # =================================================
                # K線
                # =================================================

                st.markdown(
                    "#### 📈 技術圖"
                )

                try:

                    chart_fig = (
                        create_stock_chart(
                            m,
                            params
                        )
                    )

                    if chart_fig is not None:

                        st.pyplot(
                            chart_fig
                        )

                        plt.close(
                            chart_fig
                        )

                except Exception as e:

                    st.warning(
                        f"K線圖產生失敗：{e}"
                    )

    else:

        st.warning(
            "ℹ️ 目前參數下沒有找到符合條件的股票。"
        )

# ============================================================
# 頁尾
# ============================================================

st.divider()

st.caption(
    "V2.1｜全台上市＋上櫃｜"
    "批量資料＋分層篩選｜"
    "5日均量｜週MA20｜"
    "40日創高 OR W底突破｜"
    "產業集中分析｜PDF報告"
)
