import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import twstock
import warnings
import time
import requests

from datetime import time as dt_time


# ============================================================
# Matplotlib 中文字型設定
# ============================================================

import matplotlib.font_manager as fm

available_fonts = {
    f.name
    for f in fm.fontManager.ttflist
}

chinese_fonts = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "Noto Sans CJK TC",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "PingFang TC",
    "PingFang SC",
    "Heiti TC",
    "Arial Unicode MS"
]

for font_name in chinese_fonts:

    if font_name in available_fonts:

        plt.rcParams["font.sans-serif"] = [
            font_name
        ]

        break

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 基本設定
# ============================================================

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="台股 V2 強勢突破全自動雷達",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 台股 V2 全自動選股雷達")

st.caption(
    "V2 加速版：全台上市＋上櫃｜股本過濾｜"
    "已完成交易日｜週20MA｜5日均量放量｜"
    "40日創高 OR W底突破｜產業集中分析"
)


# ============================================================
# 常數
# ============================================================

TW_TZ = "Asia/Taipei"

MIN_VOLUME_LOTS = 1000

DAILY_HISTORY_PERIOD = "1y"

CHART_DAYS = 250

BATCH_SIZE = 80

REQUEST_TIMEOUT = 15


# ============================================================
# Streamlit Cache
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_capital_data():

    """
    取得上市＋上櫃公司實收資本額。

    回傳：
        {
            "2330": 234000000000,
            ...
        }

    單位：新台幣元
    """

    capital_map = {}

    # ========================================================
    # 上市
    # ========================================================

    try:

        url_twse = (
            "https://openapi.twse.com.tw/v1/"
            "opendata/t187ap03_L"
        )

        response = requests.get(
            url_twse,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, list):

            df = pd.DataFrame(data)

            if not df.empty:

                code_col = None
                capital_col = None

                for col in df.columns:

                    col_str = str(col)

                    if (
                        "公司代號" in col_str
                        or col_str == "Code"
                    ):
                        code_col = col

                    if (
                        "實收資本額" in col_str
                        or "實收資本" in col_str
                    ):
                        capital_col = col

                if (
                    code_col is not None
                    and capital_col is not None
                ):

                    for _, row in df.iterrows():

                        code = str(
                            row[code_col]
                        ).strip()

                        capital_raw = str(
                            row[capital_col]
                        ).replace(",", "").strip()

                        try:

                            capital = float(
                                capital_raw
                            )

                            if capital > 0:

                                capital_map[
                                    code
                                ] = capital

                        except Exception:
                            pass

    except Exception:
        pass


    # ========================================================
    # 上櫃
    # ========================================================

    try:

        url_tpex = (
            "https://www.tpex.org.tw/openapi/v1/"
            "mopsfin_t187ap03_O"
        )

        response = requests.get(
            url_tpex,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, list):

            df = pd.DataFrame(data)

            if not df.empty:

                code_col = None
                capital_col = None

                for col in df.columns:

                    col_str = str(col)

                    if (
                        "公司代號" in col_str
                        or col_str == "SecuritiesCompanyCode"
                    ):
                        code_col = col

                    if (
                        "實收資本額" in col_str
                        or "實收資本" in col_str
                    ):
                        capital_col = col

                if (
                    code_col is not None
                    and capital_col is not None
                ):

                    for _, row in df.iterrows():

                        code = str(
                            row[code_col]
                        ).strip()

                        capital_raw = str(
                            row[capital_col]
                        ).replace(",", "").strip()

                        try:

                            capital = float(
                                capital_raw
                            )

                            if capital > 0:

                                capital_map[
                                    code
                                ] = capital

                        except Exception:
                            pass

    except Exception:
        pass


    return capital_map


# ============================================================
# Yahoo 欄位處理
# ============================================================

def flatten_yfinance_columns(df):

    if df is None or df.empty:
        return df

    df = df.copy()

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        level0 = df.columns.get_level_values(0)

        if "Close" in level0:

            df.columns = level0

        else:

            level1 = (
                df.columns
                .get_level_values(-1)
            )

            if "Close" in level1:

                df.columns = level1

            else:

                df.columns = level0

    return df


# ============================================================
# 台灣時間
# ============================================================

def get_taiwan_now():

    return pd.Timestamp.now(
        tz=TW_TZ
    )


def is_market_closed_for_today():

    now = get_taiwan_now()

    if now.weekday() >= 5:
        return True

    market_close = dt_time(
        13,
        30
    )

    return (
        now.time()
        >= market_close
    )


# ============================================================
# 只保留已完成交易日
# ============================================================

def prepare_completed_daily_data(
    df_day
):

    if (
        df_day is None
        or df_day.empty
    ):

        return pd.DataFrame()

    df_day = df_day.copy()

    df_day.index = pd.to_datetime(
        df_day.index
    )

    now = get_taiwan_now()

    today = now.date()

    last_date = (
        df_day.index[-1].date()
    )

    if (
        last_date == today
        and not is_market_closed_for_today()
    ):

        df_day = (
            df_day
            .iloc[:-1]
            .copy()
        )

    return df_day


# ============================================================
# 建立完整週K
# ============================================================

def build_completed_weekly_data(
    df_day
):

    if (
        df_day is None
        or df_day.empty
    ):

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

    now = get_taiwan_now()

    if now.weekday() < 5:

        if (
            not weekly.empty
            and weekly.index[-1].date()
            >= now.date()
        ):

            weekly = (
                weekly
                .iloc[:-1]
            )

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
                or
                gap > max_gap
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
                * 100
            )

            if right_to_neck_pct < 3:
                continue

            candidates.append({

                "left_idx":
                    left_idx,

                "right_idx":
                    right_idx,

                "left_foot":
                    float(left_foot),

                "right_foot":
                    float(right_foot),

                "neck_high":
                    float(neck_high),

                "foot_diff_pct":
                    float(
                        foot_diff_pct
                        * 100
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
# 股票清單
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def get_all_tw_stocks_info():

    stocks_info = {}

    for code, info in twstock.codes.items():

        if (
            info.type == "股票"
            and info.market
            in [
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

                "name":
                    info.name,

                "group":
                    (
                        info.group
                        if info.group
                        else "其他"
                    ),

                "market":
                    info.market
            }

    return stocks_info


# ============================================================
# 快速第一層篩選
# ============================================================

def fast_filter_batch(
    batch_df,
    stocks_info,
    capital_map,
    min_capital,
    vol_multiplier,
    breakout_days
):

    candidates = []

    errors = []

    if (
        batch_df is None
        or batch_df.empty
    ):

        return candidates, errors

    if not isinstance(
        batch_df.columns,
        pd.MultiIndex
    ):

        return candidates, errors

    if "Close" not in (
        batch_df.columns
        .get_level_values(0)
    ):

        return candidates, errors

    close_df = (
        batch_df["Close"]
    )

    high_df = (
        batch_df["High"]
        if "High"
        in batch_df.columns.get_level_values(0)
        else None
    )

    volume_df = (
        batch_df["Volume"]
        if "Volume"
        in batch_df.columns.get_level_values(0)
        else None
    )

    if (
        high_df is None
        or volume_df is None
    ):

        return candidates, errors

    for ticker in close_df.columns:

        if ticker not in stocks_info:
            continue

        try:

            close_series = (
                close_df[ticker]
                .dropna()
            )

            high_series = (
                high_df[ticker]
                .dropna()
            )

            volume_series = (
                volume_df[ticker]
                .dropna()
            )

            if len(close_series) < 80:
                continue

            code = (
                stocks_info[ticker]["code"]
            )

            capital = (
                capital_map.get(code)
            )

            if (
                capital is not None
                and capital < min_capital
            ):
                continue

            today = (
                get_taiwan_now()
                .date()
            )

            last_date = (
                close_series.index[-1]
                .date()
            )

            if (
                last_date == today
                and not is_market_closed_for_today()
            ):

                close_series = (
                    close_series.iloc[:-1]
                )

                high_series = (
                    high_series.iloc[:-1]
                )

                volume_series = (
                    volume_series.iloc[:-1]
                )

            if len(close_series) < 80:
                continue

            latest_close = float(
                close_series.iloc[-1]
            )

            latest_volume = float(
                volume_series.iloc[-1]
            )

            latest_volume_lots = (
                latest_volume / 1000
            )

            if (
                latest_volume_lots
                < MIN_VOLUME_LOTS
            ):
                continue

            if len(volume_series) < 6:
                continue

            previous_5_volume = (
                volume_series
                .iloc[-6:-1]
            )

            avg_5_volume = (
                previous_5_volume.mean()
            )

            if (
                not np.isfinite(
                    avg_5_volume
                )
                or avg_5_volume <= 0
            ):
                continue

            volume_ratio = (
                latest_volume
                / avg_5_volume
            )

            if (
                volume_ratio
                < vol_multiplier
            ):
                continue

            if len(high_series) <= breakout_days:
                continue

            previous_high = (
                high_series
                .iloc[
                    -(breakout_days + 1):-1
                ]
                .max()
            )

            is_breakout = (
                latest_close
                >= previous_high
            )

            candidates.append({

                "ticker":
                    ticker,

                "latest_close":
                    latest_close,

                "latest_volume":
                    latest_volume,

                "latest_volume_lots":
                    latest_volume_lots,

                "avg_5_volume":
                    avg_5_volume,

                "volume_ratio":
                    volume_ratio,

                "previous_high":
                    previous_high,

                "is_breakout":
                    is_breakout,

                "capital":
                    capital,

                "data_date":
                    close_series.index[-1]
                    .strftime("%Y-%m-%d")
            })

        except Exception as e:

            errors.append({

                "ticker":
                    ticker,

                "error":
                    repr(e)
            })

    return candidates, errors


# ============================================================
# 第二層：單股完整技術分析
# ============================================================

def analyze_candidate(
    candidate,
    stocks_info,
    params
):

    ticker = (
        candidate["ticker"]
    )

    try:

        stock_obj = yf.Ticker(
            ticker
        )

        df_day = stock_obj.history(
            period="2y",
            interval="1d",
            auto_adjust=True
        )

        if (
            df_day is None
            or df_day.empty
        ):
            return None

        df_day = flatten_yfinance_columns(
            df_day
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
        )

        if len(df_day) < 120:
            return None

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

        close_day = (
            df_day["Close"]
            .to_numpy(float)
        )

        high_day = (
            df_day["High"]
            .to_numpy(float)
        )

        low_day = (
            df_day["Low"]
            .to_numpy(float)
        )

        vol_day = (
            df_day["Volume"]
            .to_numpy(float)
        )

        close_week = (
            df_week["Close"]
            .to_numpy(float)
        )

        # ====================================================
        # 週MA
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
        # 最新日線
        # ====================================================

        latest_close = (
            float(close_day[-1])
        )

        latest_volume = (
            float(vol_day[-1])
        )

        latest_volume_lots = (
            latest_volume / 1000
        )

        # ====================================================
        # 5日均量
        # ====================================================

        previous_5_volume = (
            vol_day[-6:-1]
        )

        if len(
            previous_5_volume
        ) < 5:
            return None

        avg_5_volume = (
            np.mean(
                previous_5_volume
            )
        )

        volume_ratio = (
            latest_volume
            / avg_5_volume
        )

        if (
            latest_volume_lots
            < MIN_VOLUME_LOTS
        ):
            return None

        if (
            volume_ratio
            < params["vol_multiplier"]
        ):
            return None

        # ====================================================
        # 40日突破
        # ====================================================

        breakout_days = (
            params["breakout_days"]
        )

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
        # W底
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

        is_w_bottom = (
            w_info["is_w_bottom"]
        )

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

        code = (
            stocks_info[ticker]["code"]
        )

        capital = (
            candidate.get("capital")
        )

        return {

            "status":
                "match",

            "ticker":
                ticker,

            "code":
                code,

            "name":
                stocks_info[ticker]["name"],

            "group":
                stocks_info[ticker]["group"],

            "market":
                stocks_info[ticker]["market"],

            "capital":
                capital,

            "data_date":
                df_day.index[-1]
                .strftime("%Y-%m-%d"),

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
                    avg_5_volume / 1000,
                    0
                ),

            "volume_ratio":
                round(
                    volume_ratio,
                    2
                ),

            "ma_week_val":
                round(
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

            "previous_high":
                round(
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
                bool(
                    is_breakout
                ),

            "is_w_bottom":
                bool(
                    is_w_bottom
                ),

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
# 最後才抓股利
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
            "Dividend":
                dividends
        })

        div_df["Year"] = (
            div_df.index.year
        )

        yearly_div = (
            div_df
            .groupby("Year")
            ["Dividend"]
            .sum()
            .reset_index()
        )

        yearly_div = (
            yearly_div
            .sort_values(
                "Year"
            )
            .tail(10)
        )

        yearly_div.columns = [
            "年份",
            "現金股利"
        ]

        yearly_div[
            "現金股利"
        ] = (
            yearly_div[
                "現金股利"
            ]
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
# 股利圖
# ============================================================

def plot_dividend_bar_chart(
    div_df
):

    # --------------------------------------------------------
    # 這裡仍然使用合理的 figsize，
    # 但最後交給 Streamlit 撐滿容器
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 4)
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

    # ========================================================
    # ★ 重要修改
    # ========================================================

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


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

    plot_df = (
        df_day
        .iloc[-CHART_DAYS:]
        .copy()
    )

    if plot_df.empty:
        return

    plot_df = (
        flatten_yfinance_columns(
            plot_df
        )
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
            [
                ma_week_val
            ] * len(plot_df),
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
                    [
                        neck_high
                    ] * len(plot_df),
                    color="orange",
                    linestyle="dashdot",
                    width=1.2
                )
            )

    title_parts = [
        f"{ticker}",
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
            "W-Bottom"
        )

    # ========================================================
    # ★ K線圖本體
    #
    # 不用固定 figsize
    # 讓 Streamlit 負責容器寬度
    # ========================================================

    fig, axes = mpf.plot(

        plot_df,

        type="candle",

        style="yahoo",

        addplot=addplots,

        title="\n"
        + " | ".join(
            title_parts
        ),

        ylabel="股價（元）",

        volume=True,

        ylabel_lower="Volume",

        figratio=(16, 9),

        figscale=1.0,

        returnfig=True
    )

    # ========================================================
    # ★ 重要修改
    # ========================================================

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "🔍 V2 全自動選股控制台"
)

st.sidebar.info(
    "本版本固定掃描全台上市＋上櫃股票。"
)

st.sidebar.divider()

st.sidebar.subheader(
    "⚙️ 技術策略參數"
)


# ============================================================
# 股本
# ============================================================

min_capital_yi = (
    st.sidebar.number_input(

        "最低股本（億元）",

        min_value=0.0,

        max_value=5000.0,

        value=20.0,

        step=1.0
    )
)

min_capital = (
    min_capital_yi
    * 100_000_000
)


# ============================================================
# 5日均量
# ============================================================

vol_multiplier = (
    st.sidebar.slider(

        "放量倍數（對比前5日均量）",

        min_value=1.0,

        max_value=5.0,

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

w_tolerance = (
    st.sidebar.slider(

        "W底左右腳容錯率",

        min_value=1.0,

        max_value=15.0,

        value=6.0,

        step=0.5
    )
    / 100.0
)


pivot_window = (
    st.sidebar.number_input(

        "W底 Pivot Low 判定寬度",

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


# ============================================================
# 參數
# ============================================================

params = {

    "min_capital":
        min_capital,

    "vol_multiplier":
        vol_multiplier,

    "breakout_days":
        breakout_days,

    "ma_week":
        ma_week,

    "w_tolerance":
        w_tolerance,

    "w_lookback":
        60,

    "pivot_window":
        pivot_window,

    "w_min_gap":
        w_min_gap,

    "w_max_gap":
        w_max_gap
}


# ============================================================
# Sidebar 條件
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    f"""
### 📌 目前條件

**股本：**
≥ {min_capital_yi:.0f} 億

**最低成交量：**
≥ {MIN_VOLUME_LOTS:,} 張

**放量：**
今日量 ≥ 前5日均量 × {vol_multiplier:.1f}

**趨勢：**
股價 > 週{ma_week}MA

**型態：**
{breakout_days}日創高 OR W底突破
"""
)


# ============================================================
# 開始掃描
# ============================================================

if st.sidebar.button(
    "🚀 開始 V2 全自動雷達掃描",
    type="primary"
):

    scan_start = time.time()

    # ========================================================
    # 股票清單
    # ========================================================

    stocks_info = (
        get_all_tw_stocks_info()
    )

    if not stocks_info:

        st.error(
            "❌ 無法取得台股股票清單。"
        )

        st.stop()

    st.info(
        f"股票清單取得完成："
        f"{len(stocks_info)} 支"
    )


    # ========================================================
    # 股本資料
    # ========================================================

    with st.spinner(
        "正在取得上市＋上櫃公司股本資料..."
    ):

        capital_map = (
            get_company_capital_data()
        )

    st.info(
        f"股本資料取得完成："
        f"{len(capital_map)} 家"
    )


    # ========================================================
    # 第一層：批量下載
    # ========================================================

    tickers = list(
        stocks_info.keys()
    )

    st.subheader(
        "🔎 第一階段：批量市場資料篩選"
    )

    progress = st.progress(0)

    status = st.empty()

    fast_candidates = []

    batch_errors = []

    total_batches = (
        int(
            np.ceil(
                len(tickers)
                / BATCH_SIZE
            )
        )
    )


    for batch_number, start in enumerate(
        range(
            0,
            len(tickers),
            BATCH_SIZE
        ),
        start=1
    ):

        batch_tickers = tickers[
            start:
            start + BATCH_SIZE
        ]

        try:

            batch_df = yf.download(

                batch_tickers,

                period=
                    DAILY_HISTORY_PERIOD,

                interval="1d",

                auto_adjust=True,

                progress=False,

                group_by="column",

                threads=False
            )

            candidates, errors = (
                fast_filter_batch(

                    batch_df=batch_df,

                    stocks_info=stocks_info,

                    capital_map=capital_map,

                    min_capital=
                        min_capital,

                    vol_multiplier=
                        vol_multiplier,

                    breakout_days=
                        breakout_days
                )
            )

            fast_candidates.extend(
                candidates
            )

            batch_errors.extend(
                errors
            )

        except Exception as e:

            batch_errors.append({

                "ticker":
                    ",".join(
                        batch_tickers
                    ),

                "error":
                    repr(e)
            })

        progress.progress(
            batch_number
            / total_batches
        )

        status.text(
            f"批量掃描："
            f"{batch_number}/"
            f"{total_batches}｜"
            f"第一層候選："
            f"{len(fast_candidates)} 支"
        )

        time.sleep(0.25)


    progress.progress(1.0)


    # ========================================================
    # 第一層結果
    # ========================================================

    st.success(
        f"第一階段完成："
        f"從 {len(tickers)} 支股票中"
        f"留下 {len(fast_candidates)} 支候選股。"
    )

    if not fast_candidates:

        st.warning(
            "⚠️ 第一階段沒有候選股票。"
        )

        st.info(
            "請優先檢查："
            "股本門檻、最低成交量、"
            "5日均量放量倍數。"
        )

        st.stop()


    # ========================================================
    # 第二層
    # ========================================================

    st.subheader(
        "📐 第二階段：完整技術型態分析"
    )

    progress2 = st.progress(0)

    status2 = st.empty()

    matches = []

    total_candidates = (
        len(fast_candidates)
    )


    for i, candidate in enumerate(
        fast_candidates,
        start=1
    ):

        result = analyze_candidate(
            candidate=candidate,

            stocks_info=stocks_info,

            params=params
        )

        if result is not None:

            matches.append(
                result
            )

        progress2.progress(
            i
            / total_candidates
        )

        status2.text(
            f"完整分析："
            f"{i}/"
            f"{total_candidates}｜"
            f"目前入選："
            f"{len(matches)} 支"
        )

        time.sleep(0.05)


    progress2.progress(1.0)


    # ========================================================
    # 第三階段：股利
    # ========================================================

    st.subheader(
        "💰 第三階段：取得入選股利資料"
    )

    dividend_progress = (
        st.progress(0)
    )

    for i, m in enumerate(
        matches,
        start=1
    ):

        m["div_history"] = (
            get_dividend_history(
                m["ticker"]
            )
        )

        dividend_progress.progress(
            i
            / len(matches)
            if matches
            else 1
        )

        time.sleep(0.05)


    # ========================================================
    # 排序
    # ========================================================

    matches.sort(

        key=lambda x: (

            x["group"],

            -int(
                x["is_breakout"]
                + x["is_w_bottom"]
            ),

            -x["volume_ratio"],

            -x[
                "distance_to_week_ma_pct"
            ]
        )
    )


    elapsed = (
        time.time()
        - scan_start
    )


    # ========================================================
    # 完成
    # ========================================================

    st.success(
        f"""
🎉 V2 掃描完成！

總掃描：{len(tickers)} 支

第一層候選：
{len(fast_candidates)} 支

最終入選：
{len(matches)} 支

總耗時：
{elapsed:.1f} 秒
"""
    )


    # ========================================================
    # 產業集中分析
    # ========================================================

    if matches:

        st.subheader(
            "🏭 市場資金／強勢股產業集中度"
        )

        industry_df = (
            pd.DataFrame([
                {
                    "產業":
                        m["group"],

                    "入選家數":
                        1
                }
                for m in matches
            ])
            .groupby(
                "產業"
            )
            .sum()
            .sort_values(
                "入選家數",
                ascending=False
            )
            .reset_index()
        )

        industry_df[
            "占全部入選"
        ] = (
            industry_df["入選家數"]
            / len(matches)
            * 100
        ).round(1)

        st.dataframe(
            industry_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "產業集中度是依本次選股條件的入選股票數計算，"
            "可用來觀察目前強勢股是否集中於特定產業，"
            "不等同於真正的市場資金流向。"
        )


        # ====================================================
        # 總覽表
        # ====================================================

        st.subheader(
            f"📋 入選股票總覽（共 {len(matches)} 支）"
        )

        summary_rows = []

        for m in matches:

            capital_text = "—"

            if (
                m["capital"]
                is not None
            ):

                capital_text = (
                    f"{m['capital'] / 100_000_000:.1f}"
                )

            summary_rows.append({

                "產業":
                    m["group"],

                "股票":
                    (
                        f"{m['name']} "
                        f"({m['code']})"
                    ),

                "市場":
                    m["market"],

                "股本(億)":
                    capital_text,

                "收盤價":
                    m["close"],

                f"週{ma_week}MA":
                    m["ma_week_val"],

                "距週MA":
                    (
                        f"{m['distance_to_week_ma_pct']:.2f}%"
                    ),

                "今日量(張)":
                    f"{m['volume']:,}",

                "前5日均量":
                    f"{m['volume_avg_5']:,.0f}",

                "放量倍數":
                    f"{m['volume_ratio']:.2f}x",

                f"{breakout_days}日創高":
                    (
                        "✅"
                        if m["is_breakout"]
                        else "—"
                    ),

                "W底突破":
                    (
                        "✅"
                        if m["is_w_bottom"]
                        else "—"
                    ),

                "訊號":
                    m["signal_type"],

                "資料日期":
                    m["data_date"]
            })

        summary_df = (
            pd.DataFrame(
                summary_rows
            )
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # 詳細資料
        # ====================================================

        st.divider()

        st.subheader(
            "📊 入選股票詳細分析"
        )


        for m in matches:

            # =================================================
            # ★ 修正原本 markdown 字串
            # =================================================

            st.markdown(
                f"""
### 📌 {m['name']}（{m['code']}）

**{m['market']}｜產業：{m['group']}｜資料日期：{m['data_date']}**
"""
            )


            # ------------------------------------------------
            # 第一排
            # ------------------------------------------------

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
                    f"週{ma_week}MA",
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
                    (
                        f"{m['capital'] / 100_000_000:.1f} 億"
                        if m["capital"]
                        else "—"
                    )
                )


            # ------------------------------------------------
            # 第二排
            # ------------------------------------------------

            c1, c2, c3 = (
                st.columns(3)
            )

            with c1:

                st.metric(
                    "今日成交量",
                    f"{m['volume']:,} 張"
                )

            with c2:

                st.metric(
                    "前5日均量",
                    f"{m['volume_avg_5']:,.0f} 張"
                )

            with c3:

                st.metric(
                    "放量倍數",
                    f"{m['volume_ratio']:.2f}x"
                )


            # ------------------------------------------------
            # 突破
            # ------------------------------------------------

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    f"{breakout_days}日最高價",
                    f"{m['previous_high']:.2f}"
                )

            with c2:

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
            # W底
            # ------------------------------------------------

            if m["is_w_bottom"]:

                w = m["w_info"]

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


            # ------------------------------------------------
            # 停損
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

            st.markdown(
                f"""
🛡️ **週{ma_week}MA：**
**{m['ma_week_val']:.2f} 元**

📏 **目前價格距離週MA：**
**{distance:.2f}%**

{risk_label}

> 注意：週MA僅作為技術面停損參考，
> 並不代表實際最大損失。
"""
            )


            # ------------------------------------------------
            # 股利
            # ------------------------------------------------

            if not (
                m["div_history"]
                .empty
            ):

                st.markdown(
                    "#### 📊 近十年現金股利"
                )

                plot_dividend_bar_chart(
                    m["div_history"]
                )

            else:

                st.info(
                    "沒有可取得的近期股利資料。"
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

                ma_week_val=
                    m["ma_week_val"],

                breakout_days=
                    breakout_days,

                is_breakout=
                    m["is_breakout"],

                w_info=
                    m["w_info"]
            )

            st.divider()


    else:

        st.warning(
            "ℹ️ 目前參數下沒有符合條件的股票。"
        )


    # ========================================================
    # 批量資料錯誤
    # ========================================================

    if batch_errors:

        with st.expander(
            f"⚠️ 批量資料錯誤 "
            f"（{len(batch_errors)} 筆）"
        ):

            error_df = (
                pd.DataFrame(
                    batch_errors
                )
            )

            st.dataframe(
                error_df,
                use_container_width=True,
                hide_index=True
            )
