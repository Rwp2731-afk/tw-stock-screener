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
# 基本設定
# ============================================================

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="台股 V2.2.2 強勢突破全自動雷達",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 台股 V2.2.2 全自動選股雷達")

st.caption(
    "V2.2.2 成交量同步修正版：全台上市＋上櫃｜股本過濾｜"
    "Yahoo歷史資料＋TWSE/TPEX官方最新行情｜"
    "已完成交易日｜週20MA｜前5日均量放量｜"
    "40日創高 OR W底突破｜產業集中"
)


# ============================================================
# 常數
# ============================================================

TW_TZ = "Asia/Taipei"

# 最低成交量：1,000 張
MIN_VOLUME_LOTS = 1000

# 第一階段使用 1 年
DAILY_HISTORY_PERIOD = "1y"

# 第二階段使用 2 年
FULL_HISTORY_PERIOD = "2y"

# K 線顯示最近 250 個交易日
CHART_DAYS = 250

# Yahoo 批次下載數量
BATCH_SIZE = 80

# API timeout
REQUEST_TIMEOUT = 15

# 第一階段最低日線資料數
MIN_DAILY_ROWS = 80

# 第二階段最低完整資料數
MIN_FULL_ROWS = 120


# ============================================================
# 官方最新行情 API
# ============================================================

TWSE_STOCK_DAY_ALL_URL = (
    "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
)

TPEX_MAINBOARD_QUOTE_URL = (
    "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
)

OFFICIAL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*"
}


# ============================================================
# 公司股本資料
#
# 注意：
# 本版本暫時「不修改」股本抓不到的處理。
#
# 如果找不到股本：
#     capital_map 沒有該股票
#     capital = None
#     不會因此剔除股票
#
# 這是目前刻意保留的設計。
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_capital_data():

    capital_map = {}

    # --------------------------------------------------------
    # TWSE 上市
    # --------------------------------------------------------

    try:

        url_twse = (
            "https://openapi.twse.com.tw/v1/"
            "opendata/t187ap03_L"
        )

        response = requests.get(
            url_twse,
            headers=OFFICIAL_HEADERS,
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

                        capital_raw = (
                            str(row[capital_col])
                            .replace(",", "")
                            .strip()
                        )

                        try:

                            capital = float(
                                capital_raw
                            )

                            if capital > 0:
                                capital_map[code] = capital

                        except Exception:
                            pass

    except Exception:
        pass


    # --------------------------------------------------------
    # TPEX 上櫃
    #
    # 保留 PaidInCapital / Capital 判斷。
    # --------------------------------------------------------

    try:

        url_tpex = (
            "https://www.tpex.org.tw/openapi/v1/"
            "mopsfin_t187ap03_O"
        )

        response = requests.get(
            url_tpex,
            headers=OFFICIAL_HEADERS,
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
                        or col_str == "PaidInCapital"
                        or col_str == "Capital"
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

                        capital_raw = (
                            str(row[capital_col])
                            .replace(",", "")
                            .strip()
                        )

                        try:

                            capital = float(
                                capital_raw
                            )

                            if capital > 0:
                                capital_map[code] = capital

                        except Exception:
                            pass

    except Exception:
        pass

    return capital_map


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
# 官方日期解析
# ============================================================

def parse_official_date(date_raw):

    if date_raw is None:
        return pd.NaT

    try:

        if pd.isna(date_raw):
            return pd.NaT

    except Exception:
        pass

    text = str(
        date_raw
    ).strip()

    if not text:
        return pd.NaT


    # --------------------------------------------------------
    # 純數字日期
    #
    # 例如：
    # 1150828
    # 20260828
    # --------------------------------------------------------

    if text.isdigit():

        if len(text) == 7:

            try:

                year = (
                    int(text[:3])
                    + 1911
                )

                month = int(
                    text[3:5]
                )

                day = int(
                    text[5:7]
                )

                return pd.Timestamp(
                    year,
                    month,
                    day
                )

            except Exception:
                pass


        if len(text) == 8:

            try:

                year = int(
                    text[:4]
                )

                month = int(
                    text[4:6]
                )

                day = int(
                    text[6:8]
                )

                return pd.Timestamp(
                    year,
                    month,
                    day
                )

            except Exception:
                pass


    # --------------------------------------------------------
    # YYYY/MM/DD 或民國年
    # --------------------------------------------------------

    if "/" in text:

        parts = text.split("/")

        if len(parts) == 3:

            try:

                year = int(
                    parts[0]
                )

                month = int(
                    parts[1]
                )

                day = int(
                    parts[2]
                )

                if year < 1911:
                    year += 1911

                return pd.Timestamp(
                    year,
                    month,
                    day
                )

            except Exception:
                pass


    # --------------------------------------------------------
    # 最後交給 pandas
    # --------------------------------------------------------

    try:

        result = pd.to_datetime(
            text,
            errors="coerce"
        )

        if pd.isna(result):
            return pd.NaT

        return pd.Timestamp(
            result
        ).normalize()

    except Exception:

        return pd.NaT


# ============================================================
# 官方最新行情
#
# 回傳：
#
# {
#     "2330": {
#         "market": "上市",
#         "date": Timestamp(...),
#         "date_raw": "...",
#         "Open": ...,
#         "High": ...,
#         "Low": ...,
#         "Close": ...,
#         "Volume": ...
#     }
# }
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_official_latest_quotes():

    quotes = {}


    # ========================================================
    # TWSE
    # ========================================================

    try:

        response = requests.get(
            TWSE_STOCK_DAY_ALL_URL,
            headers=OFFICIAL_HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, list):

            for row in data:

                code = str(
                    row.get(
                        "Code",
                        ""
                    )
                ).strip()

                if not code:
                    continue

                try:

                    date_raw = str(
                        row.get(
                            "Date",
                            ""
                        )
                    ).strip()

                    official_date = (
                        parse_official_date(
                            date_raw
                        )
                    )

                    if pd.isna(
                        official_date
                    ):
                        continue


                    close = float(
                        str(
                            row.get(
                                "ClosingPrice",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    volume = float(
                        str(
                            row.get(
                                "TradeVolume",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    if (
                        close <= 0
                        or volume <= 0
                    ):
                        continue


                    open_price = float(
                        str(
                            row.get(
                                "OpeningPrice",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    high_price = float(
                        str(
                            row.get(
                                "HighestPrice",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    low_price = float(
                        str(
                            row.get(
                                "LowestPrice",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    quotes[code] = {

                        "market": "上市",

                        "date": (
                            official_date
                        ),

                        "date_raw": (
                            date_raw
                        ),

                        "Open": (
                            open_price
                        ),

                        "High": (
                            high_price
                        ),

                        "Low": (
                            low_price
                        ),

                        "Close": (
                            close
                        ),

                        "Volume": (
                            volume
                        )
                    }

                except Exception:

                    continue

    except Exception:

        pass


    # ========================================================
    # TPEX
    # ========================================================

    try:

        response = requests.get(
            TPEX_MAINBOARD_QUOTE_URL,
            headers=OFFICIAL_HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, list):

            for row in data:

                code = str(
                    row.get(
                        "SecuritiesCompanyCode",
                        ""
                    )
                ).strip()

                if not code:
                    continue

                try:

                    # ------------------------------------------------
                    # TPEX 日期欄位可能因 API 格式而不同
                    # ------------------------------------------------

                    date_raw = ""

                    possible_date_keys = [

                        "Date",

                        "date",

                        "日期",

                        "資料日期",

                        "交易日期"

                    ]

                    for key in possible_date_keys:

                        if key in row:

                            value = str(
                                row.get(
                                    key,
                                    ""
                                )
                            ).strip()

                            if value:

                                date_raw = value

                                break


                    official_date = (
                        parse_official_date(
                            date_raw
                        )
                    )

                    if pd.isna(
                        official_date
                    ):
                        continue


                    close = float(
                        str(
                            row.get(
                                "Close",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    volume = float(
                        str(
                            row.get(
                                "TradingShares",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    if (
                        close <= 0
                        or volume <= 0
                    ):
                        continue


                    open_price = float(
                        str(
                            row.get(
                                "Open",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    high_price = float(
                        str(
                            row.get(
                                "High",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    low_price = float(
                        str(
                            row.get(
                                "Low",
                                "0"
                            )
                        )
                        .replace(
                            ",",
                            ""
                        )
                        or 0
                    )


                    quotes[code] = {

                        "market": "上櫃",

                        "date": (
                            official_date
                        ),

                        "date_raw": (
                            date_raw
                        ),

                        "Open": (
                            open_price
                        ),

                        "High": (
                            high_price
                        ),

                        "Low": (
                            low_price
                        ),

                        "Close": (
                            close
                        ),

                        "Volume": (
                            volume
                        )
                    }

                except Exception:

                    continue

    except Exception:

        pass


    return quotes


# ============================================================
# 官方行情日期診斷
# ============================================================

def get_official_date_summary(
    quotes
):

    if not quotes:
        return {}

    dates = []

    for quote in quotes.values():

        date_value = quote.get(
            "date"
        )

        if (
            date_value is not None
            and not pd.isna(date_value)
        ):

            dates.append(
                pd.Timestamp(
                    date_value
                ).normalize()
            )

    if not dates:
        return {}


    summary = (
        pd.Series(dates)
        .value_counts()
        .sort_index(
            ascending=False
        )
    )

    result = {}

    for date_value, count in summary.items():

        result[
            date_value.strftime(
                "%Y-%m-%d"
            )
        ] = int(count)

    return result


def get_latest_official_date(
    quotes
):

    if not quotes:
        return None

    dates = []

    for quote in quotes.values():

        date_value = quote.get(
            "date"
        )

        if (
            date_value is not None
            and not pd.isna(date_value)
        ):

            dates.append(
                pd.Timestamp(
                    date_value
                ).normalize()
            )

    if not dates:
        return None

    return max(dates)


# ============================================================
# Yahoo 欄位處理
# ============================================================

def flatten_yfinance_columns(
    df
):

    if df is None or df.empty:
        return df

    df = df.copy()

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        level0 = (
            df.columns
            .get_level_values(0)
        )

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
# 將官方最新行情寫入 DataFrame
#
# 注意：
# 這個函式現在只負責：
#
# 1. 覆蓋相同日期
# 2. 新增官方日期
#
# 真正的「日期對齊」由後面的
# align_to_official_latest_date() 負責。
# ============================================================

def apply_official_latest_quote(
    df,
    stock_id
):

    if df is None or df.empty:
        return df

    quotes = (
        get_official_latest_quotes()
    )

    if stock_id not in quotes:
        return df

    quote = quotes[
        stock_id
    ]

    try:

        df = df.copy()

        index = pd.to_datetime(
            df.index
        )

        if getattr(
            index,
            "tz",
            None
        ) is not None:

            index = index.tz_localize(
                None
            )

        df.index = index.normalize()


        official_date = pd.Timestamp(
            quote["date"]
        ).normalize()


        today = pd.Timestamp(
            get_taiwan_now().date()
        ).normalize()


        if official_date > today:
            return df


        values = {

            "Open": quote["Open"],

            "High": quote["High"],

            "Low": quote["Low"],

            "Close": quote["Close"],

            "Volume": quote["Volume"]

        }


        if official_date in df.index:

            for col, value in values.items():

                if col in df.columns:

                    df.loc[
                        official_date,
                        col
                    ] = value

        else:

            new_row = pd.DataFrame(
                [values],
                index=[
                    official_date
                ]
            )

            df = pd.concat(
                [
                    df,
                    new_row
                ]
            )


        return df.sort_index()

    except Exception:

        return df


# ============================================================
# V2.2.2 核心：
# 強制資料不能超過「該股票官方最新交易日」
#
# 這是本次最重要的融合修正。
#
# 舉例：
#
# 官方最新 = 2026-08-27
# Yahoo     = 已經有 2026-08-28
#
# 如果只「補官方 8/27」
# 8/28 仍然會留在 DataFrame 裡。
#
# sort_index() 後：
# 8/28 反而會成為最新資料。
#
# 因此這裡必須直接刪除：
#
#     > 官方最新交易日
#
# 的所有資料。
# ============================================================

def align_to_official_latest_date(
    df,
    code,
    official_quotes
):

    if (
        df is None
        or df.empty
    ):
        return df

    quote = official_quotes.get(
        code
    )

    # 如果官方資料沒有這檔，
    # 暫時保留原資料，不新增條件。
    if quote is None:
        return df

    try:

        df = df.copy()

        official_date = pd.Timestamp(
            quote["date"]
        ).normalize()

        index = pd.to_datetime(
            df.index
        )

        if getattr(
            index,
            "tz",
            None
        ) is not None:

            index = index.tz_localize(
                None
            )

        df.index = index.normalize()

        # ----------------------------------------------------
        # 關鍵：
        # 只保留 <= 官方最新交易日
        # ----------------------------------------------------

        df = df[
            df.index <= official_date
        ].copy()

        return df.sort_index()

    except Exception:

        return df


# ============================================================
# V2.2.2：
# 官方行情同步核心
#
# 第一、第二階段共用。
#
# 流程：
#
# 1. 找官方最新行情
# 2. 用官方 OHLCV 覆蓋 / 新增
# 3. 刪除官方日期之後的 Yahoo 資料
# 4. 排序
#
# 如此可以確保：
# 第一階段與第二階段使用相同的官方日期。
# ============================================================

def synchronize_latest_volume(
    stock_df,
    code,
    official_quotes
):

    if (
        stock_df is None
        or stock_df.empty
    ):
        return stock_df

    stock_df = stock_df.copy()

    quote = official_quotes.get(
        code
    )

    if quote is None:
        return stock_df


    official_date = pd.Timestamp(
        quote["date"]
    ).normalize()


    today = pd.Timestamp(
        get_taiwan_now().date()
    ).normalize()


    if official_date > today:
        return stock_df


    # --------------------------------------------------------
    # 官方最新 OHLCV
    # --------------------------------------------------------

    values = {

        "Open": quote["Open"],

        "High": quote["High"],

        "Low": quote["Low"],

        "Close": quote["Close"],

        "Volume": quote["Volume"]

    }


    # --------------------------------------------------------
    # 覆蓋相同官方日期
    # --------------------------------------------------------

    if official_date in stock_df.index:

        for col, value in values.items():

            if col in stock_df.columns:

                stock_df.loc[
                    official_date,
                    col
                ] = value


    # --------------------------------------------------------
    # 若 Yahoo 沒有官方日期，新增
    # --------------------------------------------------------

    else:

        new_row = pd.DataFrame(
            [values],
            index=[
                official_date
            ]
        )

        stock_df = pd.concat(
            [
                stock_df,
                new_row
            ]
        )


    # --------------------------------------------------------
    # V2.2.2 關鍵：
    # 刪除官方日期之後的資料
    # --------------------------------------------------------

    stock_df = (
        stock_df[
            stock_df.index
            <= official_date
        ]
        .sort_index()
    )


    return stock_df


# ============================================================
# 只保留已完成交易日
#
# latest_allowed_date：
# 若有官方日期，優先使用官方日期作為資料上限。
# ============================================================

def prepare_completed_daily_data(
    df_day,
    latest_allowed_date=None
):

    if (
        df_day is None
        or df_day.empty
    ):
        return pd.DataFrame()

    df_day = df_day.copy()

    index = pd.to_datetime(
        df_day.index
    )

    if getattr(
        index,
        "tz",
        None
    ) is not None:

        index = index.tz_localize(
            None
        )

    df_day.index = index.normalize()


    # --------------------------------------------------------
    # 若有官方最新日期：
    # 不允許資料超過官方日期。
    # --------------------------------------------------------

    if latest_allowed_date is not None:

        latest_allowed_date = (
            pd.Timestamp(
                latest_allowed_date
            ).normalize()
        )

        df_day = df_day[
            df_day.index
            <= latest_allowed_date
        ].copy()


    if df_day.empty:
        return pd.DataFrame()


    # --------------------------------------------------------
    # 台股當日尚未收盤：
    # 不使用今日資料。
    #
    # 但若官方資料已經明確提供今日完整行情，
    # 且現在已經收盤，則保留。
    # --------------------------------------------------------

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
            df_day.iloc[:-1]
            .copy()
        )


    return df_day


# ============================================================
# 建立完整週K
#
# 本次修正：
# 不再單純依「今天星期幾」判斷。
#
# 而是依照：
# 「最後一筆已完成日線資料」
# 判斷最後一週是否完成。
#
# 這對：
# - 週末
# - 週五休市
# - 國定假日
# 更安全。
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


    df_day = df_day.copy()

    df_day = df_day.sort_index()


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


    # --------------------------------------------------------
    # 判斷最後一週是否為完整週
    #
    # 如果最後日線日期還沒到該週五，
    # 就排除最後一週。
    #
    # 如果最後交易日就是週五，
    # 則保留。
    #
    # 如果週五休市，最後交易日可能是週四，
    # 此時該週仍應視為未完成週，
    # 因為當週交易週尚未正常結束前無法單純靠星期判斷。
    #
    # 這裡採用：
    # 若最後日線日期 < weekly period end，
    # 則排除該週。
    # --------------------------------------------------------

    latest_daily_date = pd.Timestamp(
        df_day.index[-1]
    ).normalize()

    latest_week_end = pd.Timestamp(
        weekly.index[-1]
    ).normalize()


    if latest_daily_date < latest_week_end:

        weekly = (
            weekly.iloc[:-1]
            .copy()
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


    pivot_lows = (
        calculate_pivot_lows(
            lows,
            pivot_window
        )
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


            if neck_high <= max(
                left_foot,
                right_foot
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

                "left_idx": (
                    left_idx
                ),

                "right_idx": (
                    right_idx
                ),

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

        "left_idx": (
            best["left_idx"]
        ),

        "right_idx": (
            best["right_idx"]
        ),

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

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
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

                "group": (
                    info.group
                    if info.group
                    else "其他"
                ),

                "market": info.market

            }


    return stocks_info


# ============================================================
# 清理單股資料
# ============================================================

def clean_single_stock_data(
    df,
    ticker
):

    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()


    df = df.copy()


    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        try:

            if ticker in (
                df.columns
                .get_level_values(-1)
            ):

                df = df.xs(
                    ticker,
                    axis=1,
                    level=-1,
                    drop_level=True
                )


            elif ticker in (
                df.columns
                .get_level_values(0)
            ):

                df = df.xs(
                    ticker,
                    axis=1,
                    level=0,
                    drop_level=True
                )


        except Exception:

            df = (
                flatten_yfinance_columns(
                    df
                )
            )


    else:

        df = (
            flatten_yfinance_columns(
                df
            )
        )


    required_cols = [

        "Open",

        "High",

        "Low",

        "Close",

        "Volume"

    ]


    if not all(
        col in df.columns
        for col in required_cols
    ):

        return pd.DataFrame()


    df = df[
        required_cols
    ].copy()


    df = (
        df
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna()
    )


    index = pd.to_datetime(
        df.index
    )


    if getattr(
        index,
        "tz",
        None
    ) is not None:

        index = index.tz_localize(
            None
        )


    df.index = index.normalize()


    return df


# ============================================================
# 第一階段快速篩選
#
# V2.2.2：
# 第一階段先同步官方 OHLCV，
# 再做成交量與 40 日突破篩選。
# ============================================================

def fast_filter_batch(
    batch_df,
    stocks_info,
    capital_map,
    min_capital,
    vol_multiplier,
    breakout_days,
    official_quotes
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


    level0 = (
        batch_df.columns
        .get_level_values(0)
    )


    if "Close" not in level0:
        return candidates, errors

    if "High" not in level0:
        return candidates, errors

    if "Volume" not in level0:
        return candidates, errors


    close_df = (
        batch_df["Close"]
    )

    high_df = (
        batch_df["High"]
    )

    volume_df = (
        batch_df["Volume"]
    )


    today = (
        get_taiwan_now().date()
    )

    market_closed = (
        is_market_closed_for_today()
    )


    for ticker in close_df.columns:

        if ticker not in stocks_info:
            continue


        try:

            stock_df = pd.DataFrame({

                "Close": (
                    close_df[ticker]
                ),

                "High": (
                    high_df[ticker]
                ),

                "Volume": (
                    volume_df[ticker]
                )

            })


            index = pd.to_datetime(
                stock_df.index
            )


            if getattr(
                index,
                "tz",
                None
            ) is not None:

                index = index.tz_localize(
                    None
                )


            stock_df.index = (
                index.normalize()
            )


            stock_df = (
                stock_df
                .replace(
                    [
                        np.inf,
                        -np.inf
                    ],
                    np.nan
                )
                .dropna(
                    subset=[
                        "Close",
                        "High",
                        "Volume"
                    ]
                )
            )


            if len(stock_df) < (
                MIN_DAILY_ROWS
            ):

                continue


            code = stocks_info[
                ticker
            ]["code"]


            # =================================================
            # V2.2.2：
            # 官方 OHLCV 同步
            # =================================================

            stock_df = (
                synchronize_latest_volume(
                    stock_df,
                    code,
                    official_quotes
                )
            )


            if (
                stock_df is None
                or stock_df.empty
            ):

                continue


            # =================================================
            # 再次確認官方日期
            #
            # 這一層是為了讓邏輯更明確。
            # =================================================

            quote = official_quotes.get(
                code
            )


            latest_allowed_date = None

            if quote is not None:

                latest_allowed_date = (
                    pd.Timestamp(
                        quote["date"]
                    ).normalize()
                )


            stock_df = (
                prepare_completed_daily_data(
                    stock_df,
                    latest_allowed_date
                )
            )


            if stock_df.empty:
                continue


            if len(stock_df) < (
                MIN_DAILY_ROWS
            ):

                continue


            close_series = (
                stock_df["Close"]
            )

            high_series = (
                stock_df["High"]
            )

            volume_series = (
                stock_df["Volume"]
            )


            # =================================================
            # 股本
            #
            # capital = None 時：
            # 暫時不剔除。
            # =================================================

            capital = capital_map.get(
                code
            )


            if (
                capital is not None
                and capital < min_capital
            ):

                continue


            # =================================================
            # 最新價格 / 成交量
            # =================================================

            latest_close = float(
                close_series.iloc[-1]
            )


            latest_volume = float(
                volume_series.iloc[-1]
            )


            latest_volume_lots = (
                latest_volume / 1000
            )


            # =================================================
            # 最低成交量
            # =================================================

            if (
                latest_volume_lots
                < MIN_VOLUME_LOTS
            ):

                continue


            # =================================================
            # 前 5 日均量
            # =================================================

            if len(volume_series) < 6:
                continue


            previous_5_volume = (
                volume_series.iloc[-6:-1]
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


            # =================================================
            # 放量倍數
            # =================================================

            volume_ratio = (
                latest_volume
                / avg_5_volume
            )


            if (
                volume_ratio
                < vol_multiplier
            ):

                continue


            # =================================================
            # 40 日創高
            # =================================================

            if len(high_series) <= (
                breakout_days
            ):

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

                "ticker": ticker,

                "latest_close": (
                    latest_close
                ),

                "latest_volume": (
                    latest_volume
                ),

                "latest_volume_lots": (
                    latest_volume_lots
                ),

                "avg_5_volume": (
                    avg_5_volume
                ),

                "volume_ratio": (
                    volume_ratio
                ),

                "previous_high": (
                    previous_high
                ),

                "is_breakout": (
                    is_breakout
                ),

                "capital": (
                    capital
                ),

                "data_date": (
                    stock_df.index[-1]
                    .strftime(
                        "%Y-%m-%d"
                    )
                )

            })


        except Exception as e:

            errors.append({

                "ticker": ticker,

                "error": repr(e)

            })


    return candidates, errors


# ============================================================
# 第二階段完整分析
# ============================================================

def analyze_candidate_from_df(
    candidate,
    df_day,
    stocks_info,
    params,
    official_quotes
):

    ticker = candidate[
        "ticker"
    ]


    try:

        # ====================================================
        # 清理單股 Yahoo 資料
        # ====================================================

        df_day = (
            clean_single_stock_data(
                df_day,
                ticker
            )
        )


        if df_day.empty:
            return None


        code = stocks_info[
            ticker
        ]["code"]


        # ====================================================
        # V2.2.2：
        # 第二階段再次強制同步官方最新行情
        # ====================================================

        df_day = (
            synchronize_latest_volume(
                df_day,
                code,
                official_quotes
            )
        )


        if df_day.empty:
            return None


        # ====================================================
        # 官方最新日期
        # ====================================================

        quote = official_quotes.get(
            code
        )


        latest_allowed_date = None

        if quote is not None:

            latest_allowed_date = (
                pd.Timestamp(
                    quote["date"]
                ).normalize()
            )


        # ====================================================
        # 只保留已完成交易日
        # ====================================================

        df_day = (
            prepare_completed_daily_data(
                df_day,
                latest_allowed_date
            )
        )


        if df_day.empty:
            return None


        if len(df_day) < (
            MIN_FULL_ROWS
        ):

            return None


        # ====================================================
        # 完整週 K
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


        # ====================================================
        # numpy
        # ====================================================

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
        # 週 MA
        # ====================================================

        ma_week_series = (
            pd.Series(
                close_week
            )
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


        # ====================================================
        # 趨勢：
        # 最新週收盤 > 週 MA
        # ====================================================

        if (
            latest_week_close
            <= ma_week_val
        ):

            return None


        # ====================================================
        # 最新日線
        # ====================================================

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
        # 前 5 日均量
        # ====================================================

        previous_5_volume = (
            vol_day[-6:-1]
        )


        if len(
            previous_5_volume
        ) < 5:

            return None


        avg_5_volume = np.mean(
            previous_5_volume
        )


        if (
            not np.isfinite(
                avg_5_volume
            )
            or avg_5_volume <= 0
        ):

            return None


        volume_ratio = (
            latest_volume
            / avg_5_volume
        )


        # ====================================================
        # 最低成交量
        # ====================================================

        if (
            latest_volume_lots
            < MIN_VOLUME_LOTS
        ):

            return None


        # ====================================================
        # 放量
        # ====================================================

        if (
            volume_ratio
            < params["vol_multiplier"]
        ):

            return None


        # ====================================================
        # 40 日突破
        # ====================================================

        breakout_days = (
            params["breakout_days"]
        )


        previous_highs = (
            high_day[
                -(breakout_days + 1):-1
            ]
        )


        if len(
            previous_highs
        ) < breakout_days:

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
            * 100

        )


        # ====================================================
        # W底
        # ====================================================

        w_info = detect_w_bottom(

            high_day=high_day,

            low_day=low_day,

            close_day=close_day,

            tolerance=(
                params["w_tolerance"]
            ),

            lookback=(
                params["w_lookback"]
            ),

            pivot_window=(
                params["pivot_window"]
            ),

            min_gap=(
                params["w_min_gap"]
            ),

            max_gap=(
                params["w_max_gap"]
            )

        )


        is_w_bottom = (
            w_info["is_w_bottom"]
        )


        # ====================================================
        # 40日創高 OR W底
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

            signal_type = (
                "雙重訊號"
            )

        elif is_breakout:

            signal_type = (
                "區間創高"
            )

        else:

            signal_type = (
                "W底突破"
            )


        # ====================================================
        # 距離週 MA
        # ====================================================

        distance_to_week_ma_pct = (

            (
                latest_close
                - ma_week_val
            )
            / latest_close
            * 100

        )


        capital = candidate.get(
            "capital"
        )


        # ====================================================
        # 最終資料
        # ====================================================

        return {

            "status": "match",

            "ticker": ticker,

            "code": code,

            "name": stocks_info[
                ticker
            ]["name"],

            "group": stocks_info[
                ticker
            ]["group"],

            "market": stocks_info[
                ticker
            ]["market"],

            "capital": capital,

            "data_date": (
                df_day.index[-1]
                .strftime(
                    "%Y-%m-%d"
                )
            ),

            "df_day": df_day,

            "close": round(
                latest_close,
                2
            ),

            "volume": int(
                latest_volume_lots
            ),

            "volume_avg_5": round(
                avg_5_volume / 1000,
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

            "signal_type": (
                signal_type
            ),

            "reasons": reasons,

            "w_info": w_info,

            "div_history": (
                pd.DataFrame()
            )

        }


    except Exception:

        return None


# ============================================================
# 股利
#
# 最後才抓。
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
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


        dividends = (
            dividends.copy()
        )


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
            .groupby(
                "Year"
            )["Dividend"]
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

    if (
        div_df is None
        or div_df.empty
    ):

        return


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

            textcoords=(
                "offset points"
            ),

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

            [ma_week_val]
            * len(plot_df),

            color="red",

            linestyle="dashed",

            width=1.2

        )

    ]


    # ========================================================
    # W底頸線
    # ========================================================

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

        f"{ticker}",

        (
            f"Weekly MA20: "
            f"{ma_week_val:.2f}"
        )

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


    fig, axes = mpf.plot(

        plot_df,

        type="candle",

        style=mpf.make_mpf_style(

            base_mpf_style="yahoo",

            marketcolors=(
                mpf.make_marketcolors(

                    up="red",

                    down="green",

                    edge="inherit",

                    wick="inherit",

                    volume="inherit"

                )
            )

        ),

        addplot=addplots,

        title=(
            "\n"
            + " | ".join(
                title_parts
            )
        ),

        ylabel="Price (TWD)",

        volume=True,

        ylabel_lower="Volume",

        figratio=(16, 9),

        figscale=1.0,

        returnfig=True

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
    "🔍 V2.2.2 全自動選股控制台"
)


st.sidebar.info(
    "本版本固定掃描全台上市＋上櫃股票。"
)


st.sidebar.divider()


st.sidebar.subheader(
    "⚙️ 技術策略參數"
)


# ============================================================
# 最低股本
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
# 放量倍數
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
# 突破天數
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
# 週 MA
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
# W底容錯率
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


# ============================================================
# Pivot
# ============================================================

pivot_window = (
    st.sidebar.number_input(

        "W底 Pivot Low 判定寬度",

        min_value=2,

        max_value=6,

        value=3,

        step=1

    )
)


# ============================================================
# W底最小間隔
# ============================================================

w_min_gap = (
    st.sidebar.number_input(

        "W底左右腳最小間隔",

        min_value=5,

        max_value=15,

        value=7,

        step=1

    )
)


# ============================================================
# W底最大間隔
# ============================================================

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
# Parameters
# ============================================================

params = {

    "min_capital": (
        min_capital
    ),

    "vol_multiplier": (
        vol_multiplier
    ),

    "breakout_days": (
        breakout_days
    ),

    "ma_week": (
        ma_week
    ),

    "w_tolerance": (
        w_tolerance
    ),

    "w_lookback": 60,

    "pivot_window": (
        pivot_window
    ),

    "w_min_gap": (
        w_min_gap
    ),

    "w_max_gap": (
        w_max_gap
    )

}


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

**掃描範圍：**
全台上市＋上櫃
"""

)


# ============================================================
# 開始掃描
# ============================================================

if st.sidebar.button(

    "🚀 開始 V2.2.2 全自動雷達掃描",

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
    # 官方最新行情
    # ========================================================

    with st.spinner(

        "正在取得 TWSE＋TPEX 官方最新行情..."

    ):

        official_quotes = (
            get_official_latest_quotes()
        )


    latest_official_date = (
        get_latest_official_date(
            official_quotes
        )
    )


    official_date_summary = (
        get_official_date_summary(
            official_quotes
        )
    )


    if (
        latest_official_date
        is not None
    ):

        st.success(

            (
                "📅 官方行情最新資料日期："
                f"**"
                f"{latest_official_date.strftime('%Y-%m-%d')}"
                f"**"
            )

        )

    else:

        st.warning(
            "⚠️ 無法取得官方最新行情日期。"
        )


    # ========================================================
    # 官方日期診斷
    # ========================================================

    with st.expander(

        "🔍 官方行情資料日期診斷"

    ):

        if official_date_summary:

            official_diag_rows = []


            for (
                date_text,
                count
            ) in (
                official_date_summary.items()
            ):

                official_diag_rows.append({

                    "官方資料日期": (
                        date_text
                    ),

                    "股票數量": (
                        count
                    )

                })


            official_diag_df = (
                pd.DataFrame(
                    official_diag_rows
                )
            )


            st.dataframe(

                official_diag_df,

                use_container_width=True,

                hide_index=True

            )

        else:

            st.info(
                "目前沒有可供診斷的官方行情日期。"
            )


    # ========================================================
    # 股本
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
    # 第一階段
    # ========================================================

    st.subheader(
        "🔎 第一階段：批量市場資料篩選"
    )


    progress = st.progress(0)

    status = st.empty()


    fast_candidates = []

    batch_errors = []


    tickers = list(
        stocks_info.keys()
    )


    total_batches = int(

        np.ceil(

            len(tickers)
            / BATCH_SIZE

        )

    )


    # ========================================================
    # 第一階段批次
    # ========================================================

    for (

        batch_number,
        start

    ) in enumerate(

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

                period=(
                    DAILY_HISTORY_PERIOD
                ),

                interval="1d",

                auto_adjust=True,

                progress=False,

                group_by="column",

                threads=True

            )


            candidates, errors = (

                fast_filter_batch(

                    batch_df=(
                        batch_df
                    ),

                    stocks_info=(
                        stocks_info
                    ),

                    capital_map=(
                        capital_map
                    ),

                    min_capital=(
                        min_capital
                    ),

                    vol_multiplier=(
                        vol_multiplier
                    ),

                    breakout_days=(
                        breakout_days
                    ),

                    official_quotes=(
                        official_quotes
                    )

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

                "ticker": ",".join(
                    batch_tickers
                ),

                "error": repr(e)

            })


        progress.progress(

            batch_number
            / total_batches

        )


        status.text(

            (
                f"批量掃描："
                f"{batch_number}/"
                f"{total_batches}｜"
                f"第一層候選："
                f"{len(fast_candidates)} 支"
            )

        )


    progress.progress(1.0)


    # ========================================================
    # 第一階段完成
    # ========================================================

    st.success(

        (
            f"第一階段完成："
            f"從 {len(tickers)} 支股票中"
            f"留下 {len(fast_candidates)} 支候選股。"
        )

    )


    # ========================================================
    # 第一階段日期診斷
    # ========================================================

    if fast_candidates:

        fast_date_df = (

            pd.DataFrame([

                {

                    "資料日期": (
                        x["data_date"]
                    )

                }

                for x in fast_candidates

            ])

            .groupby(
                "資料日期"
            )

            .size()

            .reset_index(
                name="候選股票數"
            )

            .sort_values(

                "資料日期",

                ascending=False

            )

        )


        with st.expander(

            "📅 第一階段候選股資料日期診斷"

        ):

            st.dataframe(

                fast_date_df,

                use_container_width=True,

                hide_index=True

            )


    # ========================================================
    # 無候選
    # ========================================================

    if not fast_candidates:

        st.warning(
            "⚠️ 第一階段沒有候選股票。"
        )


        st.info(

            (
                "請優先檢查："
                "股本門檻、最低成交量、"
                "5日均量放量倍數。"
            )

        )


        st.stop()


    # ========================================================
    # 第二階段
    # ========================================================

    st.subheader(
        "📐 第二階段：批量完整技術型態分析"
    )


    progress2 = st.progress(0)

    status2 = st.empty()


    matches = []


    candidate_tickers = [

        x["ticker"]

        for x in fast_candidates

    ]


    candidate_map = {

        x["ticker"]: x

        for x in fast_candidates

    }


    total_candidate_batches = int(

        np.ceil(

            len(candidate_tickers)
            / BATCH_SIZE

        )

    )


    processed_candidates = 0


    # ========================================================
    # 第二階段批次
    # ========================================================

    for (

        batch_number,
        start

    ) in enumerate(

        range(

            0,

            len(candidate_tickers),

            BATCH_SIZE

        ),

        start=1

    ):


        batch_tickers = candidate_tickers[

            start:
            start + BATCH_SIZE

        ]


        try:

            full_batch_df = yf.download(

                batch_tickers,

                period=(
                    FULL_HISTORY_PERIOD
                ),

                interval="1d",

                auto_adjust=True,

                progress=False,

                group_by="column",

                threads=True

            )


            for ticker in batch_tickers:

                candidate = (
                    candidate_map[ticker]
                )


                try:

                    result = (

                        analyze_candidate_from_df(

                            candidate=(
                                candidate
                            ),

                            df_day=(
                                full_batch_df
                            ),

                            stocks_info=(
                                stocks_info
                            ),

                            params=(
                                params
                            ),

                            official_quotes=(
                                official_quotes
                            )

                        )

                    )


                    if result is not None:

                        matches.append(
                            result
                        )


                except Exception:

                    pass


                processed_candidates += 1


                progress2.progress(

                    processed_candidates
                    / len(
                        candidate_tickers
                    )

                )


                status2.text(

                    (
                        f"完整分析："
                        f"{processed_candidates}/"
                        f"{len(candidate_tickers)}｜"
                        f"目前入選："
                        f"{len(matches)} 支｜"
                        f"批次："
                        f"{batch_number}/"
                        f"{total_candidate_batches}"
                    )

                )


        except Exception as e:

            batch_errors.append({

                "ticker": ",".join(
                    batch_tickers
                ),

                "error": (
                    "第二階段："
                    + repr(e)
                )

            })


            processed_candidates += (
                len(batch_tickers)
            )


            progress2.progress(

                min(

                    processed_candidates
                    / len(
                        candidate_tickers
                    ),

                    1.0

                )

            )


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


    if matches:

        for (

            i,
            m

        ) in enumerate(

            matches,

            start=1

        ):


            m["div_history"] = (

                get_dividend_history(

                    m["ticker"]

                )

            )


            dividend_progress.progress(

                i / len(matches)

            )


    else:

        dividend_progress.progress(
            1.0
        )


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
🎉 V2.2.2 掃描完成！

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
    # 最終日期診斷
    # ========================================================

    if matches:

        final_date_df = (

            pd.DataFrame([

                {

                    "資料日期": (
                        m["data_date"]
                    )

                }

                for m in matches

            ])

            .groupby(
                "資料日期"
            )

            .size()

            .reset_index(
                name="最終入選股票數"
            )

            .sort_values(

                "資料日期",

                ascending=False

            )

        )


        st.subheader(
            "📅 最終入選資料日期檢查"
        )


        st.dataframe(

            final_date_df,

            use_container_width=True,

            hide_index=True

        )


        # ====================================================
        # 日期完全一致
        # ====================================================

        if (

            latest_official_date
            is not None

            and len(final_date_df) == 1

            and (
                final_date_df.iloc[0][
                    "資料日期"
                ]
                ==
                latest_official_date.strftime(
                    "%Y-%m-%d"
                )
            )

        ):

            st.success(

                (

                    "✅ 日期同步正常："
                    "所有最終入選股票均使用官方最新交易日 "
                    f"{latest_official_date.strftime('%Y-%m-%d')}"

                )

            )

        else:

            st.warning(

                (

                    "⚠️ 日期仍存在差異。"
                    "這次先不要調整選股條件，"
                    "請保留下方日期診斷結果。"

                )

            )


    # ========================================================
    # 產業集中
    # ========================================================

    industry_df = pd.DataFrame()


    if matches:

        st.subheader(
            "🏭 市場資金／強勢股產業集中度"
        )


        industry_df = (

            pd.DataFrame([

                {

                    "產業": (
                        m["group"]
                    ),

                    "入選家數": 1

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

            industry_df[
                "入選家數"
            ]

            / len(matches)

            * 100

        ).round(1)


        st.dataframe(

            industry_df,

            use_container_width=True,

            hide_index=True

        )


        st.caption(

            (

                "產業集中度是依本次選股條件的入選股票數計算，"
                "可用來觀察目前強勢股是否集中於特定產業，"
                "不等同於真正的市場資金流向。"

            )

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


            if m["capital"] is not None:

                capital_text = (

                    f"{m['capital'] / 100_000_000:.1f}"

                )


            summary_rows.append({

                "產業": (
                    m["group"]
                ),

                "股票": (

                    f"{m['name']} "
                    f"({m['code']})"

                ),

                "市場": (
                    m["market"]
                ),

                "股本(億)": (
                    capital_text
                ),

                "收盤價": (
                    m["close"]
                ),

                f"週{ma_week}MA": (
                    m["ma_week_val"]
                ),

                "距週MA": (

                    f"{m['distance_to_week_ma_pct']:.2f}%"

                ),

                "今日量(張)": (

                    f"{m['volume']:,}"

                ),

                "前5日均量": (

                    f"{m['volume_avg_5']:,.0f}"

                ),

                "放量倍數": (

                    f"{m['volume_ratio']:.2f}x"

                ),

                f"{breakout_days}日創高": (

                    "✅"

                    if m["is_breakout"]

                    else "—"

                ),

                "W底突破": (

                    "✅"

                    if m["is_w_bottom"]

                    else "—"

                ),

                "訊號": (
                    m["signal_type"]
                ),

                "資料日期": (
                    m["data_date"]
                )

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
        # 詳細資料
        # ====================================================

        st.divider()


        st.subheader(
            "📊 入選股票詳細分析"
        )


        for m in matches:

            st.markdown(

                f"""
### 📌 {m['name']}（{m['code']}）

**{m['market']}｜產業：{m['group']}｜資料日期：{m['data_date']}**
"""

            )


            # =================================================
            # 第一排
            # =================================================

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

                    (
                        f"{m['distance_to_week_ma_pct']:.2f}%"
                    )

                )


            with c4:

                st.metric(

                    "股本",

                    (

                        (

                            f"{m['capital'] / 100_000_000:.1f} 億"

                        )

                        if m["capital"] is not None

                        else "—"

                    )

                )


            # =================================================
            # 第二排
            # =================================================

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


            # =================================================
            # 第三排
            # =================================================

            c1, c2 = (
                st.columns(2)
            )


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
            # 週MA風險
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


            # =================================================
            # 股利
            # =================================================

            if not m[
                "div_history"
            ].empty:

                st.markdown(
                    "#### 📊 近十年現金股利"
                )


                plot_dividend_bar_chart(

                    m[
                        "div_history"
                    ]

                )

            else:

                st.info(
                    "沒有可取得的近期股利資料。"
                )


            # =================================================
            # K線
            # =================================================

            st.markdown(
                "#### 📈 技術圖"
            )


            plot_stock_chart(

                ticker=m["ticker"],

                df_day=m["df_day"],

                ma_week_val=m[
                    "ma_week_val"
                ],

                breakout_days=(
                    breakout_days
                ),

                is_breakout=(
                    m["is_breakout"]
                ),

                w_info=(
                    m["w_info"]
                )

            )


            st.divider()


    else:

        st.warning(
            "ℹ️ 目前參數下沒有符合條件的股票。"
        )


        st.info(

            (
                "可以優先嘗試降低放量倍數，"
                "或確認當日是否為完整交易日。"

            )

        )


    # ========================================================
    # 批次錯誤
    # ========================================================

    if batch_errors:

        with st.expander(

            f"⚠️ 批量資料錯誤（{len(batch_errors)} 筆）"

        ):

            error_df = pd.DataFrame(
                batch_errors
            )


            st.dataframe(

                error_df,

                use_container_width=True,

                hide_index=True

            )


    # ========================================================
    # 完成說明
    # ========================================================

    st.caption(

        (

            "V2.2.2 完成："
            "V2.2.1 選股條件完全保留｜"
            "成交量／OHLCV 官方日期同步｜"
            "第一、二階段使用同一官方最新交易日｜"
            "自動排除官方日期之後的 Yahoo 暫時資料｜"
            "全台上市＋上櫃批次掃描｜"
            "Yahoo歷史資料＋TWSE/TPEX官方最新行情｜"
            "產業集中｜入選股利｜資料日期診斷"

        )

    )
