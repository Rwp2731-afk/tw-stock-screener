import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, time as dt_time
import pytz

# ==========================================
# 1. 基本設定與常數定義
# ==========================================
TAIWAN_TZ = pytz.timezone("Asia/Taipei")
REQUEST_TIMEOUT = 10

TWSE_STOCK_DAY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_MAINBOARD_QUOTE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
TWSE_BWIBBU_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"

# 預設掃描的標的清單（可依需求擴充）
DEFAULT_STOCK_LIST = [
    "2330", "2317", "2454", "2308", "2382", "3231", "2356", "6669", 
    "3037", "2379", "3034", "2303", "2881", "2882", "2886", "2891"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ==========================================
# 2. 時間與交易日輔助函式
# ==========================================
def get_taiwan_now():
    return datetime.now(TAIWAN_TZ)

def is_market_closed_for_today():
    """盤後時間判斷：調延至 14:30 確保證交所 API 完成資料更新"""
    now = get_taiwan_now()
    if now.weekday() >= 5: # 週六、週日
        return True
    return now.time() >= dt_time(14, 30)

# ==========================================
# 3. 官方 API 行情與基本面抓取
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def get_official_latest_quotes():
    """從證交所與櫃買中心 OpenAPI 抓取今日最新盤後行情"""
    quotes = {}
    
    # 1. TWSE 上市
    try:
        res = requests.get(TWSE_STOCK_DAY_ALL_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            for row in res.json():
                code = row.get("Code", "").strip()
                if code:
                    try:
                        quotes[code] = {
                            "date": get_taiwan_now().strftime("%Y-%m-%d"),
                            "Open": float(row.get("OpeningPrice", 0) or 0),
                            "High": float(row.get("HighestPrice", 0) or 0),
                            "Low": float(row.get("LowestPrice", 0) or 0),
                            "Close": float(row.get("ClosingPrice", 0) or 0),
                            "Volume": float(row.get("TradeVolume", 0) or 0),
                        }
                    except ValueError:
                        continue
    except Exception:
        pass

    # 2. TPEX 上櫃
    try:
        res = requests.get(TPEX_MAINBOARD_QUOTE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            for row in res.json():
                code = row.get("SecuritiesCompanyCode", "").strip()
                if code:
                    try:
                        quotes[code] = {
                            "date": get_taiwan_now().strftime("%Y-%m-%d"),
                            "Open": float(row.get("Open", 0) or 0),
                            "High": float(row.get("High", 0) or 0),
                            "Low": float(row.get("Low", 0) or 0),
                            "Close": float(row.get("Close", 0) or 0),
                            "Volume": float(row.get("TradingShares", 0) or 0),
                        }
                    except ValueError:
                        continue
    except Exception:
        pass

    return quotes

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_pe_data():
    """取得官方本益比資料"""
    pe_map = {}
    try:
        res = requests.get(TWSE_BWIBBU_ALL_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            for row in res.json():
                code = row.get("Code", "").strip()
                pe = row.get("PEratio", None)
                if code:
                    try:
                        pe_map[code] = float(pe) if pe and pe != "-" else None
                    except ValueError:
                        pe_map[code] = None
    except Exception:
        pass
    return pe_map

# ==========================================
# 4. 技術指標計算與行情修補
# ==========================================
def apply_official_latest_quote(df, stock_id):
    """修補當日最新盤後數據至 DataFrame"""
    quotes = get_official_latest_quotes()
    if stock_id not in quotes or quotes[stock_id]["Close"] <= 0:
        return df

    quote = quotes[stock_id]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    official_date = pd.Timestamp(quote["date"]).tz_localize(None).normalize()

    if official_date in df.index:
        df.loc[official_date, ["Open", "High", "Low", "Close", "Volume"]] = [
            quote["Open"], quote["High"], quote["Low"], quote["Close"], quote["Volume"]
        ]
    else:
        new_row = pd.DataFrame([{
            "Open": quote["Open"], "High": quote["High"], 
            "Low": quote["Low"], "Close": quote["Close"], "Volume": quote["Volume"]
        }], index=[official_date])
        df = pd.concat([df, new_row])

    return df

def prepare_completed_daily_data(stock_id, period="1y"):
    """讀取 K 線並計算均線指標"""
    ticker = f"{stock_id}.TW"
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            df = yf.download(f"{stock_id}.TWO", period=period, progress=False)
            if df.empty:
                return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = apply_official_latest_quote(df, stock_id)

        # 盤中剔除未完結 K 線
        now = get_taiwan_now()
        today = pd.Timestamp(now.date()).normalize()
        if df.index[-1] == today and not is_market_closed_for_today():
            df = df.iloc[:-1].copy()

        # 計算常用技術指標
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA60"] = df["Close"].rolling(window=60).mean()
        df["Vol_MA5"] = df["Volume"].rolling(window=5).mean()

        return df
    except Exception:
        return pd.DataFrame()

# ==========================================
# 5. 選股條件判斷邏輯 (Filter Logic)
# ==========================================
def check_stock_conditions(stock_id, df, pe_map, filters):
    """依據使用者設定的篩選條件進行過濾"""
    if df.empty or len(df) < 60:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = latest["Close"]
    volume_shares = latest["Volume"]
    volume_lots = volume_shares / 1000.0 # 轉為張數

    # 1. 價格區間篩選
    if not (filters["min_price"] <= close <= filters["max_price"]):
        return None

    # 2. 成交量篩選 (張)
    if volume_lots < filters["min_volume"]:
        return None

    # 3. 均線多頭排列 (Close > MA5 > MA20 > MA60)
    if filters["ma_alignment"]:
        if not (close > latest["MA5"] > latest["MA20"] > latest["MA60"]):
            return None

    # 4. 突破 20 日均線 (昨日在 MA20 下，今日站上 MA20)
    if filters["break_ma20"]:
        if not (prev["Close"] <= prev["MA20"] and close > latest["MA20"]):
            return None

    # 5. 成交量倍增 (今日成交量 > 5日均量 * 倍率)
    if filters["volume_multiples"] > 1.0:
        if volume_shares < (latest["Vol_MA5"] * filters["volume_multiples"]):
            return None

    # 6. 本益比上限
    pe = pe_map.get(stock_id)
    if filters["max_pe"] is not None:
        if pe is None or pe > filters["max_pe"]:
            return None

    # 計算漲跌幅
    pct_change = ((close - prev["Close"]) / prev["Close"]) * 100 if prev["Close"] != 0 else 0

    return {
        "股票代碼": stock_id,
        "最新日期": df.index[-1].strftime("%Y-%m-%d"),
        "收盤價": round(close, 2),
        "漲跌幅(%)": round(pct_change, 2),
        "成交量(張)": int(volume_lots),
        "5日均價": round(latest["MA5"], 2),
        "20日均價": round(latest["MA20"], 2),
        "本益比": round(pe, 2) if pe else "N/A"
    }

# ==========================================
# 6. Streamlit 介面
# ==========================================
def main():
    st.set_page_config(page_title="台股盤後選股與即時分析器", layout="wide")
    st.title("🔍 台股盤後多功能選股器")

    # ------------------ 側邊欄：選股條件設定 ------------------
    st.sidebar.header("🎯 選股條件設定")

    if st.sidebar.button("🧹 清除快取並重讀"):
        st.cache_data.clear()
        st.rerun()

    # 條件 1：價格區間
    min_price, max_price = st.sidebar.slider(
        "股價區間 (元)", 0.0, 1000.0, (10.0, 200.0)
    )

    # 條件 2：最小成交量
    min_volume = st.sidebar.number_input("最低成交量 (張)", value=500, step=100)

    # 條件 3：均線型態
    ma_alignment = st.sidebar.checkbox("均線多頭排列 (收盤 > MA5 > MA20 > MA60)", value=False)
    break_ma20 = st.sidebar.checkbox("今日強勢突破 20 日線 (MA20)", value=False)

    # 條件 4：量能放量
    volume_multiples = st.sidebar.slider("成交量相較 5日均量放大倍數", 1.0, 5.0, 1.0, step=0.1)

    # 條件 5：基本面本益比
    enable_pe_filter = st.sidebar.checkbox("限制本益比上限", value=False)
    max_pe = st.sidebar.number_input("本益比上限", value=20.0, step=1.0) if enable_pe_filter else None

    # 觀察清單文字輸入
    st.sidebar.markdown("---")
    custom_stocks_str = st.sidebar.text_area(
        "掃描股票清單 (以逗號分隔)", 
        value=",".join(DEFAULT_STOCK_LIST),
        height=100
    )

    # 打包條件
    filters = {
        "min_price": min_price,
        "max_price": max_price,
        "min_volume": min_volume,
        "ma_alignment": ma_alignment,
        "break_ma20": break_ma20,
        "volume_multiples": volume_multiples,
        "max_pe": max_pe,
    }

    # ------------------ 主畫面執行 ------------------
    if st.button("🚀 開始條件選股與掃描", type="primary"):
        stock_list = [s.strip() for s in custom_stocks_str.split(",") if s.strip()]
        
        with st.spinner("抓取官方最新盤後行情與分析中..."):
            pe_map = get_company_pe_data()
            results = []
            
            progress_bar = st.progress(0)
            for idx, stock_id in enumerate(stock_list):
                df = prepare_completed_daily_data(stock_id)
                res = check_stock_conditions(stock_id, df, pe_map, filters)
                if res:
                    results.append(res)
                progress_bar.progress((idx + 1) / len(stock_list))

            progress_bar.empty()

            if results:
                res_df = pd.DataFrame(results)
                st.success(f"掃描完成！共有 **{len(res_df)}** 檔股票符合篩選條件：")
                st.dataframe(res_df, use_container_width=True)
            else:
                st.warning("掃描完成，目前沒有符合條件的股票。請嘗試放寬條件再次嘗試。")

if __name__ == "__main__":
    main()
