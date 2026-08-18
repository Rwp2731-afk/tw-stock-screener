import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, time as dt_time, timedelta
import pytz

# ==========================================
# 1. 基本設定與常數定義
# ==========================================
TAIWAN_TZ = pytz.timezone("Asia/Taipei")
REQUEST_TIMEOUT = 10

TWSE_STOCK_DAY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_MAINBOARD_QUOTE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
TWSE_BWIBBU_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
TPEX_BWIBBU_ALL_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio"

# 增加 Request Headers 避免被 TWSE/TPEX 防護機制攔截
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ==========================================
# 2. 時間與交易日輔助函式
# ==========================================
def get_taiwan_now():
    return datetime.now(TAIWAN_TZ)

def is_market_closed_for_today():
    """
    盤後時間判斷：
    調延至 14:30，確保證交所與櫃買中心 OpenAPI 已完成當日資料洗寫。
    """
    now = get_taiwan_now()
    if now.weekday() >= 5: # 週六、週日
        return True
    
    market_close = dt_time(14, 30)
    return now.time() >= market_close

# ==========================================
# 3. 官方 API 行情抓取 (TWSE / TPEX)
# ==========================================
# 調整 ttl 為 60 秒，避免鎖死舊行情
@st.cache_data(ttl=60, show_spinner=False)
def get_official_latest_quotes():
    """從證交所與櫃買中心 OpenAPI 抓取今日最新盤後行情"""
    quotes = {}
    
    # 1. TWSE 上市
    try:
        res = requests.get(TWSE_STOCK_DAY_ALL_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            data = res.json()
            for row in data:
                code = row.get("Code", "").strip()
                if not code:
                    continue
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
    except Exception as e:
        st.warning(f"TWSE API 讀取失敗: {e}")

    # 2. TPEX 上櫃
    try:
        res = requests.get(TPEX_MAINBOARD_QUOTE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            data = res.json()
            for row in data:
                code = row.get("SecuritiesCompanyCode", "").strip()
                if not code:
                    continue
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
    except Exception as e:
        st.warning(f"TPEX API 讀取失敗: {e}")

    return quotes

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_capital_data():
    """取得發行股數與本益比資料"""
    capital_map = {}
    
    # TWSE 股本與 PE
    try:
        res = requests.get(TWSE_BWIBBU_ALL_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            for row in res.json():
                code = row.get("Code", "").strip()
                pe = row.get("PEratio", None)
                if code:
                    capital_map[code] = {"PE": float(pe) if pe and pe != "-" else None}
    except Exception:
        pass

    return capital_map

# ==========================================
# 4. 資料整合與清洗
# ==========================================
def apply_official_latest_quote(df, stock_id):
    """將官方最新行情寫入/覆蓋至 Yahoo Finance 下載的 DataFrame 中"""
    quotes = get_official_latest_quotes()
    if stock_id not in quotes:
        return df

    quote = quotes[stock_id]
    if quote["Close"] <= 0:
        return df

    # 強制將 Index 與 官方日期統一為 Timestamp 並去除時區
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    official_date = pd.Timestamp(quote["date"]).tz_localize(None).normalize()

    # 如果當天已存在則覆蓋，不存在則 append
    if official_date in df.index:
        df.loc[official_date, "Open"] = quote["Open"]
        df.loc[official_date, "High"] = quote["High"]
        df.loc[official_date, "Low"] = quote["Low"]
        df.loc[official_date, "Close"] = quote["Close"]
        df.loc[official_date, "Volume"] = quote["Volume"]
    else:
        new_row = pd.DataFrame(
            {
                "Open": [quote["Open"]],
                "High": [quote["High"]],
                "Low": [quote["Low"]],
                "Close": [quote["Close"]],
                "Volume": [quote["Volume"]],
            },
            index=[official_date]
        )
        df = pd.concat([df, new_row])

    return df

def prepare_completed_daily_data(stock_id, period="1y"):
    """讀取股票歷史資料並修補今日最新行情"""
    ticker = f"{stock_id}.TW"
    
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            # 嘗試上櫃標的 (.TWO)
            ticker = f"{stock_id}.TWO"
            df = yf.download(ticker, period=period, progress=False)
            if df.empty:
                return pd.DataFrame()

        # 處理 MultiIndex 欄位 (yfinance 升級後特有結構)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 確保清理時區
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

        # 寫入官方最新盤後數據
        df = apply_official_latest_quote(df, stock_id)

        # 盤中未收盤時，剔除當日未完結 K 線
        now = get_taiwan_now()
        today = pd.Timestamp(now.date()).normalize()
        last_date = df.index[-1]

        if last_date == today and not is_market_closed_for_today():
            df = df.iloc[:-1].copy()

        return df

    except Exception as e:
        st.error(f"抓取 {stock_id} 資料發生錯誤: {e}")
        return pd.DataFrame()

# ==========================================
# 5. Streamlit 主介面
# ==========================================
def main():
    st.set_page_config(page_title="台股盤後即時分析器", layout="wide")
    st.title("📈 台股最新盤後行情與分析")

    # 側邊欄控制區
    st.sidebar.header("控制選單")
    
    if st.sidebar.button("🧹 清除快取並重新讀取"):
        st.cache_data.clear()
        st.rerun()

    stock_id = st.sidebar.text_input("輸入台股代碼", value="2330").strip()

    if st.sidebar.button("開始分析"):
        with st.spinner("資料讀取中..."):
            df = prepare_completed_daily_data(stock_id)

            if df.empty:
                st.error("查無資料，請確認代碼是否正確。")
                return

            last_date_str = df.index[-1].strftime('%Y-%m-%d')
            st.success(f"資料更新成功！最新 K 線日期：**{last_date_str}**")

            # 顯示最新數據卡片
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            change = latest["Close"] - prev["Close"]
            pct_change = (change / prev["Close"]) * 100 if prev["Close"] != 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("最新收盤價", f"{latest['Close']:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
            col2.metric("最高價", f"{latest['High']:.2f}")
            col3.metric("最低價", f"{latest['Low']:.2f}")
            col4.metric("成交量 (張)", f"{int(latest['Volume'] / 1000):,}")

            # 資料預覽表
            st.subheader("最近 5 個交易日行情")
            st.dataframe(df.tail(5).sort_index(ascending=False))

if __name__ == "__main__":
    main()
