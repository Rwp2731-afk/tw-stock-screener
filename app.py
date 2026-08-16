import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="台股極速篩選器", layout="wide")
st.title("📈 台股成交量篩選器 (YFinance版)")

# 常見台股代號清單 (為了快速示範，我們先取部分大型股，若要全台請替換成完整列表)
# 因為全台 1700 檔跑完需要時間，建議先測試 50 檔
stock_list = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW", "2882.TW", "1301.TW"]

if st.button("🚀 開始篩選 (測試版)"):
    results = []
    progress = st.progress(0)
    
    for i, ticker in enumerate(stock_list):
        try:
            # 獲取今日資料
            df = yf.download(ticker, period="1d", progress=False)
            if not df.empty:
                volume = df['Volume'].iloc[-1]
                price = df['Close'].iloc[-1]
                
                # 成交量門檻
                if volume > 1000000: # yfinance 的成交量是實際股數，1000張 = 1,000,000股
                    results.append({"代號": ticker, "成交量(張)": volume/1000, "收盤價": round(float(price), 2)})
        except:
            pass
        progress.progress((i + 1) / len(stock_list))
        
    if results:
        st.table(pd.DataFrame(results))
    else:
        st.warning("無符合條件標的")
