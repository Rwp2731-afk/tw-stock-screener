import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="台股成交量篩選器", layout="wide")
st.title("📈 台股成交量篩選器 (除錯版)")

# 擴大測試清單
stock_list = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW", "2882.TW", "1301.TW", "2603.TW", "2412.TW"]

if st.button("🚀 開始執行除錯篩選"):
    results = []
    debug_logs = []
    progress = st.progress(0)
    
    for i, ticker in enumerate(stock_list):
        try:
            df = yf.download(ticker, period="5d", progress=False)
            if not df.empty:
                # 兼容不同的 pandas 欄位結構
                if isinstance(df.columns, pd.MultiIndex):
                    close_val = df['Close'].iloc[-1].values[0]
                    vol_val = df['Volume'].iloc[-1].values[0]
                else:
                    close_val = df['Close'].iloc[-1]
                    vol_val = df['Volume'].iloc[-1]
                
                vol_lots = vol_val / 1000
                debug_logs.append(f"{ticker} 成功讀取：收盤價={close_val}, 成交量={vol_lots:.0f}張")
                
                # 門檻：成交量大於 100 張就先抓進來看看（降低門檻測試）
                if vol_lots >= 100:
                    results.append({
                        "代號": ticker, 
                        "成交量(張)": int(vol_lots), 
                        "收盤價": round(float(close_val), 2)
                    })
            else:
                debug_logs.append(f"{ticker}：抓到的資料是空的")
        except Exception as e:
            debug_logs.append(f"{ticker} 發生錯誤: {e}")
            
        progress.progress((i + 1) / len(stock_list))
        
    st.subheader("🔍 除錯日誌")
    for log in debug_logs:
        st.text(log)
        
    st.subheader("📊 篩選結果")
    if results:
        st.table(pd.DataFrame(results))
    else:
        st.warning("依然無符合條件標的，請檢查上方日誌是否有報錯！")
