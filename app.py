import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(page_title="台股成交量篩選器", layout="wide")
st.title("📈 台股成交量篩選器 (純HTTP防崩潰版)")

stock_list = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW", "2882.TW", "1301.TW", "2603.TW", "2412.TW"]

if st.button("🚀 開始執行純HTTP篩選"):
    results = []
    debug_logs = []
    progress = st.progress(0)
    
    for i, ticker in enumerate(stock_list):
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=1700000000&period2=2000000000&interval=1d&events=history"
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=3)
            
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                if not df.empty and 'Close' in df.columns:
                    close_val = df['Close'].iloc[-1]
                    vol_val = df['Volume'].iloc[-1]
                    vol_lots = vol_val / 1000
                    
                    debug_logs.append(f"{ticker} 成功：收盤={close_val}, 量={vol_lots:.0f}張")
                    if vol_lots >= 100:
                        results.append({
                            "代號": ticker, 
                            "成交量(張)": int(vol_lots), 
                            "收盤價": round(float(close_val), 2)
                        })
                else:
                    debug_logs.append(f"{ticker}：回傳格式不符")
            else:
                debug_logs.append(f"{ticker}：HTTP狀態碼 {res.status_code}")
        except Exception as e:
            debug_logs.append(f"{ticker} 錯誤: {str(e)}")
            
        progress.progress((i + 1) / len(stock_list))
        
    st.subheader("🔍 執行日誌")
    for log in debug_logs:
        st.text(log)
        
    st.subheader("📊 篩選結果")
    if results:
        st.table(pd.DataFrame(results))
    else:
        st.warning("沒有符合條件的標的")
