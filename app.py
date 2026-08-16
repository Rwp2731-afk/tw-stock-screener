import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="台股成交量篩選器", layout="wide")
st.title("📈 台股成交量篩選器 (純文字極速版)")

if st.button("🚀 開始透過官方通道取得行情"):
    results = []
    try:
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        
        if res.status_code == 200:
            raw_data = res.json()
            for item in raw_data:
                try:
                    vol_shares = float(item.get('TradeVolume', 0))
                    vol_lots = vol_shares / 1000 
                    close_price = float(item.get('ClosingPrice', 0))
                    
                    if vol_lots >= 1000: 
                        results.append({
                            "code": item.get('Code'),
                            "name": item.get('Name'),
                            "volume": int(vol_lots),
                            "price": close_price
                        })
                except:
                    continue
    except Exception as e:
        st.error(f"發生錯誤: {e}")
        
    if results:
        st.success(f"🎉 成功篩選出 {len(results)} 支成交量超過 1,000 張的標的！")
        st.markdown("---")
        # 改用迴圈直接印出文字，不使用 st.dataframe 避開執行緒崩潰
        for r in results:
            st.markdown(f"📌 **{r['name']} ({r['code']})** ｜ 收盤價：**{r['price']}** 元 ｜ 成交量：**{r['volume']}** 張")
    else:
        st.warning("目前沒有抓到符合的標的或連線受阻。")
