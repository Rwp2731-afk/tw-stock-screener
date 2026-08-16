import streamlit as st
import requests

st.set_page_config(page_title="台股成交量篩選器", layout="wide")
st.title("📈 台股成交量篩選器 (極速輕量版)")

if st.button("🚀 立即開始篩選", type="primary"):
    with st.spinner("正在連線至證交所讀取行情..."):
        try:
            url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
            res = requests.get(url, timeout=3)
            
            if res.status_code == 200:
                data = res.json()
                count = 0
                st.success("🎉 資料讀取成功！成交量大於 1,000 張的標的：")
                st.markdown("---")
                
                for item in data:
                    try:
                        vol = float(item.get('TradeVolume', 0)) / 1000
                        if vol >= 1000:
                            count += 1
                            name = item.get('Name', '')
                            code = item.get('Code', '')
                            price = item.get('ClosingPrice', 0)
                            st.markdown(f"📌 **{name} ({code})** ｜ 收盤：**{price}** ｜ 量：**{int(vol)}** 張")
                    except:
                        continue
                
                if count == 0:
                    st.warning("目前沒有符合條件的標的。")
            else:
                st.error(f"伺服器回應代碼：{res.status_code}")
        except Exception as e:
            st.error(f"連線逾時或發生錯誤，請稍後再試。")
