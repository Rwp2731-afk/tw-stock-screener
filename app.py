import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="台股成交量篩選器", layout="wide")
st.title("📈 台股成交量篩選器 (證交所官方通道)")

if st.button("🚀 開始透過官方通道取得行情"):
    results = []
    try:
        # 證交所每日收盤行情 JSON 接口
        url = "https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&type=ALL"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        
        if res.status_code == 200:
            data = res.json()
            # 尋找包含個股收盤行情的表格 (通常在 tables 的某個索引中，或欄位符合)
            # 這裡我們使用更穩定的備用證交所三大法人或當日收盤彙整
            st.success("成功連線至證交所伺服器！正在解析資料...")
            
            # 為了確保穩定，我們直接串接另一組公開的台股即時/收盤彙整源
            fallback_url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
            res2 = requests.get(fallback_url, headers=headers, timeout=5)
            if res2.status_code == 200:
                raw_data = res2.json()
                for item in raw_data:
                    # 欄位包含：Code, Name, TradeVolume (成交股數), Close (收盤價) etc.
                    try:
                        vol_shares = float(item.get('TradeVolume', 0))
                        vol_lots = vol_shares / 1000 # 換算成張
                        close_price = float(item.get('ClosingPrice', 0))
                        
                        if vol_lots >= 1000: # 成交量 1000 張以上
                            results.append({
                                "代號": item.get('Code'),
                                "名稱": item.get('Name'),
                                "成交量(張)": int(vol_lots),
                                "收盤價": close_price
                            })
                    except:
                        continue
        else:
            st.error(f"連線失敗，狀態碼：{res.status_code}")
    except Exception as e:
        st.error(f"發生錯誤: {e}")
        
    if results:
        df = pd.DataFrame(results)
        st.success(f"🎉 成功篩選出 {len(df)} 支成交量超過 1,000 張的標的！")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("目前沒有抓到符合的標的或連線受阻。")
