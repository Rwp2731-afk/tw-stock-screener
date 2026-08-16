import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import twstock
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股全自動極速雷達", layout="wide")
st.title("📈 台股全自動選股雷達 (全台穩定版)")
st.caption("結合 twstock 內建安全資料庫，完美過濾全台成交量 1,000 張以上標的")

def get_market_stocks(min_capital, min_vol):
    results = []
    min_cap_yuan = min_capital * 100_000_000
    
    for code, info in twstock.codes.items():
        if info.type == '股票' and info.market in ['上市', '上櫃']:
            # 檢查股本
            capital = info.capital if hasattr(info, 'capital') and info.capital else 0
            if capital < min_cap_yuan:
                continue
                
            try:
                # 利用 twstock 內建 Stock 物件讀取近期歷史資料來評估成交量與價格
                stock = twstock.Stock(code)
                if len(stock.price) < 20:
                    continue
                    
                latest_price = stock.price[-1]
                latest_vol = stock.capacity[-1] / 1000  # 轉換成張數
                
                # 成交量過濾 (預設 1000 張以上)
                if latest_vol < min_vol:
                    continue
                    
                capital_yi = round(capital / 100_000_000, 2)
                results.append({
                    "code": code,
                    "name": info.name,
                    "group": info.group if info.group else "其他",
                    "capital_yi": capital_yi,
                    "price": latest_price,
                    "volume": int(latest_vol)
                })
            except Exception:
                continue
    return results

# 網頁控制台
st.sidebar.header("🔍 全台選股參數控制台")
min_capital = st.sidebar.slider("最低股本門檻 (億元)", 1.0, 50.0, 5.0, 1.0)
min_volume = st.sidebar.slider("最低成交量 (張)", 500, 10000, 1000, 500)

if st.sidebar.button("🚀 開始全台掃描", type="primary"):
    st.info("正在掃描全台上市櫃股票，請稍候約 10~20 秒...")
    matches = get_market_stocks(min_capital, min_volume)
    
    st.success(f"🎉 掃描完畢！總共挑選出 {len(matches)} 支符合條件的標的。")
    
    if matches:
        df_result = pd.DataFrame(matches)
        df_result.columns = ["股票代號", "股票名稱", "產業類別", "股本(億)", "現價", "成交量(張)"]
        st.dataframe(df_result, use_container_width=True, hide_index=True)
    else:
        st.warning("ℹ️ 目前條件下無符合標的，可嘗試調整側邊欄門檻。")
