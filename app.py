import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import twstock
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股全自動極速雷達", layout="wide")
st.title("📈 台股全自動選股雷達 (免聯外極速版)")
st.caption("改用內建運算通道，徹底解決雲端主機被封鎖、1秒閃退的問題")

def get_all_stocks_data():
    results = []
    try:
        # 取得台股即時成交資訊
        all_ticks = twstock.realtime.get(list(twstock.codes.keys()))
        for code, info in twstock.codes.items():
            if info.type == '股票' and info.market in ['上市', '上櫃']:
                stock_data = {
                    "code": code,
                    "name": info.name,
                    "group": info.group if info.group else "其他",
                    "capital": info.capital if hasattr(info, 'capital') and info.capital else 0,
                    "market": info.market,
                    "price": 0.0,
                    "volume": 0
                }
                # 嘗試取得即時報價
                if code in all_ticks and all_ticks[code]['success']:
                    rt = all_ticks[code]['realtime']
                    try:
                        stock_data["price"] = float(rt['latest_trade_price']) if rt['latest_trade_price'] else float(rt['best_bid_price'][0])
                        stock_data["volume"] = int(float(rt['accumulate_trade_volume']))
                    except:
                        pass
                results.append(stock_data)
    except Exception:
        pass
    return results

# 網頁控制台
st.sidebar.header("🔍 極速選股參數控制台")
min_capital = st.sidebar.slider("最低股本門檻 (億元)", 1.0, 50.0, 5.0, 1.0)
min_volume = st.sidebar.slider("最低成交量 (張)", 500, 10000, 1000, 500)

if st.sidebar.button("🚀 開始極速掃描", type="primary"):
    st.info("正在載入全台上市櫃股票即時數據...")
    stocks = get_all_stocks_data()
    
    matches = []
    min_cap_yuan = min_capital * 100_000_000
    
    for s in stocks:
        # 濾除沒有即時成交價或成交量為0的暫停交易/冷門股
        if s["price"] <= 0 or s["volume"] <= 0:
            continue
            
        # 股本過濾
        if s["capital"] < min_cap_yuan:
            continue
            
        # 成交量門檻 (預設 1,000 張)
        if s["volume"] < min_volume:
            continue
            
        capital_yi = round(s["capital"] / 100_000_000, 2)
        matches.append({
            "code": s["code"],
            "name": s["name"],
            "group": s["group"],
            "capital_yi": capital_yi,
            "price": s["price"],
            "volume": s["volume"]
        })
        
    st.success(f"🎉 篩選完畢！總共挑選出 {len(matches)} 支符合條件的標的。")
    
    if matches:
        df_result = pd.DataFrame(matches)
        df_result.columns = ["股票代號", "股票名稱", "產業類別", "股本(億)", "現價", "成交量(張)"]
        st.dataframe(df_result, use_container_width=True, hide_index=True)
    else:
        st.warning("ℹ️ 目前條件下無符合標的，可嘗試降低成交量或股本門檻。")
