import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 網頁標題與設定
st.set_page_config(page_title="台股 W底放量突破選股器", layout="wide")
st.title("📈 台股 W底放量突破選股雷達")
st.caption("自動掃描符合：20週MA之上 + 60日突破 + W底型態 + 2倍爆量 + 成交量>1000張 的強勢標的")

# 畫 K 線圖函數
def plot_stock_chart(ticker, df_day):
    plot_df = df_day.iloc[-90:].copy()
    if isinstance(plot_df.columns, pd.MultiIndex):
        plot_df.columns = plot_df.columns.get_level_values(0)
        
    fig, ax = mpf.plot(
        plot_df,
        type='candle',
        style='yahoo',
        title=f"\n{ticker} - K-Chart",
        ylabel='Price (TWD)',
        volume=True,
        ylabel_lower='Volume',
        mav=(20, 60),
        figratio=(16, 9),
        figscale=1.1,
        returnfig=True
    )
    st.pyplot(fig)

# 選股核心邏輯
def run_strategy(ticker):
    try:
        df_day = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        df_week = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
        
        if len(df_day) < 60 or len(df_week) < 20:
            return None
            
        close_day = df_day['Close'].values.flatten()
        vol_day = df_day['Volume'].values.flatten()
        close_week = df_week['Close'].values.flatten()
        
        # 條件 1: 20週 MA 之上
        ma20_week = pd.Series(close_week).rolling(20).mean().iloc[-1]
        cond1 = close_week[-1] > ma20_week
        
        # 條件 5: 成交量 >= 1000張
        latest_vol_lots = vol_day[-1] / 1000
        cond5 = latest_vol_lots >= 1000
        
        # 條件 4: 放大量 (>= 20日均量 2倍)
        ma20_vol = pd.Series(vol_day).rolling(20).mean().iloc[-1]
        cond4 = vol_day[-1] >= (ma20_vol * 2.0)
        
        # 條件 2: 突破 60日新高
        latest_close = close_day[-1]
        cond2 = latest_close >= np.max(df_day['High'].values.flatten()[-60:-1])
        
        # 條件 3: W 底型態
        lows = df_day['Low'].values.flatten()[-60:]
        min1_idx = np.argmin(lows[:30])
        min2_idx = 30 + np.argmin(lows[30:-5])
        neck_high = np.max(df_day['High'].values.flatten()[-60:][min1_idx:min2_idx])
        foot1, foot2 = lows[min1_idx], lows[min2_idx]
        cond3 = (abs(foot1 - foot2) / foot1 < 0.04) and (latest_close > neck_high)
        
        if cond1 and cond2 and cond3 and cond4 and cond5:
            return {
                "ticker": ticker,
                "df_day": df_day,
                "close": round(float(latest_close), 2),
                "volume": int(latest_vol_lots)
            }
    except Exception:
        return None
    return None

# 網頁側邊欄
st.sidebar.header("🔍 選股控制台")
watchlist_input = st.sidebar.text_area(
    "輸入觀察股票代號（用逗點隔開）",
    value="2330.TW, 2317.TW, 2454.TW, 2382.TW, 3231.TW, 2308.TW, 2603.TW, 2376.TW, 2301.TW"
)

if st.sidebar.button("🚀 開始雷達掃描", type="primary"):
    watchlist = [s.strip() for s in watchlist_input.split(",") if s.strip()]
    st.info(f"正在掃描 {len(watchlist)} 支股票，請稍候...")
    
    progress_bar = st.progress(0)
    matches = []
    
    for idx, ticker in enumerate(watchlist):
        res = run_strategy(ticker)
        if res:
            matches.append(res)
        progress_bar.progress((idx + 1) / len(watchlist))
        
    st.success("掃描完成！")
    
    if matches:
        st.subheader(f"✅ 符合 5 大強勢條件的標的 (共 {len(matches)} 支)")
        for m in matches:
            st.markdown(f"### 📌 {m['ticker']} | 收盤價：**{m['close']}** 元 | 成交量：**{m['volume']}** 張")
            plot_stock_chart(m['ticker'], m['df_day'])
            st.divider()
    else:
        st.warning("ℹ️ 今日觀察名單中，暫無完全符合所有 5 個條件的股票。")
