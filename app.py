import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import twstock
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股 W底放量突破全自動雷達", layout="wide")
st.title("📈 台股全自動選股雷達 (全台股掃描)")
st.caption("自動獲取全台上市上櫃股票清單，掃描符合：20週MA之上 + 60日突破 + W底型態 + 1.1倍爆量 + 成交量>1000張 的強勢標的")

# 自動獲取全台股清單函數
@st.cache_data(ttl=86400)  # 快取 24 小時，避免重複抓取
def get_all_tw_stocks():
    codes = []
    # 抓取上市與上櫃股票
    for code, info in twstock.codes.items():
        if info.type == '股票' and info.market in ['上市', '上櫃']:
            # 加入 .TW (上市) 或 .TWO (上櫃)
            suffix = '.TW' if info.market == '上市' else '.TWO'
            codes.append(f"{code}{suffix}")
    return codes

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
        if not (close_week[-1] > ma20_week):
            return None
        
        # 條件 5: 成交量 >= 1000張
        latest_vol_lots = vol_day[-1] / 1000
        if latest_vol_lots < 1000:
            return None
        
        # 條件 4: 放大量 (>= 20日均量 1.1倍)
        ma20_vol = pd.Series(vol_day).rolling(20).mean().iloc[-1]
        if not (vol_day[-1] >= (ma20_vol * 1.1)):
            return None
        
        # 條件 2: 突破 60日新高
        latest_close = close_day[-1]
        if not (latest_close >= np.max(df_day['High'].values.flatten()[-60:-1])):
            return None
        
        # 條件 3: W 底型態
        lows = df_day['Low'].values.flatten()[-60:]
        min1_idx = np.argmin(lows[:30])
        min2_idx = 30 + np.argmin(lows[30:-5])
        neck_high = np.max(df_day['High'].values.flatten()[-60:][min1_idx:min2_idx])
        foot1, foot2 = lows[min1_idx], lows[min2_idx]
        cond3 = (abs(foot1 - foot2) / foot1 < 0.04) and (latest_close > neck_high)
        
        if cond3:
            return {
                "ticker": ticker,
                "df_day": df_day,
                "close": round(float(latest_close), 2),
                "volume": int(latest_vol_lots)
            }
    except Exception:
        return None
    return None

# 網頁控制台
st.sidebar.header("🔍 全自動選股控制台")

market_choice = st.sidebar.radio("選擇掃描範圍", ["熱門前 100 支權值股", "全台股 (上市+上櫃，約 1800+ 支)"])

if st.sidebar.button("🚀 開始全自動雷達掃描", type="primary"):
    all_stocks = get_all_tw_stocks()
    
    if market_choice == "熱門前 100 支權值股":
        target_stocks = all_stocks[:100]
    else:
        target_stocks = all_stocks
        
    st.info(f"正在全自動掃描 {len(target_stocks)} 支台灣股票，請稍候...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    matches = []
    
    for idx, ticker in enumerate(target_stocks):
        status_text.text(f"掃描中 ({idx+1}/{len(target_stocks)}): {ticker}")
        res = run_strategy(ticker)
        if res:
            matches.append(res)
        progress_bar.progress((idx + 1) / len(target_stocks))
        
    status_text.text("掃描完畢！")
    st.success("🎉 全自動掃描完成！")
    
    if matches:
        st.subheader(f"✅ 符合 5 大強勢條件的標的 (共 {len(matches)} 支)")
        for m in matches:
            st.markdown(f"### 📌 {m['ticker']} | 收盤價：**{m['close']}** 元 | 成交量：**{m['volume']}** 張")
            plot_stock_chart(m['ticker'], m['df_day'])
            st.divider()
    else:
        st.warning("ℹ️ 今日全台股市場中，暫無完全符合所有 5 個條件的股票。")
