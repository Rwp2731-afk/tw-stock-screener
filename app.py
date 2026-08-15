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
st.caption("自動獲取全台上市上櫃股票清單，掃描符合：20週MA之上 + 60日突破 + W底型態 + 1.1倍放量 + 成交量>1000張 的強勢標的")

# 自動獲取全台股清單與基本資訊
@st.cache_data(ttl=86400)
def get_all_tw_stocks_info():
    stocks_info = {}
    for code, info in twstock.codes.items():
        if info.type == '股票' and info.market in ['上市', '上櫃']:
            suffix = '.TW' if info.market == '上市' else '.TWO'
            ticker = f"{code}{suffix}"
            stocks_info[ticker] = {
                "name": info.name,
                "group": info.group if info.group else "其他"
            }
    return stocks_info

# 畫出帶有 W 底標示的 K 線圖
def plot_stock_chart(ticker, df_day, w_info=None):
    plot_df = df_day.iloc[-90:].copy()
    if isinstance(plot_df.columns, pd.MultiIndex):
        plot_df.columns = plot_df.columns.get_level_values(0)
    
    # 建立畫圖的額外標示 (W底折線)
    addplots = []
    if w_info:
        # 建立一條預設為 NaN 的序列，用於繪製 W 底連結線
        w_line = pd.Series(index=plot_df.index, data=np.nan)
        
        # 取得關鍵點的日期與價格
        p1_date = w_info['p1_date']
        neck_date = w_info['neck_date']
        p2_date = w_info['p2_date']
        latest_date = plot_df.index[-1]
        
        # 設定轉折點數值
        if p1_date in w_line.index: w_line.loc[p1_date] = w_info['p1_val']
        if neck_date in w_line.index: w_line.loc[neck_date] = w_info['neck_val']
        if p2_date in w_line.index: w_line.loc[p2_date] = w_info['p2_val']
        w_line.loc[latest_date] = plot_df['Close'].iloc[-1]
        
        # 插值連線形成 W 字軌跡
        w_line = w_line.interpolate(method='time')
        
        # 新增 W 底紅色虛線軌跡
        addplots.append(
            mpf.make_addplot(w_line, color='crimson', width=2.5, linestyle='--')
        )
        
    fig, ax = mpf.plot(
        plot_df,
        type='candle',
        style='yahoo',
        addplot=addplots if addplots else None,
        title=f"\n{ticker} - K-Chart (Red Dashed Line = W Bottom Pattern)",
        ylabel='Price (TWD)',
        volume=True,
        ylabel_lower='Volume',
        mav=(20, 60),
        figratio=(16, 9),
        figscale=1.1,
        returnfig=True
    )
    st.pyplot(fig)

def run_strategy(ticker, name, group):
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
        
        # 條件 3: W 底型態辨識與關鍵點定位
        lows = df_day['Low'].values.flatten()[-60:]
        highs = df_day['High'].values.flatten()[-60:]
        dates = df_day.index[-60:]
        
        min1_rel = np.argmin(lows[:30])
        min2_rel = 30 + np.argmin(lows[30:-5])
        neck_rel = min1_rel + np.argmax(highs[min1_rel:min2_rel])
        
        foot1, foot2 = lows[min1_rel], lows[min2_rel]
        neck_high = highs[neck_rel]
        
        cond3 = (abs(foot1 - foot2) / foot1 < 0.04) and (latest_close > neck_high)
        
        if cond3:
            w_info = {
                "p1_date": dates[min1_rel], "p1_val": foot1,
                "neck_date": dates[neck_rel], "neck_val": neck_high,
                "p2_date": dates[min2_rel], "p2_val": foot2
            }
            return {
                "ticker": ticker,
                "name": name,
                "group": group,
                "df_day": df_day,
                "close": round(float(latest_close), 2),
                "volume": int(latest_vol_lots),
                "w_info": w_info
            }
    except Exception:
        return None
    return None

# 網頁控制台
st.sidebar.header("🔍 全自動選股控制台")

market_choice = st.sidebar.radio("選擇掃描範圍", ["熱門前 100 支權值股", "全台股 (上市+上櫃，約 1800+ 支)"])

if st.sidebar.button("🚀 開始全自動雷達掃描", type="primary"):
    stocks_info = get_all_tw_stocks_info()
    all_tickers = list(stocks_info.keys())
    
    if market_choice == "熱門前 100 支權值股":
        target_tickers = all_tickers[:100]
    else:
        target_tickers = all_tickers
        
    st.info(f"正在全自動掃描 {len(target_tickers)} 支台灣股票，請稍候...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    matches = []
    
    for idx, ticker in enumerate(target_tickers):
        info = stocks_info[ticker]
        status_text.text(f"掃描中 ({idx+1}/{len(target_tickers)}): {ticker} {info['name']}")
        res = run_strategy(ticker, info['name'], info['group'])
        if res:
            matches.append(res)
        progress_bar.progress((idx + 1) / len(target_tickers))
        
    status_text.text("掃描完畢！")
    st.success("🎉 全自動掃描完成！")
    
    if matches:
        st.subheader(f"✅ 符合 5 大強勢條件的標的 (共 {len(matches)} 支)")
        for m in matches:
            st.markdown(f"### 📌 {m['name']} ({m['ticker'].split('.')[0]}) ｜ 產業：**{m['group']}** ｜ 收盤價：**{m['close']}** 元 ｜ 成交量：**{m['volume']}** 張")
            plot_stock_chart(m['ticker'], m['df_day'], m['w_info'])
            st.divider()
    else:
        st.warning("ℹ️ 今日全台股市場中，暫無完全符合所有 5 個條件的股票。")
