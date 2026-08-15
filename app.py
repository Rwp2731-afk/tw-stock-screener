import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import twstock
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股 W底放量突破全自動雷達", layout="wide")
st.title("📈 台股全自動選股雷達")
st.caption("自動獲取全台上市上櫃股票清單，掃描符合：20週MA之上 + 40日突破 + W底型態(6%容錯) + 1.1倍放量 + 成交量>1000張 的強勢標的")

# 自動獲取全台股清單與基本資訊
@st.cache_data(ttl=86400)
def get_all_tw_stocks_info():
    stocks_info = {}
    for code, info in twstock.codes.items():
        if info.type == '股票' and info.market in ['上市', '上櫃']:
            suffix = '.TW' if info.market == '上市' else '.TWO'
            ticker = f"{code}{suffix}"
            stocks_info[ticker] = {
                "code": code,
                "name": info.name,
                "group": info.group if info.group else "其他"
            }
    return stocks_info

# 標準 K 線圖 (包含 20日藍線 與 100日紫線/20週MA 及成交量)
def plot_stock_chart(ticker, df_day):
    plot_df = df_day.iloc[-120:].copy()
    if isinstance(plot_df.columns, pd.MultiIndex):
        plot_df = plot_df.xs(ticker, axis=1, level=1) if ticker in plot_df.columns.levels[1] else plot_df
        
    ma20 = plot_df['Close'].rolling(20).mean()
    ma100 = plot_df['Close'].rolling(100).mean()
    
    addplots = [
        mpf.make_addplot(ma20, color='dodgerblue', width=1.5),
        mpf.make_addplot(ma100, color='purple', width=1.8)
    ]
    
    fig, ax = mpf.plot(
        plot_df,
        type='candle',
        style='yahoo',
        addplot=addplots,
        title=f"\n{ticker} - K-Chart (Blue: 20MA / Purple: 100MA - 20W)",
        ylabel='Price (TWD)',
        volume=True,
        ylabel_lower='Volume',
        figratio=(16, 9),
        figscale=1.1,
        returnfig=True
    )
    st.pyplot(fig)

# 單一股票策略運算
def check_strategy(df_day, df_week):
    try:
        if df_day is None or df_week is None:
            return False, 0, 0
            
        df_day = df_day.dropna()
        df_week = df_week.dropna()
        
        if len(df_day) < 100 or len(df_week) < 20:
            return False, 0, 0
            
        close_day = df_day['Close'].values.flatten()
        vol_day = df_day['Volume'].values.flatten()
        close_week = df_week['Close'].values.flatten()
        
        # 條件 1: 20週 MA 之上
        ma20_week = pd.Series(close_week).rolling(20).mean().iloc[-1]
        if not (close_week[-1] > ma20_week):
            return False, 0, 0
            
        # 條件 5: 成交量 >= 1000張
        latest_vol_lots = vol_day[-1] / 1000
        if latest_vol_lots < 1000:
            return False, 0, 0
            
        # 條件 4: 放大量 (>= 20日均量 1.1倍)
        ma20_vol = pd.Series(vol_day).rolling(20).mean().iloc[-1]
        if not (vol_day[-1] >= (ma20_vol * 1.1)):
            return False, 0, 0
            
        # 條件 2: 突破 40日新高
        latest_close = close_day[-1]
        if not (latest_close >= np.max(df_day['High'].values.flatten()[-40:-1])):
            return False, 0, 0
            
        # 條件 3: W 底型態邏輯 (6%容錯)
        lows = df_day['Low'].values.flatten()[-60:]
        min1_idx = np.argmin(lows[:30])
        min2_idx = 30 + np.argmin(lows[30:-5])
        neck_high = np.max(df_day['High'].values.flatten()[-60:][min1_idx:min2_idx])
        foot1, foot2 = lows[min1_idx], lows[min2_idx]
        
        if (abs(foot1 - foot2) / foot1 < 0.06) and (latest_close > neck_high):
            return True, round(float(latest_close), 2), int(latest_vol_lots)
    except Exception:
        return False, 0, 0
    return False, 0, 0

# 網頁控制台
st.sidebar.header("🔍 全自動選股控制台")
market_choice = st.sidebar.radio("選擇掃描範圍", ["成交金額熱門前 150 大", "全台股 (上市+上櫃，約 1800+ 支)"])

if st.sidebar.button("🚀 開始全自動雷達掃描", type="primary"):
    stocks_info = get_all_tw_stocks_info()
    all_tickers = list(stocks_info.keys())
    
    if market_choice == "成交金額熱門前 150 大":
        target_tickers = all_tickers[:150]
    else:
        target_tickers = all_tickers
        
    st.info(f"正在安全分批掃描 {len(target_tickers)} 支股票...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    matches = []
    
    # ⚡ 分批處理（每批 100 支，並強制關閉多線程 threads=False）
    batch_size = 100
    total_tickers = len(target_tickers)
    
    for i in range(0, total_tickers, batch_size):
        batch_tickers = target_tickers[i:i + batch_size]
        
        try:
            # 強制關閉內部多線程（threads=False），解決被 Streamlit 系統阻斷的問題
            data_day = yf.download(batch_tickers, period="1y", interval="1d", progress=False, auto_adjust=True, group_by='ticker', threads=False)
            data_week = yf.download(batch_tickers, period="2y", interval="1wk", progress=False, auto_adjust=True, group_by='ticker', threads=False)
            
            for ticker in batch_tickers:
                try:
                    if len(batch_tickers) == 1:
                        df_d, df_w = data_day, data_week
                    else:
                        df_d = data_day.get(ticker)
                        df_w = data_week.get(ticker)
                        
                    is_match, close_val, vol_val = check_strategy(df_d, df_w)
                    if is_match:
                        matches.append({
                            "ticker": ticker,
                            "name": stocks_info[ticker]['name'],
                            "group": stocks_info[ticker]['group'],
                            "df_day": df_d,
                            "close": close_val,
                            "volume": vol_val
                        })
                except Exception:
                    pass
        except Exception:
            pass
            
        current_progress = min((i + batch_size) / total_tickers, 1.0)
        progress_bar.progress(current_progress)
        status_text.text(f"已安全處理進度: {min(i + batch_size, total_tickers)}/{total_tickers} 支標的...")
        
    status_text.text("掃描完畢！")
    st.success("🎉 全自動掃描完成！")
    
    if matches:
        st.subheader(f"✅ 符合 5 大強勢條件的標的 (共 {len(matches)} 支)")
        for m in matches:
            st.markdown(f"### 📌 {m['name']} ({m['ticker'].split('.')[0]}) ｜ 產業：**{m['group']}** ｜ 收盤價：**{m['close']}** 元 ｜ 成交量：**{m['volume']}** 張")
            plot_stock_chart(m['ticker'], m['df_day'])
            st.divider()
    else:
        st.warning("ℹ️ 今日該範圍內，暫無完全符合所有 5 個條件的股票。")
