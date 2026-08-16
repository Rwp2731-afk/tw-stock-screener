import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import twstock
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股 W底放量突破全自動雷達", layout="wide")
st.title("📈 台股全自動選股雷達 (全台股極速掃描)")
st.caption("自動獲取全台上市上櫃股票清單，依據側邊欄動態參數掃描強勢標的")

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

# 標準 K 線圖
def plot_stock_chart(ticker, df_day):
    plot_df = df_day.iloc[-120:].copy()
    if isinstance(plot_df.columns, pd.MultiIndex):
        plot_df.columns = plot_df.columns.get_level_values(0)
        
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
        title=f"\n{ticker} - K-Chart (Blue: 20MA / Purple: 100MA)",
        ylabel='Price (TWD)',
        volume=True,
        ylabel_lower='Volume',
        figratio=(16, 9),
        figscale=1.1,
        returnfig=True
    )
    st.pyplot(fig)

# 單一股票策略運算邏輯 (帶入動態參數)
def run_strategy(ticker, name, group, params):
    try:
        df_day = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        df_week = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
        
        if len(df_day) < 100 or len(df_week) < params['ma_week']:
            return None
            
        close_day = df_day['Close'].values.flatten()
        vol_day = df_day['Volume'].values.flatten()
        close_week = df_week['Close'].values.flatten()
        
        # 條件 1: 長期均線之上 (預設 20週 MA)
        ma_week_val = pd.Series(close_week).rolling(params['ma_week']).mean().iloc[-1]
        if not (close_week[-1] > ma_week_val):
            return None
        
        # 條件 2: 成交量門檻 (張數)
        latest_vol_lots = vol_day[-1] / 1000
        if latest_vol_lots < params['min_vol_lots']:
            return None
        
        # 條件 3: 放大量對比 (預設 20日均量 X 倍)
        ma_vol_val = pd.Series(vol_day).rolling(20).mean().iloc[-1]
        if not (vol_day[-1] >= (ma_vol_val * params['vol_multiplier'])):
            return None
        
        # 條件 4: 突破 N 日新高 (預設 40日)
        breakout_days = params['breakout_days']
        latest_close = close_day[-1]
        if not (latest_close >= np.max(df_day['High'].values.flatten()[-breakout_days:-1])):
            return None
        
        # 條件 5: W 底型態數學邏輯 (可調容錯率)
        tolerance = params['w_tolerance']
        lows = df_day['Low'].values.flatten()[-60:]
        min1_idx = np.argmin(lows[:30])
        min2_idx = 30 + np.argmin(lows[30:-5])
        neck_high = np.max(df_day['High'].values.flatten()[-60:][min1_idx:min2_idx])
        foot1, foot2 = lows[min1_idx], lows[min2_idx]
        cond_w = (abs(foot1 - foot2) / foot1 < tolerance) and (latest_close > neck_high)
        
        if cond_w:
            return {
                "ticker": ticker,
                "name": name,
                "group": group,
                "df_day": df_day,
                "close": round(float(latest_close), 2),
                "volume": int(latest_vol_lots)
            }
    except Exception:
        return None
    return None

# ================= 網頁控制台 (Sidebar) =================
st.sidebar.header("🔍 全自動選股控制台")

market_choice = st.sidebar.radio("選擇掃描範圍", ["成交金額熱門前 150 大", "全台股 (上市+上櫃，約 1800+ 支極速掃描)"])

st.sidebar.divider()
st.sidebar.subheader("⚙️ 策略參數動態微調")

# 動態參數控制滑桿與輸入框
params = {
    "min_vol_lots": st.sidebar.slider("最低成交量門檻 (張)", 500, 5000, 1000, 100),
    "vol_multiplier": st.sidebar.slider("放量倍數 (對比20日均量)", 1.0, 3.0, 1.1, 0.1),
    "w_tolerance": st.sidebar.slider("W底左右腳容錯率 (%)", 1.0, 15.0, 6.0, 0.5) / 100.0,
    "breakout_days": st.sidebar.number_input("突破幾日內創高", 10, 60, 40),
    "ma_week": st.sidebar.number_input("長期趨勢均線 (週MA)", 10, 40, 20)
}

st.sidebar.divider()

if st.sidebar.button("🚀 開始全自動雷達掃描", type="primary"):
    stocks_info = get_all_tw_stocks_info()
    all_tickers = list(stocks_info.keys())
    
    if market_choice == "成交金額熱門前 150 大":
        st.info("正在計算全市場最新成交金額排序，挑選前 150 大熱門標的...")
        try:
            download_df = yf.download(all_tickers, period="5d", interval="1d", progress=False, auto_adjust=True)
            if 'Close' in download_df and 'Volume' in download_df:
                latest_close = download_df['Close'].iloc[-1]
                latest_vol = download_df['Volume'].iloc[-1]
                turnover = latest_close * latest_vol
                top_turnover_tickers = turnover.sort_values(ascending=False).head(150).index.tolist()
                target_tickers = [t for t in top_turnover_tickers if t in stocks_info]
            else:
                target_tickers = all_tickers[:150]
        except Exception:
            target_tickers = all_tickers[:150]
    else:
        target_tickers = all_tickers
        
    st.info(f"正在以多線程極速引擎掃描 {len(target_tickers)} 支股票，套用動態參數中...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    matches = []
    
    # ⚡ 多線程平行計算 (開啟 16 個併行線程)
    completed_count = 0
    total_count = len(target_tickers)
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_ticker = {
            executor.submit(run_strategy, ticker, stocks_info[ticker]['name'], stocks_info[ticker]['group'], params): ticker 
            for ticker in target_tickers
        }
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            res = future.result()
            if res:
                matches.append(res)
            
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_count:
                progress_bar.progress(completed_count / total_count)
                status_text.text(f"已極速掃描: {completed_count}/{total_count} 支標的...")
                
    status_text.text("掃描完畢！")
    st.success("🎉 全自動極速掃描完成！")
    
    if matches:
        st.subheader(f"✅ 符合目前動態參數條件的標的 (共 {len(matches)} 支)")
        for m in matches:
            st.markdown(f"### 📌 {m['name']} ({m['ticker'].split('.')[0]}) ｜ 產業：**{m['group']}** ｜ 收盤價：**{m['close']}** 元 ｜ 成交量：**{m['volume']}** 張")
            plot_stock_chart(m['ticker'], m['df_day'])
            st.divider()
    else:
        st.warning("ℹ️ 在目前的參數設定下，暫無完全符合條件的股票。試著在側邊欄放寬條件（例如降低成交量或調高容錯率）看看吧！")
