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
st.caption("自動獲取全台上市上櫃股票清單，掃描符合：20週MA之上 + 40日突破 + W底型態(6%容錯) + 1.1倍放量 + 成交量>1000張 的強勢標的")

# 台股真實市值前 100 大精選權值股清單
TOP_100_CODES = [
    "2330", "2317", "2454", "2308", "2382", "2881", "2882", "3711", "2303", "2891",
    "2886", "3231", "2884", "2885", "5880", "2892", "2880", "2357", "2379", "3008",
    "6669", "2345", "3034", "2883", "2887", "2890", "5876", "2301", "2412", "3037",
    "2327", "2002", "1101", "1216", "2395", "3661", "2408", "2377", "2308", "2356",
    "3045", "4904", "2409", "3481", "2324", "2353", "2371", "3035", "6415", "2383",
    "3533", "2360", "2312", "1301", "1303", "1326", "2801", "2834", "5871", "9910",
    "9904", "2912", "2207", "2603", "2609", "2615", "2618", "2610", "1402", "2105",
    "1102", "1504", "1519", "1513", "1503", "3017", "6278", "2368", "3702", "6239",
    "3019", "2474", "2352", "2449", "3583", "8046", "6176", "2451", "5483", "6488",
    "3105", "8299", "3293", "3529", "6510", "4966", "5269", "3529", "8069", "6271"
]

# 自動獲取全台股清單與基本資訊 (代號、名稱、產業)
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
        title=f"\n{ticker} - K-Chart (Blue: 20MA / Purple: 100MA - 20W)",
        ylabel='Price (TWD)',
        volume=True,
        ylabel_lower='Volume',
        figratio=(16, 9),
        figscale=1.1,
        returnfig=True
    )
    st.pyplot(fig)

def run_strategy(ticker, name, group):
    try:
        df_day = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        df_week = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
        
        if len(df_day) < 100 or len(df_week) < 20:
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
        
        # 條件 2: 突破 40日新高
        latest_close = close_day[-1]
        if not (latest_close >= np.max(df_day['High'].values.flatten()[-40:-1])):
            return None
        
        # 條件 3: W 底型態數學邏輯 (6%容錯)
        lows = df_day['Low'].values.flatten()[-60:]
        min1_idx = np.argmin(lows[:30])
        min2_idx = 30 + np.argmin(lows[30:-5])
        neck_high = np.max(df_day['High'].values.flatten()[-60:][min1_idx:min2_idx])
        foot1, foot2 = lows[min1_idx], lows[min2_idx]
        cond3 = (abs(foot1 - foot2) / foot1 < 0.06) and (latest_close > neck_high)
        
        if cond3:
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

# 網頁控制台
st.sidebar.header("🔍 全自動選股控制台")

market_choice = st.sidebar.radio("選擇掃描範圍", ["市值前 100 大熱門權值股", "全台股 (上市+上櫃，約 1800+ 支)"])

if st.sidebar.button("🚀 開始全自動雷達掃描", type="primary"):
    stocks_info = get_all_tw_stocks_info()
    
    if market_choice == "市值前 100 大熱門權值股":
        # 精準配對前 100 大權值股
        target_tickers = [ticker for ticker, info in stocks_info.items() if info['code'] in TOP_100_CODES]
    else:
        target_tickers = list(stocks_info.keys())
        
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
            plot_stock_chart(m['ticker'], m['df_day'])
            st.divider()
    else:
        st.warning("ℹ️ 今日該範圍內，暫無完全符合所有 5 個條件的股票。")
