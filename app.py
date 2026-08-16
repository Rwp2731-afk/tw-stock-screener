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
st.title("📈 台股全自動選股雷達 (近十年股利波動折線圖)")
st.caption("自動獲取全台上市上櫃股票清單，依據動態技術參數與近十年純淨態配息折線圖掃描強勢標的")

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

# 單一股票策略運算邏輯
def run_strategy(ticker, name, group, params):
    try:
        stock_obj = yf.Ticker(ticker)
        df_day = stock_obj.history(period="1y", auto_adjust=True)
        df_week = stock_obj.history(period="2y", interval="1wk", auto_adjust=True)
        
        if len(df_day) < 100 or len(df_week) < params['ma_week']:
            return None
            
        close_day = df_day['Close'].values.flatten()
        vol_day = df_day['Volume'].values.flatten()
        close_week = df_week['Close'].values.flatten()
        
        # 條件 1: 長期均線之上
        ma_week_val = pd.Series(close_week).rolling(params['ma_week']).mean().iloc[-1]
        if not (close_week[-1] > ma_week_val):
            return None
        
        # 條件 2: 成交量固定門檻 >= 1000張
        latest_vol_lots = vol_day[-1] / 1000
        if latest_vol_lots < 1000:
            return None
        
        # 條件 3: 放大量對比
        ma_vol_val = pd.Series(vol_day).rolling(20).mean().iloc[-1]
        if not (vol_day[-1] >= (ma_vol_val * params['vol_multiplier'])):
            return None
        
        # 條件 4: 突破 N 日新高
        breakout_days = params['breakout_days']
        latest_close = close_day[-1]
        if not (latest_close >= np.max(df_day['High'].values.flatten()[-breakout_days:-1])):
            return None
        
        # 條件 5: W 底型態數學邏輯
        tolerance = params['w_tolerance']
        lows = df_day['Low'].values.flatten()[-60:]
        min1_idx = np.argmin(lows[:30])
        min2_idx = 30 + np.argmin(lows[30:-5])
        neck_high = np.max(df_day['High'].values.flatten()[-60:][min1_idx:min2_idx])
        foot1, foot2 = lows[min1_idx], lows[min2_idx]
        cond_w = (abs(foot1 - foot2) / foot1 < tolerance) and (latest_close > neck_high)
        
        if not cond_w:
            return None

        # ── 取得近十年完整股利發放數據 ──
        dividends = stock_obj.dividends
        div_history_df = pd.DataFrame(columns=["年份", "現金股利"])
        
        if not dividends.empty:
            dividends.index = pd.to_datetime(dividends.index)
            div_df = pd.DataFrame({'Dividend': dividends})
            div_df['Year'] = div_df.index.year
            yearly_div = div_df.groupby('Year')['Dividend'].sum().reset_index()
            
            yearly_div = yearly_div.sort_values(by='Year', ascending=True).tail(10)
            yearly_div.columns = ["年份", "現金股利"]
            yearly_div["現金股利"] = yearly_div["現金股利"].round(2)
            div_history_df = yearly_div

        return {
            "ticker": ticker,
            "name": name,
            "group": group,
            "df_day": df_day,
            "close": round(float(latest_close), 2),
            "volume": int(latest_vol_lots),
            "div_history": div_history_df
        }
    except Exception:
        return None

# ================= 網頁控制台 (Sidebar) =================
st.sidebar.header("🔍 全自動選股控制台")

market_choice = st.sidebar.radio("選擇掃描範圍", ["成交金額熱門前 150 大", "全台股 (上市+上櫃，約 1800+ 支極速掃描)"])

st.sidebar.divider()
st.sidebar.subheader("⚙️ 策略參數動態微調")

params = {
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
        
    st.info(f"正在以多線程極速引擎掃描 {len(target_tickers)} 支股票...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    matches = []
    
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
        st.subheader(f"✅ 符合條件的強勢標的 (共 {len(matches)} 支)")
        for m in matches:
            st.markdown(f"### 📌 {m['name']} ({m['ticker'].split('.')[0]}) ｜ 產業：**{m['group']}**")
            st.markdown(f"💰 收盤價：**{m['close']}** 元 ｜ 📈 成交量：**{m['volume']}** 張 (已過濾 >= 1000張)")
            
            # 呈現近十年現金股利折線圖與明細表 (使用 use_container_width 並帶入參數隱藏互動提示)
            if not m['div_history'].empty:
                st.markdown("**📊 近十年現金股利波動趨勢與明細：**")
                chart_data = m['div_history'].set_index("年份")
                st.line_chart(chart_data, use_container_width=True)
                st.dataframe(chart_data.T, use_container_width=True)
            else:
                st.info("該標的無近期股利發放紀錄")
                
            plot_stock_chart(m['ticker'], m['df_day'])
            st.divider()
    else:
        st.warning("ℹ️ 在目前的參數設定下，暫無符合條件的股票。")
