import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import twstock
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股 W底放量突破全自動雷達", layout="wide")
st.title("📈 台股全自動選股雷達 (批次極速版 + 月均量爆量 + 週20MA停損)")
st.caption("透過 yfinance 批次高速引擎，自動掃描全台上市上櫃強勢標的")

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
                "group": info.group if info.group else "其他",
                "capital": info.capital if hasattr(info, 'capital') and info.capital else 0
            }
    return stocks_info

# 標準 K 線圖
def plot_stock_chart(ticker, df_day, ma20_val):
    plot_df = df_day.iloc[-120:].copy()
    if isinstance(plot_df.columns, pd.MultiIndex):
        plot_df.columns = plot_df.columns.get_level_values(0)
        
    ma20 = plot_df['Close'].rolling(20).mean()
    ma100 = plot_df['Close'].rolling(100).mean()
    
    addplots = [
        mpf.make_addplot(ma20, color='dodgerblue', width=1.5),
        mpf.make_addplot(ma100, color='purple', width=1.8),
        mpf.make_addplot([ma20_val]*len(plot_df), color='red', linestyle='dashed', width=1.2)
    ]
    
    fig, ax = mpf.plot(
        plot_df,
        type='candle',
        style='yahoo',
        addplot=addplots,
        title=f"\n{ticker} - Trend & Stop-Loss Level (Red Dashed: Weekly 20MA)",
        ylabel='Price (TWD)',
        volume=True,
        ylabel_lower='Volume',
        figratio=(16, 9),
        figscale=1.1,
        returnfig=True
    )
    st.pyplot(fig)
    plt.close(fig)

# 繪製現金股利長條圖
def plot_dividend_bar_chart(div_df):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    years = div_df['年份'].astype(str).tolist()
    dividends = div_df['現金股利'].tolist()
    bars = ax.bar(years, dividends, color='teal', alpha=0.85, width=0.6)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax.set_title("Recent 10-Year Cash Dividend (TWD)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Dividend (TWD)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ================= 網頁控制台 =================
st.sidebar.header("🔍 全自動選股控制台")
market_choice = st.sidebar.radio("選擇掃描範圍", ["成交金額熱門前 150 大", "全台股 (約 1800+ 支批次掃描)"])
params = {
    "min_capital": st.sidebar.slider("最低股本門檻 (億元)", 5.0, 100.0, 20.0, 5.0),
    "vol_multiplier": st.sidebar.slider("放量倍數 (對比20日均量)", 1.0, 3.0, 1.2, 0.1),
    "w_tolerance": st.sidebar.slider("W底左右腳容錯率 (%)", 1.0, 15.0, 6.0, 0.5) / 100.0,
    "breakout_days": st.sidebar.number_input("突破幾日內創高", 10, 60, 40),
    "ma_week": st.sidebar.number_input("長期趨勢均線 (週MA)", 10, 40, 20)
}

if st.sidebar.button("🚀 開始全自動雷達掃描", type="primary"):
    stocks_info = get_all_tw_stocks_info()
    all_tickers = list(stocks_info.keys())
    
    with st.spinner("正在向 Yahoo 取得市場資料，請稍候..."):
        try:
            if market_choice == "成交金額熱門前 150 大":
                # 先抓取最近 5 天資料來計算成交金額排名
                temp_df = yf.download(all_tickers, period="5d", interval="1d", progress=False, auto_adjust=True)
                if 'Close' in temp_df and 'Volume' in temp_df:
                    latest_close = temp_df['Close'].iloc[-1]
                    latest_vol = temp_df['Volume'].iloc[-1]
                    turnover = latest_close * latest_vol
                    top_tickers = turnover.sort_values(ascending=False).head(150).index.tolist()
                    target_tickers = [t for t in top_tickers if t in stocks_info]
                else:
                    target_tickers = all_tickers[:150]
            else:
                target_tickers = all_tickers

            # 批次下載近 1 年日資料
            data_day = yf.download(target_tickers, period="1y", interval="1d", progress=False, auto_adjust=True)
            # 批次下載近 2 年週資料 (用於計算週均線)
            data_week = yf.download(target_tickers, period="2y", interval="1wk", progress=False, auto_adjust=True)
        except Exception as e:
            st.error(f"下載資料時發生錯誤: {e}")
            target_tickers = []

    if target_tickers and 'Close' in data_day:
        st.info(f"成功取得市場數據，正在進行策略篩選與運算...")
        matches = []
        min_capital_yuan = params['min_capital'] * 100_000_000

        # 針對每一支股票在記憶體內進行策略檢驗
        for ticker in target_tickers:
            try:
                info = stocks_info[ticker]
                # 1. 股本過濾
                if info['capital'] < min_capital_yuan:
                    continue
                
                # 萃取單一股票的日資料
                if len(target_tickers) == 1:
                    df_day = data_day.copy()
                    df_week = data_week.copy()
                else:
                    if ticker not in data_day['Close'].columns:
                        continue
                    df_day = pd.DataFrame({
                        'Open': data_day['Open'][ticker],
                        'High': data_day['High'][ticker],
                        'Low': data_day['Low'][ticker],
                        'Close': data_day['Close'][ticker],
                        'Volume': data_day['Volume'][ticker]
                    }).dropna()
                    
                    df_week = pd.DataFrame({
                        'Close': data_week['Close'][ticker]
                    }).dropna()

                if len(df_day) < 100 or len(df_week) < params['ma_week']:
                    continue

                close_day = df_day['Close'].values.flatten()
                vol_day = df_day['Volume'].values.flatten()
                close_week = df_week['Close'].values.flatten()

                # 2. 長期週均線之上
                ma_week_val = pd.Series(close_week).rolling(params['ma_week']).mean().iloc[-1]
                if not (close_week[-1] > ma_week_val):
                    continue

                # 3. 成交量門檻 >= 1000張
                latest_vol_lots = vol_day[-1] / 1000
                if latest_vol_lots < 1000:
                    continue

                # 4. 20日均量爆量過濾
                ma_vol_val = pd.Series(vol_day).rolling(20).mean().iloc[-1]
                if not (vol_day[-1] >= (ma_vol_val * params['vol_multiplier'])):
                    continue

                # 5. 突破 N 日新高
                breakout_days = params['breakout_days']
                latest_close = close_day[-1]
                if not (latest_close >= np.max(df_day['High'].values.flatten()[-breakout_days:-1])):
                    continue

                # 6. W 底型態數學邏輯
                tolerance = params['w_tolerance']
                lows = df_day['Low'].values.flatten()[-60:]
                min1_idx = np.argmin(lows[:30])
                min2_idx = 30 + np.argmin(lows[30:-5])
                neck_high = np.max(df_day['High'].values.flatten()[-60:][min1_idx:min2_idx])
                foot1, foot2 = lows[min1_idx], lows[min2_idx]
                cond_w = (abs(foot1 - foot2) / foot1 < tolerance) and (latest_close > neck_high)
                
                if not cond_w:
                    continue

                risk_pct = ((latest_close - ma_week_val) / latest_close) * 100
                capital_yi = round(info['capital'] / 100_000_000, 2)

                # 取得個股股利
                stock_obj = yf.Ticker(ticker)
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

                matches.append({
                    "ticker": ticker, "name": info['name'], "group": info['group'], "capital_yi": capital_yi,
                    "df_day": df_day, "close": round(float(latest_close), 2),
                    "volume": int(latest_vol_lots), "ma_week_val": round(float(ma_week_val), 2),
                    "risk_pct": round(float(risk_pct), 2), "div_history": div_history_df
                })
            except Exception:
                continue

        st.success(f"🎉 掃描完成！共找出 {len(matches)} 支符合條件的強勢標的。")

        if matches:
            st.subheader(f"✅ 符合條件的強勢標的清單")
            for m in matches:
                st.markdown(f"### 📌 {m['name']} ({m['ticker'].split('.')[0]}) ｜ 產業：**{m['group']}** ｜ 股本：**{m['capital_yi']} 億**")
                st.markdown(f"💰 收盤價：**{m['close']}** 元 ｜ 📈 成交量：**{m['volume']}** 張 (已過濾 >= 1000張)")
                
                risk_color = "red" if m['risk_pct'] < 3 else "green"
                st.markdown(f"🛡️ **停損紅線 (週20MA)：{m['ma_week_val']} 元** ｜ ⚠️ **進場風險空間：<span style='color:{risk_color}'>{m['risk_pct']}%</span>** (跌破即觸發退場)", unsafe_allow_html=True)
                
                if not m['div_history'].empty:
                    st.markdown("**📊 近十年現金股利發放長條圖：**")
                    plot_dividend_bar_chart(m['div_history'])
                else:
                    st.info("該標的無近期股利發放紀錄")
                    
                plot_stock_chart(m['ticker'], m['df_day'], m['ma_week_val'])
                st.divider()
        else:
            st.warning("ℹ️ 在目前的參數設定下，全市場暫無同時符合所有條件的股票。你可以嘗試微調側邊欄的「放量倍數」或「W底容錯率」。")
    else:
        st.error("無法取得市場資料，請檢查網路連線或稍後再試。")
