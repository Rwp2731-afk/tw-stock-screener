import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import twstock
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股 W底放量突破全自動雷達", layout="wide")
st.title("📈 台股全自動選股雷達 (月均量爆量過濾 + 週20MA停損紀律)")
st.caption("自動獲取全台上市上櫃股票清單，依據 20 日均量比對、動態技術參數、股本大小與週 20MA 停損紅線進行強勢標的掃描")

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

# 單一股票策略運算邏輯
def run_strategy(ticker, name, group, capital, params):
    try:
        min_capital_yuan = params['min_capital'] * 100_000_000
        if capital < min_capital_yuan:
            return None

        stock_obj = yf.Ticker(ticker)
        df_day = stock_obj.history(period="1y", auto_adjust=True)
        df_week = stock_obj.history(period="2y", interval="1wk", auto_adjust=True)
        
        if len(df_day) < 100 or len(df_week) < params['ma_week']:
            return None
            
        close_day = df_day['Close'].values.flatten()
        vol_day = df_day['Volume'].values.flatten()
        close_week = df_week['Close'].values.flatten()
        
        ma_week_val = pd.Series(close_week).rolling(params['ma_week']).mean().iloc[-1]
        if not (close_week[-1] > ma_week_val):
            return None
        
        latest_vol_lots = vol_day[-1] / 1000
        if latest_vol_lots < 1000:
            return None
        
        # 改回對比 20 日均量
        ma_vol_val = pd.Series(vol_day).rolling(20).mean().iloc[-1]
        if not (vol_day[-1] >= (ma_vol_val * params['vol_multiplier'])):
            return None
        
        breakout_days = params['breakout_days']
        latest_close = close_day[-1]
        if not (latest_close >= np.max(df_day['High'].values.flatten()[-breakout_days:-1])):
            return None
        
        tolerance = params['w_tolerance']
        lows = df_day['Low'].values.flatten()[-60:]
        min1_idx = np.argmin(lows[:30])
        min2_idx = 30 + np.argmin(lows[30:-5])
        neck_high = np.max(df_day['High'].values.flatten()[-60:][min1_idx:min2_idx])
        foot1, foot2 = lows[min1_idx], lows[min2_idx]
        cond_w = (abs(foot1 - foot2) / foot1 < tolerance) and (latest_close > neck_high)
        
        if not cond_w:
            return None

        risk_pct = ((latest_close - ma_week_val) / latest_close) * 100
        capital_yi = round(capital / 100_000_000, 2)
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
            "ticker": ticker, "name": name, "group": group, "capital_yi": capital_yi,
            "df_day": df_day, "close": round(float(latest_close), 2),
            "volume": int(latest_vol_lots), "ma_week_val": round(float(ma_week_val), 2),
            "risk_pct": round(float(risk_pct), 2), "div_history": div_history_df
        }
    except Exception:
        return None

# ================= 網頁控制台 =================
st.sidebar.header("🔍 全自動選股控制台")
market_choice = st.sidebar.radio("選擇掃描範圍", ["成交金額熱門前 150 大", "全台股 (約 1800+ 支極速掃描)"])
params = {
    "min_capital": st.sidebar.slider("最低股本門檻 (億元)", 5.0, 100.0, 20.0, 5.0),
    "vol_multiplier": st.sidebar.slider("放量倍數 (對比20日均量)", 1.0, 3.0, 1.2, 0.1),
    "w_tolerance": st.sidebar.slider("W底左右腳容錯率 (%)", 1.0, 15.0, 6.0, 0.5) / 100.0,
    "breakout_days": st.sidebar.number_input("突破幾日內創高", 10, 60, 40),
    "ma_week": st.sidebar.number_input("長期趨勢均線 (週MA)", 10, 40, 20)
}

if st.sidebar.button("🚀 開始全自動雷達掃描", type="primary"):
    # ... (掃描邏輯與顯示邏輯維持原樣)
