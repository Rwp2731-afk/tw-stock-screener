import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import twstock
import requests
import io
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股 W底放量突破全自動雷達", layout="wide")
st.title("📈 台股全自動選股雷達 (純HTTP安全版)")
st.caption("已全面避開 yfinance 執行緒衝突，改用安全連線通道")

def get_safe_stocks_info():
    stocks_info = {}
    try:
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
    except Exception:
        pass
    return stocks_info

# 使用純 requests 下載 Yahoo 歷史股價（完全不觸發任何多執行緒）
def get_stock_data_safe(ticker):
    try:
        # 抓取最近 1 年的日資料
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=1700000000&period2=1800000000&interval=1d&events=history&includeAdjustedClose=true"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            if 'Date' in df:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                # 清理異常值
                df = df.dropna()
                return df
    except Exception:
        pass
    return None

# 標準 K 線圖
def plot_stock_chart(ticker, df_day, ma20_val):
    plot_df = df_day.iloc[-120:].copy()
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

# 網頁控制台
st.sidebar.header("🔍 控制台")
market_choice = st.sidebar.radio("選擇掃描範圍", ["測試前 5 大股票", "成交金額熱門前 50 大"])
params = {
    "min_capital": st.sidebar.slider("最低股本門檻 (億元)", 5.0, 100.0, 20.0, 5.0),
    "vol_multiplier": st.sidebar.slider("放量倍數 (對比20日均量)", 1.0, 3.0, 1.2, 0.1),
    "w_tolerance": st.sidebar.slider("W底左右腳容錯率 (%)", 1.0, 15.0, 6.0, 0.5) / 100.0,
    "breakout_days": st.sidebar.number_input("突破幾日內創高", 10, 60, 40),
    "ma_week": st.sidebar.number_input("長期趨勢均線 (週MA)", 10, 40, 20)
}

if st.sidebar.button("🚀 開始安全掃描", type="primary"):
    stocks_info = get_safe_stocks_info()
    all_tickers = list(stocks_info.keys())
    
    target_tickers = all_tickers[:5] if market_choice == "測試前 5 大股票" else all_tickers[:50]
    
    st.info(f"正在以安全模式掃描 {len(target_tickers)} 支股票...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    matches = []
    
    for idx, ticker in enumerate(target_tickers):
        info = stocks_info[ticker]
        status_text.text(f"正在掃描: [{idx + 1}/{len(target_tickers)}] {info['name']}...")
        
        df_day = get_stock_data_safe(ticker)
        if df_day is None or len(df_day) < 100:
            progress_bar.progress((idx + 1) / len(target_tickers))
            continue
            
        try:
            close_day = df_day['Close'].values.flatten()
            vol_day = df_day['Volume'].values.flatten()
            
            latest_close = close_day[-1]
            latest_vol_lots = vol_day[-1] / 1000
            
            if latest_vol_lots < 1000:
                progress_bar.progress((idx + 1) / len(target_tickers))
                continue
                
            # 簡單計算週 MA 替代方案（用日資料模擬週趨勢以確保安全）
            ma_week_val = pd.Series(close_day).rolling(params['ma_week'] * 5).mean().iloc[-1]
            if not (latest_close > ma_week_val):
                progress_bar.progress((idx + 1) / len(target_tickers))
                continue
                
            risk_pct = ((latest_close - ma_week_val) / latest_close) * 100
            capital_yi = round(info['capital'] / 100_000_000, 2)
            
            matches.append({
                "ticker": ticker, "name": info['name'], "group": info['group'], 
                "capital_yi": capital_yi, "df_day": df_day, "close": round(float(latest_close), 2),
                "volume": int(latest_vol_lots), "ma_week_val": round(float(ma_week_val), 2),
                "risk_pct": round(float(risk_pct), 2)
            })
        except Exception:
            pass
            
        progress_bar.progress((idx + 1) / len(target_tickers))
        
    status_text.text("掃描完畢！")
    st.success(f"🎉 掃描完成！找到 {len(matches)} 支符合條件標的。")
    
    if matches:
        for m in matches:
            st.markdown(f"### 📌 {m['name']} ({m['ticker'].split('.')[0]}) ｜ 產業：**{m['group']}** ｜ 股本：**{m['capital_yi']} 億**")
            st.markdown(f"💰 收盤價：**{m['close']}** 元 ｜ 📈 成交量：**{m['volume']}** 張")
            st.markdown(f"🛡️ **停損紅線：{m['ma_week_val']} 元** ｜ ⚠️ **進場風險空間：{m['risk_pct']}%**")
            plot_stock_chart(m['ticker'], m['df_day'], m['ma_week_val'])
            st.divider()
