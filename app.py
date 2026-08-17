import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股 W底放量突破全自動雷達", layout="wide")
st.title("📈 台股全自動選股雷達 (官方資本額對應版)")
st.caption("自動獲取全台上市上櫃股票清單與真實資本額，依據動態技術參數與週 20MA 停損紅線進行強勢標的掃描")

# 從證交所/櫃買中心官方公開資料抓取真實上市櫃股票清單與資本額
@st.cache_data(ttl=86400)
def get_all_tw_stocks_info():
    stocks_info = {}
    
    # 抓取上市股票清單 (TWSE)
    try:
        url_twse = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        res = requests.get(url_twse, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for item in data:
                code = item.get('公司代號', '').strip()
                name = item.get('公司名稱', '').strip()
                group = item.get('產業別', '').strip() or "其他"
                # 實收資本額單位通常為元
                capital_str = item.get('實收資本額', '0').replace(',', '').strip()
                capital = float(capital_str) if capital_str.replace('.', '', 1).isdigit() else 0.0
                
                if code and len(code) == 4:
                    ticker = f"{code}.TW"
                    stocks_info[ticker] = {
                        "code": code,
                        "name": name,
                        "group": group,
                        "capital": capital
                    }
    except Exception:
        pass

    # 抓取上櫃股票清單 (TPEX)
    try:
        url_tpex = "https://www.tpex.org.tw/openapi/v1/t187ap03_O"
        res = requests.get(url_tpex, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for item in data:
                code = item.get('公司代號', '').strip()
                name = item.get('公司名稱', '').strip()
                group = item.get('產業別', '').strip() or "其他"
                capital_str = item.get('實收資本額', '0').replace(',', '').strip()
                capital = float(capital_str) if capital_str.replace('.', '', 1).isdigit() else 0.0
                
                if code and len(code) == 4:
                    ticker = f"{code}.TWO"
                    stocks_info[ticker] = {
                        "code": code,
                        "name": name,
                        "group": group,
                        "capital": capital
                    }
    except Exception:
        pass

    # 若官方 API 臨時異常時的備用防呆清單 (確保系統絕對不會無股票可掃)
    if not stocks_info:
        # 內建幾個主要範例讓系統不中斷
        fallback_list = [
            ("2330", "台積電", "半導體業", 259303000000, "上市"),
            ("2317", "鴻海", "電腦及週邊設備業", 138629900000, "上市"),
            ("2454", "聯發科", "半導體業", 15981000000, "上市")
        ]
        for code, name, group, cap, market in fallback_list:
            suffix = ".TW" if market == "上市" else ".TWO"
            ticker = f"{code}{suffix}"
            stocks_info[ticker] = {"code": code, "name": name, "group": group, "capital": cap}

    return stocks_info

# 標準 K 線圖（已加入週 20MA 紅色停損虛線）
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

# 繪製完全靜態的股利長條圖 (Matplotlib 產出)
def plot_dividend_bar_chart(div_df):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    years = div_df['年份'].astype(str).tolist()
    dividends = div_df['現金股利'].tolist()
    
    bars = ax.bar(years, dividends, color='teal', alpha=0.85, width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
                    
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
            "ticker": ticker,
            "name": name,
            "group": group,
            "capital_yi": capital_yi,
            "df_day": df_day,
            "close": round(float(latest_close), 2),
            "volume": int(latest_vol_lots),
            "ma_week_val": round(float(ma_week_val), 2),
            "risk_pct": round(float(risk_pct), 2),
            "div_history": div_history_df
        }
    except Exception:
        return None

# ================= 網頁控制台 (Sidebar) =================
st.sidebar.header("🔍 全自動選股控制台")

market_choice = st.sidebar.radio("選擇掃描範圍", ["成交金額熱門前 150 大", "全台股 (上市+上櫃，序列化穩定掃描)"])

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
        
    st.info(f"正在以序列化穩定引擎掃描 {len(target_tickers)} 支股票...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    matches = []
    
    completed_count = 0
    total_count = len(target_tickers)
    
    for ticker in target_tickers:
        res = run_strategy(
            ticker, 
            stocks_info[ticker]['name'], 
            stocks_info[ticker]['group'], 
            stocks_info[ticker]['capital'], 
            params
        )
        if res:
            matches.append(res)
        
        completed_count += 1
        if completed_count % 5 == 0 or completed_count == total_count:
            progress_bar.progress(completed_count / total_count)
            status_text.text(f"已掃描: {completed_count}/{total_count} 支標的...")
            
        time.sleep(0.15)
                
    status_text.text("掃描完畢！")
    st.success("🎉 全自動掃描完成！")
    
    if matches:
        st.subheader(f"✅ 符合條件的強勢標的 (共 {len(matches)} 支)")
        for m in matches:
            st.markdown(f"### 📌 {m['name']} ({m['ticker'].split('.')[0]}) ｜ 產業：**{m['group']}** ｜ 資本額：**{m['capital_yi']} 億**")
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
        st.warning("ℹ️ 在目前的參數設定下，暫無符合條件的股票。")
