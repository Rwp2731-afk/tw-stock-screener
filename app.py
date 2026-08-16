import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import twstock
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股 W底放量突破全自動雷達", layout="wide")
st.title("📈 台股全自動選股雷達 (除錯模式)")

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

st.sidebar.header("🔍 控制台")
market_choice = st.sidebar.radio("選擇範圍", ["測試前 3 大股票 (看會不會報錯)", "成交金額熱門前 150 大"])
params = {
    "min_capital": 5.0,
    "vol_multiplier": 1.0,
    "w_tolerance": 0.15,
    "breakout_days": 10,
    "ma_week": 20
}

if st.sidebar.button("🚀 開始測試", type="primary"):
    stocks_info = get_safe_stocks_info()
    all_tickers = list(stocks_info.keys())
    
    # 為了找出問題，我們只抓前 3 支股票來測，看看是不是 yfinance 連線出問題
    target_tickers = all_tickers[:3] if market_choice == "測試前 3 大股票 (看會不會報錯)" else all_tickers[:150]
    
    st.info(f"準備測試抓取 {len(target_tickers)} 支股票...")
    
    for idx, ticker in enumerate(target_tickers):
        info = stocks_info[ticker]
        st.write(f"正在連線抓取: {info['name']} ({ticker})...")
        try:
            # 嚴格測試 yfinance 是否能正常拿到資料
            df = yf.download(ticker, period="1mo", interval="1d", progress=False)
            if df.empty:
                st.error(f"{info['name']} 抓到的資料是空的！")
            else:
                st.success(f"{info['name']} 成功取得資料！筆數: {len(df)}")
        except Exception as e:
            st.error(f"{info['name']} 發生例外錯誤: {e}")
