import re
from datetime import date

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco


# -----------------------------
# ページ設定・タイトル
# -----------------------------
st.set_page_config(page_title="かんたん株式分散シミュレーター（比較）", layout="wide")
st.title("🔰 かんたん株式分散シミュレーター（比較）")
st.markdown("投資判断そのものではなく、**複数案を同じ物差しで比べる**ための過去データ・シミュレーターです。")

# -----------------------------
# 免責（社外公開向け）
# -----------------------------
with st.expander("⚠️ ご利用にあたっての重要な注意（必ずお読みください）"):
    st.markdown("""
本アプリは情報提供を目的としたものであり、特定の金融商品の購入・売却・保有を推奨・勧誘するものではありません。  
表示される結果は、過去の市場データに基づくシミュレーションであり、将来の運用成果を保証するものではありません。  
本アプリの利用によって生じたいかなる損失についても、開発者および提供者は一切の責任を負いません。  
投資に関する最終判断は、必ずご自身の責任で行ってください。
""")

# -----------------------------
# 便利関数：ティッカー & 比率パース
# -----------------------------
SEP_PATTERN = r"[,\s、，;；\n\r\t]+"

def parse_tickers(text: str):
    """カンマ/スペース/改行/全角カンマ/読点などを許容し、数字だけは .T を補完"""
    if not text:
        return []
    raw = re.split(SEP_PATTERN, text.strip())
    ts = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        # 全角英数の混在があっても最低限吸収（簡易）
        t = t.replace("　", "")  # 全角スペース除去
        t = t.upper()
        # 数字だけなら日本株想定で .T 補完
        if t.isdigit():
            t = f"{t}.T"
        ts.append(t)
    # 重複除去（順序保持）
    seen = set()
    out = []
    for t in ts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def parse_weights(text: str, n: int):
    """
    比率入力（例: 20, 30, 50 / 20% 30% 50% / 改行区切り）をfloat配列に。
    空なら None を返す（=均等配分にする）
    """
    if text is None:
        return None
    s = str(text).strip()
    if s == "":
        return None

    parts = re.split(SEP_PATTERN, s)
    vals = []
    for p in parts:
        p = p.strip().replace("%", "")
        if p == "":
            continue
        try:
            vals.append(float(p))
        except:
            return "PARSE_ERROR"

    if len(vals) != n:
        return "LEN_MISMATCH"

    w = np.array(vals, dtype=float)
    if np.any(w < 0):
        return "NEGATIVE"
    # 100基準入力を想定（合計が100に近くない場合は正規化）
    ssum = w.sum()
    if ssum == 0:
        return "ZERO"
    w = w / ssum
    return w

def portfolio_metrics(mean, cov, w, risk_free):
    """年率の期待リターン、リスク（標準偏差）、Sharpeを返す"""
    ret = float(np.sum(mean * w) * 252)
    std = float(np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252))
    sharpe = (ret - risk_free) / std if std != 0 else np.nan
    return ret, std, sharpe

def shorten(text: str, max_len: int = 22) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


# -----------------------------
# データ取得系
# -----------------------------
@st.cache_data(show_spinner=False)
def get_data(tickers, start, end):
    try:
        df = yf.download(tickers, start=start, end=end, progress=False)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            if "Adj Close" in df.columns.get_level_values(0):
                return df.xs("Adj Close", axis=1, level=0)
            if "Close" in df.columns.get_level_values(0):
                return df.xs("Close", axis=1, level=0)

        if "Adj Close" in df.columns:
            return df["Adj Close"]
        if "Close" in df.columns:
            return df["Close"]

        return df.iloc[:, 0] if df.shape[1] > 0 else df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
