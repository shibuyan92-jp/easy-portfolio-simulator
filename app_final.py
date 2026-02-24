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
st.set_page_config(page_title="かんたん株式分散シミュレーター（株数入力・比較）", layout="wide")
st.title("🔰 かんたん株式分散シミュレーター（株数入力・比較）")
st.markdown("**日本株（4桁コード）限定**。株数（数量）を入れるだけで時価比率に自動変換し、A（現状）vs B（検討）を比較します。")

# -----------------------------
# 免責
# -----------------------------
with st.expander("⚠️ ご利用にあたっての重要な注意（必ずお読みください）"):
    st.markdown("""
本アプリは情報提供を目的としたものであり、特定の金融商品の購入・売却・保有を推奨・勧誘するものではありません。  
表示される結果は、過去の市場データに基づくシミュレーションであり、将来の運用成果を保証するものではありません。  
本アプリの利用によって生じたいかなる損失についても、開発者および提供者は一切の責任を負いません。  
投資に関する最終判断は、必ずご自身の責任で行ってください。
""")

# -----------------------------
# プライバシー（データ取扱い）
# -----------------------------
with st.expander("🔒 プライバシー / データの取扱い（重要）"):
    st.markdown("""
- 本アプリは、入力された情報を使ってポートフォリオを比較するためのツールです。  
- **アップロードされたCSVファイルは保存しません。** ファイルはStreamlitの仕様上、**サーバーのメモリ（RAM）上でのみ一時的に扱われ**、取り込み後は**クリア（破棄）**されます。ユーザーがファイルを置き換える／クリアする／ブラウザタブを閉じると、アップロードデータはメモリから削除されます。  
- CSVから取り込むのは **銘柄コード（4桁）と株数（数量）のみ**です。不要な情報（氏名・口座番号等）が含まれるCSVはアップロードしないでください。  
- 取り込まれた **銘柄コード・株数**は、比較のために**ブラウザを閉じるまでの間（セッション中）**のみ画面上に保持されます。必要に応じて、画面の「Aをリセット」からいつでも消去できます。  
""")
    st.caption("（参考：st.file_uploader のファイルはRAM上で管理され、置換/クリア/タブを閉じると削除されます）")

# -----------------------------
# 日本株オンリー：ティッカー正規化
# -----------------------------
JP_TICKER_RE = re.compile(r"^\d{4}(\.T)?$")

def normalize_ticker_jp(t: str) -> str:
    """日本株のみ：4桁 or 4桁.T のみ許可。4桁は .T 補完。"""
    if t is None:
        return ""
    t = str(t).strip().replace("　", "").upper()
    if t == "":
        return ""
    if not JP_TICKER_RE.match(t):
        return "INVALID"
    if t.endswith(".T"):
        return t
    return f"{t}.T"

def shorten(text: str, max_len: int = 22) -> str:
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= max_len else s[:max_len - 1] + "…"

# -----------------------------
# データ取得
# -----------------------------
@st.cache_data(show_spinner=False)
def get_prices(tickers, start, end):
    """yfinanceから価格（Adj Close優先）を取得"""
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

        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_company_names(tickers_list):
    names = {}
    for t in tickers_list:
        try:
            info = yf.Ticker(t).info or {}
            names[t] = info.get("shortName", info.get("longName", t))
        except Exception:
            names[t] = t
    return names

def latest_prices_asof(price_df: pd.DataFrame) -> pd.Series:
    """各銘柄の直近価格（データ期間内の最終行）"""
    return price_df.ffill().iloc[-1]

def portfolio_metrics_from_weights(mean, cov, w_risky, rf, cash_w=0.0):
    """
    w_risky はリスク資産の比率（合計=1-cash_w）。
    cash_w は無リスク比率（0〜1）。
    """
    r_risky = float(np.sum(mean * w_risky) * 252)
    r_total = r_risky + cash_w * rf
    vol = float(np.sqrt(np.dot(w_risky.T, np.dot(cov, w_risky))) * np.sqrt(252))
    sharpe = (r_total - rf) / vol if vol != 0 else np.nan
    return r_total, vol, sharpe

def optimize_sharpe(mean, cov, min_w, max_w, rf):
    n = len(mean)

    def neg_sharpe(w):
        r = np.sum(mean * w) * 252
        s = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)
        if s == 0:
            return 1e9
        return -((r - rf) / s)

    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
    bnds = tuple((min_w, max_w) for _ in range(n))

    res = sco.minimize(
        neg_sharpe,
        x0=np.array([1.0 / n] * n),
        method="SLSQP",
        bounds=bnds,
        constraints=cons
    )
    return res

def clean_shares_series(s: pd.Series) -> pd.Series:
    """株数列：カンマ・文字混在を雑に吸収して数値化"""
    s = s.astype(str).str.replace(",", "", regex=False)
    s = s.str.replace("株", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

# -----------------------------
# サイドバー：共通設定
# -----------------------------
st.sidebar.header("🛠️ 設定パネル")

# 終了日：今日をデフォルト
st.session_state.setdefault("start_date", pd.to_datetime("2020-01-01").date())
st.session_state.setdefault("end_date", date.today())

def set_end_today():
    st.session_state["end_date"] = date.today()

start_date = st.sidebar.date_input("開始日", key="start_date")
c_end, c_today = st.sidebar.columns([3, 1])
with c_end:
    end_date = st.sidebar.date_input("終了日", key="end_date")
with c_today:
    st.write("")
    st.sidebar.button("今日", on_click=set_end_today)

risk_free_rate = st.sidebar.number_input("安全資産の利回り (%)", value=1.0, step=0.1, help="国債などの金利") / 100.0

st.sidebar.subheader("B案を最適化する場合の制約")
min_weight = st.sidebar.slider("最低これくらいは持ちたい (%)", 0, 20, 5, 1) / 100.0
max_weight = st.sidebar.slider("最大ここまでにしておく (%)", 20, 100, 40, 5) / 100.0

st.sidebar.subheader("Aの現金（任意）")
cash_a = st.sidebar.number_input("A: 現金（無リスク資産）", value=0.0, step=10000.0)

st.sidebar.subheader("Bの現金（任意）")
cash_b = st.sidebar.number_input("B: 現金（無リスク資産）", value=0.0, step=10000.0)

st.sidebar.subheader("B案の作り方")
b_mode = st.sidebar.radio(
    "Bの配分の作り方",
    ["🧮 株数で入力（Bも現状と同様に比較）", "🤖 Sharpe最大化でB配分を自動提案"],
)

# -----------------------------
# 入力：A/Bの株数（data_editor）
# -----------------------------
def default_holdings_df():
    return pd.DataFrame(
        {
            "ティッカー": ["8802", "7203", "6758", "8306", "9984"],
            "株数": [10, 10, 10, 10, 10],
        }
    )

st.session_state.setdefault("holdings_a", default_holdings_df())
st.session_state.setdefault("holdings_b", default_holdings_df())

st.markdown("## ① 株数入力（コピペOK / 日本株のみ）")

# -----------------------------
# A：CSVアップロード（任意）※キャッシュしない（必ず破棄したい要件のため）
# -----------------------------
st.markdown("### 🅰 A：口座CSVアップロード（任意・保存しません）")

# uploaderはキーを付けて、取り込み後に明示的にクリアできるようにする
uploaded_a = st.file_uploader(
    "Aの保有一覧CSVをアップロード（任意）",
    type=["csv"],
    key="uploader_a"
)

# Aをリセット（入力もアップロードも消す）
c_reset1, c_reset2 = st.columns([1, 3])
with c_reset1:
    if st.button("🧹 Aをリセット"):
        st.session_state["holdings_a"] = default_holdings_df()
        st.session_state["uploader_a"] = None
        st.rerun()
with c_reset2:
    st.caption("※アップロードCSVは保存しません。取り込み後は自動でクリアします。")

if uploaded_a is not None:
    # 1MB以上は拒否（口座CSVは通常数KB～）
    if getattr(uploaded_a, "size", 0) > 1 * 1024 * 1024:
        st.error("⚠️ CSVが大きすぎます（1MBまで）。口座の保有一覧など小さなCSVを想定しています。")
        st.stop()

    # ✅ その場で一度だけ読み込む（キャッシュしない）
    df_csv = pd.read_csv(uploaded_a)

    st.write("CSV列を選択してください（証券会社により列名が異なるため）")
    cols = list(df_csv.columns)
    col_t = st.selectbox("ティッカー列（銘柄コード）", cols, key="csv_col_t")
    col_s = st.selectbox("株数列（数量）", cols, key="csv_col_s")

    df_import = df_csv[[col_t, col_s]].copy()
    df_import.columns = ["ティッカー", "株数"]

    df_import["ティッカー"] = df_import["ティッカー"].map(normalize_ticker_jp)
    if (df_import["ティッカー"] == "INVALID").any():
        st.error("⚠️ 日本株のみ対応です。ティッカーは「7203」または「7203.T」の形式である必要があります。")
        st.stop()

    df_import["株数"] = clean_shares_series(df_import["株数"]).fillna(0.0)
    df_import = df_import[(df_import["ティッカー"] != "") & (df_import["株数"] > 0)].reset_index(drop=True)

    if df_import.empty:
        st.error("⚠️ 有効な行がありません（株数>0の行が必要）。")
        st.stop()

    # ✅ 必要最小限（ティッカー・株数）だけ保持
    st.session_state["holdings_a"] = df_import[["ティッカー", "株数"]].copy()

    # ✅ アップロードCSVは即クリア（破棄）
    st.session_state["uploader_a"] = None

    # ローカル変数も参照解除（念のため）
    del df_csv
