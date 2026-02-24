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
    del df_import
    del uploaded_a

    st.success("✅ CSVをAに取り込みました（アップロードファイルは破棄しました）")
    st.rerun()

# -----------------------------
# A/Bテーブル
# -----------------------------
col_in1, col_in2 = st.columns(2)

with col_in1:
    st.markdown("### 🅰 A：現状ポートフォリオ（株数）")
    tmp_a = st.data_editor(
        st.session_state["holdings_a"],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "ティッカー": st.column_config.TextColumn("ティッカー（4桁 or 4桁.T）"),
            "株数": st.column_config.NumberColumn("株数", min_value=0, step=1, format="%.0f"),
        },
        key="editor_a",
    )
    if st.button("Aに反映"):
        # 日本株チェック
        tmp_a2 = tmp_a.copy()
        tmp_a2["ティッカー"] = tmp_a2["ティッカー"].map(normalize_ticker_jp)
        if (tmp_a2["ティッカー"] == "INVALID").any():
            st.error("⚠️ 日本株のみ対応です。ティッカーは4桁（または4桁.T）で入力してください。")
        else:
            st.session_state["holdings_a"] = tmp_a2
            st.rerun()

with col_in2:
    st.markdown("### 🅱 B：検討ポートフォリオ（株数 or 最適化）")
    tmp_b = st.data_editor(
        st.session_state["holdings_b"],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "ティッカー": st.column_config.TextColumn("ティッカー（4桁 or 4桁.T）"),
            "株数": st.column_config.NumberColumn("株数", min_value=0, step=1, format="%.0f"),
        },
        key="editor_b",
    )
    if st.button("Bに反映"):
        tmp_b2 = tmp_b.copy()
        tmp_b2["ティッカー"] = tmp_b2["ティッカー"].map(normalize_ticker_jp)
        if (tmp_b2["ティッカー"] == "INVALID").any():
            st.error("⚠️ 日本株のみ対応です。ティッカーは4桁（または4桁.T）で入力してください。")
        else:
            st.session_state["holdings_b"] = tmp_b2
            st.rerun()

run = st.button("🔍 A vs B を比較する（株数→時価比率）")

# -----------------------------
# 実行：株数→時価比率→比較
# -----------------------------
if run:
    if start_date >= end_date:
        st.error("⚠️ 日付の範囲が不正です（開始日 < 終了日）")
        st.stop()
    if min_weight > max_weight:
        st.error("⚠️ 最小比率が最大比率を上回っています")
        st.stop()

    df_a = st.session_state["holdings_a"].copy()
    df_b = st.session_state["holdings_b"].copy()

    df_a["ティッカー"] = df_a["ティッカー"].map(normalize_ticker_jp)
    df_b["ティッカー"] = df_b["ティッカー"].map(normalize_ticker_jp)

    if (df_a["ティッカー"] == "INVALID").any() or (df_b["ティッカー"] == "INVALID").any():
        st.error("⚠️ 日本株のみ対応です。ティッカーは「7203」または「7203.T」の形式で入力してください。")
        st.stop()

    df_a["株数"] = pd.to_numeric(df_a["株数"], errors="coerce").fillna(0)
    df_b["株数"] = pd.to_numeric(df_b["株数"], errors="coerce").fillna(0)

    df_a = df_a[(df_a["ティッカー"] != "") & (df_a["株数"] > 0)]
    df_b = df_b[(df_b["ティッカー"] != "") & (df_b["株数"] > 0)]

    if df_a.empty or df_b.empty:
        st.error("⚠️ A/Bそれぞれ、ティッカーと株数を1行以上入力してください。")
        st.stop()

    tickers_all = list(dict.fromkeys(df_a["ティッカー"].tolist() + df_b["ティッカー"].tolist()))
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    with st.spinner("価格データ取得中..."):
        prices = get_prices(tickers_all, start_ts, end_ts)
        name_map = get_company_names(tickers_all)

    if prices is None or prices.empty:
        st.error("❌ 価格データを取得できませんでした。コードや期間を見直してください。")
        st.stop()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all").select_dtypes(include=[np.number])
    if prices.empty:
        st.error("❌ 有効な価格データが不足しています。")
        st.stop()

    last_px = latest_prices_asof(prices)
    used_date = prices.index[-1]

    df_a["価格(直近)"] = df_a["ティッカー"].map(lambda t: float(last_px.get(t, np.nan)))
    df_a = df_a.dropna(subset=["価格(直近)"])
    df_a["時価"] = df_a["株数"].astype(float) * df_a["価格(直近)"].astype(float)

    df_b["価格(直近)"] = df_b["ティッカー"].map(lambda t: float(last_px.get(t, np.nan)))
    df_b = df_b.dropna(subset=["価格(直近)"])
    df_b["時価"] = df_b["株数"].astype(float) * df_b["価格(直近)"].astype(float)

    if df_a.empty or df_b.empty:
        st.error("⚠️ A/Bの銘柄で価格が取れないものがあります。コードを確認してください。")
        st.stop()

    log_ret = np.log(prices / prices.shift(1)).dropna()

    tickers_a = df_a["ティッカー"].tolist()
    tickers_b = df_b["ティッカー"].tolist()

    lr_a = log_ret[tickers_a].dropna(how="any")
    lr_b = log_ret[tickers_b].dropna(how="any")

    mean_a, cov_a = lr_a.mean().values, lr_a.cov().values
    mean_b, cov_b = lr_b.mean().values, lr_b.cov().values

    total_a_risky = float(df_a["時価"].sum())
    total_a = total_a_risky + float(cash_a)
    if total_a <= 0:
        st.error("⚠️ Aの総額が0です。")
        st.stop()

    df_a["比率(%)"] = (df_a["時価"] / total_a) * 100.0
    w_a_risky = (df_a["時価"].values / total_a).astype(float)
    cash_w_a = float(cash_a) / total_a

    total_b_risky = float(df_b["時価"].sum())
    total_b = total_b_risky + float(cash_b)
    if total_b <= 0:
        st.error("⚠️ Bの総額が0です。")
        st.stop()

    if b_mode == "🤖 Sharpe最大化でB配分を自動提案":
        cash_w_b = float(cash_b) / total_b
        risky_budget = 1.0 - cash_w_b
        if risky_budget <= 0:
            st.error("⚠️ Bが現金100%になっています。現金を減らすか、株数を入力してください。")
            st.stop()

        res = optimize_sharpe(mean_b, cov_b, min_weight, max_weight, risk_free_rate)
        if not res.success:
            st.error("⚠️ Bの最適化に失敗しました。制約（最小/最大比率）を緩めてください。")
            st.stop()

        w_b_risky = res.x * risky_budget
        df_b["比率(%)"] = w_b_risky * 100.0
    else:
        cash_w_b = float(cash_b) / total_b
        df_b["比率(%)"] = (df_b["時価"] / total_b) * 100.0
        w_b_risky = (df_b["時価"].values / total_b).astype(float)

    ret_a, vol_a, sharpe_a = portfolio_metrics_from_weights(mean_a, cov_a, w_a_risky, risk_free_rate, cash_w=cash_w_a)
    ret_b, vol_b, sharpe_b = portfolio_metrics_from_weights(mean_b, cov_b, w_b_risky, risk_free_rate, cash_w=cash_w_b)

    st.success("✅ 比較結果ができました！")

    tab_cmp, tab_a, tab_b, tab_detail = st.tabs(["📌 比較（A vs B）", "🅰 A（現状）", "🅱 B（検討）", "🧾 前提"])

    with tab_cmp:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🅰 A（現状）")
            st.metric("💰 期待リターン（年率）", f"{ret_a:.2%}", delta=f"{(ret_a - ret_b):+.2%}（A-B）")
            st.metric("🛡️ リスク（年率）", f"{vol_a:.2%}", delta=f"{(vol_a - vol_b):+.2%}（A-B）")
            st.metric("📊 Sharpe", f"{sharpe_a:.2f}", delta=f"{(sharpe_a - sharpe_b):+.2f}（A-B）")
        with c2:
            st.subheader("🅱 B（検討）")
            st.metric("💰 期待リターン（年率）", f"{ret_b:.2%}", delta=f"{(ret_b - ret_a):+.2%}（B-A）")
            st.metric("🛡️ リスク（年率）", f"{vol_b:.2%}", delta=f"{(vol_b - vol_a):+.2%}（B-A）")
            st.metric("📊 Sharpe", f"{sharpe_b:.2f}", delta=f"{(sharpe_b - sharpe_a):+.2f}（B-A）")

        st.info(f"直近価格は **{used_date.date()} の終値（Adj Close優先）** を使用して時価比率を算出しています。")

        merged = pd.merge(
            df_a[["ティッカー", "比率(%)"]].rename(columns={"比率(%)": "A比率(%)"}),
            df_b[["ティッカー", "比率(%)"]].rename(columns={"比率(%)": "B比率(%)"}),
            on="ティッカー",
            how="outer",
        ).fillna(0.0)

        merged["社名"] = merged["ティッカー"].map(lambda t: name_map.get(t, t))
        merged["差分(B-A)(%)"] = merged["B比率(%)"] - merged["A比率(%)"]
        merged = merged[["ティッカー", "社名", "A比率(%)", "B比率(%)", "差分(B-A)(%)"]]

        st.markdown("### 配分差分（どこを増やし/減らしたか）")
        st.dataframe(
            merged,
            use_container_width=True,
            hide_index=True,
            column_config={
                "A比率(%)": st.column_config.ProgressColumn("A比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%"),
                "B比率(%)": st.column_config.ProgressColumn("B比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%"),
            },
        )

    with tab_a:
        st.subheader("🅰 A（現状）")
        col1, col2 = st.columns([1, 1])

        with col1:
            labels = [f"{shorten(name_map.get(t, t))}\n({t})" for t in df_a["ティッカー"].tolist()]
            weights = (df_a["比率(%)"].values / 100.0).astype(float)
            if cash_w_a > 0:
                labels = labels + ["Cash（無リスク）"]
                weights = np.append(weights, cash_w_a)

            fig, ax = plt.subplots()
            ax.pie(weights, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        with col2:
            view = df_a.copy()
            view["社名"] = view["ティッカー"].map(lambda t: name_map.get(t, t))
            view = view[["ティッカー", "社名", "株数", "価格(直近)", "時価", "比率(%)"]]
            st.dataframe(
                view,
                use_container_width=True,
                hide_index=True,
                column_config={"比率(%)": st.column_config.ProgressColumn("比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%")},
            )
            if cash_w_a > 0:
                st.write(f"現金（無リスク）比率：{cash_w_a*100:.1f}%")

    with tab_b:
        st.subheader("🅱 B（検討）")
        st.caption(f"作り方：{b_mode}")

        col1, col2 = st.columns([1, 1])

        with col1:
            labels = [f"{shorten(name_map.get(t, t))}\n({t})" for t in df_b["ティッカー"].tolist()]
            weights = (df_b["比率(%)"].values / 100.0).astype(float)
            if cash_w_b > 0:
                labels = labels + ["Cash（無リスク）"]
                weights = np.append(weights, cash_w_b)

            fig, ax = plt.subplots()
            ax.pie(weights, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        with col2:
            view = df_b.copy()
            view["社名"] = view["ティッカー"].map(lambda t: name_map.get(t, t))
            view = view[["ティッカー", "社名", "株数", "価格(直近)", "時価", "比率(%)"]]
            st.dataframe(
                view,
                use_container_width=True,
                hide_index=True,
                column_config={"比率(%)": st.column_config.ProgressColumn("比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%")},
            )
            if cash_w_b > 0:
                st.write(f"現金（無リスク）比率：{cash_w_b*100:.1f}%")

    with tab_detail:
        st.write("**前提（比較条件）**")
        st.write(f"- 期間：{start_date} 〜 {end_date}")
        st.write(f"- 安全資産の利回り：{risk_free_rate:.2%}")
        st.write(f"- 価格評価日：{used_date.date()}（期間内の最終営業日）")
        st.write("")
        st.write("**B最適化の制約（使用した場合）**")
        st.write(f"- 各銘柄 最小 {min_weight:.0%} / 最大 {max_weight:.0%}")
        st.write("")
        st.write("**メモ**")
        st.write("- 価格はyfinanceのAdj Close（優先）/ Close を使用します。口座の評価額とはズレることがあります。")
        st.write("- 結果は過去データに基づく比較で、将来を保証しません。")

# -----------------------------
# フッター
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ 本アプリは投資助言を目的としたものではありません。"
    "表示される結果は将来の成果を保証するものではなく、"
    "最終的な投資判断はご自身の責任で行ってください。"
)
