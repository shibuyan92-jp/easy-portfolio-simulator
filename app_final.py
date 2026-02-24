import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
from datetime import date

# -----------------------------
# ページ設定・タイトル
# -----------------------------
st.set_page_config(page_title="株式ポートフォリオ比較シミュレーター", layout="wide")
st.title("🔰 株式ポートフォリオ比較シミュレーター")
st.markdown("**株数（数量）** を入れるだけで、時価比率に自動変換して **A（現状ポートフォリオ） vs B（検討ポートフォリオ）** で比較します。")

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
# ユーティリティ
# -----------------------------
def normalize_ticker(t: str) -> str:
    """数字だけなら .T を付ける（日本株想定）。それ以外はそのまま。"""
    if t is None:
        return ""
    t = str(t).strip().replace("　", "")
    if t == "":
        return ""
    # 例: 7203 → 7203.T
    if t.isdigit():
        return f"{t}.T"
    return t

def shorten(text: str, max_len: int = 22) -> str:
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= max_len else s[:max_len - 1] + "…"

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
    # price_df: DateIndex × tickers
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

# -----------------------------
# サイドバー：共通設定
# -----------------------------
st.sidebar.header("🛠️ 設定パネル")

st.session_state.setdefault("start_date", pd.to_datetime("2020-01-01").date())
st.session_state.setdefault("end_date", pd.to_datetime("2024-12-31").date())

def set_end_today():
    st.session_state["end_date"] = date.today()

start_date = st.sidebar.date_input("開始日", key="start_date")
c_end, c_today = st.sidebar.columns([3, 1])
with c_end:
    end_date = st.date_input("終了日", key="end_date")
with c_today:
    st.write("")
    st.button("今日", on_click=set_end_today)

risk_free_rate = st.sidebar.number_input("安全資産の利回り (%)", value=1.0, step=0.1, help="国債などの金利") / 100.0

st.sidebar.subheader("B案を最適化する場合の制約")
min_weight = st.sidebar.slider("最低これくらいは持ちたい (%)", 0, 20, 5, 1) / 100.0
max_weight = st.sidebar.slider("最大ここまでにしておく (%)", 20, 100, 40, 5) / 100.0

# -----------------------------
# 入力：A/Bの株数（data_editor）
# -----------------------------
def default_holdings_df():
    return pd.DataFrame(
        {
            "ティッカー": ["8802.T", "7203.T", "6758.T", "8306.T", "9984.T"],
            "株数": [10, 10, 10, 10, 10],
        }
    )

st.session_state.setdefault("holdings_a", default_holdings_df())
st.session_state.setdefault("holdings_b", default_holdings_df())

st.sidebar.subheader("A/B入力方式")
b_mode = st.sidebar.radio(
    "Bの作り方",
    ["🧮 株数で入力（Bも現状と同様に比較）", "🤖 Sharpe最大化でB配分を自動提案"],
)

st.sidebar.subheader("Aの現金（任意）")
cash_a = st.sidebar.number_input("A: 現金（無リスク資産）", value=0.0, step=10000.0, help="現金や預り金など（通貨は銘柄に合わせてください）")

st.sidebar.subheader("Bの現金（任意）")
cash_b = st.sidebar.number_input("B: 現金（無リスク資産）", value=0.0, step=10000.0, help="現金や預り金など（通貨は銘柄に合わせてください）")

st.markdown("## ① 株数入力（コピペOK）")

col_in1, col_in2 = st.columns(2)

with col_in1:
    st.markdown("### 🅰 A：現状ポートフォリオ（株数）")
    tmp_a = st.data_editor(
        st.session_state["holdings_a"],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "ティッカー": st.column_config.TextColumn("ティッカー", help="例: 7203 / 7203.T / AAPL"),
            "株数": st.column_config.NumberColumn("株数", min_value=0, step=1, format="%.0f"),
        },
        key="editor_a",
    )
    if st.button("Aに反映"):
        st.session_state["holdings_a"] = tmp_a
        st.rerun()

with col_in2:
    st.markdown("### 🅱 B：検討ポートフォリオ（株数 or 最適化）")
    tmp_b = st.data_editor(
        st.session_state["holdings_b"],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "ティッカー": st.column_config.TextColumn("ティッカー", help="例: 7203 / 7203.T / AAPL"),
            "株数": st.column_config.NumberColumn("株数", min_value=0, step=1, format="%.0f"),
        },
        key="editor_b",
    )
    if st.button("Bに反映"):
        st.session_state["holdings_b"] = tmp_b
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

    # 整形
    df_a = st.session_state["holdings_a"].copy()
    df_b = st.session_state["holdings_b"].copy()

    df_a["ティッカー"] = df_a["ティッカー"].map(normalize_ticker)
    df_b["ティッカー"] = df_b["ティッカー"].map(normalize_ticker)

    df_a = df_a[(df_a["ティッカー"] != "") & (df_a["株数"].fillna(0) > 0)]
    df_b = df_b[(df_b["ティッカー"] != "") & (df_b["株数"].fillna(0) > 0)]

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
        st.error("❌ 価格データを取得できませんでした。ティッカーや期間を見直してください。")
        st.stop()

    # DataFrame整形（単一銘柄のときSeriesになるケース吸収）
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all").select_dtypes(include=[np.number])
    if prices.empty:
        st.error("❌ 有効な価格データが不足しています。")
        st.stop()

    # 直近価格（期間内の最終行）
    last_px = latest_prices_asof(prices)  # Series: ticker -> price
    used_date = prices.index[-1]

    # A：時価
    df_a["価格(直近)"] = df_a["ティッカー"].map(lambda t: float(last_px.get(t, np.nan)))
    df_a = df_a.dropna(subset=["価格(直近)"])
    df_a["時価"] = df_a["株数"].astype(float) * df_a["価格(直近)"].astype(float)

    # B：時価（最適化の場合でも銘柄集合に使うので計算しておく）
    df_b["価格(直近)"] = df_b["ティッカー"].map(lambda t: float(last_px.get(t, np.nan)))
    df_b = df_b.dropna(subset=["価格(直近)"])
    df_b["時価"] = df_b["株数"].astype(float) * df_b["価格(直近)"].astype(float)

    if df_a.empty or df_b.empty:
        st.error("⚠️ A/Bの銘柄で価格が取れないものがあります。ティッカー表記を確認してください。")
        st.stop()

    # リターン計算は、AとBでそれぞれの銘柄集合で計算
    log_ret = np.log(prices / prices.shift(1)).dropna()

    # Aのリスク資産集合
    tickers_a = df_a["ティッカー"].tolist()
    lr_a = log_ret[tickers_a].dropna(how="any")
    mean_a, cov_a = lr_a.mean(), lr_a.cov()

    # Bのリスク資産集合
    tickers_b = df_b["ティッカー"].tolist()
    lr_b = log_ret[tickers_b].dropna(how="any")
    mean_b, cov_b = lr_b.mean(), lr_b.cov()

    # Aの比率（現金含む）
    total_a_risky = float(df_a["時価"].sum())
    total_a = total_a_risky + float(cash_a)
    if total_a <= 0:
        st.error("⚠️ Aの総額が0です。")
        st.stop()

    df_a["比率(%)"] = (df_a["時価"] / total_a) * 100.0
    w_a_risky = (df_a["時価"].values / total_a).astype(float)  # risky比率（合計=1-cash_w）
    cash_w_a = float(cash_a) / total_a

    # Bの比率（現金含む or 最適化）
    total_b_risky = float(df_b["時価"].sum())
    total_b = total_b_risky + float(cash_b)
    if total_b <= 0:
        st.error("⚠️ Bの総額が0です。")
        st.stop()

    # Bが最適化の場合：銘柄集合は tickers_b、比率は最適化結果（現金は別枠）
    if b_mode == "🤖 Sharpe最大化でB配分を自動提案":
        # 現金比率を確定した上で、残り(1-cash_w_b)をリスク資産に配分
        cash_w_b = float(cash_b) / total_b
        risky_budget = 1.0 - cash_w_b
        if risky_budget <= 0:
            st.error("⚠️ Bが現金100%になっています。現金を減らすか、株数を入力してください。")
            st.stop()

        res = optimize_sharpe(mean_b, cov_b, min_weight, max_weight, risk_free_rate)
        if not res.success:
            st.error("⚠️ Bの最適化に失敗しました。制約（最小/最大比率）を緩めてください。")
            st.stop()

        w_b_risky_unit = res.x  # 合計1
        w_b_risky = w_b_risky_unit * risky_budget  # 合計 risky_budget

        # 表示用：Bの比率(%)は最適化配分を採用（時価・株数は参考扱い）
        df_b["比率(%)"] = w_b_risky * 100.0
        # 「時価」は参考表示のまま（入力された株数からの時価）
    else:
        # 株数→時価→比率
        df_b["比率(%)"] = (df_b["時価"] / total_b) * 100.0
        w_b_risky = (df_b["時価"].values / total_b).astype(float)
        cash_w_b = float(cash_b) / total_b

    # 指標
    ret_a, vol_a, sharpe_a = portfolio_metrics_from_weights(mean_a.values, cov_a.values, w_a_risky, risk_free_rate, cash_w=cash_w_a)
    ret_b, vol_b, sharpe_b = portfolio_metrics_from_weights(mean_b.values, cov_b.values, w_b_risky, risk_free_rate, cash_w=cash_w_b)

    # -----------------------------
    # 表示（タブ）
    # -----------------------------
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

        # 差分テーブル（銘柄集合の統合）
        df_a_view = df_a[["ティッカー", "時価", "比率(%)"]].copy()
        df_b_view = df_b[["ティッカー", "時価", "比率(%)"]].copy()
        df_a_view = df_a_view.rename(columns={"比率(%)": "A比率(%)"})
        df_b_view = df_b_view.rename(columns={"比率(%)": "B比率(%)"})

        merged = pd.merge(df_a_view[["ティッカー", "A比率(%)"]], df_b_view[["ティッカー", "B比率(%)"]], on="ティッカー", how="outer").fillna(0.0)
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
        st.caption("株数×直近価格で時価評価し、比率を自動計算しています。")

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
                column_config={
                    "比率(%)": st.column_config.ProgressColumn("比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%")
                },
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
                column_config={
                    "比率(%)": st.column_config.ProgressColumn("比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%")
                },
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
        st.write("- 価格はyfinanceのAdj Close（優先）/ Close を使用します。口座の約定・評価額とはズレることがあります。")
        st.write("- 結果は過去データに基づく比較で、将来を保証しません。")

# -----------------------------
# フッター免責
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ 本アプリは投資助言を目的としたものではありません。"
    "表示される結果は将来の成果を保証するものではなく、"
    "最終的な投資判断はご自身の責任で行ってください。"
)
