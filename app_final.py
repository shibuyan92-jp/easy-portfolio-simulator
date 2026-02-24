import re
from datetime import date

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco


# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(page_title="株式分散シミュレーター（日本株・比較/計算例）", layout="wide")
st.title("🔰 株式分散シミュレーター（日本株・比較/計算例）")
st.markdown("日本株（4桁コード）の **株数入力**から時価比率を算出し、**比較**や**計算例（最適化）**を表示します。")


# -----------------------------
# 免責・プライバシー
# -----------------------------
with st.expander("⚠️ 免責（重要）"):
    st.markdown("""
- 本アプリは情報提供を目的としたものであり、特定の金融商品の購入・売却・保有を推奨・勧誘するものではありません。  
- 表示結果は過去データに基づくシミュレーション（計算例）であり、将来の成果を保証しません。  
- 最終的な投資判断はご自身の責任で行ってください。  
""")

with st.expander("🔒 プライバシー / データの取扱い（重要）"):
    st.markdown("""
- **アップロードされたCSVは保存しません。** 取り込み後はアップローダーをクリアし、必要最小限（銘柄コード・株数）だけを画面に保持します。  
- 口座番号・氏名など不要な情報が含まれるCSVはアップロードしないでください。  
""")
    st.caption("※StreamlitのアップロードはRAM上の一時領域で扱われます。")


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


def clean_shares_series(s: pd.Series) -> pd.Series:
    """株数列：カンマ・文字混在を吸収して数値化"""
    s = s.astype(str).str.replace(",", "", regex=False)
    s = s.str.replace("株", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


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
    return price_df.ffill().iloc[-1]


# -----------------------------
# ポートフォリオ指標
# -----------------------------
def portfolio_metrics(mean, cov, w_risky, rf, cash_w=0.0):
    r_risky = float(np.sum(mean * w_risky) * 252)
    r_total = r_risky + cash_w * rf
    vol = float(np.sqrt(np.dot(w_risky.T, np.dot(cov, w_risky))) * np.sqrt(252))
    sharpe = (r_total - rf) / vol if vol != 0 else np.nan
    return r_total, vol, sharpe


# -----------------------------
# 最適化（目的：Sharpe最大 / 分散最小）
# -----------------------------
def optimize_max_sharpe(mean, cov, bounds, rf):
    n = len(mean)

    def neg_sharpe(w):
        r = np.sum(mean * w) * 252
        s = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)
        if s == 0:
            return 1e9
        return -((r - rf) / s)

    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
    bnds = tuple(bounds)
    x0 = np.array([1.0 / n] * n)
    return sco.minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bnds, constraints=cons)


def optimize_min_variance(cov, bounds):
    n = cov.shape[0]

    def var(w):
        return float(np.dot(w.T, np.dot(cov, w)))  # 日次分散（年率化は不要、相対比較で同じ）

    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
    bnds = tuple(bounds)
    x0 = np.array([1.0 / n] * n)
    return sco.minimize(var, x0=x0, method="SLSQP", bounds=bnds, constraints=cons)


def compute_weights_by_objective(objective: str, tickers: list, mean: np.ndarray, cov: np.ndarray,
                                 rf: float, bounds_unit: list, cash_w: float):
    """
    objective: 'Equal' / 'MinVol' / 'MaxSharpe'
    bounds_unit: sum(w)=1 の単位ウェイト用 bounds
    cash_w: 現金比率（0-1）
    return: w_risky（全体に対する比率、合計=1-cash_w）, status_message
    """
    n = len(tickers)
    risky_budget = 1.0 - cash_w
    if risky_budget <= 0:
        return None, "CASH_100"

    if n == 1:
        # 1銘柄なら自明
        w_unit = np.array([1.0])
        return w_unit * risky_budget, "OK"

    if objective == "Equal（均等配分）":
        w_unit = np.array([1.0 / n] * n)
        # boundsに収まるか軽くチェック
        for w, (lo, hi) in zip(w_unit, bounds_unit):
            if w < lo - 1e-9 or w > hi + 1e-9:
                # 収まらない場合は最適化へフォールバック（分散最小）
                res = optimize_min_variance(cov, bounds_unit)
                if not res.success:
                    return None, "OPT_FAIL"
                w_unit = res.x
                break
        return w_unit * risky_budget, "OK"

    if objective == "MinVol（リスク最小）":
        res = optimize_min_variance(cov, bounds_unit)
        if not res.success:
            return None, "OPT_FAIL"
        return res.x * risky_budget, "OK"

    if objective == "MaxSharpe（Sharpe最大）":
        res = optimize_max_sharpe(mean, cov, bounds_unit, rf)
        if not res.success:
            return None, "OPT_FAIL"
        return res.x * risky_budget, "OK"

    return None, "UNKNOWN"


# -----------------------------
# 初期状態
# -----------------------------
def default_holdings_df():
    return pd.DataFrame({"ティッカー": ["8802", "7203", "6758", "8306", "9984"], "株数": [10, 10, 10, 10, 10]})


st.session_state.setdefault("holdings_a", default_holdings_df())
st.session_state.setdefault("holdings_b", default_holdings_df())


# -----------------------------
# サイドバー：共通設定（終了日は今日）
# -----------------------------
st.sidebar.header("🛠️ 設定")
st.session_state.setdefault("start_date", pd.to_datetime("2020-01-01").date())
st.session_state.setdefault("end_date", date.today())

start_date = st.sidebar.date_input("開始日", key="start_date")
end_date = st.sidebar.date_input("終了日（初期値：今日）", key="end_date")
risk_free_rate = st.sidebar.number_input("安全資産の利回り (%)", value=1.0, step=0.1) / 100.0

cash_a = st.sidebar.number_input("A: 現金（任意）", value=0.0, step=10000.0)
cash_b = st.sidebar.number_input("B: 現金（任意）", value=0.0, step=10000.0)

# -----------------------------
# モード選択
# -----------------------------
st.sidebar.header("🎯 目的（モード）")
mode = st.sidebar.radio(
    "何をしたい？",
    [
        "① 比較：A vs 別案B（株数で定義）",
        "② 計算例：B銘柄集合の配分（目的を選択）",
        "③ 計算例：Aを改善する配分（目的を選択・Aベース）",
    ],
)

# 目的（②③のみ）
objective = None
if mode.startswith("②") or mode.startswith("③"):
    st.sidebar.subheader("最適化の目的（ユーザーが選択）")
    objective = st.sidebar.radio(
        "目的",
        ["Equal（均等配分）", "MinVol（リスク最小）", "MaxSharpe（Sharpe最大）"],
        help="“計算例”として、過去データ上での配分を目的関数で計算します。"
    )  # st.radio仕様 [3](https://docs.streamlit.io/develop/api-reference/widgets/st.radio)

st.sidebar.markdown("---")
st.sidebar.subheader("共通の制約（②③）")
min_w = st.sidebar.slider("最小比率（各銘柄）%", 0, 20, 0, 1) / 100.0
max_w = st.sidebar.slider("最大比率（各銘柄）%", 20, 100, 40, 5) / 100.0

# ③のみ：Aからの変更幅
delta_w = 0.0
if mode.startswith("③"):
    st.sidebar.subheader("A改善の“変更の大きさ”")
    delta_w = st.sidebar.slider("各銘柄の比率変更の上限（±%）", 0, 50, 10, 1) / 100.0
    st.sidebar.caption("※売買指示ではなく、過去データ上の“計算例”としての配分変化を示します。")

# -----------------------------
# 入力（CSV / テーブル）
# -----------------------------
st.markdown("## ① 入力（株数 / 日本株のみ）")

# CSVアップロード（Aのみ、保存しない）
st.markdown("### 🅰 A：口座CSVアップロード（任意・保存しません）")
uploaded_a = st.file_uploader("Aの保有一覧CSV（任意）", type=["csv"], key="uploader_a")

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("🧹 Aをリセット"):
        st.session_state["holdings_a"] = default_holdings_df()
        st.session_state["uploader_a"] = None
        st.rerun()
with c2:
    st.caption("※アップロードCSVは保存しません。取り込み後にクリアします。")

if uploaded_a is not None:
    if getattr(uploaded_a, "size", 0) > 1 * 1024 * 1024:
        st.error("⚠️ CSVが大きすぎます（1MBまで）。")
        st.stop()

    df_csv = pd.read_csv(uploaded_a)
    cols = list(df_csv.columns)
    col_t = st.selectbox("ティッカー列（銘柄コード）", cols, key="csv_col_t")
    col_s = st.selectbox("株数列（数量）", cols, key="csv_col_s")

    df_import = df_csv[[col_t, col_s]].copy()
    df_import.columns = ["ティッカー", "株数"]
    df_import["ティッカー"] = df_import["ティッカー"].map(normalize_ticker_jp)

    if (df_import["ティッカー"] == "INVALID").any():
        st.error("⚠️ 日本株のみ対応です（4桁/4桁.T）。")
        st.stop()

    df_import["株数"] = clean_shares_series(df_import["株数"]).fillna(0.0)
    df_import = df_import[(df_import["ティッカー"] != "") & (df_import["株数"] > 0)].reset_index(drop=True)

    st.session_state["holdings_a"] = df_import[["ティッカー", "株数"]].copy()
    st.session_state["uploader_a"] = None
    st.success("✅ CSVをAに取り込みました（アップロードはクリア済み）")
    st.rerun()

# テーブル入力
colA, colB = st.columns(2)
with colA:
    st.markdown("### 🅰 A：現状（株数）")
    edit_a = st.data_editor(
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
        tmp = edit_a.copy()
        tmp["ティッカー"] = tmp["ティッカー"].map(normalize_ticker_jp)
        if (tmp["ティッカー"] == "INVALID").any():
            st.error("⚠️ A：ティッカーは日本株（4桁/4桁.T）のみです。")
        else:
            st.session_state["holdings_a"] = tmp
            st.rerun()

with colB:
    st.markdown("### 🅱 B：入力（株数 or 銘柄リスト）")
    st.caption("②③モードでは、Bは“銘柄集合”として使われることがあります。")
    edit_b = st.data_editor(
        st.session_state["holdings_b"],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "ティッカー": st.column_config.TextColumn("ティッカー（4桁 or 4桁.T）"),
            "株数": st.column_config.NumberColumn("株数（①で使用）", min_value=0, step=1, format="%.0f"),
        },
        key="editor_b",
    )
    if st.button("Bに反映"):
        tmp = edit_b.copy()
        tmp["ティッカー"] = tmp["ティッカー"].map(normalize_ticker_jp)
        if (tmp["ティッカー"] == "INVALID").any():
            st.error("⚠️ B：ティッカーは日本株（4桁/4桁.T）のみです。")
        else:
            st.session_state["holdings_b"] = tmp
            st.rerun()


# -----------------------------
# 実行
# -----------------------------
st.markdown("## ② 実行")

explain = {
    "① 比較：A vs 別案B（株数で定義）":
        "AとBを**別ポートフォリオ**として作り、指標（リターン/リスク/Sharpe）や配分差分を比較します（Aの株数はBに引き継がれません）。",
    "② 計算例：B銘柄集合の配分（目的を選択）":
        "Bに入力した銘柄集合を使い、株数は**無視**して、目的（均等/リスク最小/Sharpe最大）に応じた比率を**計算例**として表示します。",
    "③ 計算例：Aを改善する配分（目的を選択・Aベース）":
        "Aをベースに、（A銘柄のみ／A＋B銘柄）を候補集合として、目的に応じた比率を**計算例**として表示し、Aとの差分を示します（売買指示はしません）。",
}
st.info(explain[mode])

agree = True
if mode.startswith("③"):
    agree = st.checkbox("私は、これは投資助言ではなく“過去データ上の計算例”であることを理解し、最終判断は自分で行います。", value=False)

run = st.button("🔍 計算する")

if run:
    if start_date >= end_date:
        st.error("⚠️ 日付の範囲が不正です。")
        st.stop()
    if min_w > max_w:
        st.error("⚠️ 最小比率が最大比率を上回っています。")
        st.stop()
    if mode.startswith("③") and not agree:
        st.error("⚠️ ③モードは同意チェックが必要です。")
        st.stop()

    A = st.session_state["holdings_a"].copy()
    B = st.session_state["holdings_b"].copy()

    A["ティッカー"] = A["ティッカー"].map(normalize_ticker_jp)
    B["ティッカー"] = B["ティッカー"].map(normalize_ticker_jp)

    if (A["ティッカー"] == "INVALID").any() or (B["ティッカー"] == "INVALID").any():
        st.error("⚠️ 日本株のみ対応（4桁/4桁.T）。")
        st.stop()

    A["株数"] = pd.to_numeric(A["株数"], errors="coerce").fillna(0)
    B["株数"] = pd.to_numeric(B["株数"], errors="coerce").fillna(0)

    A = A[(A["ティッカー"] != "") & (A["株数"] > 0)]
    B_any = B[(B["ティッカー"] != "")]  # ②③は株数ゼロでも銘柄集合として可

    if A.empty:
        st.error("⚠️ Aに有効な行がありません。")
        st.stop()

    # 対象銘柄集合の決定
    if mode.startswith("①"):
        B1 = B[(B["ティッカー"] != "") & (B["株数"] > 0)]
        if B1.empty:
            st.error("⚠️ ①はBにも株数>0の行が必要です。")
            st.stop()
        tickers_all = list(dict.fromkeys(A["ティッカー"].tolist() + B1["ティッカー"].tolist()))
        B_for_weights = B1.copy()

    elif mode.startswith("②"):
        if B_any.empty:
            st.error("⚠️ ②はBに銘柄を1つ以上入力してください（株数は不要）。")
            st.stop()
        tickers_all = list(dict.fromkeys(B_any["ティッカー"].tolist()))
        B_for_weights = B_any.copy()

    else:  # ③
        use_universe = st.radio(
            "③の候補銘柄セット",
            ["Aのみ", "A + B（Bに入力した銘柄も候補にする）"],
            horizontal=True
        )
        if use_universe == "Aのみ":
            tickers_all = list(dict.fromkeys(A["ティッカー"].tolist()))
        else:
            tickers_all = list(dict.fromkeys(A["ティッカー"].tolist() + B_any["ティッカー"].tolist()))
        B_for_weights = pd.DataFrame({"ティッカー": tickers_all, "株数": 0})

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    with st.spinner("価格データ取得中..."):
        prices = get_prices(tickers_all, start_ts, end_ts)
        name_map = get_company_names(tickers_all)

    if prices is None or prices.empty:
        st.error("❌ 価格データを取得できませんでした。")
        st.stop()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all").select_dtypes(include=[np.number])
    if prices.empty:
        st.error("❌ 有効な価格データが不足しています。")
        st.stop()

    last_px = latest_prices_asof(prices)
    used_date = prices.index[-1]

    # リターン統計（対象集合）
    log_ret = np.log(prices / prices.shift(1)).dropna()
    lr_all = log_ret[tickers_all].dropna(how="any")
    mean_all = lr_all.mean().values
    cov_all = lr_all.cov().values

    # Aの時価と比率（現金含む）
    A2 = A.copy()
    A2["価格(直近)"] = A2["ティッカー"].map(lambda t: float(last_px.get(t, np.nan)))
    A2 = A2.dropna(subset=["価格(直近)"])
    A2["時価"] = A2["株数"].astype(float) * A2["価格(直近)"].astype(float)

    total_a_risky = float(A2["時価"].sum())
    total_a = total_a_risky + float(cash_a)
    cash_w_a = float(cash_a) / total_a if total_a > 0 else 0.0

    # ①：A/Bを株数から比較
    if mode.startswith("①"):
        B1 = B_for_weights.copy()
        B1["価格(直近)"] = B1["ティッカー"].map(lambda t: float(last_px.get(t, np.nan)))
        B1 = B1.dropna(subset=["価格(直近)"])
        B1["時価"] = B1["株数"].astype(float) * B1["価格(直近)"].astype(float)

        total_b_risky = float(B1["時価"].sum())
        total_b = total_b_risky + float(cash_b)
        cash_w_b = float(cash_b) / total_b if total_b > 0 else 0.0

        # A指標
        tick_a = A2["ティッカー"].tolist()
        lr_a = log_ret[tick_a].dropna(how="any")
        mean_a, cov_a = lr_a.mean().values, lr_a.cov().values
        w_a = (A2["時価"].values / total_a).astype(float)

        # B指標
        tick_b = B1["ティッカー"].tolist()
        lr_b = log_ret[tick_b].dropna(how="any")
        mean_b, cov_b = lr_b.mean().values, lr_b.cov().values
        w_b = (B1["時価"].values / total_b).astype(float)

        ret_a, vol_a, sh_a = portfolio_metrics(mean_a, cov_a, w_a, risk_free_rate, cash_w=cash_w_a)
        ret_b, vol_b, sh_b = portfolio_metrics(mean_b, cov_b, w_b, risk_free_rate, cash_w=cash_w_b)

        st.success("✅ ① 比較結果（計算例）")
        cL, cR = st.columns(2)
        with cL:
            st.subheader("A（現状）")
            st.metric("期待リターン（年率）", f"{ret_a:.2%}")
            st.metric("リスク（年率）", f"{vol_a:.2%}")
            st.metric("Sharpe", f"{sh_a:.2f}")
        with cR:
            st.subheader("B（別案）")
            st.metric("期待リターン（年率）", f"{ret_b:.2%}")
            st.metric("リスク（年率）", f"{vol_b:.2%}")
            st.metric("Sharpe", f"{sh_b:.2f}")

        st.info(f"直近価格は {used_date.date()} の終値（Adj Close優先）を使用。")
        st.stop()

    # ②・③：目的選択に基づく“計算例”配分
    tick = tickers_all
    n = len(tick)

    # 現金比率の扱い：②は分かりにくさ回避で0%固定、③はA現金比率を維持
    cash_w = 0.0 if mode.startswith("②") else cash_w_a

    # bounds（単位ウェイト sum=1 用）
    if mode.startswith("②"):
        bounds_unit = [(min_w, max_w) for _ in range(n)]
    else:
        # ③：A現状比率を基準に±delta_w、かつ min/max
        w0_map = {t: 0.0 for t in tick}
        for t0, w0 in zip(A2["ティッカー"].tolist(), (A2["時価"].values / total_a).astype(float)):
            w0_map[t0] = float(w0)

        bounds_unit = []
        for t0 in tick:
            base = w0_map.get(t0, 0.0)
            lo = max(0.0, base - delta_w)
            hi = min(1.0, base + delta_w)
            lo = max(lo, min_w)
            hi = min(hi, max_w)
            # 新規追加候補（base=0）は lo を 0 にして、必ず入れる強制を避ける
            if base == 0.0:
                lo = 0.0
            if lo > hi:
                lo, hi = 0.0, max_w
            bounds_unit.append((lo, hi))

    w_risky, status = compute_weights_by_objective(
        objective=objective,
        tickers=tick,
        mean=mean_all,
        cov=cov_all,
        rf=risk_free_rate,
        bounds_unit=bounds_unit,
        cash_w=cash_w,
    )

    if status == "CASH_100":
        st.error("⚠️ 現金比率が100%になっています。")
        st.stop()
    if status != "OK" or w_risky is None:
        st.error("⚠️ 計算に失敗しました。制約（最小/最大比率、変更幅）を緩めてください。")
        st.stop()

    ret, vol, sh = portfolio_metrics(mean_all, cov_all, w_risky, risk_free_rate, cash_w=cash_w)

    title = "✅ ② 計算例：B銘柄集合の配分" if mode.startswith("②") else "✅ ③ 計算例：A改善シミュレーション"
    st.success(title)
    st.caption("※これは投資助言ではなく、過去データに基づく“計算例”です。")
    st.info(f"目的：**{objective}** / 直近価格：{used_date.date()}（Adj Close優先）")

    cM1, cM2, cM3 = st.columns(3)
    cM1.metric("期待リターン（年率）", f"{ret:.2%}")
    cM2.metric("リスク（年率）", f"{vol:.2%}")
    cM3.metric("Sharpe", f"{sh:.2f}")

    df_out = pd.DataFrame(
        {"ティッカー": tick, "社名": [name_map.get(t0, t0) for t0 in tick], "比率(%)": w_risky * 100}
    ).sort_values("比率(%)", ascending=False)

    st.markdown("### 計算された比率（上位）")
    st.dataframe(
        df_out,
        use_container_width=True,
        hide_index=True,
        column_config={"比率(%)": st.column_config.ProgressColumn("比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%")},
    )

    # ③はAとの差分表示（売買指示にはしない）
    if mode.startswith("③"):
        df_a_w = pd.DataFrame({"ティッカー": A2["ティッカー"].tolist(), "A比率(%)": (A2["時価"].values / total_a) * 100})
        df_b_w = df_out[["ティッカー", "比率(%)"]].rename(columns={"比率(%)": "計算比率(%)"})
        merged = pd.merge(df_a_w, df_b_w, on="ティッカー", how="outer").fillna(0.0)
        merged["差分(計算-A)(%)"] = merged["計算比率(%)"] - merged["A比率(%)"]
        merged["社名"] = merged["ティッカー"].map(lambda t0: name_map.get(t0, t0))
        merged = merged[["ティッカー", "社名", "A比率(%)", "計算比率(%)", "差分(計算-A)(%)"]]

        st.markdown("### Aとの差分（どこが増減する“計算例”か）")
        st.dataframe(
            merged,
            use_container_width=True,
            hide_index=True,
            column_config={
                "A比率(%)": st.column_config.ProgressColumn("A比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%"),
                "計算比率(%)": st.column_config.ProgressColumn("計算比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%"),
            },
        )

st.markdown("---")
st.caption("このアプリは投資助言ではありません。表示される結果は将来を保証しません。")
