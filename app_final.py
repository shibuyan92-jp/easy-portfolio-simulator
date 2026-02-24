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
def get_company_names(tickers_list):
    names = {}
    for t in tickers_list:
        try:
            info = yf.Ticker(t).info or {}
            names[t] = info.get("shortName", info.get("longName", t))
        except Exception:
            names[t] = t
    return names

def optimize_sharpe(mean, cov, min_w, max_w, risk_free):
    """Sharpe最大化（制約あり）"""
    n = len(mean)

    def neg_sharpe(w):
        r = np.sum(mean * w) * 252
        s = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)
        if s == 0:
            return 1e9
        return -((r - risk_free) / s)

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
st.sidebar.info("💡 目的：**現状(A)と検討(B)を比較**して、どちらが自分に合うかを議論するためのツールです。")

# 日付（終了日：今日ボタン）
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
# サイドバー：ポートフォリオA/B
# -----------------------------
st.sidebar.subheader("🅰 現状ポートフォリオ（A）")

default_a = "8802.T 7203.T 6758.T 8306.T 9984.T"
tickers_a_text = st.sidebar.text_area(
    "A: 銘柄（スペース/改行/カンマ区切りOK）",
    value=default_a,
    height=80,
    help="例: 7203 6758 8306（数字だけなら自動で .T を付けます）"
)
weights_a_text = st.sidebar.text_area(
    "A: 比率（任意）",
    value="",
    height=60,
    help="空なら均等配分。例: 20 20 20 20 20（%入力OK）"
)

st.sidebar.subheader("🅱 検討ポートフォリオ（B）")
default_b = default_a
tickers_b_text = st.sidebar.text_area(
    "B: 銘柄（スペース/改行/カンマ区切りOK）",
    value=default_b,
    height=80
)

b_mode = st.sidebar.radio(
    "Bの配分の作り方",
    ["入力した比率を使う（または均等）", "Sharpe最大化で自動計算（制約あり）"],
    help="投資仲間と『現状と変更案』を比較したい場合は、Bを最適化すると差が見えやすいです。"
)

weights_b_text = ""
if b_mode == "入力した比率を使う（または均等）":
    weights_b_text = st.sidebar.text_area(
        "B: 比率（任意）",
        value="",
        height=60,
        help="空なら均等配分。例: 10 20 20 30 20"
    )

# 実行ボタン
run = st.button("🔍 A vs B を比較する（過去データ）")


# -----------------------------
# 実行ロジック
# -----------------------------
if run:
    ts_a = parse_tickers(tickers_a_text)
    ts_b = parse_tickers(tickers_b_text)

    if len(ts_a) < 1 or len(ts_b) < 1:
        st.error("⚠️ A/Bそれぞれ1銘柄以上入力してください。")
        st.stop()

    if start_date >= end_date:
        st.error("⚠️ 日付の範囲が不正です（開始日 < 終了日）")
        st.stop()

    # まずは全部まとめてデータ取得（効率化）
    ts_all = list(dict.fromkeys(ts_a + ts_b))
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    with st.spinner("データを取得・分析中..."):
        price_all = get_data(ts_all, start_ts, end_ts)
        name_map = get_company_names(ts_all)

    if price_all is None or price_all.empty:
        st.error("❌ データ取得失敗（銘柄コード・日付範囲を見直してください）")
        st.stop()

    # 数値列だけ & 欠損処理
    price_all = price_all.dropna().select_dtypes(include=[np.number])
    if price_all.shape[1] < 1:
        st.error("⚠️ 有効なデータが不足しています。")
        st.stop()

    # A/Bで有効な銘柄（取得できたものだけ）に絞る
    valid_all = list(price_all.columns)
    valid_a = [t for t in ts_a if t in valid_all]
    valid_b = [t for t in ts_b if t in valid_all]

    if len(valid_a) < 1 or len(valid_b) < 1:
        st.error("⚠️ A/Bの銘柄のうち有効なデータが不足しています。")
        st.stop()

    # 対数リターン
    log_ret_all = np.log(price_all / price_all.shift(1)).dropna()

    # A用統計
    lr_a = log_ret_all[valid_a]
    mean_a, cov_a = lr_a.mean(), lr_a.cov()

    # B用統計
    lr_b = log_ret_all[valid_b]
    mean_b, cov_b = lr_b.mean(), lr_b.cov()

    # Aの重み
    wa = parse_weights(weights_a_text, len(valid_a))
    if wa == "PARSE_ERROR":
        st.error("⚠️ Aの比率が読み取れません。数字をスペース/改行/カンマ区切りで入力してください。")
        st.stop()
    if wa == "LEN_MISMATCH":
        st.error("⚠️ Aの比率の個数が、Aの銘柄数と一致しません。")
        st.stop()
    if wa in ["NEGATIVE", "ZERO"]:
        st.error("⚠️ Aの比率が不正です（負の値/合計0など）。")
        st.stop()
    if wa is None:
        wa = np.array([1.0 / len(valid_a)] * len(valid_a))

    # Bの重み
    if b_mode == "Sharpe最大化で自動計算（制約あり）":
        if min_weight > max_weight:
            st.error("⚠️ 最小比率が最大比率を上回っています。")
            st.stop()
        res = optimize_sharpe(mean_b, cov_b, min_weight, max_weight, risk_free_rate)
        if not res.success:
            st.warning("⚠️ Bの最適化に失敗しました。制約（最小/最大比率）を緩めてください。")
            st.stop()
        wb = res.x
    else:
        wb = parse_weights(weights_b_text, len(valid_b))
        if wb == "PARSE_ERROR":
            st.error("⚠️ Bの比率が読み取れません。数字をスペース/改行/カンマ区切りで入力してください。")
            st.stop()
        if wb == "LEN_MISMATCH":
            st.error("⚠️ Bの比率の個数が、Bの銘柄数と一致しません。")
            st.stop()
        if wb in ["NEGATIVE", "ZERO"]:
            st.error("⚠️ Bの比率が不正です（負の値/合計0など）。")
            st.stop()
        if wb is None:
            wb = np.array([1.0 / len(valid_b)] * len(valid_b))

    # 指標計算
    ret_a, std_a, sharpe_a = portfolio_metrics(mean_a, cov_a, wa, risk_free_rate)
    ret_b, std_b, sharpe_b = portfolio_metrics(mean_b, cov_b, wb, risk_free_rate)

    # -----------------------------
    # 結果表示（タブ）
    # -----------------------------
    st.success("✅ 比較結果の作成が完了しました！")

    tab_cmp, tab_a, tab_b, tab_detail = st.tabs(["📌 比較（A vs B）", "🅰 A（現状）", "🅱 B（検討）", "🧾 前提"])

    # ---- 比較タブ
    with tab_cmp:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("🅰 現状（A）")
            st.metric("💰 期待リターン（年率）", f"{ret_a:.2%}", delta=f"{(ret_a - ret_b):+.2%}（A-B）")
            st.metric("🛡️ リスク（年率）", f"{std_a:.2%}", delta=f"{(std_a - std_b):+.2%}（A-B）")
            st.metric("📊 Sharpe", f"{sharpe_a:.2f}", delta=f"{(sharpe_a - sharpe_b):+.2f}（A-B）")

        with c2:
            st.subheader("🅱 検討（B）")
            st.metric("💰 期待リターン（年率）", f"{ret_b:.2%}", delta=f"{(ret_b - ret_a):+.2%}（B-A）")
            st.metric("🛡️ リスク（年率）", f"{std_b:.2%}", delta=f"{(std_b - std_a):+.2%}（B-A）")
            st.metric("📊 Sharpe", f"{sharpe_b:.2f}", delta=f"{(sharpe_b - sharpe_a):+.2f}（B-A）")

        st.markdown("### 参考コメント（過去データ上の比較）")
        # 断定しない（助言に見えない表現）
        if np.isfinite(sharpe_a) and np.isfinite(sharpe_b):
            if sharpe_b > sharpe_a:
                st.info("Bは、過去データ上では **投資効率（Sharpe）が高い** 傾向です。")
            elif sharpe_b < sharpe_a:
                st.info("Aは、過去データ上では **投資効率（Sharpe）が高い** 傾向です。")
            else:
                st.info("AとBは、過去データ上では **投資効率（Sharpe）が同程度** です。")

        # 配分差分テーブル（A/Bの銘柄集合を統合）
        all_names = list(dict.fromkeys(valid_a + valid_b))
        df_comp = pd.DataFrame({
            "コード": all_names,
            "社名": [name_map.get(t, t) for t in all_names],
            "A比率(%)": [float(wa[valid_a.index(t)] * 100) if t in valid_a else 0.0 for t in all_names],
            "B比率(%)": [float(wb[valid_b.index(t)] * 100) if t in valid_b else 0.0 for t in all_names],
        })
        df_comp["差分(B-A)(%)"] = df_comp["B比率(%)"] - df_comp["A比率(%)"]

        st.markdown("### 配分の差分（どこを増やし/減らしたか）")
        st.dataframe(
            df_comp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "A比率(%)": st.column_config.ProgressColumn("A比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%"),
                "B比率(%)": st.column_config.ProgressColumn("B比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%"),
            }
        )

    # ---- Aタブ
    with tab_a:
        st.subheader("🅰 現状（A）")
        col1, col2 = st.columns([1, 1])

        labels_a = [f"{shorten(name_map.get(t, t))}\n({t})" for t in valid_a]
        with col1:
            fig, ax = plt.subplots()
            ax.pie(wa, labels=labels_a, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        with col2:
            df_a = pd.DataFrame({
                "コード": valid_a,
                "社名": [name_map.get(t, t) for t in valid_a],
                "比率(%)": wa * 100
            })
            st.dataframe(
                df_a,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "比率(%)": st.column_config.ProgressColumn("比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%")
                }
            )

    # ---- Bタブ
    with tab_b:
        st.subheader("🅱 検討（B）")
        st.caption(f"配分の作り方：{b_mode}")

        col1, col2 = st.columns([1, 1])

        labels_b = [f"{shorten(name_map.get(t, t))}\n({t})" for t in valid_b]
        with col1:
            fig, ax = plt.subplots()
            ax.pie(wb, labels=labels_b, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        with col2:
            df_b = pd.DataFrame({
                "コード": valid_b,
                "社名": [name_map.get(t, t) for t in valid_b],
                "比率(%)": wb * 100
            })
            st.dataframe(
                df_b,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "比率(%)": st.column_config.ProgressColumn("比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%")
                }
            )

    # ---- 詳細タブ
    with tab_detail:
        st.write("**前提（比較条件）**")
        st.write(f"- 期間：{start_date} 〜 {end_date}")
        st.write(f"- 安全資産の利回り：{risk_free_rate:.2%}")
        st.write("")
        st.write("**Bを最適化した場合の制約**")
        st.write(f"- 各銘柄 最小 {min_weight:.0%} / 最大 {max_weight:.0%}")
        st.write("")
        st.write("**メモ**")
        st.write("- 対数リターンから年率換算（252営業日換算）で算出しています。")
        st.write("- 結果は過去データに基づくシミュレーションで、将来を保証しません。")


# -----------------------------
# フッター免責
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ 本アプリは投資助言を目的としたものではありません。"
    "表示される結果は将来の成果を保証するものではなく、"
    "最終的な投資判断はご自身の責任で行ってください。"
)
