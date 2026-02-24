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
st.set_page_config(page_title="かんたん株式分散シミュレーター", layout="wide")
st.title("🔰 かんたん株式分散シミュレーター")
st.markdown("専門知識がなくても使える、**過去データに基づく資産配分シミュレーター**です。")

# -----------------------------
# 免責（社外公開向け：常時表示 + 詳細）
# -----------------------------
with st.expander("⚠️ ご利用にあたっての重要な注意（必ずお読みください）"):
    st.markdown("""
本アプリは情報提供を目的としたものであり、特定の金融商品の購入・売却・保有を推奨・勧誘するものではありません。  
表示される結果は、過去の市場データに基づくシミュレーションであり、将来の運用成果を保証するものではありません。  
本アプリの利用によって生じたいかなる損失についても、開発者および提供者は一切の責任を負いません。  
投資に関する最終判断は、必ずご自身の責任で行ってください。
""")

# -----------------------------
# サイドバー設定
# -----------------------------
st.sidebar.header("🛠️ 設定パネル")
st.sidebar.info("💡 ヒント: マウスを項目の上に乗せると、詳しい説明が表示されます。")

default_tickers = "8802.T, 7203.T, 6758.T, 8306.T, 9984.T"
tickers_input = st.sidebar.text_area(
    "銘柄コード (カンマ区切り)",
    value=default_tickers,
    height=80,
    help="例: 8802.T, 7203.T"
)

# -----------------------------
# 日付入力（終了日：今日ボタン付き）
# Streamlitの仕様：key付きwidgetを描画した後に同じkeyのsession_stateを直接書き換えると例外になる
# → on_clickコールバックで更新するのが安全（定石）[1](https://outlook.office365.com/owa/?ItemID=AAMkADcwNDQ2NzllLWRlNmEtNDVmNS05ZjkyLTBmMDVjNjhkOTRiZgBGAAAAAACXXXZLaS%2bsQZcKwnVSJtOmBwBZ3ojfmu7lR51e5bpgUtRZAAAAAAEMAABZ3ojfmu7lR51e5bpgUtRZAAGktVTCAAA%3d&exvsurl=1&viewmodel=ReadMessageItem)
# -----------------------------
st.session_state.setdefault("start_date", pd.to_datetime("2020-01-01").date())
st.session_state.setdefault("end_date", pd.to_datetime("2024-12-31").date())

def set_end_today():
    st.session_state["end_date"] = date.today()

start_date = st.sidebar.date_input("開始日", key="start_date")

col_end, col_today = st.sidebar.columns([3, 1])
with col_end:
    end_date = st.date_input("終了日", key="end_date")
with col_today:
    st.write("")
    st.button("今日", on_click=set_end_today)

st.sidebar.subheader("自分のルール")
min_weight = st.sidebar.slider(
    "最低これくらいは持ちたい (%)",
    0, 20, 5, 1,
    help="分散効果を高めるため5%程度がおすすめ"
) / 100.0

max_weight = st.sidebar.slider(
    "最大ここまでにしておく (%)",
    20, 100, 40, 5,
    help="1銘柄への集中を防ぐ上限"
) / 100.0

risk_free_rate = st.sidebar.number_input(
    "安全資産の利回り (%)",
    value=1.0,
    step=0.1,
    help="国債などの金利"
) / 100.0

# -----------------------------
# 関数群
# -----------------------------
@st.cache_data(show_spinner=False)
def get_data(tickers, start, end):
    try:
        df = yf.download(tickers, start=start, end=end, progress=False)
        if df is None or df.empty:
            return None

        # yfinanceは複数銘柄だとMultiIndexになりやすい
        if isinstance(df.columns, pd.MultiIndex):
            if "Adj Close" in df.columns.get_level_values(0):
                return df.xs("Adj Close", axis=1, level=0)
            if "Close" in df.columns.get_level_values(0):
                return df.xs("Close", axis=1, level=0)

        # 単一銘柄の場合
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

# -----------------------------
# メイン処理
# -----------------------------
if st.button("📊 シミュレーション実行（過去データ）"):
    ts = [t.strip() for t in tickers_input.split(",") if t.strip()]

    if len(ts) < 2:
        st.error("⚠️ 2銘柄以上入れてください")
    elif start_date >= end_date:
        st.error("⚠️ 日付の範囲が不正です（開始日 < 終了日）")
    elif min_weight > max_weight:
        st.error("⚠️ 最小比率が最大比率を上回っています")
    else:
        # yfinanceに渡す型を堅くする（date -> Timestamp）
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        with st.spinner("データを分析中..."):
            df = get_data(ts, start_ts, end_ts)
            name_map = get_company_names(ts)

        if df is None or df.empty:
            st.error("❌ データ取得失敗（銘柄コード・日付範囲を見直してください）")
        else:
            df = df.dropna().select_dtypes(include=[np.number])
            if df.shape[1] < 2:
                st.error("⚠️ 有効なデータ不足（銘柄や期間を変えてください）")
            else:
                try:
                    log_ret = np.log(df / df.shift(1)).dropna()
                    mean = log_ret.mean()
                    cov = log_ret.cov()
                    n = len(df.columns)

                    def neg_sharpe(w):
                        r = np.sum(mean * w) * 252
                        s = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)
                        if s == 0:
                            return 1e9
                        return -((r - risk_free_rate) / s)

                    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
                    bnds = tuple((min_weight, max_weight) for _ in range(n))

                    res = sco.minimize(
                        neg_sharpe,
                        x0=np.array([1.0 / n] * n),
                        method="SLSQP",
                        bounds=bnds,
                        constraints=cons
                    )

                    if not res.success:
                        st.warning("⚠️ 最適化に失敗しました。条件（最小/最大比率）を緩めてください。")
                    else:
                        w = res.x
                        ret = np.sum(mean * w) * 252
                        std = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)
                        sharpe = (ret - risk_free_rate) / std if std != 0 else np.nan

                        st.success("✅ 計算完了！")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("💰 期待リターン（年率）", f"{ret:.2%}")
                        c2.metric("🛡️ リスク（年率）", f"{std:.2%}")
                        c3.metric("📊 投資効率（Sharpe）", f"{sharpe:.2f}" if np.isfinite(sharpe) else "—")

                        # 表現は控えめに（社外公開向け）
                        if np.isfinite(sharpe):
                            if sharpe >= 1.0:
                                st.info("参考：過去データ上では効率が高めの構成です。")
                            elif sharpe >= 0.7:
                                st.success("参考：過去データ上ではバランスが良い構成です。")
                            else:
                                st.warning("参考：過去データ上では効率が低めの可能性があります。")

                        valid_tickers = df.columns
                        labels = [f"{name_map.get(t, t)}\n({t})" for t in valid_tickers]

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            fig, ax = plt.subplots()
                            ax.pie(w, labels=labels, autopct="%1.1f%%", startangle=90)
                            ax.axis("equal")
                            st.pyplot(fig)

                        with col2:
                            df_res = pd.DataFrame({
                                "コード": valid_tickers,
                                "社名": [name_map.get(t, t) for t in valid_tickers],
                                "推奨比率": [f"{v:.2%}" for v in w],
                            })
                            st.dataframe(df_res, use_container_width=True)

                except Exception as e:
                    st.error(f"エラー: {e}")

# -----------------------------
# 免責（短文：フッター常時表示）
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ 本アプリは投資助言を目的としたものではありません。"
    "表示される結果は将来の成果を保証するものではなく、"
    "最終的な投資判断はご自身の責任で行ってください。"
)
