import re
from datetime import date

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco


# =============================
# ページ設定
# =============================
st.set_page_config(page_title="ポートフォリオ性格診断（日本株・計算例）", layout="wide")
st.title("🧬 ポートフォリオ性格診断（日本株・計算例）")
st.markdown(
    "このツールは「どれが儲かるか」ではなく、どれがどう強いか（性格）を比べて、"
    "投資仲間と楽しく話せるネタを作るためのものです。"
)

with st.expander("⚠️ 免責（重要）"):
    st.markdown("""
- 本アプリは情報提供を目的としたものであり、特定の金融商品の購入・売却・保有を推奨・勧誘するものではありません。  
- 表示結果は過去データに基づくシミュレーション（計算例）であり、将来の成果を保証しません。  
- 最終的な投資判断はご自身の責任で行ってください。  
""")

with st.expander("🔒 プライバシー / データの取扱い（重要）"):
    st.markdown("""
- **アップロードされたCSVは保存しません。取り込み後はアップローダーをクリアし、必要最小限（銘柄コード・株数）だけを画面に保持します。  
- 口座番号・氏名など不要な情報が含まれるCSVはアップロードしないでください。  
""")
    st.caption("※アップロードファイルは一時的に扱い、取り込み後にクリアします。")


# =============================
# 日本株オンリー：ティッカー正規化
# =============================
JP_TICKER_RE = re.compile(r"^\d{4}(\.T)?$")


def normalize_ticker_jp(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip().replace("　", "").upper()
    if t == "":
        return ""
    if not JP_TICKER_RE.match(t):
        return "INVALID"
    return t if t.endswith(".T") else f"{t}.T"


def clean_shares_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False)
    s = s.str.replace("株", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def clamp(x, lo=0.0, hi=1.0):
    return max(min(x, hi), lo)


# =============================
# データ取得
# =============================
@st.cache_data(show_spinner=False)
def get_prices(tickers, start, end):
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


# =============================
# 指標計算（株式100%）
# =============================
def max_drawdown_from_nav(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan


def metrics_from_prices(prices: pd.DataFrame, weights: np.ndarray, rf: float):
    """
    prices: columns=tickers, index=date
    weights: sum=1
    rf: 年率（比較の基準値）
    """
    rets = prices.pct_change().dropna()
    if rets.empty:
        return None

    pr = (rets * weights).sum(axis=1)
    nav = (1 + pr).cumprod()

    n = len(pr)
    cagr = float(nav.iloc[-1] ** (252 / n) - 1) if n > 0 else np.nan
    vol = float(pr.std() * np.sqrt(252))
    ann_ret = float(pr.mean() * 252)
    sharpe = float((ann_ret - rf) / vol) if vol != 0 else np.nan
    mdd = max_drawdown_from_nav(nav)

    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": mdd,
        "ann_ret": ann_ret,
        "nav": nav,
        "pr": pr,
    }


def strength_axes(metrics):
    """
    会話用の5軸（0-1）
    """
    cagr = metrics["cagr"]
    vol = metrics["vol"]
    sharpe = metrics["sharpe"]
    mdd = metrics["max_dd"]

    stability = clamp(1 - vol / 0.30)                   # 30%を荒れ相場基準
    growth = clamp(cagr / 0.15)                         # 15%で上限
    blast = clamp((max(cagr, 0) / max(vol, 1e-9)) / 1.2) # ざっくり「伸び/揺れ」
    mental = clamp(1 + mdd / 0.40)                      # -40%で0
    sustain = clamp(0.5 * stability + 0.5 * mental)

    return {
        "安定性": stability,
        "成長力": growth,
        "爆発力": blast,
        "メンタル耐性": mental,
        "継続しやすさ": sustain,
    }


def stars(x01):
    n = int(round(clamp(x01) * 5))
    return "★" * n + "☆" * (5 - n)


def sharpe_label(sh):
    if np.isnan(sh):
        return "不明"
    if sh >= 1.2:
        return "かなり良い（効率高め）"
    if sh >= 0.7:
        return "良い（バランス型）"
    if sh >= 0.3:
        return "ふつう（波が出やすい）"
    return "荒れやすい（波が大きめ）"


# =============================
# 最適化（②③用）
# =============================
def optimize_max_sharpe(mean, cov, bounds, rf):
    n = len(mean)

    def neg_sharpe(w):
        r = np.sum(mean * w) * 252
        s = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)
        if s == 0:
            return 1e9
        return -((r - rf) / s)

    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
    x0 = np.array([1.0 / n] * n)
    return sco.minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=tuple(bounds), constraints=cons)


def optimize_min_variance(cov, bounds):
    n = cov.shape[0]

    def var(w):
        return float(np.dot(w.T, np.dot(cov, w)))

    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
    x0 = np.array([1.0 / n] * n)
    return sco.minimize(var, x0=x0, method="SLSQP", bounds=tuple(bounds), constraints=cons)


def compute_weights_by_objective(objective: str, mean: np.ndarray, cov: np.ndarray, rf: float, bounds: list):
    n = len(mean)
    if n == 1:
        return np.array([1.0]), "OK"

    if objective == "Equal（均等配分）":
        w = np.array([1.0 / n] * n)
        for wi, (lo, hi) in zip(w, bounds):
            if wi < lo - 1e-9 or wi > hi + 1e-9:
                res = optimize_min_variance(cov, bounds)
                if not res.success:
                    return None, "OPT_FAIL"
                return res.x, "OK"
        return w, "OK"

    if objective == "MinVol（リスク最小）":
        res = optimize_min_variance(cov, bounds)
        if not res.success:
            return None, "OPT_FAIL"
        return res.x, "OK"

    if objective == "MaxSharpe（効率重視）":
        res = optimize_max_sharpe(mean, cov, bounds, rf)
        if not res.success:
            return None, "OPT_FAIL"
        return res.x, "OK"

    return None, "UNKNOWN"


# =============================
# タイプ文言（10種）と判定
# =============================
TYPE_TEXT = {
    "夜ぐっすり安定型": {
        "tagline": "相場が荒れても、わりと普通に寝られる",
        "desc": "値動きが小さめ。下げに強い傾向。",
        "tsukkomi": "派手さはゼロ。自慢はしづらい。",
        "friend": "派手じゃないけど、生き残るタイプ。",
    },
    "ジェットコースター型": {
        "tagline": "楽しいけど、胃にくる",
        "desc": "上も下も大きい。話題性は強い。",
        "tsukkomi": "握力（メンタル）が強さ。",
        "friend": "伸びる時は最高、下げる時は修行。",
    },
    "優等生バランス型": {
        "tagline": "だいたい平均点、だいたい安心",
        "desc": "極端な弱点が少ない。",
        "tsukkomi": "逆に話題になりにくい。",
        "friend": "尖ってない分、長く付き合える。",
    },
    "修行僧メンタル型": {
        "tagline": "結果は悪くない。でも我慢が必要",
        "desc": "途中がしんどいが、耐えると報われがち。",
        "tsukkomi": "途中で売ると一番つらい。",
        "friend": "持ち続けた人が勝つタイプ。",
    },
    "一発ロマン型": {
        "tagline": "当たれば伝説、外れたら思い出",
        "desc": "爆発力に寄せた構成。",
        "tsukkomi": "メインにすると勇者。",
        "friend": "ロマン枠。語れるけど波は大きい。",
    },
    "クール耐久型": {
        "tagline": "地味だけど、生き残る",
        "desc": "下げ耐性寄り。粘り強い。",
        "tsukkomi": "気づいたら一番勝ってるやつ。",
        "friend": "静かに強い。長期で効く。",
    },
    "コツコツ積み上げ型": {
        "tagline": "毎日は地味、数年後にニヤける",
        "desc": "効率と継続を重視。",
        "tsukkomi": "途中でやめた人、だいたい後悔。",
        "friend": "急がないけど、ブレにくい。",
    },
    "感情ジェット型": {
        "tagline": "相場と一緒に気分も上下する",
        "desc": "見てて飽きないが精神コスト高め。",
        "tsukkomi": "通知オフ推奨。",
        "friend": "盛り上がるけど、付き合い方が大事。",
    },
    "我慢力ゴリラ型": {
        "tagline": "握力がすべてを決める",
        "desc": "耐えるほど強さが出るタイプ。",
        "tsukkomi": "途中離脱はもったいない。",
        "friend": "メンタル勝負。続けられる人向け。",
    },
    "玄人好み型": {
        "tagline": "分かる人には分かる",
        "desc": "効率（リターンと波のバランス）が良い傾向。",
        "tsukkomi": "説明しないと伝わらない。",
        "friend": "派手じゃないけど、数字は良い。",
    },
}

TYPE_PROFILES = {
    "夜ぐっすり安定型": {"安定性": 0.5, "継続しやすさ": 0.3, "メンタル耐性": 0.2},
    "ジェットコースター型": {"成長力": 0.6, "爆発力": 0.3, "安定性": -0.2},
    "優等生バランス型": {"安定性": 0.2, "成長力": 0.2, "爆発力": 0.2, "メンタル耐性": 0.2, "継続しやすさ": 0.2},
    "修行僧メンタル型": {"メンタル耐性": 0.55, "継続しやすさ": 0.25, "成長力": 0.2},
    "一発ロマン型": {"爆発力": 0.7, "成長力": 0.4, "安定性": -0.3},
    "クール耐久型": {"安定性": 0.55, "メンタル耐性": 0.45},
    "コツコツ積み上げ型": {"継続しやすさ": 0.4, "効率": 0.35, "安定性": 0.25},
    "感情ジェット型": {"爆発力": 0.5, "成長力": 0.4, "安定性": -0.4},
    "我慢力ゴリラ型": {"メンタル耐性": 0.7, "継続しやすさ": 0.3},
    "玄人好み型": {"効率": 0.7, "安定性": 0.3},
}


def judge_type(metrics):
    axes = strength_axes(metrics)
    # “効率”は会話用：Sharpeを0-1に圧縮
    eff = clamp(metrics["sharpe"] / 1.5) if not np.isnan(metrics["sharpe"]) else 0.0
    feat = dict(axes)
    feat["効率"] = eff

    scores = {}
    for t, wts in TYPE_PROFILES.items():
        s = 0.0
        for k, w in wts.items():
            s += feat.get(k, 0.0) * w
        scores[t] = s

    best = max(scores, key=scores.get)
    # 2位も返す
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    second = sorted_types[1][0] if len(sorted_types) > 1 else None
    return best, second, scores, axes


def radar_plot(ax_scores_a, ax_scores_b=None, label_a="A", label_b="B"):
    labels = list(ax_scores_a.keys())
    vals_a = [ax_scores_a[k] for k in labels]
    # close
    vals_a += vals_a[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.plot(angles, vals_a, linewidth=2, label=label_a)
    ax.fill(angles, vals_a, alpha=0.15)

    if ax_scores_b is not None:
        vals_b = [ax_scores_b[k] for k in labels] + [list(ax_scores_b.values())[0]]
        ax.plot(angles, vals_b, linewidth=2, label=label_b)
        ax.fill(angles, vals_b, alpha=0.10)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    return fig


def compare_comment(name_a, m_a, name_b, m_b):
    # ざっくり会話用
    if m_a is None or m_b is None:
        return "比較コメントを生成できませんでした。"
    vol_ratio = (m_b["vol"] / m_a["vol"]) if m_a["vol"] > 0 else np.nan
    cagr_diff = m_b["cagr"] - m_a["cagr"]
    dd_diff = m_b["max_dd"] - m_a["max_dd"]  # mddはマイナス値なので、より大きい(0に近い)ほど良い

    parts = []
    if not np.isnan(vol_ratio):
        if vol_ratio >= 1.25:
            parts.append(f"{name_b}は値動きが{vol_ratio:.1f}倍くらい大きめ（その分スリルあり）。")
        elif vol_ratio <= 0.8:
            parts.append(f"{name_b}は値動きが小さめ（落ち着きタイプ）。")
        else:
            parts.append(f"値動きの大きさはだいたい同じくらい。")

    if cagr_diff >= 0.02:
        parts.append(f"成長力は{name_b}の方が強め（年率で+{cagr_diff:.1%}くらい上）。")
    elif cagr_diff <= -0.02:
        parts.append(f"成長力は{name_a}の方が強め（年率で+{-cagr_diff:.1%}くらい上）。")
    else:
        parts.append("成長力は近い水準。")

    if dd_diff >= 0.05:
        parts.append(f"下げの深さは{name_b}の方が浅め（メンタルに優しい）。")
    elif dd_diff <= -0.05:
        parts.append(f"下げの深さは{name_a}の方が浅め（メンタルに優しい）。")
    else:
        parts.append("下げの深さは近い水準。")

    return " ".join(parts)


# =============================
# 初期状態
# =============================
def default_holdings_df():
    return pd.DataFrame({"ティッカー": ["8802", "7203", "6758", "8306", "9984"], "株数": [10, 10, 10, 10, 10]})


st.session_state.setdefault("holdings_a", default_holdings_df())
st.session_state.setdefault("holdings_b", default_holdings_df())


# =============================
# サイドバー設定
# =============================
st.sidebar.header("🛠️ 設定（株式100%）")
st.session_state.setdefault("start_date", pd.to_datetime("2020-01-01").date())
st.session_state.setdefault("end_date", date.today())

start_date = st.sidebar.date_input("開始日", key="start_date")
end_date = st.sidebar.date_input("終了日（初期値：今日）", key="end_date")

rf = st.sidebar.number_input(
    "比較の基準となる利回り（通常はそのままでOK, %）",
    value=1.0,
    step=0.1,
    help="""
このツールで「どのポートフォリオが安定しているか」を
比較するための【基準となる数字】です。

・実際に現金を持つ、投資する、という意味ではありません
・値動きの大きさに対して、どれだけ効率よくリターンが出ているかを比べるために使います
・分からなければ初期値のままで問題ありません

※ 専門的には「Sharpe比」の計算に使われます
""",
) / 100.0

st.sidebar.header("🎯 モード（A=既存 / B=新規）")
mode = st.sidebar.radio(
    "何をしたい？",
    [
        "① 比較：既存(A) vs 新規(B)",
        "② 計算例：新規(B)の銘柄集合で配分を計算",
        "③ 計算例：既存(A)をベースに配分を計算",
    ],
)

objective = None
if mode.startswith("②") or mode.startswith("③"):
    st.sidebar.subheader("目的（ユーザーが選択）")
    objective = st.sidebar.radio(
        "目的",
        ["Equal（均等配分）", "MinVol（リスク最小）", "MaxSharpe（効率重視）"],
        help="過去データ上の配分を、目的に沿って“計算例”として算出します。",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("制約（②③）")
min_w = st.sidebar.slider("最小比率（各銘柄）%", 0, 20, 0, 1) / 100.0
max_w = st.sidebar.slider("最大比率（各銘柄）%", 20, 100, 40, 5) / 100.0

delta_w = 0.0
if mode.startswith("③"):
    st.sidebar.subheader("③の変更幅（既存Aから）")
    delta_w = st.sidebar.slider("各銘柄の比率変更の上限（±%）", 0, 50, 10, 1) / 100.0


# =============================
# 入力（A/B）
# =============================
st.markdown("## ① 入力（株数 / 日本株のみ）")

st.markdown("### 🅰 既存ポートフォリオ（A）：CSVアップロード（任意・保存しません）")
uploaded_a = st.file_uploader("既存(A)の保有一覧CSV（任意）", type=["csv"], key="uploader_a")

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("🧹 既存(A)をリセット"):
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
    st.success("✅ CSVを既存(A)に取り込みました（アップロードはクリア済み）")
    st.rerun()

colA, colB = st.columns(2)
with colA:
    st.markdown("### 🅰 既存ポートフォリオ（A）：株数")
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
    if st.button("既存(A)に反映"):
        tmp = edit_a.copy()
        tmp["ティッカー"] = tmp["ティッカー"].map(normalize_ticker_jp)
        if (tmp["ティッカー"] == "INVALID").any():
            st.error("⚠️ 既存(A)：ティッカーは日本株（4桁/4桁.T）のみです。")
        else:
            st.session_state["holdings_a"] = tmp
            st.rerun()

with colB:
    st.markdown("### 🅱 新規ポートフォリオ（B）：株数（①用）/ 銘柄リスト（②③用）")
    st.caption("②③ではBの株数は使わず、銘柄集合として扱います。")
    edit_b = st.data_editor(
        st.session_state["holdings_b"],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "ティッカー": st.column_config.TextColumn("ティッカー（4桁 or 4桁.T）"),
            "株数": st.column_config.NumberColumn("株数（①の比較で使用）", min_value=0, step=1, format="%.0f"),
        },
        key="editor_b",
    )
    if st.button("新規(B)に反映"):
        tmp = edit_b.copy()
        tmp["ティッカー"] = tmp["ティッカー"].map(normalize_ticker_jp)
        if (tmp["ティッカー"] == "INVALID").any():
            st.error("⚠️ 新規(B)：ティッカーは日本株（4桁/4桁.T）のみです。")
        else:
            st.session_state["holdings_b"] = tmp
            st.rerun()


# =============================
# 実行
# =============================
st.markdown("## ② 実行")
st.caption("※ Sharpe比は『値動きの割に、どれだけ効率よくリターンを出したか』を見る指標です（ここでは比較の物差しとして使用）。")

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
        st.error("⚠️ ③は同意チェックが必要です。")
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
    B_any = B[(B["ティッカー"] != "")]

    if A.empty:
        st.error("⚠️ 既存(A)に有効な行がありません。")
        st.stop()

    # 対象銘柄集合
    if mode.startswith("①"):
        B1 = B[(B["ティッカー"] != "") & (B["株数"] > 0)]
        if B1.empty:
            st.error("⚠️ ①は新規(B)にも株数>0の行が必要です。")
            st.stop()
        tickers_all = list(dict.fromkeys(A["ティッカー"].tolist() + B1["ティッカー"].tolist()))
    elif mode.startswith("②"):
        if B_any.empty:
            st.error("⚠️ ②は新規(B)に銘柄を1つ以上入力してください（株数は不要）。")
            st.stop()
        tickers_all = list(dict.fromkeys(B_any["ティッカー"].tolist()))
    else:
        use_universe = st.radio("③の候補銘柄セット", ["既存(A)のみ", "既存(A) + 新規(B)"], horizontal=True)
        if use_universe == "既存(A)のみ":
            tickers_all = list(dict.fromkeys(A["ティッカー"].tolist()))
        else:
            tickers_all = list(dict.fromkeys(A["ティッカー"].tolist() + B_any["ティッカー"].tolist()))

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
    prices = prices.ffill().dropna()
    if prices.empty:
        st.error("❌ 有効な価格データが不足しています。")
        st.stop()

    used_date = prices.index[-1]

    # Aの現状weights（銘柄集合外は無視）
    last_px = latest_prices_asof(prices)

    def build_current_weights(df_holdings, tickers_subset):
        df = df_holdings.copy()
        df["価格(直近)"] = df["ティッカー"].map(lambda t: float(last_px.get(t, np.nan)))
        df = df.dropna(subset=["価格(直近)"])
        df = df[df["ティッカー"].isin(tickers_subset)]
        df["時価"] = df["株数"].astype(float) * df["価格(直近)"].astype(float)
        total = float(df["時価"].sum())
        if total <= 0:
            return None, None, None
        tick = df["ティッカー"].tolist()
        w = (df["時価"].values / total).astype(float)
        return tick, w, df

    # 共通：log returns for optimization stats
    log_ret = np.log(prices / prices.shift(1)).dropna()
    lr_all = log_ret[tickers_all].dropna(how="any")
    mean_all = lr_all.mean().values
    cov_all = lr_all.cov().values

    # 表示ユーティリティ
    def render_portfolio_block(title, metrics, ptype, ptype2, axes_scores):
        txt = TYPE_TEXT.get(ptype, {})
        st.subheader(title)
        st.markdown(f"### 💬 ひとことで\n**「{txt.get('tagline','')}」**")
        st.markdown(f"**タイプ：{ptype}**（サブ要素：{ptype2}）")
        st.caption(f"特徴：{txt.get('desc','')}")
        st.caption(f"ツッコミどころ：{txt.get('tsukkomi','')}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("成長（CAGR）", f"{metrics['cagr']:.2%}")
        c2.metric("値動き（年率）", f"{metrics['vol']:.2%}")
        c3.metric("効率（Sharpe）", f"{metrics['sharpe']:.2f}")
        c4.metric("最大下落（MaxDD）", f"{metrics['max_dd']:.1%}")

        st.markdown("### 🏅 強さ（★）")
        st.write(
            f"- 安定性：{stars(axes_scores['安定性'])}\n"
            f"- 成長力：{stars(axes_scores['成長力'])}\n"
            f"- 爆発力：{stars(axes_scores['爆発力'])}\n"
            f"- メンタル耐性：{stars(axes_scores['メンタル耐性'])}\n"
            f"- 継続しやすさ：{stars(axes_scores['継続しやすさ'])}"
        )

        st.markdown("### 📢 友達に説明するなら")
        st.info(txt.get("friend", ""))

        st.caption(f"Sharpeの雰囲気：{sharpe_label(metrics['sharpe'])}")

    # =========================
    # ① 比較：A vs B（現状同士）
    # =========================
    if mode.startswith("①"):
        # 既存A
        tick_a, w_a, df_a = build_current_weights(A, tickers_all)
        # 新規B
        B1 = B[(B["ティッカー"] != "") & (B["株数"] > 0)]
        tick_b, w_b, df_b = build_current_weights(B1, tickers_all)

        if tick_a is None or tick_b is None:
            st.error("⚠️ AまたはBの時価計算に失敗しました（価格取得できない銘柄がある可能性）。")
            st.stop()

        prices_a = prices[tick_a]
        prices_b = prices[tick_b]

        mA = metrics_from_prices(prices_a, w_a, rf)
        mB = metrics_from_prices(prices_b, w_b, rf)

        if mA is None or mB is None:
            st.error("⚠️ 指標計算に失敗しました。")
            st.stop()

        tA, tA2, _, axesA = judge_type(mA)
        tB, tB2, _, axesB = judge_type(mB)

        # 結論（あくまで会話用）
        winner = "A" if (mA["sharpe"] >= mB["sharpe"]) else "B"
        st.success(f"✅ 今日のひとこと：この条件だと **{winner}** の方が「効率よく安定してる」っぽい（※過去データの計算例）")
        st.caption(f"直近価格は {used_date.date()} の終値（Adj Close優先）を使用。")

        # レーダー
        st.markdown("## 📊 強さレーダー")
        fig = radar_plot(axesA, axesB, label_a="既存(A)", label_b="新規(B)")
        st.pyplot(fig)

        tabs = st.tabs(["🅰 既存(A)", "🅱 新規(B)", "⚔ 比較コメント", "📋 共有用テキスト"])
        with tabs[0]:
            render_portfolio_block("🅰 既存ポートフォリオ（A）", mA, tA, tA2, axesA)
        with tabs[1]:
            render_portfolio_block("🅱 新規ポートフォリオ（B）", mB, tB, tB2, axesB)
        with tabs[2]:
            st.subheader("⚔ 比較すると…")
            st.write(compare_comment("既存(A)", mA, "新規(B)", mB))
        with tabs[3]:
            share = (
                f"🧬ポートフォリオ性格診断（{start_date}〜{end_date}）\n"
                f"🅰既存(A): {tA}「{TYPE_TEXT[tA]['tagline']}」/ CAGR {mA['cagr']:.1%} / Vol {mA['vol']:.1%} / Sharpe {mA['sharpe']:.2f} / MaxDD {mA['max_dd']:.1%}\n"
                f"🅱新規(B): {tB}「{TYPE_TEXT[tB]['tagline']}」/ CAGR {mB['cagr']:.1%} / Vol {mB['vol']:.1%} / Sharpe {mB['sharpe']:.2f} / MaxDD {mB['max_dd']:.1%}\n"
                f"⚔比較: {compare_comment('既存(A)', mA, '新規(B)', mB)}\n"
                f"※過去データ上の計算例（将来を保証しません）"
            )
            st.text_area("コピペして仲間に送る用（編集OK）", value=share, height=200)

        st.stop()

    # =========================
    # ② 計算例：B銘柄集合
    # ③ 計算例：Aベース
    # =========================
    n = len(tickers_all)
    if n == 0:
        st.error("⚠️ 対象銘柄が空です。")
        st.stop()

    # bounds 設計
    if mode.startswith("②"):
        bounds = [(min_w, max_w) for _ in range(n)]
        base_label = "新規(B)の銘柄集合"
        # baseline（B株数が入っていれば比較用に作る）
        B1 = B[(B["ティッカー"] != "") & (B["株数"] > 0)]
        base_tick, base_w, _ = build_current_weights(B1, tickers_all) if not B1.empty else (None, None, None)
    else:
        base_label = "既存(A)"
        # A現状比率から±delta
        tick_a, w_a, df_a = build_current_weights(A, tickers_all)
        if tick_a is None:
            st.error("⚠️ 既存(A)の時価計算に失敗しました。")
            st.stop()

        # Aの“全体”に対する比率をtickers_allへ埋め込み
        w0_map = {t: 0.0 for t in tickers_all}
        df_a_map = dict(zip(df_a["ティッカー"].tolist(), (df_a["時価"].values / df_a["時価"].sum()).astype(float)))
        for t in tickers_all:
            w0_map[t] = float(df_a_map.get(t, 0.0))

        bounds = []
        for t in tickers_all:
            base = w0_map.get(t, 0.0)
            lo = max(0.0, base - delta_w)
            hi = min(1.0, base + delta_w)
            lo = max(lo, min_w)
            hi = min(hi, max_w)
            # 新規候補は強制投入しない
            if base == 0.0:
                lo = 0.0
            if lo > hi:
                lo, hi = 0.0, max_w
            bounds.append((lo, hi))

        base_tick, base_w = tick_a, w_a  # baselineはA

    # w_calc（計算例）
    w_calc, status = compute_weights_by_objective(objective, mean_all, cov_all, rf, bounds)
    if status != "OK" or w_calc is None:
        st.error("⚠️ 計算に失敗しました。制約（最小/最大比率、変更幅）を緩めてください。")
        st.stop()

    # metrics
    prices_sel = prices[tickers_all]
    m_calc = metrics_from_prices(prices_sel, w_calc, rf)
    if m_calc is None:
        st.error("⚠️ 指標計算に失敗しました。")
        st.stop()

    tC, tC2, _, axesC = judge_type(m_calc)

    st.success(f"✅ 計算例のタイプ：**{tC}**（{TYPE_TEXT[tC]['tagline']}）")
    st.caption(f"目的：{objective} / 直近価格：{used_date.date()}（Adj Close優先）/ 対象：{base_label}")

    # baseline metrics（あれば比較）
    m_base = None
    axes_base = None
    tB0 = None
    tB02 = None
    if base_tick is not None and base_w is not None:
        m_base = metrics_from_prices(prices[base_tick], base_w, rf)
        if m_base is not None:
            tB0, tB02, _, axes_base = judge_type(m_base)

    st.markdown("## 📊 強さレーダー（会話用）")
    fig = radar_plot(axes_base, axesC, label_a=f"{base_label}", label_b="計算例（今回）") if axes_base else radar_plot(axesC, None, label_a="計算例（今回）")
    st.pyplot(fig)

    tabs = st.tabs(["🧬 計算例（今回）", "🔎 元のポートフォリオ", "⚔ 比較コメント", "📋 共有用テキスト", "📈 配分（計算例）"])
    with tabs[0]:
        render_portfolio_block("🧬 計算例（今回）", m_calc, tC, tC2, axesC)

    with tabs[1]:
        if m_base is None:
            st.info("元のポートフォリオが株数で定義されていないため、比較表示はありません（②は株数なしでもOKです）。")
        else:
            render_portfolio_block(f"🔎 元のポートフォリオ（{base_label}）", m_base, tB0, tB02, axes_base)

    with tabs[2]:
        st.subheader("⚔ 比較すると…")
        if m_base is None:
            st.write("比較対象がないため、コメントは計算例単体です。")
            st.write(f"この計算例は『{TYPE_TEXT[tC]['tagline']}』寄りになりがち。")
        else:
            st.write(compare_comment(base_label, m_base, "計算例（今回）", m_calc))

    with tabs[3]:
        base_line = ""
        if m_base is not None:
            base_line = f"{base_label}: {tB0}「{TYPE_TEXT[tB0]['tagline']}」/ CAGR {m_base['cagr']:.1%} / Vol {m_base['vol']:.1%} / Sharpe {m_base['sharpe']:.2f} / MaxDD {m_base['max_dd']:.1%}\n"
        share = (
            f"🧬ポートフォリオ性格診断（{start_date}〜{end_date}）\n"
            f"{base_line}"
            f"計算例（今回）: {tC}「{TYPE_TEXT[tC]['tagline']}」/ CAGR {m_calc['cagr']:.1%} / Vol {m_calc['vol']:.1%} / Sharpe {m_calc['sharpe']:.2f} / MaxDD {m_calc['max_dd']:.1%}\n"
            f"⚔コメント: {compare_comment(base_label, m_base, '計算例（今回）', m_calc) if m_base else TYPE_TEXT[tC]['friend']}\n"
            f"※過去データ上の計算例（将来を保証しません）"
        )
        st.text_area("コピペして仲間に送る用（編集OK）", value=share, height=200)

    with tabs[4]:
        df_out = pd.DataFrame({"ティッカー": tickers_all, "社名": [name_map.get(t, t) for t in tickers_all], "比率(%)": w_calc * 100})
        df_out = df_out.sort_values("比率(%)", ascending=False)
        st.dataframe(
            df_out,
            use_container_width=True,
            hide_index=True,
            column_config={"比率(%)": st.column_config.ProgressColumn("比率(%)", min_value=0.0, max_value=100.0, format="%.1f%%")},
        )

st.markdown("---")
st.caption("このアプリは投資助言ではありません。表示される結果は将来を保証しません。")

