%%writefile app_final.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

st.set_page_config(page_title="かんたん株式分散シミュレーター", layout="wide")
st.title("🔰 かんたん株式分散シミュレーター")
st.markdown("専門知識がなくても大丈夫。AIが**「リスクを抑えて利益を狙う」**ための最適な配分を計算します。")
%%writefile -a app_final.py

# --- サイドバー設定 ---
st.sidebar.header("🛠️ 設定パネル")
st.sidebar.info("💡 ヒント: マウスを項目の上に乗せると、詳しい説明が表示されます。")

default_tickers = "8802.T, 7203.T, 6758.T, 8306.T, 9984.T"
tickers_input = st.sidebar.text_area("銘柄コード (カンマ区切り)", value=default_tickers, height=80, help="例: 8802.T, 7203.T")
start_date = st.sidebar.date_input("開始日", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("終了日", pd.to_datetime("2024-12-31"))

st.sidebar.subheader("自分のルール")
min_weight = st.sidebar.slider("最低これくらいは持ちたい (%)", 0, 20, 5, 1, help="分散効果を高めるため5%程度がおすすめ") / 100.0
max_weight = st.sidebar.slider("最大ここまでにしておく (%)", 20, 100, 40, 5, help="1銘柄への集中を防ぐ上限") / 100.0
risk_free_rate = st.sidebar.number_input("安全資産の利回り (%)", value=1.0, step=0.1, help="国債などの金利") / 100.0

# --- 関数群 ---
def get_data(tickers, start, end):
    try:
        df = yf.download(tickers, start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.get_level_values(0): return df.xs('Adj Close', axis=1, level=0)
            if 'Close' in df.columns.get_level_values(0): return df.xs('Close', axis=1, level=0)
        if 'Adj Close' in df.columns: return df['Adj Close']
        if 'Close' in df.columns: return df['Close']
        return df.iloc[:, 0] if df.shape[1] > 0 else df
    except: return None

@st.cache_data
def get_company_names(tickers_list):
    names = {}
    for t in tickers_list:
        try:
            ticker_info = yf.Ticker(t)
            name = ticker_info.info.get('shortName', ticker_info.info.get('longName', t))
            names[t] = name
        except:
            names[t] = t
    return names
  %%writefile -a app_final.py

# --- メイン処理 ---
if st.button("🚀 AIに計算させる"):
    # エラー回避のため、安全な書き方に変更
    raw_ts = tickers_input.split(',')
    ts = []
    for t in raw_ts:
        if t.strip():
            ts.append(t.strip())
    
    if len(ts) < 2:
        st.error("⚠️ 2銘柄以上入れてください")
    else:
        with st.spinner('データを分析中...'):
            df = get_data(ts, start_date, end_date)
            name_map = get_company_names(ts)
            
        if df is None or df.empty:
            st.error("❌ データ取得失敗")
        else:
            df = df.dropna().select_dtypes(include=[np.number])
            if df.shape[1] < 2:
                st.error("⚠️ 有効なデータ不足")
            else:
                try:
                    log_ret = np.log(df/df.shift(1)).dropna()
                    mean = log_ret.mean(); cov = log_ret.cov()
                    n = len(df.columns)
                    
                    def neg_sharpe(w):
                        r = np.sum(mean*w)*252
                        s = np.sqrt(np.dot(w.T,np.dot(cov,w)))*252**0.5
                        return -(r-risk_free_rate)/s
                    
                    cons = ({'type':'eq','fun':lambda x:np.sum(x)-1})
                    bnds = tuple((min_weight,max_weight) for _ in range(n))
                    
                    res = sco.minimize(neg_sharpe, [1./n]*n, method='SLSQP', bounds=bnds, constraints=cons)
                    
                    if res.success:
                        w = res.x
                        ret = np.sum(mean*w)*252
                        std = np.sqrt(np.dot(w.T,np.dot(cov,w)))*252**0.5
                        sharpe = (ret-risk_free_rate)/std
                        
                        st.success("✅ 計算完了！")
                        c1,c2,c3 = st.columns(3)
                        c1.metric("💰 期待リターン", f"{ret:.2%}")
                        c2.metric("🛡️ リスク", f"{std:.2%}")
                        c3.metric("📊 投資効率", f"{sharpe:.2f}")
                        
                        if sharpe>=1.0: st.info("🌟 素晴らしい構成です！")
                        elif sharpe>=0.7: st.success("👍 良いバランスです")
                        else: st.warning("⚠️ 少し効率が悪いです")
                        
                        valid_tickers = df.columns
                        labels = [f"{name_map.get(t, t)}\n({t})" for t in valid_tickers]
                        
                        col1, col2 = st.columns([1,1])
                        with col1:
                            fig, ax = plt.subplots()
                            ax.pie(w, labels=labels, autopct='%1.1f%%', startangle=90)
                            st.pyplot(fig)
                        with col2:
                            df_res = pd.DataFrame({
                                "コード": valid_tickers,
                                "社名": [name_map.get(t, t) for t in valid_tickers],
                                "推奨比率": [f"{v:.2%}" for v in w]
                            })
                            st.dataframe(df_res, use_container_width=True)
                    else: st.warning("条件を緩めてください")
                except Exception as e: st.error(f"エラー: {e}")
                  import subprocess
import time

open('tunnel.log', 'w').close()
!pkill -f streamlit
!pkill -f cloudflared

print("🚀 アプリを再起動しています...")
subprocess.Popen(["streamlit", "run", "app_final.py", "--server.port", "8501"])

with open('tunnel.log', 'w') as log_file:
    subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8501"], stdout=log_file, stderr=log_file)

print("⏳ URL発行待ち (10秒)...")
time.sleep(10)

print("-" * 50)
print("↓↓ 以下のURLをクリックしてください ↓↓")
!grep -o 'https://.*\.trycloudflare.com' tunnel.log | head -n 1
print("-" * 50)

while True:
    time.sleep(3600)
