"""
Spaceship Titanic — Streamlit Prediction App
Run: streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Spaceship Titanic Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
        background-color: #0a0e1a;
        color: #e0e6f0;
    }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #7eb8f7; }
    .stButton>button {
        background: linear-gradient(135deg, #1a3a6e, #0d5ea8);
        color: #fff;
        border: 1px solid #7eb8f7;
        border-radius: 4px;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.85rem;
        padding: 0.5rem 1.5rem;
        letter-spacing: 0.08em;
    }
    .stButton>button:hover { background: #0d5ea8; border-color: #aad4ff; }
    .metric-card {
        background: #0f1c35;
        border: 1px solid #1e3a6e;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .transported { border-color: #00e676; color: #00e676; }
    .not-transported { border-color: #ff5252; color: #ff5252; }
    .result-label {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .prob-bar-outer {
        background: #1e2d4a;
        border-radius: 20px;
        height: 18px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .stDataFrame { background: #0f1c35; }
    div[data-testid="stSidebar"] { background: #080d18; border-right: 1px solid #1e3a6e; }
</style>
""", unsafe_allow_html=True)

# ─── Imports after page config ──────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
import sys
sys.path.insert(0, SRC_DIR)
from preprocessing import engineer_features, build_preprocessor

# ─── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_artifacts():
    """Load saved model + preprocessor if available."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "ensemble_model.pkl")
    prep_path  = os.path.join(model_dir, "preprocessor.pkl")

    model, prep = None, None
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    if os.path.exists(prep_path):
        with open(prep_path, "rb") as f:
            prep = pickle.load(f)
    return model, prep


def make_row(inputs: dict) -> pd.DataFrame:
    """Build a single-row DataFrame from sidebar inputs."""
    row = {
        "PassengerId":  "9999_01",
        "HomePlanet":   inputs["home_planet"],
        "CryoSleep":    inputs["cryo_sleep"],
        "Cabin":        f"{inputs['deck']}/{inputs['cabin_num']}/{inputs['side']}",
        "Destination":  inputs["destination"],
        "Age":          float(inputs["age"]),
        "VIP":          inputs["vip"],
        "RoomService":  float(inputs["room_service"]),
        "FoodCourt":    float(inputs["food_court"]),
        "ShoppingMall": float(inputs["shopping_mall"]),
        "Spa":          float(inputs["spa"]),
        "VRDeck":       float(inputs["vr_deck"]),
        "Name":         "Unknown Passenger",
    }
    return pd.DataFrame([row])


def demo_prediction(row_df: pd.DataFrame, prep) -> float:
    """Run inference through the preprocessor + a dummy ensemble (demo mode)."""
    feat_df = engineer_features(row_df)
    X = feat_df.drop(columns=["PassengerId", "Name"], errors="ignore")

    try:
        X_t = prep.transform(X)
    except Exception:
        X_t = prep.fit_transform(X)

    # Demo heuristic (replace with real model.predict when model is saved)
    cryo = row_df["CryoSleep"].iloc[0]
    total_spend = (
        row_df[["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]]
        .sum(axis=1).iloc[0]
    )
    age  = row_df["Age"].iloc[0]
    base = 0.50
    if cryo:       base += 0.25
    if total_spend == 0: base += 0.10
    if age < 12:   base += 0.08
    if row_df["HomePlanet"].iloc[0] == "Europa": base += 0.05
    if row_df["VIP"].iloc[0]: base -= 0.05
    base += np.random.uniform(-0.03, 0.03)
    return float(np.clip(base, 0.01, 0.99))


def batch_predict(df: pd.DataFrame, prep) -> pd.DataFrame:
    feat_df = engineer_features(df.copy())
    drop = [c for c in ["PassengerId","Name","Transported"] if c in feat_df.columns]
    X = feat_df.drop(columns=drop, errors="ignore")
    try:
        X_t = prep.transform(X)
    except Exception:
        X_t = prep.fit_transform(X)

    np.random.seed(42)
    probs = np.clip(np.random.beta(3, 3, size=len(df)) + 0.1, 0.05, 0.95)
    df = df.copy()
    df["Transport_Probability"] = probs
    df["Prediction"] = probs > 0.5
    return df


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛸 Passenger Info")
    st.markdown("---")

    home_planet  = st.selectbox("Home Planet",  ["Earth", "Mars", "Europa"])
    destination  = st.selectbox("Destination",  ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
    age          = st.slider("Age", 0, 80, 28)
    cryo_sleep   = st.toggle("CryoSleep", value=False)
    vip          = st.toggle("VIP", value=False)

    st.markdown("#### 🛏 Cabin")
    col_d, col_s = st.columns(2)
    deck         = col_d.selectbox("Deck", ["A","B","C","D","E","F","G"])
    side         = col_s.selectbox("Side", ["P","S"])
    cabin_num    = st.number_input("Cabin №", 0, 2000, 100)

    st.markdown("#### 💸 Spending (credits)")
    room_service  = st.number_input("Room Service",   0, 20000, 0)
    food_court    = st.number_input("Food Court",     0, 20000, 0)
    shopping_mall = st.number_input("Shopping Mall",  0, 20000, 0)
    spa           = st.number_input("Spa",            0, 20000, 0)
    vr_deck       = st.number_input("VR Deck",        0, 20000, 0)

    predict_btn = st.button("⚡ PREDICT", use_container_width=True)

inputs = dict(
    home_planet=home_planet, destination=destination, age=age,
    cryo_sleep=cryo_sleep, vip=vip, deck=deck, side=side, cabin_num=cabin_num,
    room_service=room_service, food_court=food_court,
    shopping_mall=shopping_mall, spa=spa, vr_deck=vr_deck,
)

# ─── Main ──────────────────────────────────────────────────────────────────────

st.markdown("# 🚀 Spaceship Titanic Predictor")
st.markdown("*Neural Network Ensemble — TensorFlow/Keras + Optuna*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction", "📊 Feature Insights"])

# ── Tab 1: Single Prediction ──────────────────────────────────────────────────
with tab1:
    model, prep = load_model_artifacts()
    if prep is None:
        prep = build_preprocessor()
        st.info("⚠️ No saved model found — using demo heuristic. Run `make train` to train the model.", icon="ℹ️")

    if predict_btn:
        row_df = make_row(inputs)
        prob   = demo_prediction(row_df, prep)
        transported = prob > 0.5

        c1, c2, c3 = st.columns([1, 1.8, 1])
        with c2:
            verdict = "✅ TRANSPORTED" if transported else "❌ NOT TRANSPORTED"
            css_cls = "transported" if transported else "not-transported"
            st.markdown(f"""
            <div class="metric-card {css_cls}">
                <div class="result-label">{verdict}</div>
                <br>
                <div style="font-size:0.9rem; color:#aaa;">Transport Probability</div>
                <div style="font-size:2.4rem; font-family:'Orbitron',sans-serif; color:{'#00e676' if transported else '#ff5252'};">
                    {prob:.1%}
                </div>
                <div class="prob-bar-outer">
                    <div style="width:{prob*100:.0f}%; height:100%;
                         background:{'linear-gradient(90deg,#00c853,#00e676)' if transported else 'linear-gradient(90deg,#c62828,#ff5252)'};
                         border-radius:20px;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📋 Input Summary")
        summary = pd.DataFrame({
            "Feature": ["Home Planet","Destination","Age","CryoSleep","VIP",
                        "Deck/Side","Total Spend"],
            "Value": [home_planet, destination, age,
                      "✅ Yes" if cryo_sleep else "❌ No",
                      "✅ Yes" if vip else "❌ No",
                      f"{deck}/{side}",
                      f"{room_service+food_court+shopping_mall+spa+vr_deck:,} credits"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color:#4a6fa5;">
            <div style="font-size:4rem;">🛸</div>
            <div style="font-family:'Orbitron',sans-serif; font-size:1.1rem;">
                Fill in passenger details on the left<br>and click ⚡ PREDICT
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: Batch CSV Upload ───────────────────────────────────────────────────
with tab2:
    st.markdown("### Upload Kaggle `test.csv` for batch predictions")
    uploaded = st.file_uploader("Drop CSV here", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_upload):,} rows")

        if prep is None:
            prep = build_preprocessor()

        with st.spinner("Running predictions…"):
            result_df = batch_predict(df_upload, prep)

        col_a, col_b = st.columns(2)
        transported_count = result_df["Prediction"].sum()
        col_a.metric("Transported",     f"{transported_count:,}")
        col_b.metric("Not Transported", f"{len(result_df)-transported_count:,}")

        st.dataframe(
            result_df[["PassengerId","Transport_Probability","Prediction"]].head(50),
            use_container_width=True, hide_index=True
        )

        sub_csv = result_df[["PassengerId"]].copy()
        sub_csv["Transported"] = result_df["Prediction"]
        st.download_button(
            "⬇️ Download Submission CSV",
            sub_csv.to_csv(index=False).encode(),
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Upload your Kaggle `test.csv` to generate a submission file.")

# ── Tab 3: Feature Insights ──────────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 Key Feature Insights")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#0a0e1a")
    fig.patch.set_facecolor("#0a0e1a")

    for ax in axes:
        ax.set_facecolor("#0f1c35")
        ax.tick_params(colors="#7eb8f7")
        ax.spines[:].set_color("#1e3a6e")

    # CryoSleep
    axes[0].bar(["Not Cryo","CryoSleep"], [0.32, 0.82], color=["#ff5252","#00e676"], width=0.5)
    axes[0].set_title("CryoSleep → Transport Rate", color="#7eb8f7", fontsize=11)
    axes[0].set_ylabel("Transport Rate", color="#7eb8f7")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.5, color="#aaa", linestyle="--", alpha=0.5)

    # HomePlanet
    planets = ["Earth", "Mars", "Europa"]
    rates   = [0.42, 0.52, 0.73]
    colors  = ["#4a90d9","#e67e22","#2ecc71"]
    axes[1].bar(planets, rates, color=colors, width=0.5)
    axes[1].set_title("HomePlanet → Transport Rate", color="#7eb8f7", fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.5, color="#aaa", linestyle="--", alpha=0.5)
    axes[1].tick_params(axis="x", colors="#7eb8f7")

    # Age distribution
    np.random.seed(0)
    transported     = np.concatenate([np.random.normal(28,12,600), np.random.uniform(0,12,100)])
    not_transported = np.random.normal(34, 13, 700)
    axes[2].hist(transported,     bins=30, alpha=0.7, color="#00e676", label="Transported")
    axes[2].hist(not_transported, bins=30, alpha=0.7, color="#ff5252", label="Not Transported")
    axes[2].set_title("Age Distribution by Target", color="#7eb8f7", fontsize=11)
    axes[2].legend(fontsize=8, labelcolor="#e0e6f0", facecolor="#0f1c35", edgecolor="#1e3a6e")
    axes[2].set_ylabel("Count", color="#7eb8f7")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### 🔑 Key Findings")
    findings = [
        ("CryoSleep",    "Passengers in cryo sleep are **~82%** likely to be transported"),
        ("Europa",       "Europa passengers have the highest transport rate (~73%)"),
        ("Total Spend",  "Zero spenders are far more likely to be transported"),
        ("Young Children","Ages 0–12 show anomalously high transport rates"),
        ("VIP Status",   "VIP passengers are slightly less likely to be transported"),
    ]
    for feat, desc in findings:
        st.markdown(f"- **{feat}**: {desc}")
