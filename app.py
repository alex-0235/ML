import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

st.title("🍷 Wine Quality Predictor (Red Wine)")
st.write("화학 성분 11개를 입력하면 예측된 품질 점수와 등급(좋음/보통/나쁨)을 보여줍니다.")

# --- Sidebar: thresholds ---
st.sidebar.header("등급 기준 설정 (사용자 정의)")
good_threshold = st.sidebar.number_input("좋은 와인 기준 (≥)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
bad_threshold  = st.sidebar.number_input("나쁜 와인 기준 (≤)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
st.sidebar.caption("기본: ≥7 좋음, ≤4 나쁨, 그 사이 보통")

# --- Load data ---
st.subheader("1) 데이터 불러오기")
st.write("- 방법 A: 아래에서 CSV 업로드 (권장)\n- 방법 B: `winequality-red.csv` 파일을 이 스크립트와 같은 폴더에 두면 자동 로드")

uploaded = st.file_uploader("`winequality-red.csv` 업로드 (UCI Red Wine Quality 데이터)", type=["csv"])

def load_default_csv():
    try:
        df = pd.read_csv("winequality-red.csv")
        return df
    except Exception:
        return None

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("업로드한 CSV를 로드했습니다.")
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
else:
    df = load_default_csv()
    if df is not None:
        st.info("로컬 폴더의 `winequality-red.csv`를 로드했습니다.")
    else:
        st.warning("CSV가 없으면 예측은 데모 모드로 동작합니다(학습 없이 기본값 사용).")

# --- Expected feature columns ---
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

st.subheader("2) 화학 성분 입력")
st.caption("※ % 기호를 붙여도 됩니다(예: 0.7%). 단, 실제 데이터 단위는 %가 아닌 경우가 많아 단위는 참고용으로만 표시됩니다. 입력값은 숫자만 사용됩니다.")

def parse_percentish(s: str) -> float:
    if s is None:
        return 0.0
    s = str(s).strip()
    # remove percent sign and commas
    s = s.replace("%", "").replace(",", "")
    # keep digits, dot, minus
    import re
    m = re.findall(r"[-]?\d*\.?\d+", s)
    if not m:
        return 0.0
    try:
        return float(m[0])
    except:
        return 0.0

cols = st.columns(2)
inputs = {}
for i, feat in enumerate(FEATURES):
    with cols[i % 2]:
        val_str = st.text_input(f"{feat}", value="0")
        inputs[feat] = parse_percentish(val_str)

X_input = pd.DataFrame([inputs])

# --- Train a simple model if data available ---
model = None
train_info = ""

if df is not None and all(col in df.columns for col in FEATURES + ["quality"]):
    X = df[FEATURES].copy()
    y = df["quality"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    train_info = f"학습 완료 · 테스트 MAE={mae:.3f}, R²={r2:.3f} (랜덤포레스트 회귀)"
    st.success(train_info)
else:
    st.info("데이터가 없어서 학습을 건너뜁니다. 아래 예측은 임의의 간단한 규칙으로만 동작합니다.")

# --- Prediction ---
st.subheader("3) 예측 및 등급 출력")
def classify(score: float) -> str:
    if score >= good_threshold:
        return "좋음"
    elif score <= bad_threshold:
        return "나쁨"
    return "보통"

if model is not None:
    pred = float(model.predict(X_input)[0])
    st.metric("예측 품질 점수", f"{pred:.2f}")
    st.metric("등급", classify(pred))
else:
    # Fallback heuristic: alcohol 높고 volatile acidity 낮으면 점수 ↑ (데모용)
    score = 5.0 + 0.5 * (inputs["alcohol"] - 10) - 2.0 * max(0.0, inputs["volatile acidity"] - 0.6)
    score = max(0.0, min(10.0, score))
    st.metric("예측(데모) 품질 점수", f"{score:.2f}")
    st.metric("등급", classify(score))

st.divider()
st.caption("참고: UCI Wine Quality (Red) 데이터셋의 특성을 사용합니다. 실제 단위는 %가 아닌 g/dm³ 등 다양한 단위를 포함합니다.")