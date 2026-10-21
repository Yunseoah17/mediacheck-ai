import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error
import joblib

st.set_page_config(page_title="MediaCheck AI", page_icon="🎬", layout="wide")

FEATURES = [
    "brightness_mean", "brightness_std", "brightness_change",
    "frame_difference_mean", "scene_change_rate", "motion_proxy"
]

def extract_video_features(path, sample_fps=2, max_samples=600):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다. MP4 파일인지 확인해 주세요.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps > 0 else 0
    interval = max(1, int(fps / sample_fps)) if fps > 0 else 1

    brightness, diffs, cuts, sampled = [], [], [], []
    prev_gray = None
    idx = 0
    while len(sampled) < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            b = float(np.mean(gray) / 255.0)
            brightness.append(b)
            if prev_gray is not None:
                d = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
                diffs.append(d)
                if d > 0.22:
                    cuts.append(1)
                else:
                    cuts.append(0)
            sampled.append(idx)
            prev_gray = gray
        idx += 1
    cap.release()
    if not brightness:
        raise ValueError("분석할 프레임을 읽지 못했습니다.")
    diffs_arr = np.array(diffs) if diffs else np.array([0.0])
    return {
        "duration_sec": round(float(duration), 2),
        "sampled_frames": int(len(sampled)),
        "brightness_mean": float(np.mean(brightness)),
        "brightness_std": float(np.std(brightness)),
        "brightness_change": float(np.mean(np.abs(np.diff(brightness)))) if len(brightness)>1 else 0.0,
        "frame_difference_mean": float(np.mean(diffs_arr)),
        "scene_change_rate": float(np.mean(cuts)) if cuts else 0.0,
        "motion_proxy": float(np.percentile(diffs_arr, 75)),
    }

def demo_score(f):
    # Transparent demonstration heuristic, NOT a trained or validated predictor.
    raw = 10 * (0.45 * min(f["frame_difference_mean"] / 0.25, 1)
                + 0.30 * min(f["brightness_change"] / 0.15, 1)
                + 0.25 * min(f["scene_change_rate"] / 0.35, 1))
    return float(np.clip(raw, 0, 10))

st.title("🎬 MediaCheck AI")
st.caption("영상 특징 분석 기반 불편감 예측 시제품 · 연구/교육용 데모")

with st.sidebar:
    st.header("모델 설정")
    st.write("실험 데이터 CSV를 업로드하면 회귀모델을 학습할 수 있습니다.")
    st.caption("CSV에는 아래 영상 특징 열과 discomfort_score 열이 필요합니다.")
    st.code(", ".join(FEATURES + ["discomfort_score"]), language="text")
    labels_file = st.file_uploader("학습 데이터 CSV (선택)", type=["csv"])
    if labels_file is not None:
        try:
            train_df = pd.read_csv(labels_file)
            missing = [c for c in FEATURES + ["discomfort_score"] if c not in train_df.columns]
            if missing:
                st.error("필수 열이 없습니다: " + ", ".join(missing))
            elif len(train_df) < 5:
                st.warning("학습 행이 5개 미만입니다. 모델은 작동할 수 있지만 성능을 신뢰하기 어렵습니다.")
            else:
                if st.button("회귀모델 학습"):
                    X = train_df[FEATURES].apply(pd.to_numeric, errors="coerce")
                    y = pd.to_numeric(train_df["discomfort_score"], errors="coerce")
                    valid = y.notna()
                    X, y = X.loc[valid], y.loc[valid]
                    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
                    model.fit(X, y)
                    joblib.dump(model, "model.joblib")
                    st.session_state["model"] = model
                    st.success(f"모델 학습 완료: {len(y)}개 영상")
                    if len(y) >= 6:
                        try:
                            pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
                            st.metric("LOO 교차검증 MAE", f"{mean_absolute_error(y, pred):.2f}점")
                            st.caption("영상 수가 적으면 이 점수의 변동이 매우 클 수 있습니다.")
                        except Exception:
                            pass
        except Exception as e:
            st.error(f"CSV를 읽지 못했습니다: {e}")

uploaded = st.file_uploader("분석할 영상 업로드", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    st.video(uploaded)
    if st.button("영상 분석하기", type="primary"):
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            with st.spinner("프레임을 분석하고 있습니다..."):
                features = extract_video_features(tmp_path)
            st.session_state["features"] = features
            st.session_state["filename"] = uploaded.name
        except Exception as e:
            st.error(str(e))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

if "features" in st.session_state:
    f = st.session_state["features"]
    st.divider()
    st.subheader("분석 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("영상 길이", f'{f["duration_sec"]:.1f}초')
    c2.metric("평균 밝기", f'{f["brightness_mean"]:.3f}')
    c3.metric("프레임 변화량", f'{f["frame_difference_mean"]:.3f}')
    c4, c5, c6 = st.columns(3)
    c4.metric("밝기 변화량", f'{f["brightness_change"]:.3f}')
    c5.metric("장면 전환 추정 비율", f'{f["scene_change_rate"]:.3f}')
    c6.metric("움직임 대리 지표", f'{f["motion_proxy"]:.3f}')

    model = st.session_state.get("model")
    if model is None and os.path.exists("model.joblib"):
        try:
            model = joblib.load("model.joblib")
        except Exception:
            model = None
    if model is not None:
        score = float(np.clip(model.predict(pd.DataFrame([{k:f[k] for k in FEATURES}]))[0], 0, 10))
        st.subheader(f"모델 예상 불편감: {score:.1f} / 10")
        st.success("학습 데이터로 학습한 회귀모델의 출력입니다. 소규모 데이터에서는 오차가 클 수 있습니다.")
    else:
        score = demo_score(f)
        st.subheader(f"시연용 지표: {score:.1f} / 10")
        st.warning("이 점수는 영상 변화량을 조합한 임시 시연용 지표이며, 사람의 실제 불편감을 학습한 AI 예측값이 아닙니다. 발표에서는 '시연용 점수'로 표시하세요.")

    st.write("**영상 특징 표**")
    display = pd.DataFrame([{"특징":k, "값":round(float(v), 4) if isinstance(v, (float, int, np.floating, np.integer)) else v} for k,v in f.items()])
    st.dataframe(display, use_container_width=True, hide_index=True)
    csv = pd.DataFrame([{**{"filename":st.session_state.get("filename","video")}, **f}]).to_csv(index=False).encode("utf-8-sig")
    st.download_button("추출 특징 CSV 다운로드", data=csv, file_name="video_features.csv", mime="text/csv")

st.divider()
st.caption("주의: 이 도구는 영상의 시각적 변화량을 분석하는 연구 시제품입니다. 의료·심리 진단, 콘텐츠 유해성 판정, 자동 삭제에 사용하지 마세요.")
