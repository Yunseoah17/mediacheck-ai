import os
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MediaCheck AI · 연구용", page_icon="🎬", layout="wide")
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
FEATURE_VERSION = "v2"
TARGET = "discomfort_score"
FEATURES = [
    "brightness_mean", "brightness_std", "brightness_change",
    "frame_difference_mean", "frame_difference_p90", "scene_change_rate",
    "motion_proxy", "dark_frame_rate", "high_change_rate",
    "low_brightness_motion", "duration_sec",
]

st.title("🎬 MediaCheck AI")
st.caption("연구용 시제품 · 영상 특징 + 참가자 설문을 이용한 불편감 추정")
st.info(
    "EEG는 현재 이 앱의 예측 입력으로 사용하지 않습니다. EEG 실험은 자기보고 설문과 생리 반응의 관계를 탐구하는 별도 분석에 활용하고, "
    "이 모델은 영상 특징과 설문 점수의 관계를 학습해 새 영상의 불편감을 추정합니다. 검증 전에는 연구용 보조 지표로만 사용하세요."
)

def extract_video_features(path, sample_fps=2.0, max_samples=1200):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다. 코덱이 지원되는 MP4 파일인지 확인해 주세요.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    interval = max(1, int(round(fps / sample_fps))) if fps > 0 else 1
    brightness, diffs, indices = [], [], []
    prev_gray, idx = None, 0
    try:
        while len(indices) < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                brightness.append(float(gray.mean() / 255.0))
                if prev_gray is not None:
                    diffs.append(float(cv2.absdiff(gray, prev_gray).mean() / 255.0))
                indices.append(idx)
                prev_gray = gray
            idx += 1
    finally:
        cap.release()
    if len(brightness) < 2:
        raise ValueError("특징을 계산하기에 프레임이 부족합니다. 더 긴 영상을 사용해 주세요.")
    b = np.asarray(brightness, dtype=np.float32)
    d = np.asarray(diffs if diffs else [0.0], dtype=np.float32)
    dark_rate = float(np.mean(b < 0.20))
    return {
        "feature_version": FEATURE_VERSION,
        "duration_sec": float(duration),
        "sampled_frames": int(len(indices)),
        "sampling_fps": float(sample_fps),
        "brightness_mean": float(np.mean(b)),
        "brightness_std": float(np.std(b)),
        "brightness_change": float(np.mean(np.abs(np.diff(b)))) if len(b) > 1 else 0.0,
        "frame_difference_mean": float(np.mean(d)),
        "frame_difference_p90": float(np.percentile(d, 90)),
        "scene_change_rate": float(np.mean(d >= 0.22)),
        "motion_proxy": float(np.percentile(d, 75)),
        "dark_frame_rate": dark_rate,
        "high_change_rate": float(np.mean(d >= 0.12)),
        "low_brightness_motion": float(np.mean(d) * dark_rate),
    }

def make_model(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(alpha)))

def clean_training_data(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("CSV 필수 열이 없습니다: " + ", ".join(missing))
    out = df.copy()
    for c in FEATURES + [TARGET]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    out = out[out[TARGET].between(0, 10)]
    if out.empty:
        raise ValueError("유효한 discomfort_score가 없습니다. 0~10 숫자로 입력해 주세요.")
    return out

def cross_validate_model(X, y, groups, alpha):
    if groups is not None:
        groups = pd.Series(groups).fillna("unknown").astype(str).to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups >= 3:
            n_splits = min(5, n_groups)
            pred = cross_val_predict(make_model(alpha), X, y, cv=GroupKFold(n_splits=n_splits), groups=groups)
            return pred, f"GroupKFold {n_splits}겹"
        return None, "그룹 수가 3개 미만이라 그룹 교차검증을 수행할 수 없습니다."
    if len(y) >= 6:
        n_splits = min(5, len(y))
        pred = cross_val_predict(make_model(alpha), X, y, cv=KFold(n_splits=n_splits, shuffle=True, random_state=42))
        return pred, f"KFold {n_splits}겹"
    return None, "표본이 6개 미만이라 교차검증을 생략했습니다."

with st.sidebar:
    st.header("1. 설문 데이터로 학습")
    st.write("영상 특징 CSV에 참가자의 불편감 자기보고 점수를 연결해 업로드하세요.")
    st.code(", ".join(FEATURES + [TARGET]), language="text")
    st.caption("선택 열: video_id, participant_id, group_id, eeg_summary")
    train_file = st.file_uploader("학습 데이터 CSV", type=["csv"])
    alpha = st.slider("Ridge 규제 강도", 0.1, 100.0, 10.0, 0.1)
    group_cv = st.checkbox("그룹 기반 교차검증 사용", True)
    if train_file is not None:
        try:
            train_df = clean_training_data(pd.read_csv(train_file))
            st.write(f"유효한 평가 행: **{len(train_df)}개**")
            if len(train_df) < 10:
                st.warning("표본이 적습니다. 학습은 가능해도 새 영상의 예측 성능은 신뢰하기 어렵습니다.")
            if st.button("모델 학습 및 검증", type="primary"):
                X, y = train_df[FEATURES], train_df[TARGET].astype(float)
                group_col = next((c for c in ["group_id", "video_id"] if c in train_df.columns), None) if group_cv else None
                groups = train_df[group_col] if group_col else None
                pred, cv_method = cross_validate_model(X, y, groups, alpha)
                model = make_model(alpha)
                model.fit(X, y)
                bundle = {"model": model, "features": FEATURES, "feature_version": FEATURE_VERSION,
                          "target": TARGET, "alpha": alpha, "n_rows": len(train_df), "group_column": group_col}
                joblib.dump(bundle, MODEL_PATH)
                st.session_state["model_bundle"] = bundle
                st.session_state["cv_summary"] = {
                    "method": cv_method,
                    "mae": float(mean_absolute_error(y, pred)) if pred is not None else None,
                    "rmse": float(np.sqrt(mean_squared_error(y, pred))) if pred is not None else None,
                    "r2": float(r2_score(y, pred)) if pred is not None and len(y) >= 3 else None,
                    "group": group_col,
                }
                st.success(f"모델 학습 완료: {len(train_df)}개 평가 행")
        except Exception as e:
            st.error(f"학습 데이터 오류: {e}")
    if "cv_summary" in st.session_state:
        s = st.session_state["cv_summary"]
        st.subheader("교차검증")
        if s["mae"] is None:
            st.warning(s["method"])
        else:
            st.metric("MAE (낮을수록 좋음)", f'{s["mae"]:.2f} / 10')
            st.metric("RMSE", f'{s["rmse"]:.2f} / 10')
            if s["r2"] is not None:
                st.metric("R²", f'{s["r2"]:.2f}')
            st.caption(f'검증: {s["method"]} · 그룹 열: {s["group"] or "없음"}')

st.header("2. 새 영상 분석")
uploaded = st.file_uploader("분석할 영상 업로드", type=["mp4", "mov", "avi", "mkv"], key="video")
sample_fps = st.select_slider("초당 분석 프레임 수", options=[1.0, 2.0, 3.0, 5.0], value=2.0)
if uploaded is not None:
    st.video(uploaded)
    if st.button("영상 특징 추출", type="primary"):
        tmp_path = None
        try:
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            with st.spinner("영상 프레임을 분석 중입니다..."):
                st.session_state["features"] = extract_video_features(tmp_path, sample_fps)
                st.session_state["filename"] = uploaded.name
        except Exception as e:
            st.error(str(e))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

if "features" in st.session_state:
    f = st.session_state["features"]
    st.subheader("추출된 영상 특징")
    metric_items = [
        ("평균 밝기", "brightness_mean"), ("밝기 표준편차", "brightness_std"),
        ("밝기 변화량", "brightness_change"), ("프레임 변화 평균", "frame_difference_mean"),
        ("프레임 변화 P90", "frame_difference_p90"), ("장면 전환 추정 비율", "scene_change_rate"),
        ("움직임 대리 지표", "motion_proxy"), ("어두운 프레임 비율", "dark_frame_rate"),
        ("큰 변화 프레임 비율", "high_change_rate"),
    ]
    cols = st.columns(3)
    for i, (label, key) in enumerate(metric_items):
        cols[i % 3].metric(label, f'{f[key]:.3f}')
    st.caption(f'영상 길이 {f["duration_sec"]:.1f}초 · 샘플 프레임 {f["sampled_frames"]}개 · 특징 버전 {f["feature_version"]}')

    bundle = st.session_state.get("model_bundle")
    if bundle is None and MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
        except Exception:
            bundle = None
    if bundle is None:
        st.warning("학습 모델이 없습니다. 현재는 특징만 추출했습니다. 설문 데이터 모델이 없으므로 임의의 불편감 점수는 표시하지 않습니다.")
    elif bundle.get("feature_version") != FEATURE_VERSION or bundle.get("features") != FEATURES:
        st.error("저장된 모델의 특징 버전이 다릅니다. 현재 특징 CSV로 다시 학습해 주세요.")
    else:
        x_new = pd.DataFrame([{k: f[k] for k in FEATURES}])
        score = float(np.clip(bundle["model"].predict(x_new)[0], 0, 10))
        st.subheader(f"설문 데이터 기반 예상 불편감: {score:.1f} / 10")
        st.caption("영상 특징과 학습 설문 점수 사이의 관계를 추정한 값입니다. EEG를 직접 입력받거나 생리 반응을 직접 측정하는 값은 아닙니다.")

    row = {"filename": st.session_state.get("filename", "video"), **f}
    data = pd.DataFrame([row]).to_csv(index=False).encode("utf-8-sig")
    st.download_button("영상 특징 CSV 다운로드", data, "video_features_v2.csv", "text/csv")
    st.caption("다운로드한 CSV에 discomfort_score를 추가하세요. 참가자별 평가를 기록할 때는 한 행에 한 번의 시청/평가를 넣고 video_id와 participant_id를 보존하는 것이 좋습니다.")

st.divider()
st.subheader("연구 설계 메모")
st.markdown("""
- `discomfort_score`: 시청 직후 참가자가 직접 평가한 불편감 점수(0~10).
- `video_id`: 같은 영상을 식별하는 ID. 동일 영상의 여러 평가를 그룹으로 묶을 때 사용합니다.
- `participant_id`: 익명 참가자 ID.
- `group_id`: 교차검증에서 분리할 그룹. 기본적으로 `video_id`를 권장합니다.
- `eeg_summary`: EEG 분석 결과를 별도 분석에 연결할 선택 열이며, 현재 예측 모델 입력에는 포함하지 않습니다.
- 같은 영상의 평가가 학습과 검증에 동시에 들어가면 성능이 과대평가될 수 있으므로 그룹 기반 검증을 우선 사용하세요.
""")
st.warning("과학적 검증 전에는 자동 검열·차단·삭제의 단독 근거로 사용하지 마세요. 시각적 특징은 콘텐츠의 의미, 소리, 개인차를 충분히 포착하지 못할 수 있습니다.")
