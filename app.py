import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="MediaCheck AI | Research", page_icon="🎬", layout="wide")

# Keep this list synchronized between feature extraction and training CSV.
FEATURES = [
    "duration_sec",
    "brightness_mean",
    "brightness_std",
    "brightness_change_mean",
    "brightness_change_p95",
    "frame_difference_mean",
    "frame_difference_p95",
    "scene_change_rate",
    "motion_proxy",
    "edge_density_mean",
    "color_saturation_mean",
    "color_saturation_std",
]
TARGET = "discomfort_score"
GROUP_COL_CANDIDATES = ["video_id", "video_code", "video", "filename"]

def extract_video_features(path, sample_fps=2, max_samples=900):
    """Extract deterministic, simple visual features. These are not psychological measurements."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다. MP4/MOV/AVI 파일인지 확인해 주세요.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    interval = max(1, round(fps / sample_fps)) if fps > 0 else 1

    brightness, saturation, edges, diffs, sampled = [], [], [], [], []
    prev_gray = None
    idx = 0
    while len(sampled) < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            # Normalized features in approximately 0..1 range.
            brightness.append(float(np.mean(gray) / 255.0))
            saturation.append(float(np.mean(hsv[:, :, 1]) / 255.0))
            edges.append(float(np.mean(cv2.Canny(gray, 80, 160) > 0)))
            if prev_gray is not None:
                diffs.append(float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0))
            sampled.append(idx)
            prev_gray = gray
        idx += 1
    cap.release()

    if not brightness:
        raise ValueError("분석할 프레임을 읽지 못했습니다. 영상 파일을 확인해 주세요.")

    b = np.asarray(brightness, dtype=float)
    s = np.asarray(saturation, dtype=float)
    d = np.asarray(diffs if diffs else [0.0], dtype=float)
    # Threshold is a simple visual-change heuristic, not a validated scene detector.
    return {
        "duration_sec": float(round(duration, 3)),
        "brightness_mean": float(np.mean(b)),
        "brightness_std": float(np.std(b)),
        "brightness_change_mean": float(np.mean(np.abs(np.diff(b)))) if len(b) > 1 else 0.0,
        "brightness_change_p95": float(np.percentile(np.abs(np.diff(b)), 95)) if len(b) > 1 else 0.0,
        "frame_difference_mean": float(np.mean(d)),
        "frame_difference_p95": float(np.percentile(d, 95)),
        "scene_change_rate": float(np.mean(d > 0.22)) if len(d) else 0.0,
        "motion_proxy": float(np.percentile(d, 75)),
        "edge_density_mean": float(np.mean(edges)),
        "color_saturation_mean": float(np.mean(s)),
        "color_saturation_std": float(np.std(s)),
        "sampled_frames": int(len(sampled)),
    }

def build_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0)
    )

def find_group_col(df):
    return next((c for c in GROUP_COL_CANDIDATES if c in df.columns), None)

def validate_dataset(df):
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, None, None, f"필수 열 누락: {', '.join(missing)}"
    clean = df.copy()
    for c in FEATURES + [TARGET]:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna(subset=[TARGET])
    clean = clean[np.isfinite(clean[TARGET])]
    clean = clean[(clean[TARGET] >= 0) & (clean[TARGET] <= 10)]
    if len(clean) < 5:
        return None, None, None, "유효한 라벨 행이 5개 미만입니다. 데이터가 더 필요합니다."
    return clean, clean[FEATURES], clean[TARGET], None

st.title("🎬 MediaCheck AI")
st.caption("연구용 시제품 · 영상 특징 추출 → 자기보고 점수 기반 학습 → 검증")

with st.sidebar:
    st.header("연구 데이터")
    st.write("영상별 특징과 실제 불편감 평균 점수가 포함된 CSV를 업로드하세요.")
    st.caption("참가자별 행이 아니라, 기본 분석은 영상별 평균 점수 한 행을 권장합니다.")
    st.download_button(
        "CSV 템플릿 다운로드",
        data=pd.DataFrame(columns=["video_id"] + FEATURES + [TARGET]).to_csv(index=False).encode("utf-8-sig"),
        file_name="mediacheck_training_template.csv",
        mime="text/csv"
    )
    train_file = st.file_uploader("학습 CSV", type=["csv"], key="training_csv")
    if train_file is not None:
        try:
            raw_df = pd.read_csv(train_file)
            clean, X, y, error = validate_dataset(raw_df)
            if error:
                st.error(error)
            else:
                st.success(f"유효한 라벨 행: {len(clean)}개")
                group_col = find_group_col(clean)
                st.caption(f"영상 그룹 열: {group_col if group_col else '없음 — 행 단위 검증'}")
                if group_col:
                    n_groups = clean[group_col].nunique()
                    st.caption(f"고유 영상 수: {n_groups}개")
                if st.button("모델 학습 및 검증", type="primary"):
                    if group_col and clean[group_col].nunique() >= 3:
                        groups = clean[group_col].astype(str)
                        n_splits = min(5, groups.nunique())
                        cv = GroupKFold(n_splits=n_splits)
                        pred = cross_val_predict(build_model(), X, y, cv=cv, groups=groups)
                        split_note = f"영상 ID 그룹 분할 ({n_splits} folds)"
                    elif len(clean) >= 6:
                        cv = KFold(n_splits=min(5, len(clean)), shuffle=True, random_state=42)
                        pred = cross_val_predict(build_model(), X, y, cv=cv)
                        split_note = "행 단위 K-fold (같은 영상 중복 행이 있으면 누수 위험)"
                    else:
                        pred = None
                        split_note = "데이터가 너무 적어 교차검증을 생략"
                    model = build_model()
                    model.fit(X, y)
                    # Save in Streamlit's writable runtime; persistence across restarts is not guaranteed.
                    joblib.dump(model, "model.joblib")
                    st.session_state["trained_model"] = model
                    st.session_state["model_features"] = FEATURES
                    st.success("모델 학습 완료. 앱 런타임 내에서 사용할 수 있습니다.")
                    if pred is not None:
                        mae = mean_absolute_error(y, pred)
                        rmse = mean_squared_error(y, pred) ** 0.5
                        r2 = r2_score(y, pred) if len(y) >= 3 else float("nan")
                        st.metric("교차검증 MAE", f"{mae:.2f} / 10")
                        a, b = st.columns(2)
                        a.metric("RMSE", f"{rmse:.2f}")
                        b.metric("R²", f"{r2:.2f}" if np.isfinite(r2) else "계산 불가")
                        st.caption(f"검증 방식: {split_note}. 소규모 자료의 점수는 불확실하며 외부 영상에서의 성능을 보장하지 않습니다.")
                        st.session_state["cv_results"] = pd.DataFrame({
                            "actual_discomfort": y.to_numpy(),
                            "cross_validated_prediction": np.clip(pred, 0, 10)
                        })
        except Exception as e:
            st.error(f"CSV 처리 중 오류: {e}")

st.subheader("1. 영상 분석")
uploaded = st.file_uploader("분석할 영상 업로드", type=["mp4", "mov", "avi", "mkv"], key="video_file")
if uploaded:
    st.video(uploaded)
    if st.button("영상 특징 추출", type="primary"):
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                temp_path = tmp.name
            with st.spinner("영상 프레임을 분석하고 있습니다..."):
                feats = extract_video_features(temp_path)
            st.session_state["features"] = feats
            st.session_state["filename"] = uploaded.name
            st.session_state["video_id"] = os.path.splitext(uploaded.name)[0]
        except Exception as e:
            st.error(str(e))
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if "features" in st.session_state:
    f = st.session_state["features"]
    st.subheader("2. 추출된 영상 특징")
    st.info("아래 수치는 밝기·시각적 변화 등 영상의 물리적 특징입니다. 불편감이나 유해성을 직접 측정한 값이 아닙니다.")
    st.metric("영상 길이", f'{f["duration_sec"]:.1f}초')
    view_keys = [k for k in FEATURES if k in f]
    feature_df = pd.DataFrame({"feature": view_keys, "value": [f[k] for k in view_keys]})
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    feature_csv = pd.DataFrame([{"video_id": st.session_state.get("video_id", "video"), **{k:f[k] for k in FEATURES}}])
    st.download_button("학습용 특징 CSV 다운로드", feature_csv.to_csv(index=False).encode("utf-8-sig"),
                       file_name="video_features_for_labeling.csv", mime="text/csv")

    st.subheader("3. 점수 출력")
    model = st.session_state.get("trained_model")
    if model is None and os.path.exists("model.joblib"):
        try:
            model = joblib.load("model.joblib")
        except Exception:
            model = None
    if model is not None:
        x_one = pd.DataFrame([{k: f[k] for k in FEATURES}])
        score = float(np.clip(model.predict(x_one)[0], 0, 10))
        st.metric("모델 예상 불편감", f"{score:.1f} / 10")
        st.warning("이 값은 업로드된 설문 라벨로 학습한 모델의 추정치입니다. 의료·심리 진단이나 확정적인 유해성 판정이 아닙니다.")
    else:
        st.warning("아직 학습된 모델이 없습니다. 현재는 실제 불편감 점수를 출력하지 않습니다.")
        st.write("왼쪽에서 실제 설문 라벨이 포함된 CSV로 모델을 학습한 뒤 다시 확인하세요.")
    st.caption("‘검토 필요’ 기준은 실제 검증 데이터에서 정한 임계값이 있을 때만 설정하세요. 현재는 자동 삭제/검열 기능을 실행하지 않습니다.")

if "cv_results" in st.session_state:
    st.subheader("4. 교차검증 결과")
    st.dataframe(st.session_state["cv_results"], use_container_width=True, hide_index=True)
    st.download_button("검증 결과 CSV 다운로드",
                       st.session_state["cv_results"].to_csv(index=False).encode("utf-8-sig"),
                       file_name="cross_validation_results.csv", mime="text/csv")

st.divider()
st.caption("연구 시제품입니다. 소규모·편향된 표본에서 학습한 결과를 일반적인 인간 반응으로 일반화하지 마세요. 뇌파 특징은 실험 분석에 사용할 수 있지만, 이 최종 모델은 영상 특징만 입력받습니다.")
