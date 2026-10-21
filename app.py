import os, tempfile, io
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="MediaCheck AI · 연구용", page_icon="🎬", layout="wide")
FEATURES = [
    "brightness_mean", "brightness_std", "brightness_change", "contrast_mean",
    "saturation_mean", "frame_difference_mean", "frame_difference_p90",
    "scene_change_rate", "motion_proxy", "edge_density"
]
TARGET = "discomfort_score"
MODEL_PATH = "mediacheck_ridge_model.joblib"


def extract_video_features(path, sample_fps=1.5, max_samples=900):
    """프레임을 표본 추출해 시각적 대리 지표를 계산합니다."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다. 파일 형식과 코덱을 확인해 주세요.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    interval = max(1, int(round(fps / sample_fps))) if fps > 0 else 1
    brightness, contrast, saturation, diffs, cuts, motions, edges = [], [], [], [], [], [], []
    prev_gray, idx, n = None, 0, 0
    try:
        while n < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                g = gray.astype(np.float32) / 255.0
                brightness.append(float(np.mean(g)))
                contrast.append(float(np.std(g)))
                saturation.append(float(np.mean(hsv[:, :, 1].astype(np.float32) / 255.0)))
                edges.append(float(np.mean(cv2.Canny(gray, 80, 160) > 0)))
                if prev_gray is not None:
                    d = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
                    diffs.append(d)
                    cuts.append(1.0 if d >= 0.25 else 0.0)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
                    mag = cv2.magnitude(flow[..., 0], flow[..., 1])
                    motions.append(float(np.mean(mag) / max(gray.shape)))
                prev_gray = gray
                n += 1
            idx += 1
    finally:
        cap.release()
    if not brightness:
        raise ValueError("분석할 프레임을 읽지 못했습니다.")
    da = np.asarray(diffs or [0.0], dtype=float)
    ba = np.asarray(brightness, dtype=float)
    return {
        "duration_sec": round(duration, 2), "sampled_frames": int(n),
        "brightness_mean": float(np.mean(ba)), "brightness_std": float(np.std(ba)),
        "brightness_change": float(np.mean(np.abs(np.diff(ba)))) if len(ba) > 1 else 0.0,
        "contrast_mean": float(np.mean(contrast)), "saturation_mean": float(np.mean(saturation)),
        "frame_difference_mean": float(np.mean(da)), "frame_difference_p90": float(np.percentile(da, 90)),
        "scene_change_rate": float(np.mean(cuts)) if cuts else 0.0,
        "motion_proxy": float(np.mean(motions)) if motions else 0.0,
        "edge_density": float(np.mean(edges))
    }


def make_model(alpha=1.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(alpha)))


def prepare_data(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing))
    data = df.copy()
    for c in FEATURES + [TARGET]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET].between(0, 10)]
    data = data.dropna(subset=FEATURES, how="all")
    if len(data) < 5:
        raise ValueError("유효한 설문 응답이 5개 미만입니다. 점수 범위와 데이터를 확인해 주세요.")
    return data


def grouped_cv(X, y, groups):
    n_groups = pd.Series(groups).nunique()
    if n_groups < 2:
        raise ValueError("교차검증에는 서로 다른 video_id가 최소 2개 필요합니다.")
    splitter = LeaveOneGroupOut() if n_groups == 2 else GroupKFold(n_splits=min(5, n_groups))
    pred = cross_val_predict(make_model(), X, y, cv=splitter, groups=groups)
    return pred, n_groups


st.title("🎬 MediaCheck AI · 연구용 시제품")
st.caption("영상 시각 특징 + 참가자 설문 → 회귀모델 학습 및 영상 단위 검증")
with st.expander("연구상 한계와 주의사항", expanded=True):
    st.markdown("""
    - 밝기·움직임·색상은 시각적 특징이지, 공포감이나 불편감 자체가 아닙니다.
    - 모델은 참가자 설문으로 학습해야 하며 데이터의 대표성과 양에 따라 성능이 달라집니다.
    - 동일 영상의 응답은 같은 `video_id`로 묶습니다. 검증 시 영상 단위로 분리해 데이터 누출을 줄입니다.
    - 참가자 동의와 익명화를 지키고, 이 도구를 진단이나 자동 콘텐츠 판정에 사용하지 마세요.
    """)

tab_video, tab_train, tab_help = st.tabs(["🎞️ 영상 분석", "🧪 학습·검증", "📄 CSV 안내"])

with tab_video:
    video = st.file_uploader("분석할 영상 업로드", type=["mp4", "mov", "avi", "mkv"])
    sample_fps = st.select_slider("초당 분석 프레임 수", options=[0.5, 1.0, 1.5, 2.0, 3.0], value=1.5)
    if video:
        st.video(video)
        if st.button("영상 특징 추출", type="primary"):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.name)[1] or ".mp4") as tmp:
                    tmp.write(video.getbuffer())
                    tmp_path = tmp.name
                with st.spinner("영상 프레임을 분석 중입니다..."):
                    st.session_state["features"] = extract_video_features(tmp_path, float(sample_fps))
                    st.session_state["filename"] = video.name
            except Exception as e:
                st.error(f"분석 실패: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
    if "features" in st.session_state:
        f = st.session_state["features"]
        cols = st.columns(4)
        cols[0].metric("길이", f'{f["duration_sec"]:.1f}초')
        cols[1].metric("표본 프레임", str(f["sampled_frames"]))
        cols[2].metric("평균 밝기", f'{f["brightness_mean"]:.3f}')
        cols[3].metric("프레임 변화량", f'{f["frame_difference_mean"]:.3f}')
        st.dataframe(pd.DataFrame([{"특징": k, "값": v} for k, v in f.items()]), use_container_width=True, hide_index=True)
        feature_csv = pd.DataFrame([{"video_id": os.path.splitext(st.session_state["filename"])[0], **f}])
        st.download_button("영상 특징 CSV 다운로드", feature_csv.to_csv(index=False).encode("utf-8-sig"),
                           "video_features.csv", "text/csv")
        model = st.session_state.get("model")
        if model is None and os.path.exists(MODEL_PATH):
            try:
                saved = joblib.load(MODEL_PATH)
                model = saved["model"] if isinstance(saved, dict) else saved
            except Exception:
                model = None
        if model is None:
            st.info("학습된 모델이 없습니다. ‘학습·검증’ 탭에서 설문 데이터로 먼저 학습하세요.")
        else:
            x = pd.DataFrame([{k: f.get(k, np.nan) for k in FEATURES}])
            score = float(np.clip(model.predict(x)[0], 0, 10))
            st.subheader(f"설문 학습 모델 추정치: {score:.1f} / 10")
            st.caption("학습 데이터에서 배운 관계를 적용한 추정치이며 개인의 실제 감정을 확정하지 않습니다.")

with tab_train:
    st.markdown("""
    한 파일로 관리한다면 통합 CSV를 업로드하세요. 각 행은 참가자 한 명이 영상 하나를 평가한 기록입니다.
    분리 파일 방식에서는 설문 CSV와 영상 특징 CSV를 `video_id`로 결합합니다.
    """)
    mode = st.radio("데이터 형식", ["통합 CSV", "설문 CSV + 영상 특징 CSV"], horizontal=True)
    data = None
    if mode == "통합 CSV":
        file = st.file_uploader("통합 학습 CSV", type=["csv"], key="combined")
        if file:
            try:
                data = pd.read_csv(file)
            except Exception as e:
                st.error(f"CSV 읽기 실패: {e}")
    else:
        survey_file = st.file_uploader("설문 CSV", type=["csv"], key="survey")
        feat_file = st.file_uploader("영상 특징 CSV", type=["csv"], key="feat")
        if survey_file and feat_file:
            try:
                survey, video_features = pd.read_csv(survey_file), pd.read_csv(feat_file)
                if "video_id" not in survey or "video_id" not in video_features:
                    st.error("두 파일 모두 video_id 열이 필요합니다.")
                elif video_features["video_id"].duplicated().any():
                    st.error("영상 특징 CSV는 video_id마다 한 행이어야 합니다.")
                elif TARGET not in survey:
                    st.error("설문 CSV에 discomfort_score 열이 필요합니다.")
                else:
                    data = survey.merge(video_features[["video_id"] + [c for c in FEATURES if c in video_features]],
                                        on="video_id", how="inner", validate="many_to_one")
                    st.info(f"결합된 설문 응답: {len(data)}개")
            except Exception as e:
                st.error(f"CSV 결합 실패: {e}")
    alpha = st.slider("Ridge alpha", 0.1, 20.0, 1.0, 0.1)
    if data is not None:
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)
        try:
            clean = prepare_data(data)
            if "video_id" not in clean:
                st.warning("video_id 열이 없습니다. 영상 단위 교차검증을 위해 추가하세요.")
            st.write(f"유효 응답 {len(clean)}개 · 고유 영상 {clean['video_id'].nunique() if 'video_id' in clean else '알 수 없음'}개")
            if st.button("학습 및 교차검증 실행", type="primary"):
                X = clean[FEATURES]
                y = clean[TARGET].astype(float)
                if "video_id" in clean and clean["video_id"].nunique() >= 2:
                    try:
                        pred, ng = grouped_cv(X, y, clean["video_id"].astype(str).to_numpy())
                        c1, c2, c3 = st.columns(3)
                        c1.metric("MAE", f"{mean_absolute_error(y, pred):.2f}점")
                        c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y, pred)):.2f}점")
                        c3.metric("교차검증 R²", f"{r2_score(y, pred):.3f}" if len(y) > 1 else "계산 불가")
                        st.caption(f"video_id 그룹 {ng}개 기준 교차검증입니다. 영상 수가 적으면 지표가 불안정합니다.")
                        result = pd.DataFrame({"video_id": clean["video_id"].astype(str),
                                               "실제 점수": y, "교차검증 예측": np.clip(pred, 0, 10),
                                               "절대 오차": np.abs(y - pred)})
                        st.dataframe(result, use_container_width=True, hide_index=True)
                        st.download_button("교차검증 결과 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                                           "cross_validation_predictions.csv", "text/csv")
                    except Exception as e:
                        st.warning(f"교차검증 실패: {e}")
                else:
                    st.warning("서로 다른 video_id가 2개 이상 있어야 영상 단위 교차검증이 가능합니다.")
                model = make_model(alpha)
                model.fit(X, y)
                payload = {"model": model, "features": FEATURES, "target": TARGET,
                           "alpha": alpha, "n_rows": len(clean),
                           "n_videos": int(clean["video_id"].nunique()) if "video_id" in clean else None}
                joblib.dump(payload, MODEL_PATH)
                st.session_state["model"] = model
                buf = io.BytesIO()
                joblib.dump(payload, buf)
                st.success("최종 모델을 전체 데이터로 학습했습니다. 교차검증 결과와 최종 모델의 학습 성능을 혼동하지 마세요.")
                st.download_button("학습 모델 다운로드 (.joblib)", buf.getvalue(),
                                   "mediacheck_ridge_model.joblib", "application/octet-stream")
        except Exception as e:
            st.error(str(e))

with tab_help:
    st.markdown("**설문 CSV 필수 열**")
    st.code("participant_id,video_id,discomfort_score", language="text")
    st.markdown("**영상 특징 CSV 필수 열**")
    st.code("video_id," + ",".join(FEATURES), language="text")
    st.markdown("**통합 CSV 필수 열**")
    st.code("video_id,participant_id,discomfort_score," + ",".join(FEATURES), language="text")
    st.markdown("""
    - `discomfort_score`: 참가자가 직접 평가한 0~10점.
    - `participant_id`: 익명화한 참가자 코드. 이름 등 식별 정보는 넣지 마세요.
    - 같은 영상은 참가자가 달라도 동일한 `video_id`를 사용하세요.
    - 각 영상 특징은 영상 분석 탭에서 추출해 CSV로 저장할 수 있습니다.
    """)

st.divider()
st.caption("MediaCheck AI 연구용 시제품 · 의료/심리 진단 및 자동 콘텐츠 유해성 판정용이 아닙니다.")
