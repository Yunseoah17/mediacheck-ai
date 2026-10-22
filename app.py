import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="MediaCheck AI", page_icon="🎬", layout="wide")
st.title("🎬 MediaCheck AI")
st.caption("영상만 입력하면 가설 기반 불쾌도 추정 지표(0~10)를 출력하는 연구 시제품")
st.warning("이 점수는 실제 사람의 반응으로 학습·검증된 AI 예측값이 아니라, 연구 가설을 반영한 규칙 기반 추정치입니다.")

def extract_video_features(path, sample_fps=2, max_samples=600):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다. MP4/MOV 파일인지 확인해 주세요.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    interval = max(1, int(round(fps / sample_fps))) if fps > 0 else 1
    brightness, diffs, cuts, saturation, edges = [], [], [], [], []
    prev_gray = None
    idx = sampled = 0
    while sampled < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            b = float(np.mean(gray) / 255.0)
            brightness.append(b)
            saturation.append(float(np.mean(hsv[:, :, 1]) / 255.0))
            edges.append(float(np.mean(cv2.Canny(gray, 80, 160) > 0)))
            if prev_gray is not None:
                d = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
                diffs.append(d)
                cuts.append(float(d > 0.22))
            prev_gray = gray
            sampled += 1
        idx += 1
    cap.release()
    if not brightness:
        raise ValueError("분석할 프레임을 읽지 못했습니다.")
    b = np.asarray(brightness, dtype=float)
    d = np.asarray(diffs if diffs else [0.0], dtype=float)
    return {
        "영상 길이(초)": round(duration, 2),
        "평균 밝기": float(np.mean(b)),
        "밝기 변동": float(np.mean(np.abs(np.diff(b)))) if len(b) > 1 else 0.0,
        "프레임 변화량 평균": float(np.mean(d)),
        "프레임 변화량 상위 90%": float(np.percentile(d, 90)),
        "장면 변화 추정 비율": float(np.mean(cuts)) if cuts else 0.0,
        "색상 채도 평균": float(np.mean(saturation)),
        "윤곽 밀도": float(np.mean(edges)),
        "샘플 프레임 수": int(sampled),
    }

def estimate_score(f):
    # 가설 기반 규칙: 어두움이 주 요인, 움직임/장면전환은 낮은 가중치.
    darkness = np.clip((0.65 - f["평균 밝기"]) / 0.55, 0, 1)
    brightness_instability = np.clip(f["밝기 변동"] / 0.18, 0, 1)
    motion = np.clip(f["프레임 변화량 평균"] / 0.30, 0, 1)
    cuts = np.clip(f["장면 변화 추정 비율"] / 0.40, 0, 1)
    score = 0.5 + 6.0 * darkness + 1.5 * brightness_instability + 1.0 * motion + 0.5 * cuts
    return float(np.clip(score, 0, 10))

uploaded = st.file_uploader("분석할 영상 업로드", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    st.video(uploaded)
    if st.button("영상 분석 및 점수 산출", type="primary", use_container_width=True):
        tmp_path = None
        try:
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            with st.spinner("영상 특징을 분석하고 있습니다..."):
                features = extract_video_features(tmp_path)
            st.session_state["last_features"] = features
            st.session_state["last_score"] = estimate_score(features)
            st.session_state["last_filename"] = uploaded.name
        except Exception as e:
            st.error(f"분석 오류: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

if "last_score" in st.session_state:
    st.divider()
    st.subheader("분석 결과")
    score = st.session_state["last_score"]
    st.metric("가설 기반 불쾌도 추정 지표", f"{score:.1f} / 10")
    st.progress(int(round(score * 10)))
    st.caption("어두움에 가장 큰 가중치를 두고, 밝기 불안정·움직임·장면 전환은 보조적으로 반영합니다. 실제 불쾌감 예측으로 검증된 점수는 아닙니다.")
    st.subheader("추출된 영상 특징")
    f = st.session_state["last_features"]
    st.dataframe(pd.DataFrame([{"항목": k, "값": round(v, 4) if isinstance(v, (float, int, np.floating, np.integer)) else v} for k,v in f.items()]), use_container_width=True, hide_index=True)
    out = pd.DataFrame([{"filename": st.session_state["last_filename"], **f, "estimated_discomfort_index": score}])
    st.download_button("분석 결과 CSV 다운로드", out.to_csv(index=False).encode("utf-8-sig"), file_name="mediacheck_result.csv", mime="text/csv")

st.divider()
st.caption("연구용 프로토타입입니다. 의료·심리 진단이나 자동 삭제·확정적 유해성 판정에 사용하지 마세요.")
