import os
import tempfile
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="MediaCheck AI | Research", page_icon="🎬", layout="wide")

# Features computed from the video only. They are not direct measures of human discomfort.
FEATURES = [
    "brightness_mean", "brightness_std", "brightness_change_mean",
    "frame_diff_mean", "frame_diff_p90", "high_change_rate",
    "edge_density_mean", "color_saturation_mean", "color_saturation_std",
    "dark_frame_rate", "bright_frame_rate", "duration_sec"
]
TARGET = "discomfort_score"
MODEL_PATH = "mediacheck_model.joblib"
MODEL_META_PATH = "mediacheck_model_meta.json"

st.title("🎬 MediaCheck AI — 연구용 시제품")
st.caption("영상 시각 특징 추출 · 설문 라벨 기반 회귀 학습 · 검증 | 의료/심리 진단 도구가 아닙니다.")

st.info(
    "중요: 실제 시청자 설문으로 학습된 모델이 없으면 영상의 불편감을 신뢰성 있게 예측할 수 없습니다. "
    "이 앱은 임의 점수를 실제 예측값처럼 표시하지 않습니다."
)

@st.cache_data(show_spinner=False)
def analyze_video_bytes(video_bytes, suffix, sample_fps=2, max_samples=900):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다. 다른 MP4 파일로 시도해 주세요.")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        interval = max(1, int(round(fps / sample_fps))) if fps > 0 else 1

        brightness, diffs, edges, saturations = [], [], [], []
        high_change, dark, bright = [], [], []
        prev_gray = None
        idx = 0
        sampled = 0

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
                saturations.append(float(np.mean(hsv[:, :, 1]) / 255.0))
                edges.append(float(np.mean(cv2.Canny(gray, 80, 160) > 0)))
                dark.append(float(b < 0.12))
                bright.append(float(b > 0.88))
                if prev_gray is not None:
                    d = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
                    diffs.append(d)
                    high_change.append(float(d > 0.22))
                prev_gray = gray
                sampled += 1
            idx += 1
        cap.release()

        if not brightness:
            raise ValueError("분석 가능한 프레임이 없습니다.")

        diff = np.asarray(diffs if diffs else [0.0], dtype=float)
        b_arr = np.asarray(brightness, dtype=float)
        return {
            "duration_sec": round(float(duration), 3),
            "brightness_mean": float(np.mean(b_arr)),
            "brightness_std": float(np.std(b_arr)),
            "brightness_change_mean": float(np.mean(np.abs(np.diff(b_arr)))) if len(b_arr) > 1 else 0.0,
            "frame_diff_mean": float(np.mean(diff)),
            "frame_diff_p90": float(np.percentile(diff, 90)),
            "high_change_rate": float(np.mean(high_change)) if high_change else 0.0,
            "edge_density_mean": float(np.mean(edges)),
            "color_saturation_mean": float(np.mean(saturations)),
            "color_saturation_std": float(np.std(saturations)),
            "dark_frame_rate": float(np.mean(dark)),
            "bright_frame_rate": float(np.mean(bright)),
            "sampled_frames": int(sampled),
            "source_fps": round(fps, 3),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=2.0))
    ])

def normalize_labels(df):
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV에 필요한 열이 없습니다: " + ", ".join(missing))
    out = df.copy()
    for col in FEATURES + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[TARGET])
    out = out[out[TARGET].between(0, 10)]
    out = out.dropna(subset=FEATURES, how="all")
    if len(out) < 5:
        raise ValueError("유효한 데이터가 5개 미만입니다. 모델 학습보다 데이터 정리를 먼저 해 주세요.")
    return out

def cv_evaluate(df):
    X = df[FEATURES]
    y = df[TARGET]
    n = len(df)
    if n < 6:
        return None, "검증에 사용할 데이터가 너무 적습니다(최소 6개 권장)."
    # If multiple rows belong to the same video, split by video ID to avoid leakage.
    if "video_id" in df.columns and df["video_id"].nunique() >= 3:
        groups = df["video_id"].astype(str)
        n_splits = min(5, groups.nunique())
        splitter = GroupKFold(n_splits=n_splits)
        preds = cross_val_predict(make_model(), X, y, cv=splitter, groups=groups)
        method = f"영상 단위 GroupKFold ({n_splits} folds)"
    else:
        n_splits = min(5, n)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        preds = cross_val_predict(make_model(), X, y, cv=splitter)
        method = f"KFold ({n_splits} folds)"
        if "video_id" not in df.columns:
            method += " — 각 행이 서로 다른 영상인지 확인 필요"
    baseline = np.full(len(y), float(y.mean()))
    metrics = {
        "method": method,
        "n_rows": int(n),
        "n_videos": int(df["video_id"].nunique()) if "video_id" in df.columns else int(n),
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "baseline_mae": float(mean_absolute_error(y, baseline)),
        "r2": float(r2_score(y, preds)) if len(np.unique(y)) > 1 else None,
        "actual": y.to_numpy(),
        "predicted": np.clip(preds, 0, 10),
    }
    return metrics, None

with st.sidebar:
    st.header("실험 데이터로 모델 학습")
    st.write("CSV는 영상 특징과 실제 참여자 불편감 점수를 포함해야 합니다.")
    st.caption("필수 열:")
    st.code(", ".join(FEATURES + [TARGET]), language="text")
    st.caption("선택 열: video_id (같은 영상의 여러 참여자 행을 묶어 검증할 때 사용)")
    training_file = st.file_uploader("학습용 CSV 업로드", type=["csv"], key="training_csv")
    if training_file is not None:
        try:
            raw_df = pd.read_csv(training_file)
            train_df = normalize_labels(raw_df)
            st.success(f"유효한 행: {len(train_df)}개")
            st.caption("각 행은 참가자 1명×영상 1개여도 되지만, 검증은 반드시 영상 단위로 분리해야 합니다.")
            if st.button("검증 후 모델 학습", type="primary"):
                metrics, err = cv_evaluate(train_df)
                if err:
                    st.error(err)
                else:
                    st.session_state["cv_metrics"] = metrics
                    st.session_state["cv_actual"] = metrics.pop("actual").tolist()
                    st.session_state["cv_predicted"] = metrics.pop("predicted").tolist()
                    model = make_model()
                    model.fit(train_df[FEATURES], train_df[TARGET])
                    joblib.dump(model, MODEL_PATH)
                    meta = {
                        "features": FEATURES,
                        "target": TARGET,
                        "training_rows": int(len(train_df)),
                        "unique_videos": int(train_df["video_id"].nunique()) if "video_id" in train_df.columns else int(len(train_df)),
                        "note": "Research prototype; not validated for safety or clinical use."
                    }
                    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    st.session_state["model"] = model
                    st.success("검증을 마치고 모델을 학습했습니다.")
        except Exception as e:
            st.error(f"CSV 처리 오류: {e}")

tab_video, tab_data, tab_help = st.tabs(["영상 분석", "학습/검증 결과", "CSV 준비 방법"])

with tab_video:
    video = st.file_uploader("분석할 영상 업로드", type=["mp4", "mov", "avi", "mkv"], key="video_upload")
    if video is not None:
        st.video(video)
        if st.button("영상 특징 추출", type="primary"):
            try:
                suffix = os.path.splitext(video.name)[1] or ".mp4"
                with st.spinner("영상 프레임을 분석하고 있습니다..."):
                    feats = analyze_video_bytes(video.getvalue(), suffix)
                st.session_state["current_features"] = feats
                st.session_state["current_filename"] = video.name
            except Exception as e:
                st.error(f"분석 실패: {e}")

    if "current_features" in st.session_state:
        f = st.session_state["current_features"]
        st.subheader("추출된 영상 특징")
        c1, c2, c3 = st.columns(3)
        c1.metric("영상 길이", f'{f["duration_sec"]:.1f}초')
        c2.metric("평균 밝기", f'{f["brightness_mean"]:.3f}')
        c3.metric("프레임 변화량 평균", f'{f["frame_diff_mean"]:.3f}')
        c4, c5, c6 = st.columns(3)
        c4.metric("밝기 변화량", f'{f["brightness_change_mean"]:.3f}')
        c5.metric("큰 프레임 변화 비율", f'{f["high_change_rate"]:.1%}')
        c6.metric("가장 어두운 프레임 비율", f'{f["dark_frame_rate"]:.1%}')

        model = st.session_state.get("model")
        if model is None and os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
            except Exception:
                model = None

        if model is not None:
            X_new = pd.DataFrame([{col: f[col] for col in FEATURES}])
            score = float(np.clip(model.predict(X_new)[0], 0, 10))
            st.subheader(f"모델 예측 불편감 점수: {score:.1f} / 10")
            st.caption("설문 데이터로 학습한 모델의 출력입니다. 이 점수는 개인의 반응이나 콘텐츠 유해성을 확정하지 않습니다.")
            if score >= 6:
                st.warning("검토 후보: 설정한 점수 기준 이상입니다. 자동 검열이나 삭제가 아닌 사람의 검토를 권장합니다.")
            else:
                st.info("현재 기준에서는 검토 후보 점수 미만입니다. 불편감이 없거나 안전하다는 의미는 아닙니다.")
        else:
            st.warning("예측 모델이 아직 없습니다. 아래는 영상 특징 분석만 완료된 상태입니다.")
            st.write("**불편감 예측: 산출 불가** — 실제 설문 라벨로 학습된 모델을 먼저 만들어야 합니다.")
            st.caption("특징값만으로 임의의 불편감 점수를 만들지 않습니다. 그렇게 만든 숫자는 검증된 예측이 아닙니다.")

        shown = {k: v for k, v in f.items()}
        st.dataframe(pd.DataFrame([{"특징": k, "값": round(float(v), 4) if isinstance(v, (float, int, np.floating, np.integer)) else v} for k,v in shown.items()]), use_container_width=True, hide_index=True)
        export_features = {k: f[k] for k in FEATURES}
        export_features["video_id"] = os.path.splitext(st.session_state.get("current_filename", "video"))[0]
        csv = pd.DataFrame([export_features]).to_csv(index=False).encode("utf-8-sig")
        st.download_button("학습용 특징 CSV 다운로드", csv, file_name="video_features_for_labels.csv", mime="text/csv")
        st.caption("이 CSV에는 설문 정답 점수가 없습니다. 참여자 설문을 수집해 discomfort_score 열을 추가해야 학습할 수 있습니다.")

with tab_data:
    st.subheader("검증 결과")
    if "cv_metrics" not in st.session_state:
        st.write("학습용 CSV를 올리고 사이드바에서 '검증 후 모델 학습'을 누르면 결과가 표시됩니다.")
    else:
        m = st.session_state["cv_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f'{m["mae"]:.2f}점')
        c2.metric("RMSE", f'{m["rmse"]:.2f}점')
        c3.metric("평균값 기준 MAE", f'{m["baseline_mae"]:.2f}점')
        st.write(f'검증 방법: {m["method"]}')
        st.write(f'행 수: {m["n_rows"]} · 영상 수(추정/고유): {m["n_videos"]}')
        if m["r2"] is not None:
            st.write(f'R²: {m["r2"]:.3f}')
        if m["mae"] >= m["baseline_mae"]:
            st.warning("교차검증에서 모델이 단순 평균 예측보다 낫지 않았습니다. 예측 성능이 입증되지 않았으므로 특징/데이터를 개선해야 합니다.")
        else:
            st.info("교차검증에서 평균값 기준보다 MAE가 낮았습니다. 다만 소규모 표본 결과이며 새로운 영상에서의 성능을 보장하지 않습니다.")
        plot_df = pd.DataFrame({"실제 불편감": st.session_state["cv_actual"], "교차검증 예측": st.session_state["cv_predicted"]})
        st.line_chart(plot_df)
        st.caption("주의: 같은 영상의 여러 참여자 기록이 있다면 video_id를 반드시 제공해야 영상 단위 누출을 줄일 수 있습니다.")

with tab_help:
    st.subheader("CSV는 어떻게 준비하나요?")
    st.markdown("""
1. 각 영상 파일을 이 앱에 넣고 **학습용 특징 CSV 다운로드**를 누릅니다.
2. 영상마다 추출한 특징값을 하나의 표에 모읍니다.
3. 실제 시청자 설문에서 해당 영상의 불편감 점수(0~10)를 구해 `discomfort_score` 열에 입력합니다.
4. `video_id` 열에는 영상마다 고유한 ID를 넣습니다. 같은 영상에 대한 여러 참가자 기록은 같은 ID를 사용합니다.
5. CSV를 사이드바에 올려 **검증 후 모델 학습**을 누릅니다.

가능하면 참가자별 행을 유지하세요. 단, 같은 영상의 여러 참가자 행이 학습과 평가에 나뉘어 들어가면 성능이 과대평가될 수 있으므로 검증은 영상 단위로 분리해야 합니다.
""")
    st.code("video_id," + ",".join(FEATURES) + ",discomfort_score\\nclip_01,...,4.5", language="text")
    st.warning("현재 앱의 특징 추출은 시각 정보 중심이며 오디오 음량, 의미/상황 이해, 뇌파 분석은 포함하지 않습니다. 영상 특징만으로 불편감을 완벽히 판단할 수 없습니다.")

st.divider()
st.caption("연구용 프로토타입입니다. 의료·심리 진단, 개인 상태 판정, 자동 삭제 또는 확정적인 콘텐츠 검열에 사용하지 마세요.")
