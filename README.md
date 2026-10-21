# MediaCheck AI — 연구용 시제품

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 사용 순서
1. 앱에서 영상을 분석하고 `video_features_v2.csv`를 다운로드합니다.
2. 참가자가 영상을 본 직후 불편감 자기보고 점수(0~10)를 기록합니다.
3. 각 평가 행에 `discomfort_score`를 추가하고, 가능하면 `video_id`, `participant_id`를 보존합니다.
4. 여러 참가자/영상의 데이터를 합쳐 CSV로 만들고 앱 사이드바에 올린 뒤 `모델 학습 및 검증`을 누릅니다.
5. 새 영상의 특징을 추출하면 학습된 모델이 예상 점수를 출력합니다.

필수 열: `brightness_mean, brightness_std, brightness_change, frame_difference_mean, frame_difference_p90, scene_change_rate, motion_proxy, dark_frame_rate, high_change_rate, low_brightness_motion, duration_sec, discomfort_score`

선택 열: `video_id`, `participant_id`, `group_id`, `eeg_summary`

## EEG 관련 한계
현재 예측 모델은 EEG를 입력받지 않습니다. EEG와 자기보고 설문을 함께 측정하면 불편감 라벨의 타당성을 연구할 수 있지만, EEG에서 나타난 특징을 영상 특징만으로 자동 추정할 수 있다고 곧바로 결론 내릴 수는 없습니다. 영상 특징과 설문 점수로 모델을 학습하고, 분리된 영상/참가자 데이터에서 성능을 검증해야 합니다. EEG 분석은 별도 단계로 수행해 설문 점수와의 관계를 평가하고 영상 특징 모델과 비교하세요.

## 검증 주의
- 같은 영상에 대한 여러 참가자 평가가 있다면 `video_id`를 그룹 열로 써서 동일 영상이 학습/검증 양쪽에 섞이지 않도록 하세요.
- 표본 수가 적으면 MAE/RMSE가 매우 불안정합니다.
- 특징은 밝기와 시각 변화 중심이며 의미, 맥락, 소리, 개인차를 충분히 표현하지 못합니다.
- 검증 전에는 자동 검열/차단의 단독 근거로 사용하지 마세요.
