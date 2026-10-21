# MediaCheck AI — Research prototype

## 기능
- 영상에서 밝기, 밝기 변화, 프레임 차이, 장면 전환 추정, 색 채도, 에지 밀도 등 시각 특징 추출
- 실제 설문 불편감 점수(0~10)를 라벨로 사용하는 Ridge 회귀
- 교차검증 MAE/RMSE/R² 표시
- 영상 ID가 있으면 GroupKFold로 같은 영상의 행이 학습/검증에 섞이는 위험을 줄임
- 모델이 없을 때 가짜 점수를 출력하지 않음

## 실행
`pip install -r requirements.txt`
`streamlit run app.py`

## 학습 CSV
필수 열: `video_id` (권장), 아래 특징 열, `discomfort_score`

- duration_sec
- brightness_mean
- brightness_std
- brightness_change_mean
- brightness_change_p95
- frame_difference_mean
- frame_difference_p95
- scene_change_rate
- motion_proxy
- edge_density_mean
- color_saturation_mean
- color_saturation_std
- discomfort_score

가장 좋은 구조는 영상별 한 행으로 만들고, 각 영상에 대해 여러 참여자의 불편감 점수를 평균내어 `discomfort_score`로 넣는 것입니다. 특징 열은 앱이 추출한 것과 동일한 방식으로 계산되어야 합니다. 앱에서 영상 특징 CSV를 내려받아 실제 설문 점수를 추가해도 됩니다.

## 중요한 제한
이 코드는 학습 알고리즘을 제공하지만, 학습 데이터를 포함하지 않습니다. 실제 설문 데이터 없이 실제 불편감을 잘 예측한다고 보장할 수 없습니다. 임의의 점수나 합성 라벨로 학습한 결과를 실제 정확도로 주장하지 마세요. EEG는 연구에서 자기보고 점수와의 관계를 분석할 수 있으나, 최종 영상-only 모델의 입력은 영상 특징뿐입니다. 검열·삭제 대신 검토 보조로 사용하세요.
