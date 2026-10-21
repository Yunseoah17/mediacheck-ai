# MediaCheck AI

영상 파일을 업로드하면 시각적 특징을 추출하고 결과를 보여주는 Streamlit 시제품입니다.

## 실행 방법
Python 3.10 또는 3.11 권장.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub에 올리고 공개 웹앱으로 배포하기
1. GitHub에서 새 저장소를 만듭니다.
2. `app.py`, `requirements.txt`, `README.md`를 저장소 루트에 올립니다.
3. Streamlit Community Cloud에 로그인하고 **Create app**에서 해당 저장소와 `app.py`를 선택합니다.
4. 배포가 끝나면 생성된 웹 주소를 발표 시연에 사용합니다.

## 모델 학습 CSV
CSV에는 다음 열이 필요합니다.

- `brightness_mean`
- `brightness_std`
- `brightness_change`
- `frame_difference_mean`
- `scene_change_rate`
- `motion_proxy`
- `discomfort_score`

각 행은 **서로 다른 영상 하나**를 나타내는 것이 가장 좋습니다. 특징값은 앱에서 영상별로 추출한 값과 같은 방식으로 만들어야 합니다. `discomfort_score`는 해당 영상에 대한 실제 참여자 불편감 평균(0~10)입니다.

현재 앱은 CSV 학습 데이터가 없으면 영상 변화량을 조합한 **시연용 지표**를 표시합니다. 이 점수는 실제 시청자의 불편감을 학습한 AI 예측값이 아닙니다. 실제 설문 라벨로 학습한 CSV를 넣어야 회귀모델의 예측 결과로 사용할 수 있습니다.

## 제한 사항
- 현재는 영상의 시각적 특징만 추출합니다. 오디오 음량 분석은 포함되어 있지 않습니다.
- 장면 전환과 움직임은 단순한 프레임 차이 기반 추정치입니다.
- 영상 수가 적으면 모델 성능 추정이 불안정합니다.
- 의료적 판단, 개인 심리 진단, 콘텐츠 자동 검열에 사용하지 마세요.
