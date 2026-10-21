# MediaCheck AI 연구용 시제품

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 데이터 형식
통합 CSV 각 행은 참가자 한 명이 영상 하나를 평가한 기록입니다. 필수 열은 `video_id`, `discomfort_score`와 app.py에 정의된 영상 특징 10개입니다. `participant_id`는 익명화한 코드로 사용하는 것을 권장합니다.

분리 CSV를 쓸 경우 설문 파일은 `participant_id, video_id, discomfort_score`, 영상 특징 파일은 `video_id`와 특징 10개를 포함합니다. 특징 파일은 video_id마다 한 행이어야 합니다.

## 주의
영상 특징은 시각적 대리 지표일 뿐 감정을 직접 측정하지 않습니다. 검증은 video_id 그룹 단위로 수행합니다. 표본이 적으면 평가 지표가 불안정하며, 연구용 외 진단이나 자동 판정에 사용하지 마세요.
