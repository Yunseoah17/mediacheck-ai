# MediaCheck AI — Research Prototype

This app extracts video-only visual features, trains a Ridge regression model from human-reported discomfort scores, and evaluates it with cross-validation.

## Deploy
Upload `app.py` and `requirements.txt` to the root of your GitHub repository and deploy with Streamlit Community Cloud.

## Important
No pretrained discomfort predictor is included. Without real human-rated labels, the app reports that prediction is unavailable instead of inventing a score. A model trained on a small dataset is not validated for content moderation, clinical use, or safety decisions.

## Training CSV columns
Required: `brightness_mean`, `brightness_std`, `brightness_change_mean`, `frame_diff_mean`, `frame_diff_p90`, `high_change_rate`, `edge_density_mean`, `color_saturation_mean`, `color_saturation_std`, `dark_frame_rate`, `bright_frame_rate`, `duration_sec`, `discomfort_score`.
Recommended: `video_id`, identical for rows from the same video, so cross-validation can group all rows for one video together.
