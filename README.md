# Credit Default Risk — Streamlit Deployment

## Files
- `app.py` — Streamlit app. Loads `model.pkl` (+ optional `vectorizer.pkl`/preprocessor) and serves predictions.
- `train_model.py` — Optional template to actually train and produce `model.pkl` + `vectorizer.pkl` from `application_data.csv`. Not required if you already have your own pickles.
- `requirements.txt` — Python dependencies.

## What you still need to add
1. **`model.pkl`** — your trained classifier (e.g. from `train_model.py`, or your own training code).
2. **`vectorizer.pkl`** (optional) — your fitted preprocessor/encoder (e.g. a `ColumnTransformer`, `StandardScaler`, or `OneHotEncoder`). Note: since this is tabular data, not text, "vectorizer" here really means "preprocessor" — rename if you like, just update `VECTORIZER_PATH` in `app.py`.
3. Update `build_input_frame()` in `app.py` so the input columns/order exactly match what your model/preprocessor was trained on.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder (with `model.pkl` and `vectorizer.pkl` included) to a GitHub repo.
2. Go to https://share.streamlit.io, connect the repo, and set `app.py` as the entry point.
3. Streamlit Cloud installs `requirements.txt` automatically and deploys.

Keep pickle files under GitHub's 100MB file limit — if your model is larger, use Git LFS or load it from cloud storage (S3/GCS) inside `app.py` instead of committing it directly.
