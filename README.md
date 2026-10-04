# News Classification System

Classifies news articles (title + description) into **World, Sports, Business, Sci/Tech** using
**TF-IDF + Logistic Regression**, served with **FastAPI** and deployable on **Render**.

| Class ID | Category |
|---|---|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

## Results (provided test set, 7,600 articles)

| Metric | Value |
|---|---|
| Accuracy | **92.43%** |
| Macro F1 | **92.42%** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| World | 0.936 | 0.917 | 0.926 |
| Sports | 0.961 | 0.985 | 0.973 |
| Business | 0.898 | 0.893 | 0.896 |
| Sci/Tech | 0.902 | 0.903 | 0.902 |

Sports is easiest (distinct vocabulary); most errors are Business <-> Sci/Tech, which overlap heavily
(tech-company news). See `reports/` for the confusion matrix and full metrics.

## Approach

1. **Input**: title and description are concatenated (title repeated once for extra weight).
2. **Preprocessing** (`src/preprocess.py`): HTML-entity decoding, tag/URL removal, removal of the
   literal `\` artefacts in the dataset, lowercasing, punctuation removal, whitespace normalisation.
3. **Features**: TF-IDF with unigrams + bigrams, English stop-words removed, `min_df=2`,
   `max_df=0.9`, sublinear TF, up to 300k features.
4. **Model**: Logistic Regression; `C` tuned over `[1, 5, 10, 20]` on a stratified 10% validation split,
   then refit on the full training set (best `C=5`).
5. **Evaluation**: accuracy, macro-F1, per-class report and confusion matrix on `test.csv`.

Preprocessing is part of the saved sklearn `Pipeline`, so training and serving can never drift apart.

## Project structure

```
news-classifier/
├── app/
│   ├── main.py            # FastAPI app (endpoints, model loaded once at startup)
│   └── schemas.py         # Pydantic request/response models
├── src/
│   ├── config.py          # paths, label map, hyper-parameters
│   ├── data_loader.py     # dataset loading + validation
│   ├── preprocess.py      # text cleaning
│   ├── model.py           # TF-IDF + LogisticRegression pipeline
│   ├── train.py           # tune -> refit -> evaluate -> save
│   ├── evaluate.py        # metrics + confusion matrix
│   └── predict.py         # inference wrapper
├── models/news_classifier.joblib   # trained model (13 MB, committed so deploys need no training)
├── reports/               # metrics, confusion matrix, tuning results, training log
├── tests/                 # preprocessing + API tests
├── data/                  # put train.csv and test.csv here (not committed)
├── Dockerfile, render.yaml, requirements*.txt
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + model status |
| POST | `/predict` | Classify one article |
| POST | `/predict/batch` | Classify up to 100 articles |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Apple unveils new chip for MacBook", "description": "The processor promises faster AI performance."}'
```
```json
{
  "class_id": 4,
  "category": "Sci/Tech",
  "confidence": 0.9774,
  "probabilities": {"World": 0.0024, "Sports": 0.001, "Business": 0.0192, "Sci/Tech": 0.9774}
}
```
Interactive docs (Swagger UI) are available at `/docs`.

