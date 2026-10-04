"""FastAPI application exposing the news classifier."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.config import LABEL_MAP
from src.predict import NewsClassifier

from .schemas import Article, BatchRequest, BatchResponse, HealthResponse, Prediction

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at start-up
    state["clf"] = NewsClassifier()
    yield
    state.clear()


app = FastAPI(
    title="News Classification API",
    description="Classifies news articles (title + description) into World, Sports, Business or Sci/Tech.",
    version="1.0.0",
    lifespan=lifespan,
)


def _clf() -> NewsClassifier:
    clf = state.get("clf")
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return clf


@app.get("/", tags=["info"])
def root():
    return {
        "service": "News Classification API",
        "docs": "/docs",
        "categories": LABEL_MAP,
        "endpoints": ["/health", "/predict", "/predict/batch"],
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    return HealthResponse(status="ok", model_loaded="clf" in state)


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(article: Article):
    return _clf().predict(article.title, article.description)


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(req: BatchRequest):
    items = [(a.title, a.description) for a in req.articles]
    return BatchResponse(predictions=_clf().predict_batch(items))
