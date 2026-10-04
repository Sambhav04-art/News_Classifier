"""Pydantic request / response models."""
from pydantic import BaseModel, Field


class Article(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, examples=["Wall St. rallies as oil prices fall"])
    description: str = Field("", max_length=5000, examples=["Stocks closed higher on Friday as investors cheered cooling inflation data."])


class BatchRequest(BaseModel):
    articles: list[Article] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    class_id: int = Field(..., description="1=World, 2=Sports, 3=Business, 4=Sci/Tech")
    category: str
    confidence: float
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
