from pydantic import BaseModel
from typing import Optional


class ResultBase(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    detection_score: float

    variety: Optional[str]
    variety_score: Optional[float]

    maturity: Optional[str]
    maturity_score: Optional[float]


class ResultCreate(ResultBase):
    pass


class ResultResponse(ResultBase):
    id: int

    class Config:
        from_attributes = True