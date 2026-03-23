from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.schemas.result_schema import ResultCreate, ResultResponse


class HistoryBase(BaseModel):
    image: str
    report: Optional[str]


class HistoryCreate(HistoryBase):
    results: List[ResultCreate]


class HistoryResponse(HistoryBase):
    id: int
    user_id: int
    created_at: datetime

    results: List[ResultResponse] = []

    class Config:
        from_attributes = True
        
        
class HistoryListResponse(BaseModel):
    id: int
    image: str
    created_at: datetime

    class Config:
        from_attributes = True