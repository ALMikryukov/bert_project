from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос по документации")


class AskResponse(BaseModel):
    answer: str
    found: bool
    source: Optional[Dict[str, Any]] = None
    scores: Optional[Dict[str, float]] = None
