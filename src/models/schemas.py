# src/models/schemas.py
from pydantic import BaseModel
from typing import List, Dict
from enum import Enum

class SentimentType(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    OTHER = "OTHER"

class DetailSentimentType(str, Enum):
    # positive
    JOY = "JOY"
    LOVE = "LOVE"
    GRATITUDE = "GRATITUDE"
    # negative
    ANGER = "ANGER"
    SADNESS = "SADNESS"
    FEAR = "FEAR"
    # other
    NEUTRAL = "NEUTRAL"

class CommentSentimentDetail(BaseModel):
    apiCommentId: str
    content: str
    sentimentType: SentimentType
    detailSentimentTypes: List[DetailSentimentType]  # 0~3개

class AIAnalysisResponse(BaseModel):
    videoId: int
    apiVideoId: str
    summation: str
    isWarning: bool
    keywords: List[str]
    sentimentComments: List[CommentSentimentDetail]
    languageRatio: Dict[str, float]
    sentimentRatio: Dict[str, float]