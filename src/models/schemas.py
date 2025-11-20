# src/models/schemas.py
from pydantic import BaseModel
from typing import List, Dict
from enum import Enum

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    OTHER = "other"

class DetailSentimentType(str, Enum):
    # positive
    JOY = "joy"
    LOVE = "love"
    GRATITUDE = "gratitude"
    # negative
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    # other
    NEUTRAL = "neutral"

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
    languageRatio: Dict[str, int]
    sentimentRatio: Dict[str, int]

