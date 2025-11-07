from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

from src.pipelines.summarize import summarize_comments_with_gpt
from src.pipelines.sentiment import analyze_sentiment_async
from src.pipelines.keywords import extract_keywords_tfidf
from src.pipelines.lang_ratio import detect_languages
from src.pipelines.controversy import is_video_controversial


class AnalysisRequest(BaseModel):
    videoId: str
    comments: Dict[str, str]


class AnalysisResponse(BaseModel):
    videoId: Optional[str]
    apiVideoId: str
    summation: str
    isWarning: bool
    keywords: List[str]
    sentimentComments: Dict[str, str]
    languageRatio: Dict[str, float]
    sentimentRatio: Dict[str, float]


app = FastAPI(
    title="YouTube Comment Analyzer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "YouTube Comment Analyzer API", "status": "running"}


@app.post("/analyze", response_model=AnalysisResponse)
@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    print("/analyze 요청 도착")

    try:
        video_id = request.videoId
        comments_dict = request.comments

        if not comments_dict:
            raise HTTPException(status_code=400, detail="댓글 데이터가 없습니다.")

        comment_texts = list(comments_dict.values())

        summary = summarize_comments_with_gpt(comment_texts)

        sentiment_comments, sentiment_ratio = await analyze_sentiment_async(comments_dict)

        keywords = extract_keywords_tfidf(comment_texts, top_n=5)

        language_ratio = detect_languages(comment_texts)

        is_warning = await is_video_controversial(comment_texts)

        response = AnalysisResponse(
            videoId=None,
            apiVideoId=video_id,
            summation=summary,
            isWarning=is_warning,
            keywords=keywords,
            sentimentComments=sentiment_comments,
            languageRatio=language_ratio,
            sentimentRatio=sentiment_ratio,
        )

        print("분석 완료, 응답 반환 중...")
        return response

    except Exception as e:
        print(f"[ERROR] 분석 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")