"""
YouTube 댓글 분석 API 서버
백엔드로부터 댓글 데이터를 받아 AI 분석 결과를 반환합니다.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import asyncio
import logging

# 우리가 만든 분석 파이프라인들 불러오기
from src.pipelines.summarize import summarize_comments_with_gpt
from src.pipelines.sentiment import analyze_sentiment_async
from src.pipelines.keywords import extract_keywords_tfidf
from src.pipelines.lang_ratio import detect_languages
from src.pipelines.controversy import is_video_controversial

# Pydantic 모델 불러오기
from src.models.schemas import (
    AIAnalysisResponse
)

# ============================================================
# 로깅 설정
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 요청(Request) 형식 정의
# ============================================================
class AnalysisRequest(BaseModel):
    """
    백엔드에서 보내는 요청 형식

    예시:
    {
      "videoId": "dQw4w9WgXcQ",
      "comments": {
        "comment_001": "정말 유익한 영상이네요!",
        "comment_002": "최악이에요"
      }
    }
    """
    videoId: str  # YouTube 비디오 ID (예: "dQw4w9WgXcQ")
    comments: Dict[str, str]  # {댓글ID: 댓글내용}


# ============================================================
# FastAPI 앱 생성
# ============================================================
app = FastAPI(
    title="YouTube Comment Analyzer",
    description="유튜브 댓글 종합 분석 API",
    version="2.2.0",
)


# ============================================================
# 비동기 헬퍼 함수
# ============================================================
async def safe_analyze_sentiment(comments_dict: Dict[str, str]):
    try:
        logger.info("[감정 분석] 시작 (번역 포함)")

        result = await asyncio.wait_for(
            analyze_sentiment_async(comments_dict),
            timeout=90.0
        )

        logger.info("[감정 분석] 완료")
        return result

    except asyncio.TimeoutError:
        logger.error("[감정 분석] 타임아웃 (90초)")
        return [], {"positive": 33, "negative": 33, "other": 34}, []

    except Exception as e:
        logger.error(f"[감정 분석] 에러: {e}", exc_info=True)
        return [], {"positive": 33, "negative": 33, "other": 34}, []


async def safe_extract_keywords(comment_texts: List[str], top_n: int = 5):
    """키워드 추출"""
    try:
        logger.info("[키워드 추출] 시작")

        result = await asyncio.wait_for(
            asyncio.to_thread(extract_keywords_tfidf, comment_texts, top_n),
            timeout=30.0
        )

        logger.info("[키워드 추출] 완료")
        return result

    except asyncio.TimeoutError:
        logger.error("[키워드 추출] 타임아웃 (30초)")
        return []

    except Exception as e:
        logger.error(f"[키워드 추출] 에러: {e}", exc_info=True)
        return []


async def safe_detect_languages(comment_texts: List[str]):
    try:
        logger.info("[언어 감지] 시작")

        result = await asyncio.wait_for(
            asyncio.to_thread(detect_languages, comment_texts),
            timeout=20.0
        )

        logger.info("[언어 감지] 완료")
        return result

    except asyncio.TimeoutError:
        logger.error("[언어 감지] 타임아웃 (20초)")
        return {"한국어": 100.0}

    except Exception as e:
        logger.error(f"[언어 감지] 에러: {e}", exc_info=True)
        return {"한국어": 100.0}


async def safe_check_controversy(translated_texts: List[str]):
    try:
        logger.info(f"[논란 감지] 시작: {len(translated_texts)}개 댓글")

        is_warning = await asyncio.wait_for(
            is_video_controversial(translated_texts),
            timeout=30.0
        )

        logger.info(f"[논란 감지] 완료: {'감지됨' if is_warning else '정상'}")
        return is_warning

    except asyncio.TimeoutError:
        logger.error("[논란 감지] 타임아웃 (30초) - False 반환")
        return False

    except Exception as e:
        logger.error(f"[논란 감지] 에러: {e} - False 반환", exc_info=True)
        return False


async def safe_summarize(comment_texts: List[str]):
    try:
        logger.info(f"[요약 생성] 시작: {len(comment_texts)}개 댓글")

        summary = await asyncio.wait_for(
            asyncio.to_thread(summarize_comments_with_gpt, comment_texts[:50]),
            timeout=60.0
        )

        logger.info(f"[요약 생성] 완료: {len(summary)}자")
        return summary

    except asyncio.TimeoutError:
        logger.error("[요약 생성] 타임아웃 (60초)")
        return "댓글 요약을 생성할 수 없습니다."

    except Exception as e:
        logger.error(f"[요약 생성] 에러: {e}", exc_info=True)
        return "댓글 요약을 생성할 수 없습니다."


# ============================================================
# 메인 분석 엔드포인트
# ============================================================
@app.post("/analyze", response_model=AIAnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    유튜브 댓글 종합 분석 API

    [처리 과정]
    1. 감정 분석 (GoEmotions 모델)
    2. 댓글 요약 (GPT)
    3. 키워드 추출 (TF-IDF)
    4. 언어 비율 분석
    5. 논란 감지

    [입력]
    - videoId: YouTube 비디오 ID
    - comments: 댓글 딕셔너리

    [출력]
    - AIAnalysisResponse: 종합 분석 결과
    """

    logger.info("=" * 70)
    logger.info("[새 분석 요청 도착]")
    logger.info("=" * 70)
    logger.info(f"비디오 ID: {request.videoId}")
    logger.info(f"댓글 개수: {len(request.comments)}개")
    logger.info("=" * 70)

    try:
        # 입력 데이터 검증
        video_id = request.videoId
        comments_dict = request.comments

        # 댓글이 없으면 에러
        if not comments_dict:
            raise HTTPException(
                status_code=400,
                detail="댓글 데이터가 없습니다."
            )

        # 댓글 텍스트만 추출 (키워드/요약에 사용)
        comment_texts = list(comments_dict.values())

        # ============================================================
        # PHASE 1: 병렬 분석 (감정 + 키워드 + 언어)
        # ============================================================
        logger.info("[Phase 1/3] 병렬 분석 시작 (감정 + 키워드 + 언어)")

        sentiment_task = safe_analyze_sentiment(comments_dict)  # 번역 포함, 번역 결과도 반환
        keywords_task = safe_extract_keywords(comment_texts, top_n=5)
        language_task = safe_detect_languages(comment_texts)

        results = await asyncio.gather(
            sentiment_task,
            keywords_task,
            language_task,
            return_exceptions=True
        )

        (sentiment_comments, sentiment_ratio, translated_texts), keywords, language_ratio = results

        logger.info(f"감정 분석 완료: 긍정 {sentiment_ratio.get('positive', 0)}%, "
                    f"부정 {sentiment_ratio.get('negative', 0)}%, "
                    f"기타 {sentiment_ratio.get('other', 0)}%")
        logger.info(f"키워드 추출 완료: {len(keywords)}개")
        logger.info(f"언어 감지 완료: {language_ratio}")

        # ============================================================
        # PHASE 2: 댓글 요약 (GPT)
        # ============================================================
        logger.info("[Phase 2/3] 댓글 요약 중 (GPT)")
        summary = await safe_summarize(comment_texts)
        logger.info(f"요약 완료: {len(summary)}자")

        # ============================================================
        # PHASE 3: 논란 감지 (번역된 텍스트 재사용)
        # ============================================================
        logger.info("[Phase 3/3] 논란 감지 중")
        is_warning = await safe_check_controversy(translated_texts)
        logger.info(f"논란 감지 완료: {'감지됨' if is_warning else '정상'}")

        # ============================================================
        # 최종 응답 생성
        # ============================================================
        try:
            video_id_int = int(video_id) if video_id.isdigit() else hash(video_id) % 1000000
        except:
            video_id_int = hash(video_id) % 1000000  # 해시값 사용

        response = AIAnalysisResponse(
            videoId=video_id_int,
            apiVideoId=video_id,
            summation=summary,
            isWarning=is_warning,
            keywords=keywords,
            sentimentComments=sentiment_comments,  # ✅ List[CommentSentimentDetail]
            languageRatio=language_ratio,
            sentimentRatio=sentiment_ratio,
        )

        logger.info("=" * 70)
        logger.info("[분석 완료]")
        logger.info("=" * 70)
        logger.info(f"긍정: {sentiment_ratio.get('positive', 0)}%")
        logger.info(f"부정: {sentiment_ratio.get('negative', 0)}%")
        logger.info(f"기타: {sentiment_ratio.get('other', 0)}%")
        logger.info(f"키워드: {', '.join(keywords)}")
        logger.info(f"논란: {'감지됨' if is_warning else '없음'}")
        logger.info(f"요약: {summary[:50]}...")
        logger.info("=" * 70)

        return response

    except HTTPException as he:
        # 이미 정의된 HTTP 예외는 그대로 전달
        raise he

    except Exception as e:
        logger.error("=" * 70)
        logger.error("[에러 발생]")
        logger.error("=" * 70)
        logger.error(f"에러 상세: {str(e)}", exc_info=True)
        logger.error("=" * 70)

        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}"
        )


# ============================================================
# 서버 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 70)
    logger.info("YouTube Comment Analyzer API 서버 시작")
    logger.info("=" * 70)
    logger.info("로컬: http://localhost:7777")
    logger.info("API 문서: http://localhost:7777/docs")
    logger.info("Redoc: http://localhost:7777/redoc")
    logger.info("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",  # 외부 접속 허용
        port=7777,  # 포트 번호
        reload=True  # 코드 변경시 자동 재시작 (개발 모드)
    )