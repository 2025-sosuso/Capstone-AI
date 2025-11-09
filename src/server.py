"""
YouTube 댓글 분석 API 서버
백엔드로부터 댓글 데이터를 받아 AI 분석 결과를 반환합니다.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

# 우리가 만든 분석 파이프라인들 불러오기
from src.pipelines.summarize import summarize_comments_with_gpt
from src.pipelines.sentiment import analyze_sentiment_async
from src.pipelines.keywords import extract_keywords_tfidf
from src.pipelines.lang_ratio import detect_languages
from src.pipelines.controversy import is_video_controversial

# Pydantic 모델 불러오기
from src.models.schemas import (
    SentimentType,
    DetailSentimentType,
    CommentSentimentDetail,
    AIAnalysisResponse
)


# ============================================================
# 📥 요청(Request) 형식 정의
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
# 🚀 FastAPI 앱 생성
# ============================================================
app = FastAPI(
    title="YouTube Comment Analyzer",
    description="유튜브 댓글 종합 분석 API",
    version="1.0.0",
)

# CORS 설정 (다른 도메인에서도 API 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


# ============================================================
# 🏠 루트 엔드포인트 (서버 상태 확인용)
# ============================================================
@app.get("/")
async def root():
    """
    서버가 정상 작동하는지 확인
    
    브라우저에서 접속하면:
    {"message": "YouTube Comment Analyzer API", "status": "running"}
    """
    return {
        "message": "YouTube Comment Analyzer API",
        "status": "running"
    }


# ============================================================
# 📊 메인 분석 엔드포인트
# ============================================================
@app.post("/analyze", response_model=AIAnalysisResponse)
@app.post("/analyze/", response_model=AIAnalysisResponse)  # 슬래시 있어도 작동
async def analyze(request: AnalysisRequest):
    """
    📌 유튜브 댓글 종합 분석 API
    
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
    
    # ============================================================
    # 📝 요청 정보 로그
    # ============================================================
    print("\n" + "=" * 70)
    print("🔔 [새 분석 요청 도착]")
    print("=" * 70)
    print(f"📹 비디오 ID: {request.videoId}")
    print(f"💬 댓글 개수: {len(request.comments)}개")
    print("=" * 70 + "\n")

    try:
        # ============================================================
        # 🔍 입력 데이터 검증
        # ============================================================
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
        # 🎭 STEP 1: 감정 분석 (가장 오래 걸림)
        # ============================================================
        print("🎭 [1/5] 감정 분석 중...")
        sentiment_comments, sentiment_ratio = await analyze_sentiment_async(comments_dict)
        print(f"   ✅ 완료: 긍정 {sentiment_ratio.get('POSITIVE', 0):.1f}%, "
              f"부정 {sentiment_ratio.get('NEGATIVE', 0):.1f}%, "
              f"기타 {sentiment_ratio.get('OTHER', 0):.1f}%")

        # ============================================================
        # 📝 STEP 2: 댓글 요약
        # ============================================================
        print("📝 [2/5] 댓글 요약 중...")
        summary = summarize_comments_with_gpt(comment_texts)
        print(f"   ✅ 완료: {len(summary)}자 요약 생성")

        # ============================================================
        # 🔑 STEP 3: 키워드 추출
        # ============================================================
        print("🔑 [3/5] 키워드 추출 중...")
        keywords = extract_keywords_tfidf(comment_texts, top_n=5)
        print(f"   ✅ 완료: {len(keywords)}개 키워드 추출")

        # ============================================================
        # 🌍 STEP 4: 언어 비율 분석
        # ============================================================
        print("🌍 [4/5] 언어 비율 분석 중...")
        language_ratio = detect_languages(comment_texts)
        print(f"   ✅ 완료: {language_ratio}")

        # ============================================================
        # ⚠️ STEP 5: 논란 감지
        # ============================================================
        print("⚠️  [5/5] 논란 감지 중...")
        is_warning = await is_video_controversial(comment_texts)
        print(f"   ✅ 완료: {'🚨 논란 감지!' if is_warning else '✅ 정상'}")

        # ============================================================
        # 📦 최종 응답 생성
        # ============================================================
        # videoId를 int로 변환 시도 (백엔드가 int 기대할 경우)
        try:
            video_id_int = int(video_id) if video_id.isdigit() else hash(video_id) % 1000000
        except:
            video_id_int = hash(video_id) % 1000000  # 해시값 사용
        
        response = AIAnalysisResponse(
            videoId=video_id_int,           # int 타입으로 변환
            apiVideoId=video_id,            # 원본 string 유지
            summation=summary,
            isWarning=is_warning,
            keywords=keywords,
            sentimentComments=sentiment_comments,  # ✅ List[CommentSentimentDetail]
            languageRatio=language_ratio,
            sentimentRatio=sentiment_ratio,
        )

        # ============================================================
        # ✅ 성공 로그
        # ============================================================
        print("\n" + "=" * 70)
        print("✅ [분석 완료!]")
        print("=" * 70)
        print(f"📊 긍정: {sentiment_ratio.get('POSITIVE', 0):.1f}%")
        print(f"📊 부정: {sentiment_ratio.get('NEGATIVE', 0):.1f}%")
        print(f"📊 기타: {sentiment_ratio.get('OTHER', 0):.1f}%")
        print(f"🔍 키워드: {', '.join(keywords)}")
        print(f"⚠️  논란: {'🚨 감지됨' if is_warning else '✅ 없음'}")
        print("=" * 70 + "\n")
        
        return response

    except HTTPException as he:
        # 이미 정의된 HTTP 예외는 그대로 전달
        raise he
    
    except Exception as e:
        # 예상치 못한 에러 처리
        print("\n" + "=" * 70)
        print("❌ [에러 발생!]")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        print("=" * 70 + "\n")
        
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}"
        )


# ============================================================
# 🏃 서버 실행 (개발 모드)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 YouTube Comment Analyzer API 서버 시작...")
    print("=" * 70)
    print("📍 로컬: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("📊 Redoc: http://localhost:8000/redoc")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # 외부 접속 허용
        port=7777,       # 포트 번호
        reload=True      # 코드 변경시 자동 재시작 (개발 모드)
    )