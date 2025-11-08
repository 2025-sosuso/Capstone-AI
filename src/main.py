# src/main.py
"""
YouTube 댓글 종합 분석 파이프라인
- 감정 분석, 요약, 논란 감지, 키워드 추출, 언어 비율 분석을 통합 실행
- AIAnalysisResponse 형식으로 결과 출력
"""
import os
import sys
import asyncio
import argparse
import re
from typing import Dict

# ============================================================
# Windows 한글 깨짐 방지 (최상단 배치)
# ============================================================
if sys.platform == 'win32':
    # 1. 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 2. 표준 출력/에러 스트림 재설정
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
    
    # 3. Windows 콘솔 코드 페이지를 UTF-8로 설정
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, 
                      capture_output=True, check=False)
    except Exception:
        pass

from src.config import YOUTUBE_API_KEY, VIDEO_KEY
from src.utils.youtube import fetch_youtube_comments, fetch_youtube_comment_map
from src.models.schemas import AIAnalysisResponse
from src.pipelines.sentiment import analyze_sentiment_async
from src.pipelines.summarize import summarize_comments_with_gpt
from src.pipelines.controversy import is_video_controversial
from src.pipelines.keywords import extract_keywords_tfidf
from src.pipelines.lang_ratio import detect_languages

# 11자리 YouTube videoId 검증용 정규식
_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')


async def run_full_analysis(api_video_id: str, max_pages: int = 1, page_size: int = 100):
    """
    전체 분석을 수행하여 AIAnalysisResponse 반환
    
    Args:
        api_video_id: YouTube API 비디오 ID
        max_pages: 수집할 최대 페이지 수
        page_size: 페이지당 댓글 수
    
    Returns:
        AIAnalysisResponse: 모든 분석 결과를 포함한 응답 객체
    """
    print(f"\n{'='*60}")
    print(f"[INFO] YouTube 비디오 '{api_video_id}' 전체 분석 시작")
    print(f"{'='*60}\n")
    
    # ============================================================
    # STEP 1: YouTube 댓글 수집 (Dict 형태로)
    # ============================================================
    print("[1/6] 댓글 수집 중...")
    comments_dict: Dict[str, str] = fetch_youtube_comment_map(
        video_id=api_video_id,
        api_key=YOUTUBE_API_KEY,
        max_pages=max_pages,
        page_size=page_size,
        include_replies=False,
        apply_cleaning=True,
    )
    
    if not comments_dict:
        print("[WARNING] 수집된 댓글이 없습니다.")
        # 빈 응답 반환
        return AIAnalysisResponse(
            videoId=0,  # 임시 ID
            apiVideoId=api_video_id,
            summation="분석할 댓글이 없습니다.",
            isWarning=False,
            keywords=[],
            sentimentComments=[],
            languageRatio={},
            sentimentRatio={"POSITIVE": 0.0, "NEGATIVE": 0.0, "OTHER": 0.0}
        )
    
    print(f"[SUCCESS] {len(comments_dict)}개 댓글 수집 완료\n")
    
    # 댓글 텍스트만 추출 (일부 함수는 텍스트 리스트만 필요)
    comment_texts = list(comments_dict.values())
    
    # ============================================================
    # STEP 2: 감정 분석 (비동기)
    # ============================================================
    print("[2/6] 감정 분석 중...")
    sentiment_comments, sentiment_ratio = await analyze_sentiment_async(comments_dict)
    print(f"[SUCCESS] 감정 분석 완료 - {len(sentiment_comments)}개 댓글\n")
    
    # ============================================================
    # STEP 3: 요약 생성
    # ============================================================
    print("[3/6] 댓글 요약 생성 중...")
    summation = summarize_comments_with_gpt(comment_texts)
    print(f"[SUCCESS] 요약 완료\n")
    
    # ============================================================
    # STEP 4: 논란 감지 (비동기)
    # ============================================================
    print("[4/6] 논란 감지 중...")
    is_warning = await is_video_controversial(comment_texts, ratio_threshold=0.10)
    print(f"[SUCCESS] 논란 감지 완료 - {'⚠️ 경고' if is_warning else '✅ 정상'}\n")
    
    # ============================================================
    # STEP 5: 키워드 추출
    # ============================================================
    print("[5/6] 키워드 추출 중...")
    keywords = extract_keywords_tfidf(comment_texts, top_n=10)
    print(f"[SUCCESS] 키워드 추출 완료 - {len(keywords)}개\n")
    
    # ============================================================
    # STEP 6: 언어 비율 분석
    # ============================================================
    print("[6/6] 언어 비율 분석 중...")
    language_ratio = detect_languages(comment_texts)
    print(f"[SUCCESS] 언어 비율 분석 완료\n")
    
    # ============================================================
    # 최종 응답 생성
    # ============================================================
    response = AIAnalysisResponse(
        videoId=0,  # 백엔드 DB ID (임시로 0 사용)
        apiVideoId=api_video_id,
        summation=summation,
        isWarning=is_warning,
        keywords=keywords,
        sentimentComments=sentiment_comments,
        languageRatio=language_ratio,
        sentimentRatio=sentiment_ratio
    )
    
    print(f"{'='*60}")
    print("[COMPLETE] 전체 분석 완료!")
    print(f"{'='*60}\n")
    
    return response


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YouTube 댓글 종합 분석 시스템")
    ap.add_argument("--video", help="YouTube 11자 videoId (URL 아님)", default=None)
    ap.add_argument("--pages", type=int, default=1, help="수집할 페이지 수 (기본: 1)")
    ap.add_argument("--page-size", type=int, default=10, help="페이지당 댓글 수 (기본: 100)")
    args = ap.parse_args()
    
    # ============================================================
    # 비디오 ID 확인
    # ============================================================
    video_id = args.video or VIDEO_KEY
    if not video_id:
        print("[ERROR] 영상 ID가 없습니다.")
        print("방법 1: --video 인자로 지정")
        print("  예: python -m src.main --video dQw4w9WgXcQ")
        print("방법 2: .env 파일에 VIDEO_KEY 설정")
        print("  예: VIDEO_KEY=dQw4w9WgXcQ")
        sys.exit(1)
    
    if not _ID_RE.fullmatch(video_id):
        print(f"[ERROR] 유효한 11자리 YouTube videoId가 아닙니다: {video_id}")
        sys.exit(1)
    
    if not YOUTUBE_API_KEY:
        print("[ERROR] YOUTUBE_API_KEY가 없습니다. .env에 추가하세요.")
        sys.exit(1)
    
    # ============================================================
    # 전체 분석 실행
    # ============================================================
    try:
        result = asyncio.run(run_full_analysis(
            api_video_id=video_id,
            max_pages=args.pages,
            page_size=args.page_size
        ))
        
        # ============================================================
        # JSON 형태로 최종 결과 출력
        # ============================================================
        print("=" * 60)
        print("[최종 결과] AIAnalysisResponse (JSON 형식)")
        print("=" * 60)
        print(result.model_dump_json(indent=2, ensure_ascii=False))
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n[INFO] 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)