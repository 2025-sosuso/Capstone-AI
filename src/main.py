# src/main.py
import argparse
import asyncio
import sys
import re
from src.config import YOUTUBE_API_KEY, VIDEO_KEY
from src.utils.youtube import fetch_youtube_comments, fetch_youtube_comment_map

# 11자리 YouTube videoId 검증용 정규식
_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')


def _load_comments_or_die(video_or_id: str | None, with_replies: bool, pages: int, page_size: int):
    """
    영상 ID를 가져오고 댓글을 로드.
    - 우선순위: CLI --video → .env VIDEO_KEY
    - 둘 다 없거나 11자 ID가 아니면 종료
    """
    source = video_or_id or VIDEO_KEY  # ✅ .env 값 기본 사용
    if not source:
        print("[ERROR] 영상 ID가 없습니다. --video 인자 또는 .env의 VIDEO_KEY를 설정하세요.")
        sys.exit(1)

    if not _ID_RE.fullmatch(source):
        print(f"[ERROR] 유효한 11자리 YouTube videoId가 아닙니다: {source}")
        sys.exit(1)

    if not YOUTUBE_API_KEY:
        print("[ERROR] YOUTUBE_API_KEY가 없습니다. .env에 추가하세요.")
        sys.exit(1)

    comments = fetch_youtube_comments(
        video_id=source,
        api_key=YOUTUBE_API_KEY,
        max_pages=pages,
        page_size=page_size,
        include_replies=with_replies,
        apply_cleaning=True,
    )
    if not comments:
        print("[ERROR] 댓글을 가져오지 못했습니다. 영상의 댓글이 비활성화되었거나 API 쿼터/네트워크 문제가 있을 수 있습니다.")
        sys.exit(1)

    return comments, source  # ✅ videoId도 함께 반환


def run_sum(comments):
    from src.pipelines.summarize import summarize_comments_with_gpt
    print("=== Summarize ===")
    print(summarize_comments_with_gpt(comments))


def run_keywords(comments):
    from src.pipelines.keywords import extract_keywords_tfidf
    print("=== Keywords ===")
    print(extract_keywords_tfidf(comments, top_n=10))


def run_lang(comments):
    from src.pipelines.lang_ratio import detect_languages
    print("=== Language Ratio ===")
    print(detect_languages(comments))


async def run_sentiment(comments, video_id: str):
    """댓글 감정 분석 (commentId 기준)"""
    from src.pipelines.sentiment import analyze_sentiment_async
    print("=== Sentiment ===")

    try:
        # ✅ commentId→text 형태로 다시 가져오기
        comment_map = fetch_youtube_comment_map(
            video_id=video_id,
            api_key=YOUTUBE_API_KEY,
            max_pages=3,
            page_size=100,
            include_replies=False,
            apply_cleaning=True,
        )
        if comment_map:
            mapping, ratio = await analyze_sentiment_async(comment_map)
            print(mapping)
            print(ratio)
            return
    except Exception as e:
        print(f"[WARN] commentId 맵 로드 실패. 인덱스로 대체합니다. ({e})")

    # 폴백: 리스트 인덱스를 key로 사용
    mapping, ratio = await analyze_sentiment_async({str(i): c for i, c in enumerate(comments)})
    print(mapping)
    print(ratio)


async def run_controversy(comments):
    from src.pipelines.controversy import is_video_controversial
    print("=== Controversy ===")
    print(await is_video_controversial(comments, ratio_threshold=0.10))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sum", "sent", "kw", "lang", "contro"])
    ap.add_argument("--video", help="YouTube 11자 videoId (URL 아님)", default=None)
    ap.add_argument("--with-replies", action="store_true")
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--page-size", type=int, default=100)
    args = ap.parse_args()

    # ✅ comments와 video_id 둘 다 받기
    comments, video_id = _load_comments_or_die(args.video, args.with_replies, args.pages, args.page_size)

    if args.task == "sum":
        run_sum(comments)
    elif args.task == "kw":
        run_keywords(comments)
    elif args.task == "lang":
        run_lang(comments)
    elif args.task == "sent":
        asyncio.run(run_sentiment(comments, video_id))  # ✅ .env의 VIDEO_KEY 자동 전달
    elif args.task == "contro":
        asyncio.run(run_controversy(comments))
