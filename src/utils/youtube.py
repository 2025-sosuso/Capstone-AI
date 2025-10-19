# src/utils/youtube.py
"""
YouTube 댓글 수집 및 전처리 유틸리티
- fetch_youtube_comments: 댓글 텍스트만 리스트 형태로 반환
- fetch_youtube_comment_map: {commentId: text} 형태로 반환
"""
import re
import requests
from typing import List, Dict

YOUTUBE_COMMENTS_API = "https://www.googleapis.com/youtube/v3/commentThreads"


# -------------------------------
# (1) 댓글 전처리 함수
# -------------------------------
def clean_comment(text: str) -> str:
    """댓글 텍스트 정제"""
    text = re.sub(r"[^\w\s.,!?ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z]", "", text)  # 특수문자 제거
    text = re.sub(r"http\S+|www\.\S+", "", text)                   # URL 제거
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)                     # 3회 이상 반복 문자 축소
    text = text.replace('\n', ' ').replace('\r', ' ')              # 개행 제거
    return text.strip()


# -------------------------------
# (2) 댓글 수집 함수 (Map 버전)
# -------------------------------
def fetch_youtube_comment_map(
    video_id: str,
    api_key: str,
    max_pages: int = 3,
    page_size: int = 100,
    include_replies: bool = False,
    apply_cleaning: bool = True,
) -> Dict[str, str]:
    """
    YouTube Data API를 이용해 댓글을 {commentId: text} 형태로 반환
    """
    params = {
        "key": api_key,
        "videoId": video_id,
        "part": "snippet,replies" if include_replies else "snippet",
        "maxResults": page_size,
        "textFormat": "plainText",
        "order": "time",
    }

    comment_map: Dict[str, str] = {}
    next_page = None
    pages = 0

    while pages < max_pages:
        if next_page:
            params["pageToken"] = next_page
        resp = requests.get(YOUTUBE_COMMENTS_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            # 상위 댓글
            top = item["snippet"]["topLevelComment"]["snippet"]
            cid = item["snippet"]["topLevelComment"]["id"]
            txt = top.get("textDisplay", "")
            if apply_cleaning:
                txt = clean_comment(txt)
            if txt:
                comment_map[cid] = txt

            # 대댓글 (옵션)
            if include_replies:
                for r in item.get("replies", {}).get("comments", []):
                    rcid = r["id"]
                    rtxt = r["snippet"].get("textDisplay", "")
                    if apply_cleaning:
                        rtxt = clean_comment(rtxt)
                    if rtxt:
                        comment_map[rcid] = rtxt

        next_page = data.get("nextPageToken")
        pages += 1
        if not next_page:
            break

    return comment_map


# -------------------------------
# (3) 댓글 수집 함수 (List 버전)
# -------------------------------
def fetch_youtube_comments(
    video_id: str,
    api_key: str,
    max_pages: int = 3,
    page_size: int = 100,
    include_replies: bool = False,
    apply_cleaning: bool = True,
) -> List[str]:
    """
    YouTube Data API로 댓글을 수집하여 텍스트 리스트로 반환
    (fetch_youtube_comment_map()의 값만 추출)
    """
    comment_map = fetch_youtube_comment_map(
        video_id=video_id,
        api_key=api_key,
        max_pages=max_pages,
        page_size=page_size,
        include_replies=include_replies,
        apply_cleaning=apply_cleaning,
    )
    return list(comment_map.values())
