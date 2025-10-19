# src/pipelines/summarize.py
"""
[기능1] AI 기반 자동 전체 댓글 요약
- OpenAI Chat Completions API 사용
- 입력: 댓글 문자열 리스트
- 출력: 한국어 요약문(2문장 내외)
"""
from typing import List
from openai import OpenAI
from src.config import OPENAI_API_KEY

# OpenAI 클라이언트는 모듈 로드 시 1회 생성 (재사용)
client = OpenAI(api_key=OPENAI_API_KEY)

def summarize_comments_with_gpt(comments: List[str]) -> str:
    """
    GPT를 이용한 댓글 요약.
    - comments: 댓글 문자열 리스트
    - 반환: 한국어 요약문 (문자열)
    """
    if not comments:
        return "요약할 댓글이 없습니다."
    joined = "\n".join(comments[:500])  # 안전: 과도한 길이 방지(확실하지 않음: 길이 제한은 필요시 조정)

    prompt = (
        "당신은 텍스트 요약 전문가입니다.\n"
        "다음은 어떤 유튜브 영상에 달린 다양한 댓글들입니다.\n"
        "이 댓글들의 전체 분위기와 주요 논점만 2문장 이내로 자연스럽고 간결하게 한국어로 요약해주세요.\n"
        "중복 의견은 묶고, 인상적인 반응은 반영해 주세요.\n\n"
        f"댓글 목록:\n{joined}\n"
    )

    # 모델명은 환경/요금정책에 따라 조정 가능(확실하지 않음)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()