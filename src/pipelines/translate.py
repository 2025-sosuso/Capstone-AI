"""
댓글 번역 모듈

DeepL API를 사용하여 비영어 댓글을 영어로 번역한다.
배치 처리로 API 호출을 최소화하고, 영어 텍스트는 자동 스킵한다.

Example::

    from src.pipelines.translate import translate_batch

    texts = ["안녕하세요", "Hello", "감사합니다"]
    translated = await translate_batch(texts)
    # translated: ["Hello", "Hello", "Thank you"]
"""
import re
from typing import List

import httpx

try:
    from src.config import DEEPL_API_KEY
except ImportError:
    from ..config import DEEPL_API_KEY


def should_skip_translation(text: str) -> bool:
    """
    번역을 건너뛸지 확인한다.

    빈 문자열이거나 영어/숫자만 포함된 경우 번역을 스킵한다.

    Args:
        text: 확인할 텍스트

    Returns:
        True면 번역 스킵, False면 번역 필요

    Example::

        should_skip_translation("Hello World")  # True
        should_skip_translation("안녕하세요")     # False
        should_skip_translation("")             # True
        should_skip_translation("Hello 안녕")   # False
    """
    if not text or not text.strip():
        return True

    cleaned = re.sub(r"[^\w\s]", "", text)
    return bool(re.fullmatch(r"[A-Za-z0-9\s]+", cleaned))


async def translate_batch(texts: List[str]) -> List[str]:
    """
    댓글들을 배치로 번역한다.

    영어 텍스트는 자동 스킵하고, 비영어만 모아서
    1번의 HTTP 요청으로 처리한다. 이를 통해 API 비용과
    네트워크 오버헤드를 최소화한다.

    Args:
        texts: 번역할 텍스트 리스트

    Returns:
        번역된 텍스트 리스트. 영어 텍스트는 원본 유지.

    Raises:
        httpx.HTTPError: DeepL API 호출 실패 시 (원문 반환으로 대체)

    Note:
        - DeepL API 키가 없으면 원본을 그대로 반환한다.
        - 빈 리스트 입력 시 빈 리스트를 반환한다.

    Example::

        texts = ["안녕하세요", "Hello", "감사합니다"]
        result = await translate_batch(texts)
        # result: ["Hello", "Hello", "Thank you"]

        # 모두 영어인 경우 API 호출 없음
        texts = ["Hello", "World"]
        result = await translate_batch(texts)  # API 호출 0번
        # result: ["Hello", "World"]
    """
    if not DEEPL_API_KEY:
        print("[번역] DeepL API 키 없음 - 원본 반환")
        return texts

    if not texts:
        return texts

    # 번역 필요한 텍스트만 필터링
    non_english_indices: List[int] = []
    non_english_texts: List[str] = []

    for i, text in enumerate(texts):
        if not should_skip_translation(text):
            non_english_indices.append(i)
            non_english_texts.append(text)

    skip_count = len(texts) - len(non_english_texts)
    print(f"[번역] 전체 {len(texts)}개 중 {len(non_english_texts)}개 번역 필요 ({skip_count}개 스킵)")

    # 모두 영어면 API 호출 생략
    if not non_english_texts:
        print("[번역] 모든 댓글 스킵 - API 호출 생략")
        return texts

    # 배치 번역
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.deepl.com/v2/translate",
                data={
                    "auth_key": DEEPL_API_KEY,
                    "text": non_english_texts,
                    "target_lang": "EN",
                },
            )
            resp.raise_for_status()
            translations = resp.json()["translations"]

        print(f"[번역] 완료 (1번 요청으로 {len(non_english_texts)}개 처리)")

    except Exception as e:
        print(f"[번역] 실패: {e} - 원문 반환")
        return texts

    result = list(texts)
    for idx, trans in zip(non_english_indices, translations):
        result[idx] = trans["text"]

    return result