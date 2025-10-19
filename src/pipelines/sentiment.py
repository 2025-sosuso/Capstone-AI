# src/pipelines/sentiment.py
from __future__ import annotations
import re
import asyncio
from collections import Counter
from typing import Dict, List, Tuple, Union # union 추가함!!!!!!

import httpx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.config import DEEPL_API_KEY

_MNAME = "cardiffnlp/twitter-roberta-base-sentiment"

# 전역 캐시(지연 로딩)
_tok = None
_model = None
_pipe = None

def _get_sentiment_pipeline():
    """
    처음 호출될 때만 모델/토크나이저 로드.
    - safetensors만 사용 (torch 2.6 미만에서도 OK)
    - GPU가 있으면 자동 사용
    """
    global _tok, _model, _pipe
    if _pipe is None:
        _tok = AutoTokenizer.from_pretrained(_MNAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            _MNAME,
            use_safetensors=True,   # 🔴 핵심: .bin 대신 safetensors만 사용
        )
        _pipe = pipeline(
            task="sentiment-analysis",
            model=_model,
            tokenizer=_tok,
            device=0 if torch.cuda.is_available() else -1,  # GPU 자동 사용
            truncation=True,
            max_length=128,
            return_all_scores=False,  # 최고 점수 1개 라벨만 반환
        )
    return _pipe


def _is_english(text: str) -> bool:
    cleaned = re.sub(r"[^\w\s.,!?\'\"-]", "", text)
    return bool(re.fullmatch(r"[A-Za-z0-9\s\.,;:'\"!?()\[\]{}@#$%^&*_\-=+/<>|~]+", cleaned))

async def _translate_text_async(client: httpx.AsyncClient, text: str, api_key: str, target_lang: str = "EN") -> str:
    """
    DeepL 비동기 번역 (키 없으면 원문 유지)
    """
    if not api_key:
        return text
    url = "https://api.deepl.com/v2/translate"
    data = {"auth_key": api_key, "text": text, "target_lang": target_lang}
    try:
        resp = await client.post(url, data=data, timeout=15.0)
        resp.raise_for_status()
        return resp.json()["translations"][0]["text"]
    except Exception:
        return text

# !!!!!!!!!!!!!!
def _normalize_input(
    comments: Union[Dict[str, str], List[Tuple[str, str]], List[str]]
) -> Tuple[List[str], List[str]]:
    """
    다양한 입력을 (ids, texts)로 정규화
    - Dict[commentId, text]        -> (ids, texts)  ✅ commentId 유지
    - List[Tuple[commentId, text]] -> (ids, texts)  ✅ commentId 유지
    - List[str]                    -> (['0','1',...], texts)  (id가 없을 때만 인덱스)
    """
    if isinstance(comments, dict):
        ids, texts = zip(*comments.items()) if comments else ([], [])
        return list(ids), list(texts)

    if isinstance(comments, list) and comments and isinstance(comments[0], tuple):
        ids = [cid for cid, _t in comments]
        texts = [_t for _cid, _t in comments]
        return ids, texts

    # 그 외(List[str] 등) → 인덱스 키 사용(후방호환)
    if isinstance(comments, list):
        texts = list(comments)
        ids = [str(i) for i in range(len(texts))]
        return ids, texts

    # fallback
    return [], []

async def analyze_sentiment_async(
    comments_dict: Union[Dict[str, str], List[Tuple[str, str]], List[str]]
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    입력:
      - Dict[commentId, text]
      - List[Tuple[commentId, text]]
      - List[text] (기존처럼 index 사용)
    출력:
      - (각 댓글의 감정 결과 Map, 전체 비율 Map)
    """
    # ✅ 입력을 (ids, texts)로 정규화
    ids, texts = _normalize_input(comments_dict)
    if not texts:
        return {}, {}

    # ✅ 영어가 아닌 경우 DeepL 번역
    async with httpx.AsyncClient() as http_client:
        processed = [
            t if _is_english(t) else await _translate_text_async(http_client, t, DEEPL_API_KEY)
            for t in texts
        ]

    pipe = _get_sentiment_pipeline()
    results = pipe(list(processed), batch_size=64)

    # ✅ 결과 매핑
    label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
    per_comment: Dict[str, str] = {
        cid: label_map.get(r["label"], "NEUTRAL") for cid, r in zip(ids, results)
    }

    # ✅ 전체 비율 계산
    cnt = Counter(per_comment.values())
    total = max(sum(cnt.values()), 1)
    ratio = {
        "positive": round(cnt.get("POSITIVE", 0) / total * 100, 2),
        "neutral":  round(cnt.get("NEUTRAL", 0) / total * 100, 2),
        "negative": round(cnt.get("NEGATIVE", 0) / total * 100, 2),
    }

    # ✅ commentId 기준 결과 반환
    return per_comment, ratio
