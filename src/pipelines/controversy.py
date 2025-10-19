# src/pipelines/controversy.py
"""
[기능5] AI 기반 논란 의심 댓글 탐지 (안정/배치 버전)
- facebook/bart-large-mnli (safetensors, GPU 자동)
- 비영어 텍스트는 선택적 번역
- 빈 문자열/짧은 문자열 필터 + 배치 추론 + 예외 가드
"""
from __future__ import annotations
import re
from typing import List, Tuple

import httpx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.config import DEEPL_API_KEY

_MNAME = "facebook/bart-large-mnli"
_labels = ["controversial", "non-controversial"]
_hypo = "This text is {}."

_tok = None
_model = None
_clf = None

def _get_classifier():
    """지연 로딩(+ safetensors), GPU 자동 사용"""
    global _tok, _model, _clf
    if _clf is None:
        _tok = AutoTokenizer.from_pretrained(_MNAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            _MNAME,
            use_safetensors=True,  # .bin 로드 회피
        )
        _clf = pipeline(
            task="zero-shot-classification",
            model=_model,
            tokenizer=_tok,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _clf

def _is_english(text: str) -> bool:
    cleaned = re.sub(r"[^\w\s.,!?\'\"-]", "", text or "")
    return bool(re.fullmatch(r"[A-Za-z0-9\s\.,;:'\"!?()\[\]{}@#$%^&*_\-=+/<>|~]+", cleaned))

async def _translate_to_en_batch(texts: List[str]) -> List[str]:
    """DeepL로 일괄 번역(키 없으면 원문 반환). 네트워크 이슈 시 원문 사용."""
    if not DEEPL_API_KEY:
        return texts
    url = "https://api.deepl.com/v2/translate"
    out: List[str] = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        for t in texts:
            if _is_english(t):
                out.append(t)
                continue
            data = {"auth_key": DEEPL_API_KEY, "text": t, "target_lang": "EN"}
            try:
                r = await client.post(url, data=data)
                r.raise_for_status()
                out.append(r.json()["translations"][0]["text"])
            except Exception:
                out.append(t)
    return out

def _clean_and_filter(texts: List[str]) -> List[str]:
    """공백/너무 짧은 항목 제거 (파이프라인 빈 입력 방지)"""
    cleaned = [(t or "").strip() for t in texts]
    cleaned = [t for t in cleaned if len(t) >= 3]
    return cleaned

async def _controversy_scores_batch(texts: List[str]) -> List[float]:
    """
    배치로 논란 점수(0~1) 계산: label='controversial'의 score 반환.
    빈 입력이 오면 빈 리스트 반환(예외 방지).
    """
    seqs = _clean_and_filter(texts)
    if not seqs:
        return []

    # 번역(선택)
    seqs = await _translate_to_en_batch(seqs)

    clf = _get_classifier()
    try:
        outputs = clf(
            seqs,
            candidate_labels=_labels,
            hypothesis_template=_hypo,
            batch_size=16,      # GPU 효율 ↑
            multi_label=False,  # 둘 중 하나를 선택
        )
    except ValueError as e:
        # "at least one label and at least one sequence"류 예외 방지 가드
        return []

    scores: List[float] = []
    for out in outputs:
        # out["labels"]는 ['controversial','non-controversial'] 순서가 아닐 수 있음
        lbls = out.get("labels", [])
        scrs = out.get("scores", [])
        score = 0.0
        for lbl, sc in zip(lbls, scrs):
            if lbl == "controversial":
                score = float(sc)
                break
        scores.append(score)
    return scores

async def is_video_controversial(comments: List[str], ratio_threshold: float = 0.10) -> bool:
    """
    영상 전체에서 'controversial' 비율이 ratio_threshold 이상이면 True
    """
    if not comments:
        return False

    # 배치 추론로 변경
    scores = await _controversy_scores_batch(comments)
    if not scores:
        return False

    flagged = sum(1 for s in scores if s >= 0.7)  # 임계값: 0.7
    ratio = flagged / max(1, len(scores))
    return ratio >= ratio_threshold
