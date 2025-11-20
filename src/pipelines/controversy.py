# src/pipelines/controversy.py
"""
[기능5] AI 기반 논란 의심 댓글 탐지 (안정/배치 버전)
- facebook/bart-large-mnli (safetensors, GPU 자동)
- 비영어 텍스트는 선택적 번역
- 빈 문자열/짧은 문자열 필터 + 배치 추론 + 예외 가드
- 논란 레이블 세분화: 사기 의혹, 뒷광고 논란, 허위정보/조작
"""
from __future__ import annotations

import os
import sys
import re
from typing import List, Tuple

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

import httpx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ============================================================
# API 키 및 유틸리티 불러오기
# ============================================================
try:
    from src.config import DEEPL_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY
    from src.utils.youtube import fetch_youtube_comments
except ImportError:
    from ..config import DEEPL_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY
    from ..utils.youtube import fetch_youtube_comments

_MNAME = "facebook/bart-large-mnli"
_labels = [
    "direct accusation: this is fraud, scam, or undisclosed paid promotion",  # 사기/뒷광고 통합
    "general comment, opinion, or complaint"  # 일반 댓글
]
_hypo = "This comment is: {}."  # 단순화
_controversy_labels = _labels[:1]  # 첫 번째만 논란 레이블

_tok = None
_model = None
_clf = None


def _get_classifier():
    """지연 로딩(+ safetensors), GPU 자동 사용"""
    global _tok, _model, _clf
    if _clf is None:
        print("[INFO] BART 논란 감지 모델 로딩 중...")
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
        print("[SUCCESS] BART 논란 감지 모델 로딩 완료!")
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


async def _controversy_scores_batch(texts: List[str], debug: bool = False) -> List[float]:
    """
    배치로 논란 점수(0~1) 계산:
    처음 3개 레이블(논란 카테고리) 중 최대 점수 반환.
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
            batch_size=16,  # GPU 효율 ↑
            multi_label=False,  # ⭐ 하나만 선택 (가장 적합한 레이블)
        )
    except ValueError as e:
        # "at least one label and at least one sequence"류 예외 방지 가드
        print(f"[ERROR] ValueError in classification: {e}")
        return []

    scores: List[float] = []

    for idx, out in enumerate(outputs):
        lbls = out.get("labels", [])
        scrs = out.get("scores", [])

        # 🔍 디버깅: 각 댓글의 레이블별 점수 출력
        if debug:
            print(f"\n[DEBUG] 댓글 #{idx + 1}: {seqs[idx][:50]}...")
            print(f"  레이블별 점수:")
            for lbl, sc in zip(lbls, scrs):
                controversy_mark = "⚠️" if lbl in _controversy_labels else "✅"
                print(f"    {controversy_mark} {lbl}: {sc:.4f}")

        # ⭐ multi_label=False: 가장 높은 점수의 레이블(첫 번째)만 확인
        top_label = lbls[0] if lbls else ""
        top_score = scrs[0] if scrs else 0.0

        # 최고 점수 레이블이 논란 카테고리인지 확인
        if top_label in _controversy_labels:
            controversy_score = float(top_score)
        else:
            controversy_score = 0.0  # 일반 의견으로 분류됨

        if debug:
            print(f"  → 최고 레이블: {top_label}")
            print(f"  → 최종 논란 점수: {controversy_score:.4f}")

        scores.append(controversy_score)

    return scores


async def is_video_controversial(comments: List[str], ratio_threshold: float = 0.20, debug: bool = False) -> bool:
    """
    영상 전체에서 'controversial' 비율이 ratio_threshold 이상이면 True
    """
    if not comments:
        return False

    # 배치 추론로 변경
    scores = await _controversy_scores_batch(comments, debug=debug)
    if not scores:
        return False

    flagged = sum(1 for s in scores if s >= 0.35)  # ⭐ 임계값: 0.35 (사기 의혹 점수 반영)
    ratio = flagged / max(1, len(scores))

    # 🔍 디버깅: 전체 통계 출력
    if debug:
        print(f"\n{'=' * 60}")
        print(f"[DEBUG] 논란 감지 통계")
        print(f"{'=' * 60}")
        print(f"  전체 댓글 수: {len(scores)}개")
        print(f"  논란 댓글 수: {flagged}개 (임계값 >= 0.35)")
        print(f"  논란 비율: {ratio * 100:.1f}%")
        print(f"  임계값: {ratio_threshold * 100:.1f}%")
        print(f"  결과: {'⚠️ 논란 있음 (True)' if ratio >= ratio_threshold else '✅ 정상 (False)'}")
        print(f"{'=' * 60}\n")

    return ratio >= ratio_threshold


# ============================================================
# 사용 예시 (테스트용 코드)
# ============================================================
if __name__ == "__main__":
    import asyncio

    # config.py에서 VIDEO_KEY 확인
    if not VIDEO_KEY:
        print("[ERROR] VIDEO_KEY가 .env 파일에 설정되지 않았습니다.")
        print("[INFO] 테스트 데이터로 실행합니다.")

        print("\n" + "=" * 60)
        print("[테스트] 논란 댓글 탐지 (샘플 데이터)")
        print("=" * 60 + "\n")

        # 테스트 데이터 1: 논란이 없는 댓글들
        print("[테스트 1] 평화로운 댓글들")
        peaceful_comments = [
            "정말 유익한 영상이네요!",
            "감사합니다. 많은 도움이 되었어요.",
            "설명이 정말 잘 되어있네요.",
            "좋은 정보 감사합니다!",
            "구독하고 갑니다~",
        ]

        result1 = asyncio.run(is_video_controversial(peaceful_comments, ratio_threshold=0.20, debug=True))
        print(f"  댓글 수: {len(peaceful_comments)}개")
        print(f"  논란 여부: {'⚠️ 경고 (True)' if result1 else '✅ 정상 (False)'}")
        print()

        # 테스트 데이터 2: 논란이 있는 댓글들
        print("[테스트 2] 논란이 있는 댓글들")
        controversial_comments = [
            "이건 완전 사기네요!",
            "거짓말 그만하세요. 증거가 있습니다.",
            "이렇게 논란이 많은데도 사과 안 하나요?",
            "법적 조치 들어갑니다.",
            "이건 명백한 사기 행위입니다.",
            "신고했습니다.",
            "사람들 속이지 마세요.",
            "뒷광고 아닌가요?",
            "협찬 받고 거짓말 하시네요.",
        ]

        result2 = asyncio.run(is_video_controversial(controversial_comments, ratio_threshold=0.20, debug=True))
        print(f"  댓글 수: {len(controversial_comments)}개")
        print(f"  논란 여부: {'⚠️ 경고 (True)' if result2 else '✅ 정상 (False)'}")
        print()

        # 테스트 데이터 3: 혼합된 댓글들
        print("[테스트 3] 혼합된 댓글들")
        mixed_comments = [
            "좋은 영상이네요!",
            "감사합니다.",
            "이건 사기 아닌가요?",
            "구독하고 갑니다.",
            "증거 있으면 보여주세요.",
            "유익한 정보네요.",
            "법적 문제 있을 것 같은데요.",
            "좋아요 눌렀어요!",
        ]

        result3 = asyncio.run(is_video_controversial(mixed_comments, ratio_threshold=0.20, debug=True))
        print(f"  댓글 수: {len(mixed_comments)}개")
        print(f"  논란 여부: {'⚠️ 경고 (True)' if result3 else '✅ 정상 (False)'}")
        print()

        # 테스트 데이터 4: 단순 부정적 댓글들 (논란 아님)
        print("[테스트 4] 단순 부정적 댓글들 (논란 아님)")
        negative_comments = [
            "별로네요.",
            "지루해요.",
            "재미없어요.",
            "별로 유익하지 않은 것 같아요.",
            "기대 이하였습니다.",
        ]

        result4 = asyncio.run(is_video_controversial(negative_comments, ratio_threshold=0.20, debug=True))
        print(f"  댓글 수: {len(negative_comments)}개")
        print(f"  논란 여부: {'⚠️ 경고 (True)' if result4 else '✅ 정상 (False)'}")
        print()

        print("=" * 60)
        print("[완료] 논란 댓글 탐지 테스트 완료 (샘플 데이터)")
        print("=" * 60)
    else:
        if not YOUTUBE_API_KEY:
            print("[ERROR] YOUTUBE_API_KEY가 .env 파일에 설정되지 않았습니다.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print(f"[테스트] YouTube 비디오 '{VIDEO_KEY}' 논란 탐지")
        print("=" * 60 + "\n")

        try:
            # YouTube 댓글 수집
            print("[1/2] YouTube 댓글 수집 중...")
            youtube_comments = fetch_youtube_comments(
                video_id=VIDEO_KEY,
                api_key=YOUTUBE_API_KEY,
                max_pages=1,  # 1페이지 (100개)
                page_size=100,  # 페이지당 100개
                include_replies=False,  # 대댓글 제외
                apply_cleaning=True,  # 텍스트 전처리 적용
            )

            print(f"[SUCCESS] {len(youtube_comments)}개 댓글 수집 완료!\n")

            if not youtube_comments:
                print("[WARNING] 수집된 댓글이 없습니다.")
                sys.exit(0)

            # 논란 탐지 실행
            print("[2/2] 논란 댓글 탐지 중...")
            is_controversial = asyncio.run(is_video_controversial(
                youtube_comments,
                ratio_threshold=0.20,  # 20% 이상의 댓글이 논란이면 경고
                debug=True  # 🔍 디버깅 모드 활성화
            ))

            # ============================================================
            # 결과 출력
            # ============================================================
            print("\n" + "=" * 60)
            print("[결과] 논란 탐지 결과:")
            print("=" * 60)
            print(f"  영상 ID: {VIDEO_KEY}")
            print(f"  분석 댓글 수: {len(youtube_comments)}개")
            print(f"  논란 여부: {'⚠️ 경고 (True)' if is_controversial else '✅ 정상 (False)'}")

            if is_controversial:
                print("\n  ⚠️  이 영상은 논란이 있는 것으로 판단됩니다.")
                print("      댓글 중 20% 이상이 사기 의혹, 뒷광고, 허위정보 등을 제기하고 있습니다.")
            else:
                print("\n  ✅  이 영상은 논란이 없는 것으로 판단됩니다.")
                print("      댓글 대부분이 평화롭고 건전합니다.")

            print("=" * 60)

        except Exception as e:
            print(f"[ERROR] 처리 실패: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)