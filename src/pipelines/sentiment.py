# # src/pipelines/sentiment.py
# from __future__ import annotations
#
# import os
# import sys
# import warnings
# from pathlib import Path
#
# # ============================================================
# # Windows 한글 깨짐 방지 (최상단 배치)
# # ============================================================
# if sys.platform == 'win32':
#     # 1. 환경 변수 설정
#     os.environ['PYTHONIOENCODING'] = 'utf-8'
#
#     # 2. 표준 출력/에러 스트림 재설정
#     if hasattr(sys.stdout, 'reconfigure'):
#         try:
#             sys.stdout.reconfigure(encoding='utf-8')
#         except Exception:
#             pass
#
#     if hasattr(sys.stderr, 'reconfigure'):
#         try:
#             sys.stderr.reconfigure(encoding='utf-8')
#         except Exception:
#             pass
#
#     # 3. Windows 콘솔 코드 페이지를 UTF-8로 설정
#     try:
#         import subprocess
#
#         subprocess.run(['chcp', '65001'], shell=True,
#                        capture_output=True, check=False)
#     except Exception:
#         pass
#
# # ============================================================
# # HuggingFace symlink 경고 제거
# # ============================================================
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# warnings.filterwarnings("ignore", message=".*symlink.*")
#
# # 프로젝트 루트 디렉토리를 Python 경로에 추가
# project_root = Path(__file__).parent.parent.parent  # capstone-ai 디렉토리
# sys.path.insert(0, str(project_root))
#
# import asyncio  # 비동기 모듈, fast api랑 연결하기 위해 지정함
# from collections import Counter  # 개수 count
# from typing import Dict, List, Tuple, Union  # 타입 힌팅을 위한 도구들
#
# import torch  # 딥러닝 프레임워크, 신경망 모델 학습 및 실행할 때 사용됨.
# import httpx
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
#
# # ============================================================
# # API 키 불러오기 (config.py에서)
# # ============================================================
# try:
#     from src.config import DEEPL_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY  # 모듈로 실행할 때
#     from src.utils.youtube import fetch_youtube_comment_map  # YouTube 댓글 수집 함수
#     from src.models.schemas import CommentSentimentDetail, SentimentType, DetailSentimentType  # Pydantic 모델
# except ImportError:
#     from ..config import DEEPL_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY  # 패키지 내부에서 실행할 때
#     from ..utils.youtube import fetch_youtube_comment_map
#     from ..models.schemas import CommentSentimentDetail, SentimentType, DetailSentimentType
#
# # ============================================================
# # 1. 영어 GoEmotions 모델 사용 (이미 학습된 모델)
# # ============================================================
# _MNAME = "SamLowe/roberta-base-go_emotions"
#
# _tok = None  # 토크나이저 (텍스트를 숫자로 변환하는 도구)
# _model = None  # 감정 분류 모델
# _pipe = None  # 파이프라인 (입력→전처리→모델→후처리를 한번에 처리)
#
#
# def _get_sentiment_pipeline():
#     """
#     영어 GoEmotions 파이프라인 로드 (지연 로딩)
#
#     작동 원리:
#     1. 처음 호출될 때만 모델/토크나이저를 메모리에 로드
#     2. 이후 호출에서는 이미 로드된 것을 재사용 (빠름!)
#     3. GPU가 있으면 자동으로 GPU 사용 (속도 향상)
#
#     반환: transformers의 pipeline 객체
#     """
#     global _tok, _model, _pipe
#
#     # 이미 로드되어 있으면 그대로 반환
#     if _pipe is None:
#         print("[INFO] GoEmotions 모델 로딩 중...")
#
#         # 토크나이저 로드: 텍스트를 모델이 이해할 수 있는 숫자로 변환
#         _tok = AutoTokenizer.from_pretrained(_MNAME)
#
#         # 모델 로드: 실제 감정 분류를 수행하는 신경망
#         _model = AutoModelForSequenceClassification.from_pretrained(_MNAME)
#
#         # 🔄 변경됨: top_k=1 → top_k=3 (여러 감정 반환)
#         # 파이프라인 생성: 전처리→모델 예측→후처리를 한번에
#         _pipe = pipeline(
#             task="text-classification",  # 감정 분석 작업
#             model=_model,  # 위에서 로드한 모델
#             tokenizer=_tok,  # 위에서 로드한 토크나이저
#             device=0 if torch.cuda.is_available() else -1,  # GPU 있으면 0번 GPU 사용, 없으면 CPU(-1)
#             top_k=3,  # 🔄 상위 3개 감정 반환 (기존: top_k=1)
#         )
#         print("[SUCCESS] GoEmotions 모델 로딩 완료!")
#         print(f"[INFO] 라벨 개수: {len(_model.config.id2label)}개")
#
#     return _pipe
#
#
# async def _translate_to_english(text: str) -> str:
#     """
#     한국어를 영어로 번역 (DeepL API 사용) - 단일 텍스트
#
#     ⚠️ 권장하지 않음: 여러 댓글을 번역할 때는 translate_comments_batch를 사용하세요!
#
#     Args:
#         text: 번역할 한국어 텍스트
#
#     Returns:
#         번역된 영어 텍스트 (실패 시 원문 반환)
#     """
#     if not DEEPL_API_KEY:
#         print("[WARNING] DeepL API 키가 없어 번역 생략")
#         return text
#
#     async with httpx.AsyncClient() as client:
#         try:
#             resp = await client.post(
#                 "https://api.deepl.com/v2/translate",
#                 data={
#                     "auth_key": DEEPL_API_KEY,
#                     "text": text,
#                     "target_lang": "EN",
#                 },
#                 timeout=15.0,
#             )
#             resp.raise_for_status()
#             return resp.json()["translations"][0]["text"]
#         except Exception as e:
#             print(f"[ERROR] 번역 실패: {e}, 원문 사용")
#             return text
#
#
# # ============================================================
# # ⭐ 배치 번역 함수 (export용)
# # ============================================================
# async def translate_comments_batch(texts: List[str]) -> List[str]:
#     """
#     댓글들을 배치로 번역 (DeepL API 사용)
#
#     ✨ 이 함수는 server.py에서 Phase 0에서 호출되어 한 번만 번역을 수행합니다.
#
#     Args:
#         texts: 번역할 텍스트 리스트
#
#     Returns:
#         번역된 텍스트 리스트 (실패 시 원문 반환)
#     """
#     if not DEEPL_API_KEY:
#         print("[WARNING] DeepL API 키가 없어 번역 생략")
#         return texts
#
#     print(f"[INFO] {len(texts)}개 댓글 병렬 번역 중...")
#
#     # 병렬 번역 (asyncio.gather 사용)
#     tasks = [_translate_to_english(t) for t in texts]
#     translated = await asyncio.gather(*tasks)
#
#     print(f"[INFO] 번역 완료!")
#
#     return list(translated)
#
#
# # 어떤 형태로 유튜브 댓글을 넣어도 tuple로 normalize 해주는 함수
# def _normalize_input(
#         comments: Union[Dict[str, str], List[Tuple[str, str]], List[str]]
# ) -> Tuple[List[str], List[str]]:
#     """
#     다양한 형태의 입력을 통일된 형태로 변환
#
#     입력 가능한 형태:
#     1. Dict[commentId, text]        → commentId를 그대로 사용
#     2. List[Tuple[commentId, text]] → commentId를 그대로 사용
#     3. List[str]                    → 자동으로 0, 1, 2... 인덱스 부여
#
#     반환: (댓글 ID 리스트, 댓글 텍스트 리스트)
#
#     예시:
#     - {"c1": "좋아요", "c2": "싫어"} → (["c1", "c2"], ["좋아요", "싫어"])
#     - [("c1", "좋아요"), ("c2", "싫어")] → (["c1", "c2"], ["좋아요", "싫어"])
#     - ["좋아요", "싫어"] → (["0", "1"], ["좋아요", "싫어"])
#     """
#     # 경우 1: 딕셔너리 입력
#     if isinstance(comments, dict):
#         if comments:
#             ids, texts = zip(*comments.items())  # 딕셔너리를 (키, 값) 튜플로 분리
#             return list(ids), list(texts)
#         return [], []
#
#     # 경우 2: 튜플 리스트 입력 [(id, text), ...]
#     if isinstance(comments, list) and comments and isinstance(comments[0], tuple):
#         ids = [cid for cid, _t in comments]
#         texts = [_t for _cid, _t in comments]
#         return ids, texts
#
#     # 경우 3: 단순 텍스트 리스트 ["text1", "text2", ...]
#     if isinstance(comments, list):
#         texts = list(comments)
#         ids = [str(i) for i in range(len(texts))]  # 0, 1, 2... 인덱스 부여
#         return ids, texts
#
#     # 그 외의 경우: 빈 리스트 반환
#     return [], []
#
#
# async def analyze_sentiment_async(
#         comments_dict: Union[Dict[str, str], List[Tuple[str, str]], List[str]],
#         translated_texts: List[str] = None  # ⭐ 새로 추가된 파라미터
# ) -> Tuple[List[CommentSentimentDetail], Dict[str, float], List[str]]:
#     """
#     댓글들의 감정을 분석하는 비동기 함수
#
#     입력:
#       - comments_dict: 댓글 데이터 (Dict/List 형태)
#       - translated_texts: (선택) 이미 번역된 텍스트 리스트 ⭐ 중복 번역 방지!
#
#     출력:
#       - (CommentSentimentDetail 리스트, 긍정/부정/기타 비율, 번역된 텍스트 리스트)
#
#     처리 과정:
#     1. 입력을 표준 형태로 변환
#     2. 한국어 → 영어 번역 (이미 번역된 경우 생략!) ⭐
#     3. 모델을 사용해 각 댓글의 감정 예측 (GoEmotions 28개 중 상위 3개)
#     4. GoEmotions의 28개 감정 → 7개 감정으로 그룹핑
#     5. 7개 감정을 POSITIVE/NEGATIVE/OTHER로 분류
#     6. CommentSentimentDetail 객체 리스트 생성
#     7. 번역된 텍스트 반환
#     """
#     # ============================================================
#     # STEP 1: 입력 정규화
#     # ============================================================
#     ids, texts = _normalize_input(comments_dict)
#
#     # 빈 입력이면 빈 결과 반환
#     if not texts:
#         return [], {}, []
#
#     # ============================================================
#     # STEP 2: 번역 (이미 번역되었으면 생략!) ⭐ 핵심 최적화
#     # ============================================================
#     if translated_texts is None:
#         # 번역된 텍스트가 없으면 직접 번역
#         print(f"[INFO] {len(texts)}개 댓글 병렬 번역 중...")
#         tasks = [_translate_to_english(t) for t in texts]
#         translated = await asyncio.gather(*tasks)
#         print(f"[INFO] 번역 완료!")
#     else:
#         # 번역된 텍스트가 제공됨 (재사용)
#         print(f"[INFO] ✨ 번역된 텍스트 재사용 ({len(translated_texts)}개)")
#         translated = translated_texts
#
#     # ============================================================
#     # STEP 3: GoEmotions 예측 (28개 감정 중 상위 3개)
#     # ============================================================
#     pipe = _get_sentiment_pipeline()
#     results = pipe(translated, batch_size=64)  # 64개씩 배치로 처리 (속도 향상)
#
#     # ============================================================
#     # STEP 4: GoEmotions 28개 → 프로젝트 7개 감정으로 매핑
#     # ============================================================
#     # GoEmotions 원본 라벨을 우리가 원하는 7개 카테고리로 그룹핑
#     label_map = {
#         # 긍정 감정들 → joy (기쁨)
#         "admiration": "joy", "amusement": "joy", "approval": "joy",
#         "excitement": "joy", "joy": "joy", "optimism": "joy",
#         "pride": "joy", "relief": "joy",
#
#         # 애정 관련 → love (사랑)
#         "caring": "love", "desire": "love", "love": "love",
#
#         # 감사 → gratitude (감사)
#         "gratitude": "gratitude",
#
#         # 분노 관련 → anger (분노)
#         "anger": "anger", "annoyance": "anger",
#         "disapproval": "anger", "disgust": "anger",
#
#         # 슬픔 관련 → sadness (슬픔)
#         "disappointment": "sadness", "embarrassment": "sadness",
#         "grief": "sadness", "remorse": "sadness", "sadness": "sadness",
#
#         # 두려움 관련 → fear (두려움)
#         "fear": "fear", "nervousness": "fear",
#
#         # 중립/기타 → neutral
#         "confusion": "neutral", "curiosity": "neutral",
#         "neutral": "neutral", "realization": "neutral", "surprise": "neutral",
#     }
#
#     # 7개 감정을 POSITIVE/NEGATIVE/OTHER로 매핑
#     detail_to_sentiment_map = {
#         "joy": "positive",
#         "love": "positive",
#         "gratitude": "positive",
#         "anger": "negative",
#         "sadness": "negative",
#         "fear": "negative",
#         "neutral": "other",
#     }
#
#     # ============================================================
#     # 🔄 STEP 5: CommentSentimentDetail 리스트 생성 (변경됨)
#     # ============================================================
#     sentiment_comments: List[CommentSentimentDetail] = []
#     sentiment_category_counter = Counter()  # POSITIVE/NEGATIVE/OTHER 카운트
#
#     for cid, text, result in zip(ids, texts, results):
#         # 🔄 변경됨: result는 이제 리스트 (top_k=3이므로 최대 3개)
#         # 각 감정의 확률(score)이 20% 이상인 것만 선택
#         detail_emotions = []
#
#         for pred in result:
#             original_label = pred["label"]
#             score = pred["score"]
#
#             # 🔄 확률이 15% 이상인 감정만 포함 (임계값)
#             if score >= 0.15:
#                 detail_emotion = label_map.get(original_label, "neutral")
#                 detail_emotions.append(detail_emotion)
#
#         # 🔄 감정이 없으면 neutral 추가 (안전장치)
#         if not detail_emotions:
#             detail_emotions = ["neutral"]
#
#         # 🔄 중복 제거 (같은 감정이 여러 번 나올 수 있음)
#         # 예: ["joy", "joy", "love"] → ["joy", "love"]
#         detail_emotions = list(dict.fromkeys(detail_emotions))
#
#         # 🔄 가장 높은 점수의 감정(첫 번째)으로 전체 sentiment_type 결정
#         primary_emotion = detail_emotions[0]
#         sentiment_type = detail_to_sentiment_map[primary_emotion]
#
#         # 카운트 증가
#         sentiment_category_counter[sentiment_type] += 1
#
#         # 🔄 CommentSentimentDetail 객체 생성 (여러 세부 감정 포함)
#         comment_detail = CommentSentimentDetail(
#             apiCommentId=cid,
#             content=text,
#             sentimentType=SentimentType(sentiment_type),
#             detailSentimentTypes=[DetailSentimentType(e) for e in detail_emotions]  # 🔄 리스트로 변환
#         )
#         sentiment_comments.append(comment_detail)
#
#     # ============================================================
#     # STEP 6: 긍정/부정/기타 비율 계산
#     # ============================================================
#     total = max(sum(sentiment_category_counter.values()), 1)
#     sentiment_ratio = {
#         "positive": round(sentiment_category_counter.get("positive", 0) / total * 100),
#         "negative": round(sentiment_category_counter.get("negative", 0) / total * 100),
#         "other": round(sentiment_category_counter.get("other", 0) / total * 100),
#     }
#
#     # ============================================================
#     # 반환: (CommentSentimentDetail 리스트, 비율, 번역된 텍스트)
#     # ============================================================
#     return sentiment_comments, sentiment_ratio, translated
#
#
# # ============================================================
# # 사용 예시 (테스트용 코드)
# # ============================================================
# if __name__ == "__main__":
#     # config.py에서 VIDEO_KEY 확인
#     if not VIDEO_KEY:
#         print("[ERROR] VIDEO_KEY가 .env 파일에 설정되지 않았습니다.")
#         print("[INFO] 테스트 데이터로 실행합니다.")
#
#         # 테스트 데이터
#         test_comments = {
#             "c1": "오늘 정말 행복한 하루였어요!",
#             "c2": "너무 화가 나네요",
#             "c3": "감사합니다",
#             "c4": "무섭고 두려워요",
#         }
#         sentiment_comments, sentiment_ratio, translated = asyncio.run(
#             analyze_sentiment_async(test_comments)
#         )
#     else:
#         if not YOUTUBE_API_KEY:
#             print("[ERROR] YOUTUBE_API_KEY가 .env 파일에 설정되지 않았습니다.")
#             sys.exit(1)
#
#         print(f"\n[INFO] YouTube 비디오 '{VIDEO_KEY}'에서 댓글 수집 중...")
#
#         try:
#             # YouTube 댓글 수집 (최대 100개)
#             youtube_comments = fetch_youtube_comment_map(
#                 video_id=VIDEO_KEY,
#                 api_key=YOUTUBE_API_KEY,
#                 max_pages=1,  # 1페이지 -> 시간 오래걸려서 3페이지에서 1페이지로 바꿈
#                 page_size=100,  # 페이지당 100개
#                 include_replies=False,  # 대댓글 제외
#                 apply_cleaning=True,  # 텍스트 전처리 적용
#             )
#
#             print(f"[SUCCESS] {len(youtube_comments)}개 댓글 수집 완료!")
#
#             if not youtube_comments:
#                 print("[WARNING] 수집된 댓글이 없습니다.")
#                 sys.exit(0)
#
#             # 감정 분석 실행
#             sentiment_comments, sentiment_ratio, translated = asyncio.run(
#                 analyze_sentiment_async(youtube_comments)
#             )
#
#         except Exception as e:
#             print(f"[ERROR] 댓글 수집 실패: {e}")
#             sys.exit(1)
#
#     # ============================================================
#     # 결과 출력
#     # ============================================================
#     print("\n" + "=" * 60)
#


# src/pipelines/sentiment.py
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

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

# ============================================================
# HuggingFace symlink 경고 제거
# ============================================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*symlink.*")

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent  # capstone-ai 디렉토리
sys.path.insert(0, str(project_root))

import asyncio  # 비동기 모듈, fast api랑 연결하기 위해 지정함
from collections import Counter  # 개수 count
from typing import Dict, List, Tuple, Union  # 타입 힌팅을 위한 도구들

import torch  # 딥러닝 프레임워크, 신경망 모델 학습 및 실행할 때 사용됨.
import httpx
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ============================================================
# API 키 불러오기 (config.py에서)
# ============================================================
try:
    from src.config import DEEPL_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY  # 모듈로 실행할 때
    from src.utils.youtube import fetch_youtube_comment_map  # YouTube 댓글 수집 함수
    from src.models.schemas import CommentSentimentDetail, SentimentType, DetailSentimentType  # Pydantic 모델
except ImportError:
    from ..config import DEEPL_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY  # 패키지 내부에서 실행할 때
    from ..utils.youtube import fetch_youtube_comment_map
    from ..models.schemas import CommentSentimentDetail, SentimentType, DetailSentimentType

# ============================================================
# 1. 영어 GoEmotions 모델 사용 (이미 학습된 모델)
# ============================================================
_MNAME = "SamLowe/roberta-base-go_emotions"

_tok = None  # 토크나이저 (텍스트를 숫자로 변환하는 도구)
_model = None  # 감정 분류 모델
_pipe = None  # 파이프라인 (입력→전처리→모델→후처리를 한번에 처리)


def _get_sentiment_pipeline():
    """
    영어 GoEmotions 파이프라인 로드 (지연 로딩)

    작동 원리:
    1. 처음 호출될 때만 모델/토크나이저를 메모리에 로드
    2. 이후 호출에서는 이미 로드된 것을 재사용 (빠름!)
    3. GPU가 있으면 자동으로 GPU 사용 (속도 향상)

    반환: transformers의 pipeline 객체
    """
    global _tok, _model, _pipe

    # 이미 로드되어 있으면 그대로 반환
    if _pipe is None:
        print("[INFO] GoEmotions 모델 로딩 중...")

        # 토크나이저 로드: 텍스트를 모델이 이해할 수 있는 숫자로 변환
        _tok = AutoTokenizer.from_pretrained(_MNAME)

        # 모델 로드: 실제 감정 분류를 수행하는 신경망
        _model = AutoModelForSequenceClassification.from_pretrained(_MNAME)

        # ⭐ 파이프라인 생성: truncation, max_length 추가 (토큰 길이 제한)
        _pipe = pipeline(
            task="text-classification",  # 감정 분석 작업
            model=_model,  # 위에서 로드한 모델
            tokenizer=_tok,  # 위에서 로드한 토크나이저
            device=0 if torch.cuda.is_available() else -1,  # GPU 있으면 0번 GPU 사용, 없으면 CPU(-1)
            top_k=3,  # 상위 3개 감정 반환
            truncation=True,  # ⭐ 추가: 긴 텍스트 자동 자르기
            max_length=512,  # ⭐ 추가: 최대 512 토큰까지만 처리
            padding=True  # ⭐ 추가: 배치 처리 시 길이 맞추기
        )
        print("[SUCCESS] GoEmotions 모델 로딩 완료!")
        print(f"[INFO] 라벨 개수: {len(_model.config.id2label)}개")
        print(f"[INFO] 최대 토큰 길이: 512 (자동 truncation)")

    return _pipe


async def _translate_to_english(text: str) -> str:
    """
    한국어를 영어로 번역 (DeepL API 사용) - 단일 텍스트

    ⚠️ 권장하지 않음: 여러 댓글을 번역할 때는 translate_comments_batch를 사용하세요!

    Args:
        text: 번역할 한국어 텍스트

    Returns:
        번역된 영어 텍스트 (실패 시 원문 반환)
    """
    if not DEEPL_API_KEY:
        print("[WARNING] DeepL API 키가 없어 번역 생략")
        return text

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "https://api.deepl.com/v2/translate",
                data={
                    "auth_key": DEEPL_API_KEY,
                    "text": text,
                    "target_lang": "EN",
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()["translations"][0]["text"]
        except Exception as e:
            print(f"[ERROR] 번역 실패: {e}, 원문 사용")
            return text


# ============================================================
# ⭐ 배치 번역 함수 (export용)
# ============================================================
async def translate_comments_batch(texts: List[str]) -> List[str]:
    """
    댓글들을 배치로 번역 (DeepL API 사용)

    ✨ 이 함수는 server.py에서 Phase 0에서 호출되어 한 번만 번역을 수행합니다.

    Args:
        texts: 번역할 텍스트 리스트

    Returns:
        번역된 텍스트 리스트 (실패 시 원문 반환)
    """
    if not DEEPL_API_KEY:
        print("[WARNING] DeepL API 키가 없어 번역 생략")
        return texts

    print(f"[INFO] {len(texts)}개 댓글 병렬 번역 중...")

    # 병렬 번역 (asyncio.gather 사용)
    tasks = [_translate_to_english(t) for t in texts]
    translated = await asyncio.gather(*tasks)

    print(f"[INFO] 번역 완료!")

    return list(translated)


# 어떤 형태로 유튜브 댓글을 넣어도 tuple로 normalize 해주는 함수
def _normalize_input(
        comments: Union[Dict[str, str], List[Tuple[str, str]], List[str]]
) -> Tuple[List[str], List[str]]:
    """
    다양한 형태의 입력을 통일된 형태로 변환

    입력 가능한 형태:
    1. Dict[commentId, text]        → commentId를 그대로 사용
    2. List[Tuple[commentId, text]] → commentId를 그대로 사용
    3. List[str]                    → 자동으로 0, 1, 2... 인덱스 부여

    반환: (댓글 ID 리스트, 댓글 텍스트 리스트)

    예시:
    - {"c1": "좋아요", "c2": "싫어"} → (["c1", "c2"], ["좋아요", "싫어"])
    - [("c1", "좋아요"), ("c2", "싫어")] → (["c1", "c2"], ["좋아요", "싫어"])
    - ["좋아요", "싫어"] → (["0", "1"], ["좋아요", "싫어"])
    """
    # 경우 1: 딕셔너리 입력
    if isinstance(comments, dict):
        if comments:
            ids, texts = zip(*comments.items())  # 딕셔너리를 (키, 값) 튜플로 분리
            return list(ids), list(texts)
        return [], []

    # 경우 2: 튜플 리스트 입력 [(id, text), ...]
    if isinstance(comments, list) and comments and isinstance(comments[0], tuple):
        ids = [cid for cid, _t in comments]
        texts = [_t for _cid, _t in comments]
        return ids, texts

    # 경우 3: 단순 텍스트 리스트 ["text1", "text2", ...]
    if isinstance(comments, list):
        texts = list(comments)
        ids = [str(i) for i in range(len(texts))]  # 0, 1, 2... 인덱스 부여
        return ids, texts

    # 그 외의 경우: 빈 리스트 반환
    return [], []


async def analyze_sentiment_async(
        comments_dict: Union[Dict[str, str], List[Tuple[str, str]], List[str]],
        translated_texts: List[str] = None  # ⭐ 새로 추가된 파라미터
) -> Tuple[List[CommentSentimentDetail], Dict[str, float], List[str]]:
    """
    댓글들의 감정을 분석하는 비동기 함수

    입력:
      - comments_dict: 댓글 데이터 (Dict/List 형태)
      - translated_texts: (선택) 이미 번역된 텍스트 리스트 ⭐ 중복 번역 방지!

    출력:
      - (CommentSentimentDetail 리스트, 긍정/부정/기타 비율, 번역된 텍스트 리스트)

    처리 과정:
    1. 입력을 표준 형태로 변환
    2. 한국어 → 영어 번역 (이미 번역된 경우 생략!) ⭐
    3. 모델을 사용해 각 댓글의 감정 예측 (GoEmotions 28개 중 상위 3개)
    4. GoEmotions의 28개 감정 → 7개 감정으로 그룹핑
    5. 7개 감정을 POSITIVE/NEGATIVE/OTHER로 분류
    6. CommentSentimentDetail 객체 리스트 생성
    7. 번역된 텍스트 반환
    """
    # ============================================================
    # STEP 1: 입력 정규화
    # ============================================================
    ids, texts = _normalize_input(comments_dict)

    # 빈 입력이면 빈 결과 반환
    if not texts:
        return [], {}, []

    # ============================================================
    # STEP 2: 번역 (이미 번역되었으면 생략!) ⭐ 핵심 최적화
    # ============================================================
    if translated_texts is None:
        # 번역된 텍스트가 없으면 직접 번역
        print(f"[INFO] {len(texts)}개 댓글 병렬 번역 중...")
        tasks = [_translate_to_english(t) for t in texts]
        translated = await asyncio.gather(*tasks)
        print(f"[INFO] 번역 완료!")
    else:
        # 번역된 텍스트가 제공됨 (재사용)
        print(f"[INFO] ✨ 번역된 텍스트 재사용 ({len(translated_texts)}개)")
        translated = translated_texts

    # ============================================================
    # STEP 3: GoEmotions 예측 (28개 감정 중 상위 3개)
    # ============================================================
    pipe = _get_sentiment_pipeline()

    # ⭐ 이중 안전장치: pipe 호출 시에도 truncation 명시
    results = pipe(
        translated,
        batch_size=64,  # 64개씩 배치로 처리
        truncation=True,  # ⭐ 긴 텍스트 자동 자르기
        max_length=512  # ⭐ 최대 512 토큰
    )

    # ============================================================
    # STEP 4: GoEmotions 28개 → 프로젝트 7개 감정으로 매핑
    # ============================================================
    # GoEmotions 원본 라벨을 우리가 원하는 7개 카테고리로 그룹핑
    label_map = {
        # 긍정 감정들 → joy (기쁨)
        "admiration": "joy", "amusement": "joy", "approval": "joy",
        "excitement": "joy", "joy": "joy", "optimism": "joy",
        "pride": "joy", "relief": "joy",

        # 애정 관련 → love (사랑)
        "caring": "love", "desire": "love", "love": "love",

        # 감사 → gratitude (감사)
        "gratitude": "gratitude",

        # 분노 관련 → anger (분노)
        "anger": "anger", "annoyance": "anger",
        "disapproval": "anger", "disgust": "anger",

        # 슬픔 관련 → sadness (슬픔)
        "disappointment": "sadness", "embarrassment": "sadness",
        "grief": "sadness", "remorse": "sadness", "sadness": "sadness",

        # 두려움 관련 → fear (두려움)
        "fear": "fear", "nervousness": "fear",

        # 중립/기타 → neutral
        "confusion": "neutral", "curiosity": "neutral",
        "neutral": "neutral", "realization": "neutral", "surprise": "neutral",
    }

    # 7개 감정을 POSITIVE/NEGATIVE/OTHER로 매핑
    detail_to_sentiment_map = {
        "joy": "positive",
        "love": "positive",
        "gratitude": "positive",
        "anger": "negative",
        "sadness": "negative",
        "fear": "negative",
        "neutral": "other",
    }

    # ============================================================
    # STEP 5: CommentSentimentDetail 리스트 생성
    # ============================================================
    sentiment_comments: List[CommentSentimentDetail] = []
    sentiment_category_counter = Counter()  # POSITIVE/NEGATIVE/OTHER 카운트

    for cid, text, result in zip(ids, texts, results):
        # result는 이제 리스트 (top_k=3이므로 최대 3개)
        # 각 감정의 확률(score)이 15% 이상인 것만 선택
        detail_emotions = []

        for pred in result:
            original_label = pred["label"]
            score = pred["score"]

            # 확률이 15% 이상인 감정만 포함 (임계값)
            if score >= 0.15:
                detail_emotion = label_map.get(original_label, "neutral")
                detail_emotions.append(detail_emotion)

        # 감정이 없으면 neutral 추가 (안전장치)
        if not detail_emotions:
            detail_emotions = ["neutral"]

        # 중복 제거 (같은 감정이 여러 번 나올 수 있음)
        detail_emotions = list(dict.fromkeys(detail_emotions))

        # 가장 높은 점수의 감정(첫 번째)으로 전체 sentiment_type 결정
        primary_emotion = detail_emotions[0]
        sentiment_type = detail_to_sentiment_map[primary_emotion]

        # 카운트 증가
        sentiment_category_counter[sentiment_type] += 1

        # CommentSentimentDetail 객체 생성 (여러 세부 감정 포함)
        comment_detail = CommentSentimentDetail(
            apiCommentId=cid,
            content=text,
            sentimentType=SentimentType(sentiment_type),
            detailSentimentTypes=[DetailSentimentType(e) for e in detail_emotions]
        )
        sentiment_comments.append(comment_detail)

    # ============================================================
    # STEP 6: 긍정/부정/기타 비율 계산
    # ============================================================
    total = max(sum(sentiment_category_counter.values()), 1)
    sentiment_ratio = {
        "positive": round(sentiment_category_counter.get("positive", 0) / total * 100),
        "negative": round(sentiment_category_counter.get("negative", 0) / total * 100),
        "other": round(sentiment_category_counter.get("other", 0) / total * 100),
    }

    # ============================================================
    # 반환: (CommentSentimentDetail 리스트, 비율, 번역된 텍스트)
    # ============================================================
    return sentiment_comments, sentiment_ratio, translated


# ============================================================
# 사용 예시 (테스트용 코드)
# ============================================================
if __name__ == "__main__":
    # config.py에서 VIDEO_KEY 확인
    if not VIDEO_KEY:
        print("[ERROR] VIDEO_KEY가 .env 파일에 설정되지 않았습니다.")
        print("[INFO] 테스트 데이터로 실행합니다.")

        # 테스트 데이터
        test_comments = {
            "c1": "오늘 정말 행복한 하루였어요!",
            "c2": "너무 화가 나네요",
            "c3": "감사합니다",
            "c4": "무섭고 두려워요",
        }
        sentiment_comments, sentiment_ratio, translated = asyncio.run(
            analyze_sentiment_async(test_comments)
        )
    else:
        if not YOUTUBE_API_KEY:
            print("[ERROR] YOUTUBE_API_KEY가 .env 파일에 설정되지 않았습니다.")
            sys.exit(1)

        print(f"\n[INFO] YouTube 비디오 '{VIDEO_KEY}'에서 댓글 수집 중...")

        try:
            # YouTube 댓글 수집 (최대 100개)
            youtube_comments = fetch_youtube_comment_map(
                video_id=VIDEO_KEY,
                api_key=YOUTUBE_API_KEY,
                max_pages=1,  # 1페이지
                page_size=100,  # 페이지당 100개
                include_replies=False,  # 대댓글 제외
                apply_cleaning=True,  # 텍스트 전처리 적용
            )

            print(f"[SUCCESS] {len(youtube_comments)}개 댓글 수집 완료!")

            if not youtube_comments:
                print("[WARNING] 수집된 댓글이 없습니다.")
                sys.exit(0)

            # 감정 분석 실행
            sentiment_comments, sentiment_ratio, translated = asyncio.run(
                analyze_sentiment_async(youtube_comments)
            )

        except Exception as e:
            print(f"[ERROR] 댓글 수집 실패: {e}")
            sys.exit(1)

    # ============================================================
    # 결과 출력
    # ============================================================
    print("\n" + "=" * 60)
    print("[결과] 개별 댓글 감정 분석 (샘플 5개):")
    print("=" * 60)
    for i, comment in enumerate(sentiment_comments[:5], 1):
        print(f"  {i}. {comment.apiCommentId}")
        print(f"     내용: {comment.content[:50]}...")
        print(f"     감정 타입: {comment.sentimentType.value}")
        print(f"     세부 감정: {[d.value for d in comment.detailSentimentTypes]}")
    if len(sentiment_comments) > 5:
        print(f"  ... (총 {len(sentiment_comments)}개 댓글)")

    print("\n" + "=" * 60)
    print("[통계] 긍정/부정/기타 비율:")
    print("=" * 60)
    for sentiment, percentage in sentiment_ratio.items():
        print(f"  {sentiment:12s}: {percentage:6.2f}%")
    print("=" * 60)
