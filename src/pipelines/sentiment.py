# src/pipelines/sentiment.py
from __future__ import annotations # 타입 힌팅할 때 사용함, 자기 참조하는 타입을 따옴표 없이 쓸 수 있게 해주는 것.

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

import asyncio # 비동기 모듈, fast api랑 연결하기 위해 지정함
from collections import Counter # 개수 count
from typing import Dict, List, Tuple, Union # 타입 힌팅을 위한 도구들

import torch # 딥러닝 프레임워크, 신경망 모델 학습 및 실행할 때 사용됨.
import httpx
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    # AutoTokenizer: 텍스트를 숫자로 변환해주는 도구
    # AutoModelForSequenceClassification: 텍스트 분류를 위한 모델
    # pipeline: 전처리 -> 모델 -> 후처리를 한 번에 처리해주는 도구

# ============================================================
# API 키 불러오기 (config.py에서)
# ============================================================
try:
    from src.config import DEEPL_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY  # 모듈로 실행할 때
    from src.utils.youtube import fetch_youtube_comment_map  # YouTube 댓글 수집 함수
    from src.models.schemas import CommentSentimentDetail, SentimentType, DetailSentimentType  # Pydantic 모델
except ImportError:
    from ..config import DEEPL_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY    # 패키지 내부에서 실행할 때
    from ..utils.youtube import fetch_youtube_comment_map
    from ..models.schemas import CommentSentimentDetail, SentimentType, DetailSentimentType

# ============================================================
# 1. 영어 GoEmotions 모델 사용 (이미 학습된 모델)
# ============================================================
_MNAME = "SamLowe/roberta-base-go_emotions"

_tok = None      # 토크나이저 (텍스트를 숫자로 변환하는 도구)
_model = None    # 감정 분류 모델
_pipe = None     # 파이프라인 (입력→전처리→모델→후처리를 한번에 처리)


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
        
        # 파이프라인 생성: 전처리→모델 예측→후처리를 한번에
        _pipe = pipeline(
            task="text-classification",           # 감정 분석 작업
            model=_model,                         # 위에서 로드한 모델
            tokenizer=_tok,                       # 위에서 로드한 토크나이저
            device=0 if torch.cuda.is_available() else -1,  # GPU 있으면 0번 GPU 사용, 없으면 CPU(-1)
            top_k=1,                              # 최고 점수 1개만 반환
        )
        print("[SUCCESS] GoEmotions 모델 로딩 완료!")
        print(f"[INFO] 라벨 개수: {len(_model.config.id2label)}개")
    
    return _pipe


async def _translate_to_english(text: str) -> str:
    """
    한국어를 영어로 번역 (DeepL API 사용)
    
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
    comments_dict: Union[Dict[str, str], List[Tuple[str, str]], List[str]]
) -> Tuple[List[CommentSentimentDetail], Dict[str, float]]:
    """
    댓글들의 감정을 분석하는 비동기 함수
    
    입력:
      - Dict[commentId, text]: 댓글 ID와 텍스트 쌍의 딕셔너리
      - List[Tuple[commentId, text]]: 댓글 ID와 텍스트 튜플의 리스트
      - List[text]: 텍스트만 있는 리스트
    
    출력:
      - (CommentSentimentDetail 리스트, 긍정/부정/기타 비율)
      - 예: ([CommentSentimentDetail(...), ...], {"POSITIVE": 50.0, "NEGATIVE": 25.0, "OTHER": 25.0})
    
    처리 과정:
    1. 입력을 표준 형태로 변환
    2. 한국어 → 영어 번역 (DeepL)
    3. 모델을 사용해 각 댓글의 감정 예측 (GoEmotions 28개)
    4. GoEmotions의 28개 감정 → 7개 감정으로 그룹핑
    5. 7개 감정을 POSITIVE/NEGATIVE/OTHER로 분류
    6. CommentSentimentDetail 객체 리스트 생성
    """
    # ============================================================
    # STEP 1: 입력 정규화
    # ============================================================
    ids, texts = _normalize_input(comments_dict)
    
    # 빈 입력이면 빈 결과 반환
    if not texts:
        return [], {}
    
    # ============================================================
    # STEP 2: 한국어 → 영어 번역
    # ============================================================
    print(f"[INFO] {len(texts)}개 댓글 번역 중...")
    translated = [await _translate_to_english(t) for t in texts]
    
    # ============================================================
    # STEP 3: GoEmotions 예측 (28개 감정)
    # ============================================================
    pipe = _get_sentiment_pipeline()
    results = pipe(translated, batch_size=64)  # 64개씩 배치로 처리 (속도 향상)
    
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
        "joy": "POSITIVE",
        "love": "POSITIVE",
        "gratitude": "POSITIVE",
        "anger": "NEGATIVE",
        "sadness": "NEGATIVE",
        "fear": "NEGATIVE",
        "neutral": "OTHER",
    }
    
    # ============================================================
    # STEP 5: CommentSentimentDetail 리스트 생성
    # ============================================================
    sentiment_comments: List[CommentSentimentDetail] = []
    sentiment_category_counter = Counter()  # POSITIVE/NEGATIVE/OTHER 카운트
    
    for cid, text, result in zip(ids, texts, results):
        # GoEmotions 28개를 7개 감정으로 변환
        original_label = result[0]["label"]
        detail_emotion = label_map.get(original_label, "neutral")
        
        # 7개 감정을 POSITIVE/NEGATIVE/OTHER로 변환
        sentiment_type = detail_to_sentiment_map[detail_emotion]
        
        # 카운트 증가
        sentiment_category_counter[sentiment_type] += 1
        
        # CommentSentimentDetail 객체 생성
        comment_detail = CommentSentimentDetail(
            apiCommentId=cid,
            content=text,
            sentimentType=SentimentType(sentiment_type),
            detailSentimentTypes=[DetailSentimentType(detail_emotion.upper())]  # 리스트로 감싸기
        )
        sentiment_comments.append(comment_detail)
    
    # ============================================================
    # STEP 6: 긍정/부정/기타 비율 계산
    # ============================================================
    total = max(sum(sentiment_category_counter.values()), 1)
    sentiment_ratio = {
        "POSITIVE": round(sentiment_category_counter.get("POSITIVE", 0) / total * 100, 2),
        "NEGATIVE": round(sentiment_category_counter.get("NEGATIVE", 0) / total * 100, 2),
        "OTHER": round(sentiment_category_counter.get("OTHER", 0) / total * 100, 2),
    }
    
    # ============================================================
    # 반환: (CommentSentimentDetail 리스트, 긍정/부정/기타 비율)
    # ============================================================
    return sentiment_comments, sentiment_ratio


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
        sentiment_comments, sentiment_ratio = asyncio.run(analyze_sentiment_async(test_comments))
    else:
        if not YOUTUBE_API_KEY:
            print("[ERROR] YOUTUBE_API_KEY가 .env 파일에 설정되지 않았습니다.")
            sys.exit(1)
        
        print(f"\n[INFO] YouTube 비디오 '{VIDEO_KEY}'에서 댓글 수집 중...")
        
        try:
            # YouTube 댓글 수집 (최대 300개: 3페이지 × 100개)
            youtube_comments = fetch_youtube_comment_map(
                video_id=VIDEO_KEY,
                api_key=YOUTUBE_API_KEY,
                max_pages=1,           # 1페이지 -> 시간 오래걸려서 3페이지에서 1페이지로 바꿈
                page_size=100,         # 페이지당 100개
                include_replies=False, # 대댓글 제외
                apply_cleaning=True,   # 텍스트 전처리 적용
            )
            
            print(f"[SUCCESS] {len(youtube_comments)}개 댓글 수집 완료!")
            
            if not youtube_comments:
                print("[WARNING] 수집된 댓글이 없습니다.")
                sys.exit(0)
            
            # 감정 분석 실행
            sentiment_comments, sentiment_ratio = asyncio.run(analyze_sentiment_async(youtube_comments))
            
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