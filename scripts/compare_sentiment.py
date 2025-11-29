"""
감정 분석 모델 성능 평가: SamLowe/roberta-base-go_emotions
- test_comments_100.py 데이터셋 사용
- GoEmotions 28개 → 7개 → 3개(positive/negative/other) 매핑
- DeepL API로 한국어 → 영어 번역 후 분석
"""
from __future__ import annotations

import re
import time
import asyncio
import httpx
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 테스트 데이터 import
try:
    from scripts.test_comments_100 import TEST_COMMENTS, CATEGORY_INFO, get_stats
except ModuleNotFoundError:
    from test_comments_100 import TEST_COMMENTS, CATEGORY_INFO, get_stats

# DeepL API 키 import
try:
    from src.config import DEEPL_API_KEY
except ModuleNotFoundError:
    try:
        from config import DEEPL_API_KEY
    except ModuleNotFoundError:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

# ============================================================
# 모델 설정
# ============================================================
_MNAME = "SamLowe/roberta-base-go_emotions"

# GoEmotions 28개 → 7개 감정 매핑
LABEL_MAP = {
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

# 7개 감정 → 3개 카테고리 매핑
DETAIL_TO_SENTIMENT = {
    "joy": "positive",
    "love": "positive",
    "gratitude": "positive",
    "anger": "negative",
    "sadness": "negative",
    "fear": "negative",
    "neutral": "other",
}

# 공유 모델
_pipe = None


def _get_pipeline():
    """모델 지연 로딩"""
    global _pipe
    if _pipe is None:
        print("[INFO] GoEmotions 모델 로딩 중...")
        tok = AutoTokenizer.from_pretrained(_MNAME)
        model = AutoModelForSequenceClassification.from_pretrained(_MNAME)
        _pipe = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tok,
            device=0 if torch.cuda.is_available() else -1,
            top_k=3,  # 상위 3개 감정 반환
            truncation=True,
            max_length=512,
        )
        print("[SUCCESS] GoEmotions 모델 로딩 완료!")
    return _pipe


# ============================================================
# DeepL 번역 함수
# ============================================================
def _is_english(text: str) -> bool:
    """영어인지 확인"""
    cleaned = re.sub(r"[^\w\s.,!?\'\"-]", "", text or "")
    return bool(re.fullmatch(r"[A-Za-z0-9\s\.,;:'\"!?()\[\]{}@#$%^&*_\-=+/<>|~]+", cleaned))


async def translate_to_english(texts: List[str]) -> List[str]:
    """DeepL API로 한국어 → 영어 번역 (비동기)"""
    if not DEEPL_API_KEY:
        print("[WARN] DEEPL_API_KEY가 없습니다. 원문으로 분석합니다.")
        return texts
    
    url = "https://api.deepl.com/v2/translate"
    translated: List[str] = []
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        for text in texts:
            if _is_english(text):
                translated.append(text)
                continue
            
            try:
                response = await client.post(
                    url,
                    data={
                        "auth_key": DEEPL_API_KEY,
                        "text": text,
                        "target_lang": "EN"
                    }
                )
                response.raise_for_status()
                result = response.json()
                translated.append(result["translations"][0]["text"])
            except Exception as e:
                print(f"[WARN] 번역 실패: {text[:20]}... → 원문 사용")
                translated.append(text)
    
    return translated


def translate_sync(texts: List[str]) -> List[str]:
    """동기 버전 번역 함수"""
    return asyncio.run(translate_to_english(texts))


# ============================================================
# 결과 데이터 클래스
# ============================================================
@dataclass
class SentimentResult:
    text: str
    translated: str
    expected: str  # 정답 라벨 (positive/negative/other)
    predicted: str  # 예측 라벨
    detail_emotions: List[str]  # 7개 감정 중 감지된 것들
    correct: bool


# ============================================================
# 감정 분석 함수
# ============================================================
def analyze_sentiment(texts: List[str]) -> Tuple[List[str], List[List[str]], float]:
    """
    감정 분석 수행
    
    Returns:
        (예측 라벨 리스트, 세부 감정 리스트, 처리 시간)
    """
    pipe = _get_pipeline()
    
    start = time.time()
    results = pipe(texts, batch_size=64, truncation=True, max_length=512)
    elapsed_ms = (time.time() - start) * 1000
    
    predictions = []
    detail_emotions_list = []
    
    for result in results:
        # 감정과 점수를 함께 저장
        emotion_scores = {}
        
        for pred in result:
            original_label = pred["label"]
            score = pred["score"]
            
            # 15% 이상인 감정만 포함
            if score >= 0.15:
                detail_emotion = LABEL_MAP.get(original_label, "neutral")
                if detail_emotion not in emotion_scores or score > emotion_scores[detail_emotion]:
                    emotion_scores[detail_emotion] = score
        
        # neutral과 다른 감정이 함께 있을 때 처리
        if "neutral" in emotion_scores and len(emotion_scores) > 1:
            neutral_score = emotion_scores["neutral"]
            other_emotions = {k: v for k, v in emotion_scores.items() if k != "neutral"}
            max_other_score = max(other_emotions.values())
            
            if max_other_score >= neutral_score:
                del emotion_scores["neutral"]
            else:
                emotion_scores = {"neutral": neutral_score}
        
        # 감정이 없으면 neutral 추가
        if not emotion_scores:
            emotion_scores = {"neutral": 0.0}
        
        # 점수 높은 순으로 정렬
        detail_emotions = sorted(
            emotion_scores.keys(),
            key=lambda x: emotion_scores[x],
            reverse=True
        )
        
        # 최종 감정 결정 (가장 높은 점수)
        primary_emotion = detail_emotions[0]
        sentiment_type = DETAIL_TO_SENTIMENT[primary_emotion]
        
        predictions.append(sentiment_type)
        detail_emotions_list.append(detail_emotions)
    
    return predictions, detail_emotions_list, elapsed_ms


# ============================================================
# 메트릭 계산
# ============================================================
def calculate_metrics(predictions: List[str], ground_truth: List[str]) -> Dict:
    """다중 클래스 분류 메트릭 계산"""
    classes = ["positive", "negative", "other"]
    
    # 전체 정확도
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions) if predictions else 0
    
    # 클래스별 메트릭
    class_metrics = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != cls and g == cls)
        tn = sum(1 for p, g in zip(predictions, ground_truth) if p != cls and g != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": tp + fn  # 실제 해당 클래스 개수
        }
    
    # Macro Average (클래스별 평균)
    macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(classes)
    macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(classes)
    macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(classes)
    
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
    }


# ============================================================
# Confusion Matrix 생성
# ============================================================
def create_confusion_matrix(predictions: List[str], ground_truth: List[str]) -> Dict:
    """Confusion Matrix 생성"""
    classes = ["positive", "negative", "other"]
    matrix = {actual: {pred: 0 for pred in classes} for actual in classes}
    
    for pred, actual in zip(predictions, ground_truth):
        matrix[actual][pred] += 1
    
    return matrix


# ============================================================
# 메인 평가 함수
# ============================================================
def evaluate_sentiment():
    """감정 분석 모델 평가 실행"""
    print("\n" + "=" * 70)
    print("🎭 감정 분석 모델 성능 평가: GoEmotions")
    print("=" * 70)
    print("📁 데이터: test_comments_100.py (100개 댓글)")
    print("🌐 번역: DeepL API (한국어 → 영어)")
    print(f"🤖 모델: {_MNAME}")
    
    # API 키 확인
    if DEEPL_API_KEY:
        print("✅ DEEPL_API_KEY 로드 완료")
    else:
        print("⚠️ DEEPL_API_KEY 없음 - 원문으로 분석됩니다")
    
    # 통계 출력
    stats = get_stats()
    print(f"\n[테스트 데이터 구성]")
    print(f"  총 댓글: {stats['total']}개")
    print(f"  😊 긍정 (positive): {stats['positive']}개")
    print(f"  😢 부정 (negative): {stats['negative']}개")
    print(f"  😐 중립 (other): {stats['other']}개")
    
    # 데이터 추출
    texts = [t[0] for t in TEST_COMMENTS]
    ground_truth = [t[2] for t in TEST_COMMENTS]  # 감정 라벨
    
    print(f"\n{'='*70}")
    print(f"[평가 시작] 총 {len(texts)}개 댓글 분석")
    print(f"{'='*70}\n")
    
    # 1. 번역
    print("[번역] DeepL API로 한국어 → 영어 번역 중...")
    translate_start = time.time()
    translated_texts = translate_sync(texts)
    translate_time = (time.time() - translate_start) * 1000
    print(f"       완료! ({translate_time:.1f}ms)")
    
    # 번역 샘플
    print("\n[번역 샘플]")
    for i in [0, 20, 55]:
        if i < len(texts):
            print(f"  원문: {texts[i][:30]}...")
            print(f"  번역: {translated_texts[i][:50]}...")
            print()
    
    # 2. 감정 분석
    print("[분석] GoEmotions 모델로 감정 분석 중...")
    predictions, detail_emotions_list, analysis_time = analyze_sentiment(translated_texts)
    print(f"       완료! ({analysis_time:.1f}ms)")
    
    # 3. 결과 생성
    results = []
    for i, (text, expected, pred, details) in enumerate(zip(texts, ground_truth, predictions, detail_emotions_list)):
        results.append(SentimentResult(
            text=text,
            translated=translated_texts[i],
            expected=expected,
            predicted=pred,
            detail_emotions=details,
            correct=(expected == pred)
        ))
    
    # 4. 메트릭 계산
    metrics = calculate_metrics(predictions, ground_truth)
    confusion = create_confusion_matrix(predictions, ground_truth)
    
    # 5. 결과 출력
    print_results(results, metrics, confusion, analysis_time)
    
    # 6. 카테고리별 분석
    analyze_by_category(results)
    
    print("\n" + "=" * 70)
    print("✅ 감정 분석 평가 완료!")
    print("=" * 70 + "\n")


def print_results(results: List[SentimentResult], metrics: Dict, 
                  confusion: Dict, time_ms: float):
    """결과 출력"""
    
    print(f"\n{'='*70}")
    print("📊 개별 댓글 분석 결과")
    print(f"{'='*70}")
    print(f"{'#':<4} {'정답':<10} {'예측':<10} {'결과':<6} {'세부감정':<20} 댓글")
    print("-" * 70)
    
    emoji_map = {
        "positive": "😊긍정",
        "negative": "😢부정",
        "other": "😐중립"
    }
    
    for i, r in enumerate(results, 1):
        expected_str = emoji_map[r.expected]
        predicted_str = emoji_map[r.predicted]
        correct_str = "✅" if r.correct else "❌"
        details_str = ", ".join(r.detail_emotions[:2])
        text_preview = r.text[:20] + "..." if len(r.text) > 20 else r.text
        
        print(f"{i:<4} {expected_str:<10} {predicted_str:<10} {correct_str:<6} {details_str:<20} {text_preview}")
    
    # 성능 요약
    print(f"\n{'='*70}")
    print("📈 성능 요약")
    print(f"{'='*70}")
    print(f"{'메트릭':<25} {'값':<15}")
    print("-" * 40)
    print(f"{'정확도 (Accuracy)':<25} {metrics['accuracy']*100:>6.1f}%")
    print(f"{'Macro Precision':<25} {metrics['macro_precision']*100:>6.1f}%")
    print(f"{'Macro Recall':<25} {metrics['macro_recall']*100:>6.1f}%")
    print(f"{'Macro F1 Score':<25} {metrics['macro_f1']*100:>6.1f}%")
    print(f"{'처리 시간':<25} {time_ms:>6.1f}ms")
    
    # 클래스별 성능
    print(f"\n{'='*70}")
    print("📊 클래스별 성능")
    print(f"{'='*70}")
    print(f"{'클래스':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Support':<10}")
    print("-" * 58)
    
    for cls, m in metrics["class_metrics"].items():
        cls_name = emoji_map[cls]
        print(f"{cls_name:<12} {m['precision']*100:>6.1f}%{'':<5} {m['recall']*100:>6.1f}%{'':<5} "
              f"{m['f1']*100:>6.1f}%{'':<5} {m['support']:>4}개")
    
    # Confusion Matrix
    print(f"\n{'='*70}")
    print("🔢 Confusion Matrix")
    print(f"{'='*70}")
    print(f"\n{'':>15} {'예측:긍정':>12} {'예측:부정':>12} {'예측:중립':>12}")
    
    for actual in ["positive", "negative", "other"]:
        actual_name = {"positive": "실제:긍정", "negative": "실제:부정", "other": "실제:중립"}[actual]
        row = confusion[actual]
        print(f"{actual_name:>15} {row['positive']:>12} {row['negative']:>12} {row['other']:>12}")


def analyze_by_category(results: List[SentimentResult]):
    """카테고리별 분석"""
    
    print(f"\n{'='*70}")
    print("📂 카테고리별 분석 결과")
    print(f"{'='*70}")
    
    for cat_name, info in CATEGORY_INFO.items():
        start, end = info["range"]
        expected_sentiment = info.get("expected_sentiment", "mixed")
        
        cat_results = results[start:end]
        correct_count = sum(1 for r in cat_results if r.correct)
        total = len(cat_results)
        
        # 예측 분포
        pred_counter = Counter(r.predicted for r in cat_results)
        
        print(f"\n[{cat_name}] ({total}개)")
        print(f"  기대 감정: {expected_sentiment}")
        print(f"  정확도: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
        print(f"  예측 분포: 😊{pred_counter.get('positive', 0)} | 😢{pred_counter.get('negative', 0)} | 😐{pred_counter.get('other', 0)}")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    evaluate_sentiment()