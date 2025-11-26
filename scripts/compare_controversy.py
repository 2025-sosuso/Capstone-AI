"""
논란 탐지 모델 성능 비교: 1학기 vs 2학기
- 1학기: ["controversial", "non-controversial"] 라벨
- 2학기: ["direct accusation: fraud/scam/undisclosed promotion", "general comment"] 라벨
- DeepL API로 한국어 → 영어 번역 후 분석
"""
from __future__ import annotations

import re
import time
import asyncio
import httpx
from typing import List, Dict, Tuple
from dataclasses import dataclass

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
_MNAME = "facebook/bart-large-mnli"

# 1학기 설정
V1_LABELS = ["controversial", "non-controversial"]
V1_HYPO = "This text is {}."
V1_THRESHOLD = 0.7
V1_RATIO_THRESHOLD = 0.10

# 2학기 설정
V2_LABELS = [
    "direct accusation: this is fraud, scam, or undisclosed paid promotion",
    "general comment, opinion, or complaint"
]
V2_HYPO = "This comment is: {}."
V2_THRESHOLD = 0.35
V2_RATIO_THRESHOLD = 0.20
V2_CONTROVERSY_LABELS = V2_LABELS[:1]

# 공유 모델
_clf = None


def _get_classifier():
    """모델 지연 로딩"""
    global _clf
    if _clf is None:
        print("[INFO] BART 모델 로딩 중...")
        tok = AutoTokenizer.from_pretrained(_MNAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            _MNAME, use_safetensors=True
        )
        _clf = pipeline(
            task="zero-shot-classification",
            model=model,
            tokenizer=tok,
            device=0 if torch.cuda.is_available() else -1,
        )
        print("[SUCCESS] 모델 로딩 완료!")
    return _clf


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
    
    # DeepL API URL (유료 버전)
    url = "https://api.deepl.com/v2/translate"
    translated: List[str] = []
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        for text in texts:
            # 이미 영어면 번역 스킵
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
class CommentResult:
    text: str
    translated: str
    v1_score: float
    v2_score: float
    v1_flagged: bool
    v2_flagged: bool
    expected: bool


@dataclass
class ComparisonSummary:
    total_comments: int
    v1_flagged_count: int
    v2_flagged_count: int
    v1_accuracy: float
    v2_accuracy: float
    v1_precision: float
    v2_precision: float
    v1_recall: float
    v2_recall: float
    v1_f1: float
    v2_f1: float
    v1_time_ms: float
    v2_time_ms: float


# ============================================================
# 1학기 버전 (V1)
# ============================================================
def v1_controversy_scores(texts: List[str]) -> Tuple[List[float], float]:
    """1학기 방식: controversial vs non-controversial"""
    clf = _get_classifier()
    
    start = time.time()
    outputs = clf(
        texts,
        candidate_labels=V1_LABELS,
        hypothesis_template=V1_HYPO,
        batch_size=16,
        multi_label=False,
    )
    elapsed_ms = (time.time() - start) * 1000
    
    scores = []
    for out in outputs:
        lbls = out.get("labels", [])
        scrs = out.get("scores", [])
        score = 0.0
        for lbl, sc in zip(lbls, scrs):
            if lbl == "controversial":
                score = float(sc)
                break
        scores.append(score)
    
    return scores, elapsed_ms


# ============================================================
# 2학기 버전 (V2)
# ============================================================
def v2_controversy_scores(texts: List[str]) -> Tuple[List[float], float]:
    """2학기 방식: 구체적 사기/뒷광고 레이블"""
    clf = _get_classifier()
    
    start = time.time()
    outputs = clf(
        texts,
        candidate_labels=V2_LABELS,
        hypothesis_template=V2_HYPO,
        batch_size=16,
        multi_label=False,
    )
    elapsed_ms = (time.time() - start) * 1000
    
    scores = []
    for out in outputs:
        lbls = out.get("labels", [])
        scrs = out.get("scores", [])
        
        top_label = lbls[0] if lbls else ""
        top_score = scrs[0] if scrs else 0.0
        
        if top_label in V2_CONTROVERSY_LABELS:
            controversy_score = float(top_score)
        else:
            controversy_score = 0.0
        
        scores.append(controversy_score)
    
    return scores, elapsed_ms


# ============================================================
# 성능 메트릭 계산
# ============================================================
def calculate_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict[str, float]:
    """정확도, 정밀도, 재현율, F1 계산"""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    
    accuracy = (tp + tn) / len(predictions) if predictions else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


# ============================================================
# 메인 비교 함수
# ============================================================
def compare_versions(test_data: List[Tuple[str, bool]]) -> Tuple[List[CommentResult], ComparisonSummary, Dict, Dict]:
    """두 버전 비교 실행 (번역 포함)"""
    texts = [t[0] for t in test_data]
    ground_truth = [t[1] for t in test_data]
    
    print(f"\n{'='*70}")
    print(f"[비교 시작] 총 {len(texts)}개 댓글 분석")
    print(f"{'='*70}\n")
    
    # ============================================================
    # 1단계: DeepL 번역
    # ============================================================
    print("[번역] DeepL API로 한국어 → 영어 번역 중...")
    translate_start = time.time()
    translated_texts = translate_sync(texts)
    translate_time = (time.time() - translate_start) * 1000
    print(f"       완료! ({translate_time:.1f}ms, {len(translated_texts)}개 번역)")
    
    # 번역 샘플 출력
    print("\n[번역 샘플]")
    for i in [0, 40, 55]:  # 긍정, 논란, 질문 카테고리에서 샘플
        if i < len(texts):
            print(f"  원문: {texts[i][:30]}...")
            print(f"  번역: {translated_texts[i][:50]}...")
            print()
    
    # ============================================================
    # 2단계: V1 실행 (번역된 텍스트 사용)
    # ============================================================
    print("[V1] 1학기 버전 실행 중... (controversial/non-controversial)")
    v1_scores, v1_time = v1_controversy_scores(translated_texts)
    v1_flagged = [s >= V1_THRESHOLD for s in v1_scores]
    print(f"     완료! ({v1_time:.1f}ms)")
    
    # ============================================================
    # 3단계: V2 실행 (번역된 텍스트 사용)
    # ============================================================
    print("[V2] 2학기 버전 실행 중... (fraud/scam/promotion)")
    v2_scores, v2_time = v2_controversy_scores(translated_texts)
    v2_flagged = [s >= V2_THRESHOLD for s in v2_scores]
    print(f"     완료! ({v2_time:.1f}ms)")
    
    # 개별 결과 생성
    results = []
    for i, (text, expected) in enumerate(test_data):
        results.append(CommentResult(
            text=text,
            translated=translated_texts[i],
            v1_score=v1_scores[i],
            v2_score=v2_scores[i],
            v1_flagged=v1_flagged[i],
            v2_flagged=v2_flagged[i],
            expected=expected
        ))
    
    # 메트릭 계산
    v1_metrics = calculate_metrics(v1_flagged, ground_truth)
    v2_metrics = calculate_metrics(v2_flagged, ground_truth)
    
    summary = ComparisonSummary(
        total_comments=len(texts),
        v1_flagged_count=sum(v1_flagged),
        v2_flagged_count=sum(v2_flagged),
        v1_accuracy=v1_metrics["accuracy"],
        v2_accuracy=v2_metrics["accuracy"],
        v1_precision=v1_metrics["precision"],
        v2_precision=v2_metrics["precision"],
        v1_recall=v1_metrics["recall"],
        v2_recall=v2_metrics["recall"],
        v1_f1=v1_metrics["f1"],
        v2_f1=v2_metrics["f1"],
        v1_time_ms=v1_time,
        v2_time_ms=v2_time
    )
    
    return results, summary, v1_metrics, v2_metrics


def print_results(results: List[CommentResult], summary: ComparisonSummary, 
                  v1_metrics: Dict, v2_metrics: Dict):
    """결과 출력"""
    
    print(f"\n{'='*70}")
    print("📊 개별 댓글 분석 결과")
    print(f"{'='*70}")
    print(f"{'#':<3} {'정답':<6} {'V1점수':<8} {'V1판정':<8} {'V2점수':<8} {'V2판정':<8} 댓글")
    print("-" * 70)
    
    for i, r in enumerate(results, 1):
        expected_str = "🔴논란" if r.expected else "🟢정상"
        v1_flag_str = "⚠️탐지" if r.v1_flagged else "✅정상"
        v2_flag_str = "⚠️탐지" if r.v2_flagged else "✅정상"
        
        v1_correct = "✓" if r.v1_flagged == r.expected else "✗"
        v2_correct = "✓" if r.v2_flagged == r.expected else "✗"
        
        text_preview = r.text[:25] + "..." if len(r.text) > 25 else r.text
        
        print(f"{i:<3} {expected_str:<6} {r.v1_score:<8.3f} {v1_flag_str}{v1_correct:<3} "
              f"{r.v2_score:<8.3f} {v2_flag_str}{v2_correct:<3} {text_preview}")
    
    # 요약 통계
    print(f"\n{'='*70}")
    print("📈 성능 비교 요약")
    print(f"{'='*70}")
    print(f"{'메트릭':<20} {'1학기 (V1)':<20} {'2학기 (V2)':<20} {'차이':<15}")
    print("-" * 70)
    
    metrics = [
        ("정확도 (Accuracy)", summary.v1_accuracy, summary.v2_accuracy),
        ("정밀도 (Precision)", summary.v1_precision, summary.v2_precision),
        ("재현율 (Recall)", summary.v1_recall, summary.v2_recall),
        ("F1 Score", summary.v1_f1, summary.v2_f1),
    ]
    
    for name, v1, v2 in metrics:
        diff = v2 - v1
        diff_str = f"+{diff*100:.1f}%p" if diff >= 0 else f"{diff*100:.1f}%p"
        better = "⬆️" if diff > 0 else ("⬇️" if diff < 0 else "➡️")
        print(f"{name:<20} {v1*100:>6.1f}%{'':<12} {v2*100:>6.1f}%{'':<12} {diff_str} {better}")
    
    print("-" * 70)
    print(f"{'처리 시간':<20} {summary.v1_time_ms:>6.1f}ms{'':<12} {summary.v2_time_ms:>6.1f}ms")
    print(f"{'탐지 댓글 수':<20} {summary.v1_flagged_count:>6}개{'':<12} {summary.v2_flagged_count:>6}개")
    
    # Confusion Matrix
    print(f"\n{'='*70}")
    print("🔢 Confusion Matrix")
    print(f"{'='*70}")
    
    print("\n[V1 - 1학기]")
    print(f"              예측: 정상    예측: 논란")
    print(f"  실제 정상:    {v1_metrics['tn']:>4}         {v1_metrics['fp']:>4}")
    print(f"  실제 논란:    {v1_metrics['fn']:>4}         {v1_metrics['tp']:>4}")
    
    print("\n[V2 - 2학기]")
    print(f"              예측: 정상    예측: 논란")
    print(f"  실제 정상:    {v2_metrics['tn']:>4}         {v2_metrics['fp']:>4}")
    print(f"  실제 논란:    {v2_metrics['fn']:>4}         {v2_metrics['tp']:>4}")
    
    # 결론
    print(f"\n{'='*70}")
    print("📝 결론")
    print(f"{'='*70}")
    
    if summary.v2_f1 > summary.v1_f1:
        improvement = (summary.v2_f1 - summary.v1_f1) * 100
        print(f"✅ 2학기 버전이 F1 Score 기준 {improvement:.1f}%p 향상되었습니다.")
    elif summary.v1_f1 > summary.v2_f1:
        decline = (summary.v1_f1 - summary.v2_f1) * 100
        print(f"⚠️ 2학기 버전이 F1 Score 기준 {decline:.1f}%p 하락했습니다.")
    else:
        print("➡️ 두 버전의 F1 Score가 동일합니다.")
    
    print(f"\n[라벨 비교]")
    print(f"  V1: {V1_LABELS}")
    print(f"  V2: {V2_LABELS}")
    print(f"\n[임계값 비교]")
    print(f"  V1: 개별={V1_THRESHOLD}, 비율={V1_RATIO_THRESHOLD}")
    print(f"  V2: 개별={V2_THRESHOLD}, 비율={V2_RATIO_THRESHOLD}")


# ============================================================
# 카테고리별 분석
# ============================================================
def analyze_by_category(results: List[CommentResult]):
    """카테고리별 성능 분석 (CATEGORY_INFO 사용)"""
    
    print(f"\n{'='*70}")
    print("📂 카테고리별 분석 결과")
    print(f"{'='*70}")
    
    for cat_name, info in CATEGORY_INFO.items():
        start, end = info["range"]
        is_controversy_cat = info["expected"]
        
        cat_results = results[start:end]
        
        v1_correct = sum(1 for r in cat_results if r.v1_flagged == r.expected)
        v2_correct = sum(1 for r in cat_results if r.v2_flagged == r.expected)
        
        v1_flagged = sum(1 for r in cat_results if r.v1_flagged)
        v2_flagged = sum(1 for r in cat_results if r.v2_flagged)
        
        total = len(cat_results)
        
        print(f"\n[{cat_name}] ({total}개)")
        print(f"  정답 라벨: {'🔴 논란' if is_controversy_cat else '🟢 정상'}")
        print(f"  V1 정확도: {v1_correct}/{total} ({v1_correct/total*100:.1f}%) | 탐지: {v1_flagged}개")
        print(f"  V2 정확도: {v2_correct}/{total} ({v2_correct/total*100:.1f}%) | 탐지: {v2_flagged}개")
        
        diff = v2_correct - v1_correct
        if diff > 0:
            print(f"  → V2가 {diff}개 더 정확 ⬆️")
        elif diff < 0:
            print(f"  → V1이 {-diff}개 더 정확 ⬇️")
        else:
            print(f"  → 동일 ➡️")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔬 논란 탐지 모델 성능 비교: 1학기 vs 2학기")
    print("=" * 70)
    print("📁 데이터: test_comments_100.py (100개 댓글)")
    print("🌐 번역: DeepL API (한국어 → 영어)")
    
    # DeepL API 키 확인
    if DEEPL_API_KEY:
        print(f"✅ DEEPL_API_KEY 로드 완료")
    else:
        print(f"⚠️ DEEPL_API_KEY 없음 - 원문으로 분석됩니다")
    
    # 통계 출력
    stats = get_stats()
    print(f"\n[테스트 데이터 구성]")
    print(f"  총 댓글: {stats['total']}개")
    print(f"  ├─ 🔴 논란 댓글 (뒷광고/협찬): {stats['controversy']}개")
    print(f"  └─ 🟢 일반 댓글: {stats['normal']}개")
    
    for cat_name, info in CATEGORY_INFO.items():
        start, end = info["range"]
        label = "🔴" if info["expected"] else "🟢"
        print(f"       ├─ {label} {cat_name}: {end - start}개")
    
    # 비교 실행
    results, summary, v1_metrics, v2_metrics = compare_versions(TEST_COMMENTS)
    
    # 결과 출력
    print_results(results, summary, v1_metrics, v2_metrics)
    
    # 카테고리별 분석
    analyze_by_category(results)
    
    print("\n" + "=" * 70)
    print("✅ 비교 분석 완료!")
    print("=" * 70 + "\n")