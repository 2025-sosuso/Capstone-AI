# scripts/test_finetuned_model.py
# fine tuning 한 모델이 실제 유튜브 댓글에서 잘 감정 세분화를 하는지 테스트하는 코드
"""
Fine-tuned KoELECTRA 모델 테스트
뒷광고 의심 댓글로 감정 분류 성능 확인
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# ============================================================
# 테스트 데이터
# ============================================================
TEST_COMMENTS = {
    "comment_001": "뒷광고 아닌가요?",
    "comment_002": "협찬 받으셨나요?",
    "comment_003": "돈 받고 홍보하시는 거죠?",
    "comment_004": "스폰서십 표기 안 하셨네요",
    "comment_005": "광고인지 밝히세요",
    "comment_006": "이건 명백한 광고인데요",
    "comment_007": "협찬 받고 거짓 리뷰",
    "comment_008": "유료 광고 표시 안 하셨네요",
    "comment_009": "돈 받고 추천하는 거 맞죠?",
    "comment_010": "뒷광고 신고합니다"
}

# ============================================================
# Fine-tuned 모델 로드
# ============================================================
MODEL_PATH = PROJECT_ROOT / "saved_models" / "ko-emotions_finetuned"

print("=" * 70)
print("🔬 Fine-tuned KoELECTRA 모델 테스트")
print("=" * 70)

print(f"\n📁 모델 경로: {MODEL_PATH}")
print(f"📊 테스트 댓글 수: {len(TEST_COMMENTS)}개\n")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 디바이스: {device}")

# 토크나이저 & 모델 로드
print("\n🔄 모델 로딩 중...")
tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH)
model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()  # 평가 모드
print("✅ 모델 로드 완료!\n")

# Label 정보 확인
id2label = model.config.id2label
label2id = model.config.label2id
print(f"🏷️  감정 레이블: {list(id2label.values())}")
print(f"🏷️  총 {len(id2label)}개 클래스\n")

# ============================================================
# 감정 분류 실행
# ============================================================
print("=" * 70)
print("📊 감정 분류 결과")
print("=" * 70)

results = []
# ⭐ 임계값 설정 (15% 이상인 감정만 포함)
THRESHOLD = 0.15

for comment_id, text in TEST_COMMENTS.items():
    # 토크나이징
    inputs = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # GPU로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 확률 계산 (softmax)
        probs = torch.softmax(logits, dim=-1)[0]
        
        # 가장 높은 확률의 감정
        pred_id = torch.argmax(probs).item()
        pred_label = id2label[pred_id]
        pred_prob = probs[pred_id].item()
        
        # 상위 3개 감정
        top3_probs, top3_ids = torch.topk(probs, k=3)
        top3_emotions = [(id2label[idx.item()], prob.item()) 
                        for idx, prob in zip(top3_ids, top3_probs)]
        
        # ⭐ 임계값 이상의 모든 감정 추출
        detected_emotions = []
        for i, prob in enumerate(probs):
            if prob.item() >= THRESHOLD:
                emotion = id2label[i]
                detected_emotions.append((emotion, prob.item()))
        
        # 확률 높은 순으로 정렬
        detected_emotions.sort(key=lambda x: x[1], reverse=True)
        
        # 감정이 없으면 neutral 추가 (안전장치)
        if not detected_emotions:
            detected_emotions = [("neutral", probs[label2id["neutral"]].item())]
    
    # 결과 저장
    result = {
        'id': comment_id,
        'text': text,
        'predicted': pred_label,  # 주요 감정
        'confidence': pred_prob,
        'top3': top3_emotions,
        'all_emotions': detected_emotions  # ⭐ 모든 감정 (임계값 이상)
    }
    results.append(result)
    
    # 출력
    print(f"\n📝 {comment_id}: {text}")
    print(f"   🎯 주요 감정: {pred_label} ({pred_prob*100:.1f}%)")
    print(f"   📊 Top 3:")
    for emotion, prob in top3_emotions:
        print(f"      - {emotion:12s}: {prob*100:5.1f}%")
    
    # ⭐ 감지된 모든 감정 표시 (임계값 {THRESHOLD*100}% 이상)
    print(f"   🎨 감지된 감정들 (임계값 {THRESHOLD*100}% 이상):")
    if detected_emotions:
        emotion_labels = [f"{e}({p*100:.1f}%)" for e, p in detected_emotions]
        print(f"      → {', '.join(emotion_labels)}")
    else:
        print(f"      → (없음)")

# ============================================================
# 통계 분석
# ============================================================
print("\n" + "=" * 70)
print("📈 통계 분석")
print("=" * 70)

from collections import Counter

# 감정별 분포 (주요 감정 기준)
emotion_counter = Counter([r['predicted'] for r in results])
print(f"\n감정별 분포 (주요 감정):")
for emotion, count in emotion_counter.most_common():
    percentage = count / len(results) * 100
    print(f"  {emotion:12s}: {count:2d}개 ({percentage:5.1f}%)")

# ⭐ 감정별 발생 빈도 (중복 포함 - 임계값 이상)
all_emotions_counter = Counter()
for r in results:
    for emotion, prob in r['all_emotions']:
        all_emotions_counter[emotion] += 1

print(f"\n감정별 발생 빈도 (임계값 {THRESHOLD*100}% 이상, 중복 가능):")
for emotion, count in all_emotions_counter.most_common():
    percentage = count / len(results) * 100
    print(f"  {emotion:12s}: {count:2d}회 ({percentage:5.1f}%)")

# ⭐ 감정 조합 분석
emotion_combinations = Counter()
for r in results:
    # 감정 레이블만 추출하여 튜플로 변환
    emotions = tuple([e for e, p in r['all_emotions']])
    emotion_combinations[emotions] += 1

print(f"\n감정 조합 빈도 (임계값 {THRESHOLD*100}% 이상):")
for emotions, count in emotion_combinations.most_common():
    emotion_str = ' + '.join(emotions) if emotions else '(없음)'
    print(f"  {emotion_str:40s}: {count:2d}개")

# 평균 신뢰도
avg_confidence = sum(r['confidence'] for r in results) / len(results)
print(f"\n평균 신뢰도: {avg_confidence*100:.1f}%")

# ⭐ 평균 감정 개수
avg_emotion_count = sum(len(r['all_emotions']) for r in results) / len(results)
print(f"댓글당 평균 감정 개수: {avg_emotion_count:.2f}개")

# 감정 매핑 (POSITIVE/NEGATIVE/OTHER)
emotion_to_category = {
    "joy": "positive",
    "gratitude": "positive",
    "anger": "negative",
    "sadness": "negative",
    "fear": "negative",
    "neutral": "other",
}

category_counter = Counter()
for r in results:
    category = emotion_to_category[r['predicted']]
    category_counter[category] += 1

print(f"\n카테고리별 분포:")
for category in ['negative', 'positive', 'other']:
    count = category_counter[category]
    percentage = count / len(results) * 100
    print(f"  {category:12s}: {count:2d}개 ({percentage:5.1f}%)")

print("\n" + "=" * 70)
print("✅ 테스트 완료!")
print("=" * 70)

# ============================================================
# ⭐ 상세 결과 테이블 출력 (선택)
# ============================================================
print("\n" + "=" * 70)
print("📋 상세 결과 테이블")
print("=" * 70)
print(f"\n{'ID':<15} {'텍스트':<30} {'주요 감정':<12} {'모든 감정'}")
print("-" * 100)
for r in results:
    text_short = r['text'][:27] + '...' if len(r['text']) > 30 else r['text']
    all_emotions_str = ', '.join([e for e, p in r['all_emotions']])
    print(f"{r['id']:<15} {text_short:<30} {r['predicted']:<12} {all_emotions_str}")
print("-" * 100)