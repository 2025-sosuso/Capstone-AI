# scripts/compare_models.py
"""
Fine-tuned KoELECTRA vs 영어 GoEmotions 모델 비교
100개 YouTube 댓글로 성능 측정
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter
import asyncio

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from transformers import (
    ElectraTokenizer, 
    ElectraForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# ============================================================
# 100개 테스트 댓글
# ============================================================
from scripts.test_comments_100 import YOUTUBE_COMMENTS_100

# ============================================================
# 모델 1: Fine-tuned KoELECTRA
# ============================================================
class KoELECTRAModel:
    def __init__(self):
        self.name = "Fine-tuned KoELECTRA"
        self.model_path = PROJECT_ROOT / "saved_models" / "ko-emotions_finetuned"
        self.threshold = 0.15
        
        print(f"\n{'='*70}")
        print(f"📦 {self.name} 로딩 중...")
        print(f"{'='*70}")
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 로드
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_path)
        self.model = ElectraForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 메타데이터
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.num_labels = len(self.id2label)
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        
        # 학습 정보 로드
        with open(self.model_path / "training_config.json", "r", encoding="utf-8") as f:
            self.training_info = json.load(f)
        
        print(f"✅ 모델 로드 완료!")
        print(f"   - 감정 레이블: {list(self.id2label.values())}")
        print(f"   - 파라미터 수: {self.num_parameters:,}개")
        print(f"   - 디바이스: {self.device}")
    
    def predict(self, texts):
        """댓글 리스트를 입력받아 감정 분석"""
        results = []
        
        for text in texts:
            # 토크나이징
            inputs = self.tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 예측
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
                
                # 주요 감정
                pred_id = torch.argmax(probs).item()
                pred_label = self.id2label[pred_id]
                pred_prob = probs[pred_id].item()
                
                # 임계값 이상 감정 추출
                detected_emotions = []
                for i, prob in enumerate(probs):
                    if prob.item() >= self.threshold:
                        emotion = self.id2label[i]
                        detected_emotions.append((emotion, prob.item()))
                
                detected_emotions.sort(key=lambda x: x[1], reverse=True)
                
                if not detected_emotions:
                    detected_emotions = [(pred_label, pred_prob)]
            
            results.append({
                'primary_emotion': pred_label,
                'confidence': pred_prob,
                'all_emotions': [e for e, p in detected_emotions],
                'all_scores': {e: p for e, p in detected_emotions}
            })
        
        return results


# ============================================================
# 모델 2: 영어 GoEmotions (번역 필요)
# ============================================================
class GoEmotionsModel:
    def __init__(self):
        self.name = "SamLowe/roberta-base-go_emotions"
        self.threshold = 0.15
        
        print(f"\n{'='*70}")
        print(f"📦 {self.name} 로딩 중...")
        print(f"{'='*70}")
        
        # 디바이스 설정
        self.device = 0 if torch.cuda.is_available() else -1
        
        # 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name)
        
        # 파이프라인 생성
        self.pipe = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            top_k=None,  # 모든 감정 반환
        )
        
        # 메타데이터
        self.num_labels = len(self.model.config.id2label)
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        
        # GoEmotions → 프로젝트 감정 매핑
        self.label_map = {
            "admiration": "joy", "amusement": "joy", "approval": "joy",
            "excitement": "joy", "joy": "joy", "optimism": "joy",
            "pride": "joy", "relief": "joy",
            "caring": "love", "desire": "love", "love": "love",
            "gratitude": "gratitude",
            "anger": "anger", "annoyance": "anger",
            "disapproval": "anger", "disgust": "anger",
            "disappointment": "sadness", "embarrassment": "sadness",
            "grief": "sadness", "remorse": "sadness", "sadness": "sadness",
            "fear": "fear", "nervousness": "fear",
            "confusion": "neutral", "curiosity": "neutral",
            "neutral": "neutral", "realization": "neutral", "surprise": "neutral",
        }
        
        print(f"✅ 모델 로드 완료!")
        print(f"   - 원본 레이블: {self.num_labels}개 (GoEmotions)")
        print(f"   - 매핑 후: 7개 (joy, love, gratitude, anger, sadness, fear, neutral)")
        print(f"   - 파라미터 수: {self.num_parameters:,}개")
        print(f"   - 디바이스: {'GPU' if self.device == 0 else 'CPU'}")
    
    async def translate_batch(self, texts):
        """DeepL API로 배치 번역 (실제로는 간단히 처리)"""
        # 실제로는 DeepL API 호출
        # 여기서는 시뮬레이션
        print(f"   🌐 {len(texts)}개 댓글 번역 중...")
        await asyncio.sleep(0.1)  # 번역 시간 시뮬레이션
        return texts  # 실제로는 번역된 텍스트 반환
    
    def predict(self, texts):
        """댓글 리스트를 입력받아 감정 분석"""
        # 영어로 번역 (실제로는 DeepL 사용)
        # 여기서는 원문 그대로 사용 (시뮬레이션)
        
        results_raw = self.pipe(texts, batch_size=64)
        
        results = []
        for result in results_raw:
            # 매핑 및 임계값 적용
            mapped_emotions = {}
            for pred in result:
                original_label = pred["label"]
                score = pred["score"]
                
                if score >= self.threshold:
                    mapped_label = self.label_map.get(original_label, "neutral")
                    if mapped_label in mapped_emotions:
                        mapped_emotions[mapped_label] += score
                    else:
                        mapped_emotions[mapped_label] = score
            
            if not mapped_emotions:
                mapped_emotions = {"neutral": 1.0}
            
            # 확률 높은 순 정렬
            sorted_emotions = sorted(
                mapped_emotions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            primary_emotion = sorted_emotions[0][0]
            confidence = sorted_emotions[0][1]
            
            results.append({
                'primary_emotion': primary_emotion,
                'confidence': confidence,
                'all_emotions': [e for e, s in sorted_emotions],
                'all_scores': dict(sorted_emotions)
            })
        
        return results


# ============================================================
# 평가 함수
# ============================================================
def evaluate_model(model, comments_dict):
    """모델 평가 및 통계 계산"""
    print(f"\n{'='*70}")
    print(f"🔍 {model.name} 평가 중...")
    print(f"{'='*70}")
    
    # 댓글 리스트 생성
    comment_ids = list(comments_dict.keys())
    texts = list(comments_dict.values())
    
    # 처리 시간 측정
    start_time = time.time()
    predictions = model.predict(texts)
    processing_time = time.time() - start_time
    
    # 통계 계산
    confidences = [p['confidence'] for p in predictions]
    avg_confidence = sum(confidences) / len(confidences)
    
    emotion_counts = [len(p['all_emotions']) for p in predictions]
    avg_emotion_count = sum(emotion_counts) / len(emotion_counts)
    
    # 감정별 분포 (주요 감정)
    primary_emotions = [p['primary_emotion'] for p in predictions]
    emotion_counter = Counter(primary_emotions)
    
    # 감정별 발생 빈도 (중복 포함)
    all_emotions_counter = Counter()
    for p in predictions:
        for emotion in p['all_emotions']:
            all_emotions_counter[emotion] += 1
    
    # 카테고리 매핑
    emotion_to_category = {
        "joy": "positive",
        "love": "positive",
        "gratitude": "positive",
        "anger": "negative",
        "sadness": "negative",
        "fear": "negative",
        "neutral": "other",
    }
    
    category_counter = Counter()
    for emotion in primary_emotions:
        category = emotion_to_category.get(emotion, "other")
        category_counter[category] += 1
    
    # anger 탐지율
    anger_count = sum(1 for p in predictions if 'anger' in p['all_emotions'])
    anger_detection_rate = anger_count / len(predictions) * 100
    
    # neutral 편향도 (주요 감정이 neutral인 비율)
    neutral_count = sum(1 for p in predictions if p['primary_emotion'] == 'neutral')
    neutral_bias = neutral_count / len(predictions) * 100
    
    print(f"✅ 평가 완료! (처리 시간: {processing_time:.2f}초)")
    
    return {
        'model_name': model.name,
        'num_parameters': model.num_parameters,
        'num_labels': model.num_labels,
        'processing_time': processing_time,
        'avg_confidence': avg_confidence,
        'avg_emotion_count': avg_emotion_count,
        'emotion_distribution': dict(emotion_counter),
        'all_emotions_frequency': dict(all_emotions_counter),
        'category_distribution': dict(category_counter),
        'anger_detection_rate': anger_detection_rate,
        'neutral_bias': neutral_bias,
        'predictions': predictions,
        'comment_ids': comment_ids,
        'texts': texts,
    }


# ============================================================
# 비교 표 생성
# ============================================================
def generate_comparison_table(results_ko, results_en):
    """두 모델의 비교 표 생성"""
    
    print("\n" + "="*100)
    print("📊 모델 비교 요약표")
    print("="*100)
    
    # 기본 정보
    print(f"\n{'='*100}")
    print(f"{'항목':<30} {'Fine-tuned KoELECTRA':<35} {'GoEmotions (영어)':<35}")
    print(f"{'='*100}")
    
    print(f"{'모델 이름':<30} {results_ko['model_name']:<35} {results_en['model_name']:<35}")
    print(f"{'파라미터 수':<30} {results_ko['num_parameters']:,}개{' '*15} {results_en['num_parameters']:,}개")
    print(f"{'감정 레이블 수':<30} {results_ko['num_labels']}개{' '*30} {results_en['num_labels']}개 (원본 28개)")
    print(f"{'언어':<30} {'한국어 직접':<35} {'영어 (번역 필요)':<35}")
    print(f"{'번역 필요':<30} {'❌ 불필요':<35} {'✅ 필요 (DeepL)':<35}")
    
    # 성능 지표
    print(f"\n{'='*100}")
    print(f"{'성능 지표':<30} {'Fine-tuned KoELECTRA':<35} {'GoEmotions (영어)':<35}")
    print(f"{'='*100}")
    
    print(f"{'평균 신뢰도':<30} {results_ko['avg_confidence']*100:>6.1f}%{' '*27} {results_en['avg_confidence']*100:>6.1f}%")
    print(f"{'평균 감정 개수':<30} {results_ko['avg_emotion_count']:>6.2f}개{' '*26} {results_en['avg_emotion_count']:>6.2f}개")
    print(f"{'처리 속도 (100개)':<30} {results_ko['processing_time']:>6.2f}초{' '*26} {results_en['processing_time']:>6.2f}초")
    
    # anger 탐지율
    print(f"{'anger 탐지율':<30} {results_ko['anger_detection_rate']:>6.1f}%{' '*27} {results_en['anger_detection_rate']:>6.1f}%")
    print(f"{'neutral 편향도':<30} {results_ko['neutral_bias']:>6.1f}%{' '*27} {results_en['neutral_bias']:>6.1f}%")
    
    # 카테고리 분포
    print(f"\n{'='*100}")
    print(f"{'카테고리 분포':<30} {'Fine-tuned KoELECTRA':<35} {'GoEmotions (영어)':<35}")
    print(f"{'='*100}")
    
    for category in ['positive', 'negative', 'other']:
        ko_count = results_ko['category_distribution'].get(category, 0)
        en_count = results_en['category_distribution'].get(category, 0)
        ko_pct = ko_count / 100 * 100
        en_pct = en_count / 100 * 100
        print(f"{category:<30} {ko_count}개 ({ko_pct:>5.1f}%){' '*18} {en_count}개 ({en_pct:>5.1f}%)")
    
    # 실용성
    print(f"\n{'='*100}")
    print(f"{'실용성 평가':<30} {'Fine-tuned KoELECTRA':<35} {'GoEmotions (영어)':<35}")
    print(f"{'='*100}")
    
    print(f"{'API 비용':<30} {'무료 (번역 불필요)':<35} {'유료 (DeepL 필요)':<35}")
    print(f"{'도메인 적합성':<30} {'❌ 낮음 (대화 데이터)':<35} {'⚠️ 보통 (Reddit)':<35}")
    print(f"{'배포 용이성':<30} {'✅ 좋음 (직접 사용)':<35} {'⚠️ 보통 (번역 필요)':<35}")
    
    print(f"\n{'='*100}\n")
    
    # 상세 비교 (샘플 20개)
    print(f"\n{'='*100}")
    print(f"📋 상세 비교 (샘플 20개)")
    print(f"{'='*100}\n")
    
    print(f"{'ID':<15} {'댓글 (일부)':<40} {'KoELECTRA':<20} {'GoEmotions':<20}")
    print(f"{'-'*100}")
    
    for i in range(min(20, len(results_ko['texts']))):
        comment_id = results_ko['comment_ids'][i]
        text = results_ko['texts'][i][:37] + '...' if len(results_ko['texts'][i]) > 40 else results_ko['texts'][i]
        
        ko_emotion = results_ko['predictions'][i]['primary_emotion']
        ko_conf = results_ko['predictions'][i]['confidence'] * 100
        
        en_emotion = results_en['predictions'][i]['primary_emotion']
        en_conf = results_en['predictions'][i]['confidence'] * 100
        
        ko_str = f"{ko_emotion}({ko_conf:.0f}%)"
        en_str = f"{en_emotion}({en_conf:.0f}%)"
        
        print(f"{comment_id:<15} {text:<40} {ko_str:<20} {en_str:<20}")
    
    print(f"{'-'*100}\n")


# ============================================================
# CSV 저장
# ============================================================
def save_detailed_results(results_ko, results_en):
    """상세 결과를 CSV로 저장"""
    
    data = []
    for i in range(len(results_ko['texts'])):
        row = {
            'comment_id': results_ko['comment_ids'][i],
            'text': results_ko['texts'][i],
            
            'ko_primary': results_ko['predictions'][i]['primary_emotion'],
            'ko_confidence': results_ko['predictions'][i]['confidence'],
            'ko_all_emotions': ', '.join(results_ko['predictions'][i]['all_emotions']),
            
            'en_primary': results_en['predictions'][i]['primary_emotion'],
            'en_confidence': results_en['predictions'][i]['confidence'],
            'en_all_emotions': ', '.join(results_en['predictions'][i]['all_emotions']),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    output_path = PROJECT_ROOT / "model_comparison_results.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 상세 결과 저장 완료: {output_path}")


# ============================================================
# 메인 실행
# ============================================================
def main():
    print("\n" + "="*100)
    print("🔬 Fine-tuned KoELECTRA vs GoEmotions 모델 비교")
    print("="*100)
    print(f"📊 테스트 댓글: {len(YOUTUBE_COMMENTS_100)}개")
    
    # 모델 1: KoELECTRA
    model_ko = KoELECTRAModel()
    results_ko = evaluate_model(model_ko, YOUTUBE_COMMENTS_100)
    
    # 모델 2: GoEmotions
    model_en = GoEmotionsModel()
    results_en = evaluate_model(model_en, YOUTUBE_COMMENTS_100)
    
    # 비교 표 생성
    generate_comparison_table(results_ko, results_en)
    
    # CSV 저장
    save_detailed_results(results_ko, results_en)
    
    print("\n" + "="*100)
    print("✅ 모든 비교 완료!")
    print("="*100)


if __name__ == "__main__":
    main()
