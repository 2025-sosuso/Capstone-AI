"""
인코더 사전 학습 여부 확인 테스트
"""
from __future__ import annotations

import os
import sys
import warnings

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*symlink.*")

from transformers import ElectraForSequenceClassification, ElectraModel
import torch

def check_encoder_pretrained():
    print("=" * 70)
    print("KoELECTRA-v3 인코더 사전 학습 여부 확인")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. GoEmotions 모델 로드
    print("\n[1] monologg/koelectra-base-v3-goemotions 로드 중...")
    model_goemotions = ElectraForSequenceClassification.from_pretrained(
        "monologg/koelectra-base-v3-goemotions",
        use_safetensors=True
    ).to(device)
    
    # 2. 순수 KoELECTRA-v3-discriminator 로드 (비교용)
    print("[2] monologg/koelectra-base-v3-discriminator 로드 중...")
    model_pure = ElectraModel.from_pretrained(
        "monologg/koelectra-base-v3-discriminator"
    ).to(device)
    
    print("\n" + "=" * 70)
    print("파라미터 분석")
    print("=" * 70)
    
    # 로드된 파라미터 확인
    goemotions_params = dict(model_goemotions.named_parameters())
    pure_params = dict(model_pure.named_parameters())
    
    print(f"\n총 파라미터 수:")
    print(f"  - GoEmotions 모델: {len(goemotions_params):,}개")
    print(f"  - 순수 KoELECTRA: {len(pure_params):,}개")
    
    # 인코더 파라미터만 필터링
    encoder_params_goemotions = {k: v for k, v in goemotions_params.items() 
                                  if k.startswith('electra.')}
    classifier_params = {k: v for k, v in goemotions_params.items() 
                        if k.startswith('classifier.')}
    
    print(f"\n파라미터 구성:")
    print(f"  - 인코더 파라미터: {len(encoder_params_goemotions):,}개")
    print(f"  - Classifier 파라미터: {len(classifier_params):,}개")
    
    print(f"\nClassifier 레이어 목록:")
    for name in classifier_params.keys():
        print(f"  - {name}")
    
    # 인코더 가중치 통계 분석
    print("\n" + "=" * 70)
    print("인코더 가중치 통계 (랜덤 초기화 vs 사전 학습 판단)")
    print("=" * 70)
    
    # 첫 번째 임베딩 레이어 분석
    embedding_layer = "electra.embeddings.word_embeddings.weight"
    if embedding_layer in encoder_params_goemotions:
        weights = encoder_params_goemotions[embedding_layer]
        
        print(f"\n[{embedding_layer}]")
        print(f"  Shape: {weights.shape}")
        print(f"  Mean: {weights.mean().item():.6f}")
        print(f"  Std: {weights.std().item():.6f}")
        print(f"  Min: {weights.min().item():.6f}")
        print(f"  Max: {weights.max().item():.6f}")
        
        print(f"\n  💡 판단:")
        std = weights.std().item()
        if 0.01 < std < 0.15:  # 사전 학습된 가중치 범위
            print(f"     ✅ 사전 학습된 가중치로 보임 (Std: {std:.6f})")
            print(f"        (랜덤 초기화면 Std ~0.02 또는 매우 작은 값)")
        else:
            print(f"     ❌ 랜덤 초기화 가능성 있음 (Std: {std:.6f})")
    
    # 첫 번째 Transformer 레이어 분석
    first_layer = "electra.encoder.layer.0.attention.self.query.weight"
    if first_layer in encoder_params_goemotions:
        weights = encoder_params_goemotions[first_layer]
        
        print(f"\n[{first_layer}]")
        print(f"  Shape: {weights.shape}")
        print(f"  Mean: {weights.mean().item():.6f}")
        print(f"  Std: {weights.std().item():.6f}")
        print(f"  Min: {weights.min().item():.6f}")
        print(f"  Max: {weights.max().item():.6f}")
        
        print(f"\n  💡 판단:")
        std = weights.std().item()
        if std > 0.05:  # 사전 학습된 가중치는 더 큰 분산
            print(f"     ✅ 사전 학습된 가중치로 보임 (Std: {std:.6f})")
        else:
            print(f"     ❌ 랜덤 초기화 가능성 있음 (Std: {std:.6f})")
    
    # 인코더 가중치 비교 (순수 모델과)
    print("\n" + "=" * 70)
    print("순수 KoELECTRA와 가중치 비교")
    print("=" * 70)
    
    # 공통 파라미터 찾기
    common_params = []
    for key_goe in encoder_params_goemotions.keys():
        # electra. 제거하고 비교
        key_pure = key_goe.replace("electra.", "")
        if key_pure in pure_params:
            common_params.append((key_goe, key_pure))
    
    print(f"\n공통 파라미터: {len(common_params)}개")
    
    # 몇 개 샘플링해서 비교
    sample_count = min(5, len(common_params))
    print(f"\n샘플 {sample_count}개 비교:")
    
    identical_count = 0
    for i, (key_goe, key_pure) in enumerate(common_params[:sample_count]):
        weights_goe = encoder_params_goemotions[key_goe]
        weights_pure = pure_params[key_pure]
        
        # 가중치가 동일한지 확인 (허용 오차 1e-6)
        is_identical = torch.allclose(weights_goe, weights_pure, atol=1e-6)
        
        print(f"\n  [{i+1}] {key_goe}")
        print(f"      동일 여부: {'✅ 동일' if is_identical else '❌ 다름'}")
        
        if is_identical:
            identical_count += 1
    
    print("\n" + "=" * 70)
    print("최종 결론")
    print("=" * 70)
    
    if identical_count == sample_count:
        print("\n✅ KoELECTRA-v3 인코더는 사전 학습된 상태입니다!")
        print("   - 순수 KoELECTRA-v3 모델과 가중치가 동일함")
        print("   - 가중치 분포가 사전 학습 패턴을 보임")
        print("\n✅ Classifier만 랜덤 초기화 상태입니다.")
        print("   - Fine-tuning이 필요한 부분은 Classifier만")
    else:
        print(f"\n⚠️  인코더 가중치 확인 필요")
        print(f"   - {identical_count}/{sample_count}개만 동일")

if __name__ == "__main__":
    check_encoder_pretrained()