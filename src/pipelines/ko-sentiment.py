# ============================================================
# 1. 한국어 fine-tuned KoELECTRA 모델 사용
# ============================================================
# ⭐ 학습된 모델 경로 (변경됨)
_MODEL_PATH = Path(__file__).parent.parent.parent / 'saved_models' / 'ko-emotions_finetuned'

_tok = None  # 토크나이저
_model = None  # 감정 분류 모델
_device = None  # 디바이스 (GPU/CPU)

# ⭐ 7개 감정 레이블 (학습 시 사용한 것과 동일)
_LABELS = ['joy', 'love', 'gratitude', 'anger', 'sadness', 'fear', 'neutral']