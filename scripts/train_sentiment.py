# scripts/train_sentiment.py
"""
KoELECTRA 감정 분석 모델 Fine-tuning
- 데이터: AI Hub 감성대화말뭉치 (전처리 완료)
- 모델: monologg/koelectra-base-v3-goemotions
- 목표: 6개 감정 분류 (joy, gratitude, anger, sadness, fear, neutral)
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # 
from transformers import (
    ElectraForSequenceClassification,
    ElectraTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time

# ============================================================
# Windows 한글 설정
# ============================================================
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

# ============================================================
# 프로젝트 루트 경로 설정
# ============================================================
SCRIPT_DIR = Path(__file__).parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent     # capstone-ai/

# ============================================================
# 설정
# ============================================================
CONFIG = {
    'model_name': 'monologg/koelectra-base-v3-goemotions',
    
    # 데이터 경로 (전처리 완료된 데이터)
    'train_data_path': PROJECT_ROOT / 'data' / 'processed' / 'trainProcessed' / 'train_processed.csv',
    'val_data_path': PROJECT_ROOT / 'data' / 'processed' / 'valProcessed' / 'val_processed.csv',
    
    # 모델 저장 위치
    'output_dir': PROJECT_ROOT / 'saved_models' / 'ko-emotions_finetuned',
    
    # ⭐ 실험 ID (실험마다 변경하세요!)
    'experiment_id': 'exp_4_32',
    
    # 학습 하이퍼파라미터
    # batch size vs epoch vs iterarion
    # batch size: 전체 데이터 셋을 여러 소그룹으로 나눴을 때 하나의 소그룹에 속하는 데이터 수 / 
    #             크면: 학습 속도 느림, 메모리 부족 / 작으면: 적은 데이터로 가중치가 자주 업데이트돼서 훈련 불안정
    # epoch: 모든 데이터셋을 학습하는 횟수
    #        크면: overfitting 발생할 확률 높음 / 작으면: underfitting 발생할 확률 높음
    # iteration: 1-epoch를 마치는데 필요한 미니배치 수(=1-epoch에서 파라미터 업데이트 횟수). 따라서 전체 데이터 수 / batch size 
    'batch_size': 32,          # RTX 4090이면 64까지 가능
    'epochs': 4,               
    'learning_rate_encoder': 2e-5,      # 인코더: 미세 조정
    'learning_rate_classifier': 1e-3,   # Classifier: 새로 학습
    'max_length': 128,         # 최대 토큰 길이
    'warmup_steps': 100,       # Warmup 스텝
    'weight_decay': 0.01,      # 정규화
    
    # 감정 레이블 (전처리된 데이터와 동일)
    'labels': ['joy', 'gratitude', 'anger', 'sadness', 'fear', 'neutral']  # 데이터셋에 love가 없어서 우선 뺌
}

# ============================================================
# Dataset 클래스
# ============================================================
class EmotionDataset(Dataset):
    """
    감성대화말뭉치 Dataset
    컬럼: 사람문장1, goemotion_label
    """
    def __init__(self, csv_path, tokenizer, max_length, label2id):
        print(f"[INFO] 데이터 로딩: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        
        # 데이터 확인
        print(f"[INFO] 총 {len(self.df)}개 샘플")
        print(f"[INFO] 감정 분포:")
        print(self.df['goemotion_label'].value_counts())
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['사람문장1'])  # 컬럼명 주의!
        label = row['goemotion_label']
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label2id[label], dtype=torch.long)
        }

# ============================================================
# 평가 함수
# ============================================================
def evaluate(model, dataloader, device, id2label):
    """
    모델 평가 및 상세 메트릭 계산
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)  # NLP 는 입력에 넣는 문장들이 길이들이 같아야 함. 그래서 길이를 지정해줬고 지정된 길이보다 짧으면 가짜 토큰을 넣어줌. 
                                                                # 이때 어떤 토큰이 진짜고 가짜인지 알려주는 게 attention_mask 역할임
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    eval_time = time.time() - eval_start_time
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # 상세 리포트
    label_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    
    return avg_loss, accuracy, report, all_preds, all_labels, eval_time

# ============================================================
# 학습 함수
# ============================================================
def train():
    total_start_time = time.time()
    
    print("=" * 70)
    print(f"🔬 실험: {CONFIG['experiment_id']}")
    print("=" * 70)
    
    print(f"\n📁 프로젝트 루트: {PROJECT_ROOT}")
    print(f"📁 Training 데이터: {CONFIG['train_data_path']}")
    print(f"📁 Validation 데이터: {CONFIG['val_data_path']}")
    print(f"📁 모델 저장 위치: {CONFIG['output_dir']}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 사용 디바이스: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    # Label mapping
    label2id = {label: idx for idx, label in enumerate(CONFIG['labels'])}  # labels(감정7가지)에 index 부여 -> ex.{'joy':0, 'love':1}
    id2label = {idx: label for label, idx in label2id.items()}  # ex.{0:'joy', 1:'love'}
    
    print(f"🏷️  감정 레이블: {CONFIG['labels']}")
    print(f"🏷️  총 {len(CONFIG['labels'])}개 클래스\n")
    
    print("🔄 모델 로딩 중...")
    tokenizer = ElectraTokenizer.from_pretrained(CONFIG['model_name'])
    model = ElectraForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=len(CONFIG['labels']),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
        ignore_mismatched_sizes=True  # Classifier 크기가 다를 수 있음
    )
    model.to(device)
    print("✅ 모델 로드 완료\n")
    
    # Dataset & DataLoader
    # Dataset: 데이터를 보관하고 한 개씩 꺼내주는 곳
    # DataLoader: Dataset에서 데이터를 여러 개씩 묶어서 모델에게 운반
    print("📊 데이터셋 로딩 중...")
    train_dataset = EmotionDataset(
        CONFIG['train_data_path'],
        tokenizer,
        CONFIG['max_length'],
        label2id
    )
    val_dataset = EmotionDataset(
        CONFIG['val_data_path'],
        tokenizer,
        CONFIG['max_length'],
        label2id
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],  # batch_size: 모델에 넣을 데이터를 한 번에 몇 개 넣을건지 지정
        shuffle=True,
        num_workers=0  # Windows에서는 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"\n✅ Training 데이터: {len(train_dataset):,}개")
    print(f"✅ Validation 데이터: {len(val_dataset):,}개")
    print(f"✅ Batch size: {CONFIG['batch_size']}")
    print(f"✅ Total batches per epoch: {len(train_loader)}\n")
    
    # Optimizer & Scheduler (Differential Learning Rate)
    print("⚙️  Optimizer 설정...")
    optimizer = AdamW([
        {
            'params': model.electra.parameters(),  # 인코더
            'lr': CONFIG['learning_rate_encoder'],
            'weight_decay': CONFIG['weight_decay']  # 정규화, weight=weight-(lr*gradient)-(weight_decay*weight)->weight_decay*weight로 인해 weight 조금씩 줄임(규제)
        },
        {
            'params': model.classifier.parameters(),  # Classifier
            'lr': CONFIG['learning_rate_classifier'],  # learning rate: gradient descent에서 최적값을 찾을 때 최솟값을 내려가는 포복의 크기, lr가 크면->overshooting발생, lr가 작으면->local mininum발생, 적당한 값 찾는게 좋음
            'weight_decay': CONFIG['weight_decay']  # weight_decay: weight 값들의 증가를 제한해서 모델의 복잡도 감소 시킴, 모델이 복잡해지면?->overfitting발생
        }
    ])
    
    total_steps = len(train_loader) * CONFIG['epochs']
    # scheduler: 시간에 따라 lr 조정하는 역할 
    # ex. step 1~100->warmup, step100~2000->lr값유지, step2000~4000->lr값 decay => 이 내용은 transformers library 안에 있음
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    print(f"   Encoder LR: {CONFIG['learning_rate_encoder']}")
    print(f"   Classifier LR: {CONFIG['learning_rate_classifier']}")
    print(f"   총 학습 스텝: {total_steps:,}")
    print(f"   Warmup 스텝: {CONFIG['warmup_steps']}\n")
    
    print("=" * 70)
    print("🚀 학습 시작!")
    print("=" * 70)
    
    best_val_accuracy = 0
    best_epoch = 0
    best_val_report = None  # ⭐ Best epoch의 상세 리포트 저장
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'epoch_time': [],
        'eval_time': []
    }
    
    training_start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*70}")
        print(f"📅 Epoch {epoch + 1}/{CONFIG['epochs']}")
        print(f"{'='*70}")
        
        epoch_start_time = time.time()
        
        # Training
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = total_train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        # Validation
        print("\n🔍 Validation 시작...")
        val_loss, val_accuracy, val_report, _, _, eval_time = evaluate(
            model, val_loader, device, id2label
        )
        
        # F1 Score (macro average)
        val_f1 = val_report['macro avg']['f1-score']
        
        # 결과 저장
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        history['epoch_time'].append(epoch_time)
        history['eval_time'].append(eval_time)
        
        # 결과 출력
        print(f"\n{'='*70}")
        print(f"📊 Epoch {epoch + 1} 결과:")
        print(f"{'='*70}")
        print(f"  Train Loss    : {avg_train_loss:.4f}")
        print(f"  Val Loss      : {val_loss:.4f}")
        print(f"  Val Accuracy  : {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"  Val F1 (macro): {val_f1:.4f}")
        print(f"  ⏱️  Epoch 시간   : {epoch_time//60:.0f}분 {epoch_time%60:.0f}초")
        print(f"  ⏱️  평가 시간    : {eval_time:.1f}초")
        print(f"{'='*70}")
        
        # 감정별 성능
        print("\n📈 감정별 성능:")
        for label_name in CONFIG['labels']:
            if label_name in val_report:
                metrics = val_report[label_name]
                print(f"  {label_name:12s}: "
                      f"Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, "
                      f"F1={metrics['f1-score']:.3f}")
        
        # Best model 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_val_report = val_report  # ⭐ Best 리포트 저장
            output_dir = Path(CONFIG['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n💾 Best 모델 저장 중...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Config 저장
            config_save = {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()}
            config_save['best_val_accuracy'] = float(best_val_accuracy)
            config_save['best_val_f1'] = float(val_f1)
            config_save['best_epoch'] = best_epoch
            config_save['trained_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            config_save['label2id'] = label2id
            config_save['id2label'] = id2label
            
            with open(output_dir / 'training_config.json', 'w', encoding='utf-8') as f:
                json.dump(config_save, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Best model 저장 완료!")
            print(f"   Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"   F1 Score: {val_f1:.4f}")
    
    # 전체 학습 시간
    total_training_time = time.time() - training_start_time
    total_time = time.time() - total_start_time
    
    # ⭐ Per-class F1 추출 (Best epoch)
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    for label_name in CONFIG['labels']:
        if label_name in best_val_report:
            per_class_f1[label_name] = float(best_val_report[label_name]['f1-score'])
            per_class_precision[label_name] = float(best_val_report[label_name]['precision'])
            per_class_recall[label_name] = float(best_val_report[label_name]['recall'])
    
    # 학습 완료
    print("\n" + "="*70)
    print("🎉 학습 완료!")
    print("="*70)
    print(f"✨ Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    print(f"✨ Best Macro F1 Score: {history['val_f1'][best_epoch-1]:.4f}")
    print(f"✨ Best Epoch: {best_epoch}")
    print(f"⏱️  총 학습 시간: {total_training_time//60:.0f}분 {total_training_time%60:.0f}초")
    print(f"⏱️  전체 실행 시간: {total_time//60:.0f}분 {total_time%60:.0f}초")
    print(f"⏱️  Epoch당 평균: {total_training_time/CONFIG['epochs']//60:.0f}분 {total_training_time/CONFIG['epochs']%60:.0f}초")
    print(f"📁 모델 저장 위치: {CONFIG['output_dir']}")
    print("="*70 + "\n")
    
    # History 저장
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(CONFIG['output_dir']) / 'training_history.csv', index=False)
    print("✅ 학습 히스토리 저장 완료: training_history.csv\n")
    
    # ⭐ 실험 결과 요약 저장 (Per-class F1 포함)
    summary = {
        'experiment_id': CONFIG['experiment_id'],
        'hyperparameters': {
            'epochs': CONFIG['epochs'],
            'batch_size': CONFIG['batch_size'],
            'learning_rate_encoder': CONFIG['learning_rate_encoder'],
            'learning_rate_classifier': CONFIG['learning_rate_classifier'],
        },
        'results': {
            'best_accuracy': float(best_val_accuracy),
            'best_macro_f1': float(history['val_f1'][best_epoch-1]),
            'best_epoch': best_epoch,
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'per_class_f1': per_class_f1,  # ⭐ 추가
            'per_class_precision': per_class_precision,  # ⭐ 추가
            'per_class_recall': per_class_recall,  # ⭐ 추가
        },
        'timing': {
            'total_training_time_seconds': float(total_training_time),
            'total_time_seconds': float(total_time),
            'avg_epoch_time_seconds': float(total_training_time / CONFIG['epochs']),
            'avg_eval_time_seconds': float(np.mean(history['eval_time'])),
        }
    }
    
    with open(Path(CONFIG['output_dir']) / f'{CONFIG["experiment_id"]}_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 실험 요약 저장 완료: {CONFIG['experiment_id']}_summary.json\n")
    
    # ⭐ 콘솔에 복사 가능한 결과 출력 (확장됨)
    print("\n" + "="*70)
    print("📋 실험 결과 요약 (복사용)")
    print("="*70)
    print(f"\n🔬 실험 ID: {CONFIG['experiment_id']}")
    print(f"\n📊 하이퍼파라미터:")
    print(f"  - Epoch: {CONFIG['epochs']}")
    print(f"  - Batch Size: {CONFIG['batch_size']}")
    print(f"  - Learning Rate (Encoder): {CONFIG['learning_rate_encoder']}")
    print(f"  - Learning Rate (Classifier): {CONFIG['learning_rate_classifier']}")
    
    print(f"\n🎯 전체 성능 (Best Epoch {best_epoch}):")
    print(f"  - Accuracy: {best_val_accuracy*100:.2f}%")
    print(f"  - Macro F1 Score: {history['val_f1'][best_epoch-1]:.4f}")
    print(f"  - Train Loss: {history['train_loss'][best_epoch-1]:.4f}")
    print(f"  - Val Loss: {history['val_loss'][best_epoch-1]:.4f}")
    
    print(f"\n📈 감정별 F1 Score (Per-class):")
    print(f"  ┌{'─'*14}┬{'─'*12}┬{'─'*12}┬{'─'*12}┐")
    print(f"  │ {'Emotion':12s} │ {'Precision':>10s} │ {'Recall':>10s} │ {'F1-Score':>10s} │")
    print(f"  ├{'─'*14}┼{'─'*12}┼{'─'*12}┼{'─'*12}┤")
    for label_name in CONFIG['labels']:
        if label_name in per_class_f1:
            prec = per_class_precision[label_name]
            rec = per_class_recall[label_name]
            f1 = per_class_f1[label_name]
            print(f"  │ {label_name:12s} │ {prec:10.3f} │ {rec:10.3f} │ {f1:10.3f} │")
    print(f"  └{'─'*14}┴{'─'*12}┴{'─'*12}┴{'─'*12}┘")
    
    print(f"\n⏱️  학습 시간:")
    print(f"  - 총 학습 시간: {total_training_time//60:.0f}분 {total_training_time%60:.0f}초")
    print(f"  - Epoch당 평균: {total_training_time/CONFIG['epochs']//60:.0f}분 {total_training_time/CONFIG['epochs']%60:.0f}초")
    print(f"  - 평가 시간: {np.mean(history['eval_time']):.1f}초")
    
    print("="*70)
    
    return model, history

# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    try:
        model, history = train()
        print("✅ 모든 작업 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()