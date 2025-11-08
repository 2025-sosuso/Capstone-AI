import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time  # 시간 측정을 위한 모듈

# 감정 매핑 딕셔너리: (대분류, 소분류) -> GoEmotions 감정
EMOTION_MAPPING = {
    # joy
    ('기쁨', '기쁨'): 'joy',
    ('기쁨', '편안한'): 'joy',
    ('기쁨', '만족스러운'): 'joy',
    ('기쁨', '흥분되는'): 'joy',
    ('기쁨', '느긋한'): 'joy',
    ('기쁨', '안도하는'): 'joy',
    ('기쁨', '신이 난'): 'joy',
    ('기쁨', '자신하는'): 'joy',
    
    # love
    ('기쁨', '사랑하는'): 'love',
    
    # gratitude
    ('기쁨', '감사하는'): 'gratitude',
    
    # anger
    ('분노', '분노'): 'anger',
    ('분노', '툴툴대는'): 'anger',
    ('분노', '좌절한'): 'anger',
    ('분노', '짜증나는'): 'anger',
    ('분노', '방어적인'): 'anger',
    ('분노', '악의적인'): 'anger',
    ('분노', '안달하는'): 'anger',
    ('분노', '구역질 나는'): 'anger',
    ('분노', '노여워하는'): 'anger',
    ('분노', '성가신'): 'anger',
    ('당황', '혐오스러운'): 'anger',
    
    # sadness
    ('슬픔', '슬픔'): 'sadness',
    ('슬픔', '실망한'): 'sadness',
    ('슬픔', '비통한'): 'sadness',
    ('슬픔', '후회되는'): 'sadness',
    ('슬픔', '우울한'): 'sadness',
    ('슬픔', '마비된'): 'sadness',
    ('슬픔', '염세적인'): 'sadness',
    ('슬픔', '눈물이 나는'): 'sadness',
    ('슬픔', '낙담한'): 'sadness',
    ('슬픔', '환멸을 느끼는'): 'sadness',
    ('상처', '배신당한'): 'sadness',
    ('상처', '고립된'): 'sadness',
    ('상처', '불우한'): 'sadness',
    ('상처', '괴로워하는'): 'sadness',
    ('상처', '버려진'): 'sadness',
    ('당황', '고립된'): 'sadness',
    ('당황', '외로운'): 'sadness',
    
    # fear
    ('불안', '불안'): 'fear',
    ('불안', '두려운'): 'fear',
    ('불안', '스트레스 받는'): 'fear',
    ('불안', '취약한'): 'fear',
    ('불안', '혼란스러운'): 'fear',
    ('불안', '당혹스러운'): 'fear',
    ('불안', '걱정스러운'): 'fear',
    ('불안', '조심스러운'): 'fear',
    ('불안', '초조한'): 'fear',
    
    # neutral
    ('불안', '회의적인'): 'neutral',
    ('상처', '상처'): 'neutral',
    ('상처', '질투하는'): 'neutral',
    ('상처', '충격 받은'): 'neutral',
    ('상처', '희생된'): 'neutral',
    ('상처', '억울한'): 'neutral',
    ('당황', '당황'): 'neutral',
    ('당황', '남의 시선 의식하는'): 'neutral',
    ('당황', '열등감'): 'neutral',
    ('당황', '죄책감'): 'neutral',
    ('당황', '부끄러운'): 'neutral',
    ('당황', '한심한'): 'neutral',
    ('당황', '혼란스러운'): 'neutral',
}


def map_emotion(row):
    """
    (감정_대분류, 감정_소분류) 튜플을 GoEmotions 감정으로 매핑
    """
    key = (row['감정_대분류'], row['감정_소분류'])
    
    if key in EMOTION_MAPPING:
        return EMOTION_MAPPING[key]
    else:
        # 매핑되지 않은 경우 처리
        print(f"경고: 매핑되지 않은 감정 조합 발견 - 대분류: {row['감정_대분류']}, 소분류: {row['감정_소분류']}")
        return 'neutral'  # 기본값으로 neutral 반환


def process_dataset(input_path, output_path):
    # ============ 전처리 시작 시간 측정 ============
    start_time = time.time()
    
    print(f"읽는 중: {input_path}")
    df = pd.read_excel(input_path)
    print(f"로딩 완료! 총 {len(df)}행")
    
    print("전처리 시작...")
    
    # 컬럼명 확인
    print("원본 파일의 컬럼 목록:")
    print(df.columns.tolist())
    print(f"\n원본 데이터 개수: {len(df)}개")
    
    # 필요한 컬럼만 선택
    df_filtered = df[['사람문장1', '감정_대분류', '감정_소분류']].copy()
    
    # 결측치 제거
    df_filtered = df_filtered.dropna()
    
    # 중복 제거 (선택사항)
    df_filtered = df_filtered.drop_duplicates()
    
    # tqdm으로 진행사항 보기
    tqdm.pandas(desc="텍스트 처리")
    
    df_filtered['사람문장1'] = df_filtered['사람문장1'].progress_apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    
    # GoEmotions 감정 매핑 추가
    print("\nGoEmotions 감정으로 매핑 중...")
    tqdm.pandas(desc="감정 매핑")
    df_filtered['goemotion_label'] = df_filtered.progress_apply(map_emotion, axis=1)
    
    print(f"\n추출 후 데이터 개수: {len(df_filtered)}개")
    print(f"\n원본 감정 대분류 분포:")
    print(df_filtered['감정_대분류'].value_counts())
    print(f"\nGoEmotions 감정 분포:")
    print(df_filtered['goemotion_label'].value_counts())
    
    # 매핑되지 않은 감정 조합 확인
    print("\n감정 조합 샘플 확인:")
    print(df_filtered[['감정_대분류', '감정_소분류', 'goemotion_label']].drop_duplicates().head(20))
    
    # 출력 디렉토리가 없으면 생성
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 저장 (필요한 컬럼만 선택)
    print(f"\n저장 중: {output_path}")
    df_output = df_filtered[['사람문장1', 'goemotion_label']].copy()
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("완료!")
    
    print(f"\n최종 출력 컬럼: {df_output.columns.tolist()}")
    print(f"최종 데이터 개수: {len(df_output)}개")
    
    # ============ 전처리 종료 시간 측정 및 출력 ============
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 시간을 분:초 형식으로 변환
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    
    print(f"\n{'='*50}")
    print(f"전처리 소요 시간: {minutes}분 {seconds:.2f}초 (총 {elapsed_time:.2f}초)")
    print(f"{'='*50}\n")
    
    return df_filtered


# 사용 예시
if __name__ == "__main__":
    # ============ 전체 실행 시간 측정 시작 ============
    total_start_time = time.time()
    
    print("\n" + "="*70)
    print("Training 데이터 전처리 시작")
    print("="*70)
    
    # Training 데이터
    train_df = process_dataset(
        r'C:\CapstoneDesign\capstone-ai\data\raw\trainDataset\Training-dataset.xlsx',
        r'C:\CapstoneDesign\capstone-ai\data\processed\trainProcessed\train_processed.csv'
    )
    
    print("\n" + "="*70)
    print("Validation 데이터 전처리 시작")
    print("="*70)
    
    # Validation 데이터
    val_df = process_dataset(
        r'C:\CapstoneDesign\capstone-ai\data\raw\valDataset\Val-dataset.xlsx',
        r'C:\CapstoneDesign\capstone-ai\data\processed\valProcessed\val_processed.csv'
    )
    
    # ============ 전체 실행 시간 측정 종료 ============
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    # 전체 시간을 분:초 형식으로 변환
    total_minutes = int(total_elapsed_time // 60)
    total_seconds = total_elapsed_time % 60
    
    print("\n" + "="*70)
    print("전체 전처리 완료!")
    print("="*70)
    print(f"총 소요 시간: {total_minutes}분 {total_seconds:.2f}초 (총 {total_elapsed_time:.2f}초)")
    print(f"Training 데이터: {len(train_df)}개")
    print(f"Validation 데이터: {len(val_df)}개")
    print("="*70 + "\n")
    
    # # 샘플 확인
    # print("\n=== Training 데이터 샘플 ===")
    # print(train_df.head(10))
    
    # # GoEmotions 레이블별 샘플 확인
    # print("\n=== GoEmotions 레이블별 샘플 ===")
    # for emotion in ['joy', 'love', 'gratitude', 'anger', 'sadness', 'fear', 'neutral']:
    #     print(f"\n{emotion}:")
    #     sample = train_df[train_df['goemotion_label'] == emotion].head(2)
    #     if not sample.empty:
    #         print(sample[['사람문장1', '감정_대분류', '감정_소분류', 'goemotion_label']])