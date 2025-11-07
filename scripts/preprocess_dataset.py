## csv파일 생성된 거 확인하기!!!!!!!!!!!!!!!
import pandas as pd
from pathlib import Path
from tqdm import tqdm # 진행사항 보여주는 기능

def process_dataset(input_path, output_path):
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
    
    # tqdm 으로 진행사항 보기
    tqdm.pandas(desc="텍스트 처리")
    
    df_filtered['사람문장1'] = df_filtered['사람문장1'].progress_apply\
        (lambda x: x.strip() if isinstance(x, str) else x) 
        # progress_apply(): DataFrame 전체에 적용할 함수를 작성하는 함수
        # isinstance(x,자료형): x가 자료형에 부합하면 True, 반대면 False 반환
    
    print(f"\n추출 후 데이터 개수: {len(df_filtered)}개")
    print(f"\n감정 분포:")
    print(df_filtered['감정_대분류'].value_counts())
    
    # 출력 디렉토리가 없으면 생성
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 저장
    print(f"저장 중: {output_path}")
    df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("완료!")
    
    return df_filtered

# 사용 예시
if __name__ == "__main__":
    # Training 데이터
    train_df = process_dataset(
        r'C:\CapstoneDesign\capstone-ai\data\raw\trainDataset\train.xlsx',  # 실제 파일명으로 변경
        r'C:\CapstoneDesign\capstone-ai\data\processed\trainProcessed\train_processed.csv'
    )
    
    # Validation 데이터
    val_df = process_dataset(
        r'C:\CapstoneDesign\capstone-ai\data\raw\valDataset\val.xlsx',  # 실제 파일명으로 변경
        r'C:\CapstoneDesign\capstone-ai\data\processed\valProcessed\val_processed.csv'
    )
    
    # 샘플 확인
    print("\n=== Training 데이터 샘플 ===")
    print(train_df.head(1))
