# 데이터 전처리 할 데이터셋 속성만 뽑아내기
import pandas as pd
import sys

# 콘솔 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

file_path = r'C:\CapstoneDesign\capstone-ai\data\raw\trainDataset\train.xlsx'
df = pd.read_excel(file_path)

# for col in df.columns:
#     print(col)

# 감정_소분류 분포 확인
emotion_counts = df['감정_소분류'].value_counts()
print(emotion_counts.head(20))

# 감정_대분류 분포 확인
major_counts = df['감정_대분류'].value_counts()
print(major_counts)