# src/config.py
"""
환경변수(.env) 로딩 및 공통 설정 모듈
- .env에 들어있는 OPENAI_API_KEY / DEEPL_API_KEY 를 읽어옵니다.
- 누락 시 경고만 출력(실행은 계속)합니다.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # 같은 디렉토리 또는 상위 디렉토리의 .env를 읽음

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
VIDEO_KEY = os.getenv("VIDEO_KEY")

# (선택) 필수 키 체크: 없으면 경고 로그 출력
_required = ["OPENAI_API_KEY", "DEEPL_API_KEY", "YOUTUBE_API_KEY", "VIDEO_KEY"]
_missing = [k for k in _required if not os.getenv(k)]
if _missing:
    print(f"[WARN] Missing keys in .env: {_missing}")  # 확실하지 않음: 로깅 정책은 상황에 맞게
