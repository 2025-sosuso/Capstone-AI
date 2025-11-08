# src/pipelines/summarize.py
"""
[기능1] AI 기반 자동 전체 댓글 요약
- OpenAI Chat Completions API 사용
- 입력: 댓글 문자열 리스트
- 출력: 한국어 요약문(2문장 내외)
"""
import os
import sys
from typing import List

# ============================================================
# Windows 한글 깨짐 방지 (최상단 배치)
# ============================================================
if sys.platform == 'win32':
    # 1. 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 2. 표준 출력/에러 스트림 재설정
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
    
    # 3. Windows 콘솔 코드 페이지를 UTF-8로 설정
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, 
                      capture_output=True, check=False)
    except Exception:
        pass

from openai import OpenAI

# ============================================================
# API 키 및 유틸리티 불러오기
# ============================================================
try:
    from src.config import OPENAI_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY
    from src.utils.youtube import fetch_youtube_comments
except ImportError:
    from ..config import OPENAI_API_KEY, YOUTUBE_API_KEY, VIDEO_KEY
    from ..utils.youtube import fetch_youtube_comments

# OpenAI 클라이언트는 모듈 로드 시 1회 생성 (재사용)
client = OpenAI(api_key=OPENAI_API_KEY)

def summarize_comments_with_gpt(comments: List[str]) -> str:
    """
    GPT를 이용한 댓글 요약.
    - comments: 댓글 문자열 리스트
    - 반환: 한국어 요약문 (문자열) (AIAnalysisResponse의 summation 필드)
    """
    if not comments:
        return "요약할 댓글이 없습니다."
    joined = "\n".join(comments[:500])  # 안전: 과도한 길이 방지(확실하지 않음: 길이 제한은 필요시 조정)

    prompt = (
        "당신은 텍스트 요약 전문가입니다.\n"
        "다음은 어떤 유튜브 영상에 달린 다양한 댓글들입니다.\n"
        "이 댓글들의 전체 분위기와 주요 논점만 2문장 이내로 자연스럽고 간결하게 한국어로 요약해주세요.\n"
        "중복 의견은 묶고, 인상적인 반응은 반영해 주세요.\n\n"
        f"댓글 목록:\n{joined}\n"
    )

    # 모델명은 환경/요금정책에 따라 조정 가능(확실하지 않음)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


# ============================================================
# 사용 예시 (테스트용 코드)
# ============================================================
if __name__ == "__main__":
    import json
    
    # config.py에서 VIDEO_KEY 확인
    if not VIDEO_KEY:
        print("[ERROR] VIDEO_KEY가 .env 파일에 설정되지 않았습니다.")
        print("[INFO] 테스트 데이터로 실행합니다.\n")
        
        # 테스트 데이터
        test_comments = [
            "정말 유익한 영상이네요! 감사합니다.",
            "설명이 정말 잘 되어있어요. 이해하기 쉬웠습니다.",
            "좋은 정보 감사합니다. 도움이 많이 되었어요.",
            "구독하고 갑니다. 앞으로도 좋은 영상 부탁드려요!",
            "이런 영상 기다렸어요. 정말 유익합니다.",
        ]
        
        summation = summarize_comments_with_gpt(test_comments)
        
        # JSON 형식으로 출력
        result = {"summation": summation}
        print("=" * 60)
        print("[결과] 댓글 요약 (테스트 데이터):")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 60)
        
    else:
        if not YOUTUBE_API_KEY:
            print("[ERROR] YOUTUBE_API_KEY가 .env 파일에 설정되지 않았습니다.")
            sys.exit(1)
        
        if not OPENAI_API_KEY:
            print("[ERROR] OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
            sys.exit(1)
        
        print(f"[INFO] YouTube 비디오 '{VIDEO_KEY}'에서 댓글 수집 중...")
        
        try:
            # YouTube 댓글 수집
            youtube_comments = fetch_youtube_comments(
                video_id=VIDEO_KEY,
                api_key=YOUTUBE_API_KEY,
                max_pages=1,           # 1페이지 (100개)
                page_size=100,         # 페이지당 100개
                include_replies=False, # 대댓글 제외
                apply_cleaning=True,   # 텍스트 전처리 적용
            )
            
            print(f"[SUCCESS] {len(youtube_comments)}개 댓글 수집 완료!")
            
            if not youtube_comments:
                print("[WARNING] 수집된 댓글이 없습니다.")
                sys.exit(0)
            
            # 댓글 요약 실행
            print(f"[INFO] GPT를 이용한 댓글 요약 중...\n")
            summation = summarize_comments_with_gpt(youtube_comments)
            
            # ============================================================
            # JSON 형식으로 결과 출력 (AIAnalysisResponse의 summation 필드)
            # ============================================================
            result = {"summation": summation}
            
            print("=" * 60)
            print("[결과] 댓글 요약 (summation 필드):")
            print("=" * 60)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("=" * 60)
            
        except Exception as e:
            print(f"[ERROR] 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)