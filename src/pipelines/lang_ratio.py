# src/pipelines/lang_ratio.py
"""
[기능4] 언어 비율 분석
- langdetect 사용
- 결과를 언어 코드로 반환 (비율: 0~100)
"""
import os
import sys
from collections import Counter
from typing import Dict, List

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

from langdetect import detect

# ============================================================
# API 키 및 유틸리티 불러오기
# ============================================================
try:
    from src.config import YOUTUBE_API_KEY, VIDEO_KEY
    from src.utils.youtube import fetch_youtube_comments
except ImportError:
    from ..config import YOUTUBE_API_KEY, VIDEO_KEY
    from ..utils.youtube import fetch_youtube_comments


def detect_languages(comments: List[str]) -> Dict[str, float]:
    """
    댓글 리스트의 언어 비율(%)을 반환.
    - 매우 짧은 텍스트는 건너뜀.
    
    반환:
        Dict[str, float]: 언어 코드와 퍼센트(0~100) (AIAnalysisResponse의 languageRatio 필드)
        예: {"ko": 95, "en": 5}
    """
    langs = []
    for c in comments:
        t = (c or "").strip()
        if len(t) < 3:
            continue
        try:
            langs.append(detect(t))
        except Exception:
            continue

    if not langs:
        return {}

    cnt = Counter(langs)
    total = sum(cnt.values()) or 1
    out: Dict[str, float] = {}
    for code, n in cnt.items():
        ratio = round(n / total * 100)
        if ratio > 0:
            out[code] = ratio
    return out


# ============================================================
# 사용 예시 (테스트용 코드)
# ============================================================
if __name__ == "__main__":
    import json
    
    # config.py에서 VIDEO_KEY 확인
    if not VIDEO_KEY:
        print("[ERROR] VIDEO_KEY가 .env 파일에 설정되지 않았습니다.")
        print("[INFO] 테스트 데이터로 실행합니다.\n")
        
        # 테스트 데이터 (다양한 언어)
        test_comments = [
            "정말 유익한 영상이네요! 감사합니다.",
            "이 영상 정말 좋아요. 구독하고 갑니다.",
            "Thank you for this amazing video!",
            "This is so helpful, thanks!",
            "정말 감사합니다.",
            "한국어 댓글입니다.",
            "Great content!",
            "이해하기 쉽게 설명해주셔서 감사해요.",
        ]
        
        language_ratio = detect_languages(test_comments)
        
        # JSON 형식으로 출력
        result = {"languageRatio": language_ratio}
        print("=" * 60)
        print("[결과] 언어 비율 분석 (테스트 데이터):")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 60)
        
    else:
        if not YOUTUBE_API_KEY:
            print("[ERROR] YOUTUBE_API_KEY가 .env 파일에 설정되지 않았습니다.")
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
            
            # 언어 비율 분석 실행
            print(f"[INFO] 언어 비율 분석 중...\n")
            language_ratio = detect_languages(youtube_comments)
            
            # ============================================================
            # JSON 형식으로 결과 출력 (AIAnalysisResponse의 languageRatio 필드)
            # ============================================================
            result = {"languageRatio": language_ratio}
            
            print("=" * 60)
            print("[결과] 언어 비율 분석 (languageRatio 필드):")
            print("=" * 60)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("=" * 60)
            
        except Exception as e:
            print(f"[ERROR] 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)