# src/pipelines/lang_ratio.py
"""
[기능4] 언어 비율 분석
- langdetect 사용
- 결과를 한국어명으로 매핑
"""
from collections import Counter
from typing import Dict, List

from langdetect import detect

LANGUAGE_NAME_MAP = {
    "af": "아프리칸스어", "sq": "알바니아어", "ar": "아랍어", "az": "아제르바이잔어",
    "be": "벨라루스어", "bg": "불가리아어", "bn": "벵골어", "ca": "카탈루냐어",
    "cs": "체코어", "cy": "웨일스어", "da": "덴마크어", "de": "독일어",
    "el": "그리스어", "en": "영어", "es": "스페인어", "et": "에스토니아어",
    "fa": "페르시아어", "fi": "핀란드어", "fr": "프랑스어", "gu": "구자라트어",
    "he": "히브리어", "hi": "힌디어", "hr": "크로아티아어", "hu": "헝가리어",
    "id": "인도네시아어", "is": "아이슬란드어", "it": "이탈리아어", "ja": "일본어",
    "jv": "자바어", "ka": "조지아어", "kk": "카자흐어", "km": "크메르어",
    "kn": "칸나다어", "ko": "한국어", "lt": "리투아니아어", "lv": "라트비아어",
    "mk": "마케도니아어", "ml": "말라얄람어", "mr": "마라티어", "ms": "말레이어",
    "my": "버마어", "ne": "네팔어", "nl": "네덜란드어", "no": "노르웨이어",
    "pa": "펀자브어", "pl": "폴란드어", "pt": "포르투갈어", "ro": "루마니아어",
    "ru": "러시아어", "sk": "슬로바키아어", "sl": "슬로베니아어", "sv": "스웨덴어",
    "sw": "스와힐리어", "ta": "타밀어", "te": "텔루구어", "th": "태국어",
    "tl": "타갈로그어", "tr": "터키어", "uk": "우크라이나어", "ur": "우르두어",
    "vi": "베트남어", "zh-cn": "중국어(간체)", "zh-tw": "중국어(번체)"
}

def detect_languages(comments: List[str]) -> Dict[str, float]:
    """
    댓글 리스트의 언어 비율(%)을 반환.
    - 매우 짧은 텍스트는 건너뜀.
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
        name = LANGUAGE_NAME_MAP.get(code, f"기타({code})")
        pct = round(n / total * 100, 2)
        if pct > 0:
            out[name] = pct
    return out
