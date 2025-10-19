# src/pipelines/keywords.py
"""
[기능3] TF-IDF 기반 키워드 분석 (불용어 정제 + 부분중복 제거 버전)
- NLTK 없이 작동
- 한국어/영어 공통 불용어 필터링
- 1~2 gram 사용 + 결과에서 부분문자열(긴 구절) 중복 제거
"""
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import re

# -----------------------------
# 1) 불용어 (영어 + 한국어)
# -----------------------------
KO_STOPWORDS = {
    # 요청 예시
    "진짜", "정말", "다시", "너무",
    # 일반 잡음/구어체
    "그냥", "근데", "그리고", "하지만", "또", "또한", "이거", "저거", "그거",
    "이건", "저건", "그건", "거의", "그렇죠", "그렇다", "거든요", "뭔가",
    "에서", "으로", "이나", "하고", "같아요", "같은", "거에요", "거예요",
    "근데요", "근데요", "아니", "아니요", "막", "또요", "좀", "그럼", "그래도",
    "이게", "제발"
}
# 영어 기본 불용어 (scikit-learn 내장)
EN_STOPWORDS = set(ENGLISH_STOP_WORDS)

# 하나로 합치기
COMMON_STOPWORDS = KO_STOPWORDS.union(EN_STOPWORDS)

# -----------------------------
# 2) 간단 토크나이저
# -----------------------------
_word_pat = re.compile(r"[A-Za-z]+|[가-힣]+")

def _simple_tokenizer(text: str) -> List[str]:
    """
    - 영문/한글 토큰만 추출
    - 영어는 소문자화
    - 길이 1 토큰/불용어 제거
    """
    tokens = _word_pat.findall(text or "")
    out = []
    for t in tokens:
        if re.fullmatch(r"[A-Za-z]+", t):
            tt = t.lower()
        else:
            tt = t  # 한글은 소문자 개념 없음

        if len(tt) < 2:
            continue
        if tt in COMMON_STOPWORDS:
            continue
        out.append(tt)
    return out

# -----------------------------
# 3) 부분문자열(중복) 제거
# -----------------------------
def _dedup_substrings(candidates: List[str], top_n: int) -> List[str]:
    """
    - 상위 후보들에서 다른 항목의 부분문자열인 경우 제거
    - 예: '너무', '너무 길고' → '너무'만 남김(짧은 핵심어 우선)
    """
    filtered = []
    for kw in candidates:
        # kw가 더 긴 문구의 부분이면(= 긴 문구 존재) 짧은 쪽만 남기려면
        # '긴 문구 제거'가 아니라 '짧은 걸 남기기'니까
        # 긴 문구가 존재해도 kw를 남기고, 긴 문구는 뒤에서 자연히 걸러짐.
        # → 반대로 구현: kw가 '다른 항목의 부분문자열'이면 긴 항목을 버리도록
        if any((longer != kw and longer.find(kw) != -1) for longer in candidates):
            # 긴 항목을 제거하는 로직을 적용하기 위해,
            # 여기서는 일단 kw를 보류하고 긴 항목들이 필터링되도록 구조를 단순화.
            pass
        filtered.append(kw)

    # 이제 긴 항목(= 다른 키워드를 포함하는 항목)을 제거
    final = []
    for kw in filtered:
        if not any((short != kw and kw.find(short) != -1) for short in filtered):
            final.append(kw)

    # 고유 순서 유지 후 상위 N개 반환
    seen = set()
    uniq = []
    for w in final:
        if w not in seen:
            uniq.append(w); seen.add(w)
    return uniq[:top_n]

# -----------------------------
# 4) 메인 함수
# -----------------------------
def extract_keywords_tfidf(comments: List[str], top_n: int = 10) -> List[str]:
    """
    TF-IDF 상위 키워드 추출
    - comments: 문서 리스트
    - top_n: 상위 몇 개 단어를 반환할지
    - 불용어 제거 + 1~2그램 + 부분중복 제거
    """
    if not comments:
        return []

    vec = TfidfVectorizer(
        tokenizer=_simple_tokenizer,   # 커스텀 토크나이저
        analyzer="word",
        lowercase=False,               # we handle case in tokenizer
        ngram_range=(1, 2),            # 1~2 gram
        max_features=5000,
    )
    X = vec.fit_transform(comments)

    scores = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    pairs = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)

    # 넉넉히 뽑은 다음(중복 제거 전) 정제
    raw_top = [w for w, _ in pairs[: top_n * 5]]
    # 불용어가 포함된 빅그램 정리(두 토큰이 전부 불용어면 버림)
    cleaned = []
    for w in raw_top:
        parts = w.split()
        if len(parts) == 2 and all(p in COMMON_STOPWORDS or len(p) < 2 for p in parts):
            continue
        cleaned.append(w)

    # 부분문자열 중복 제거(짧은 핵심어 위주로 남김)
    return _dedup_substrings(cleaned, top_n)
