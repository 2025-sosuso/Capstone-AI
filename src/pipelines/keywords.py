# src/pipelines/keywords.py
"""
[기능3] TF-IDF 기반 키워드 분석 (다국어 명사 추출 버전)
- langdetect로 언어 감지 후 언어별 토크나이저 분기
- 한국어: kiwipiepy, 영어: spaCy, 일본어: fugashi, 중국어: jieba
- 기타 70개+ 언어: Stanza
- 불용어 필터링 + 1~2 gram + 부분중복 제거
"""
import os
import sys
import re
from typing import List, Set

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
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True, check=False)
    except Exception:
        pass

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from langdetect import detect, LangDetectException

# 언어별 라이브러리
from kiwipiepy import Kiwi

# ============================================================
# API 키 및 유틸리티 불러오기
# ============================================================
try:
    from src.config import YOUTUBE_API_KEY, VIDEO_KEY
    from src.utils.youtube import fetch_youtube_comments
except ImportError:
    from ..config import YOUTUBE_API_KEY, VIDEO_KEY
    from ..utils.youtube import fetch_youtube_comments

# ============================================================
# 1) 언어별 불용어
# ============================================================
KO_STOPWORDS = {
    # 일반 잡음/구어체
    "진짜", "정말", "다시", "너무", "최고",
    "그냥", "근데", "그리고", "하지만", "또", "또한", "이거", "저거", "그거",
    "이건", "저건", "그건", "거의", "그렇죠", "그렇다", "거든요", "뭔가",
    "에서", "으로", "이나", "하고", "같아요", "같은", "거에요", "거예요",
    "근데요", "아니", "아니요", "막", "또요", "좀", "그럼", "그래도",
    "이게", "제발",
    # 감탄/칭찬 표현
    "대박", "짱", "굿", "쩔어", "미쳤", "레전드", "갓", "킹",
    "천재", "존잘", "존예", "개꿀", "실화", "인정", "ㅋㅋ", "ㅎㅎ",
    "와", "우와", "헐", "오", "아",
    # 유튜브 관련 명사
    "영상", "채널", "구독", "알림", "시청", "댓글", "추천", "조회",
    "좋아요", "설정",
    # 일반적 칭찬/감정 명사
    "감사", "응원", "사랑", "최애", "공감",
    # 추상적/지시적 일반 명사
    "것", "수", "거", "때", "말", "게", "점", "번", "분", "중",
    "정도", "이유", "생각", "느낌", "기분", "마음", "부분", "경우",
    "사람", "모습", "얘기", "이야기", "내용", "정보", "도움",
}

JA_STOPWORDS = {
    "これ", "それ", "あれ", "この", "その", "あの",
    "ここ", "そこ", "あそこ", "こちら", "どこ", "だれ",
    "なに", "なん", "何", "私", "僕", "俺", "あなた",
    "こと", "もの", "ため", "よう", "ところ", "とき",
    "動画", "チャンネル", "登録", "視聴", "コメント",
    "最高", "すごい", "やばい", "神", "草", "笑",
}

ZH_STOPWORDS = {
    "的", "了", "是", "我", "你", "他", "她", "它",
    "这", "那", "哪", "什么", "怎么", "为什么",
    "很", "太", "真", "好", "吗", "呢", "啊", "吧",
    "视频", "频道", "订阅", "观看", "评论", "点赞",
    "最", "更", "非常", "特别", "一个", "这个", "那个",
}

DE_STOPWORDS = {
    # 일반 불용어
    "danke", "bitte", "mal", "auch", "noch", "schon", "sehr", "ganz",
    "ja", "nein", "nicht", "aber", "oder", "und", "mit", "für", "von",
    "das", "der", "die", "ein", "eine", "ist", "sind", "war", "haben",
    "wird", "kann", "muss", "soll", "wir", "ihr", "sie", "ich", "du",
    # 유튜브 관련
    "video", "kanal", "abo", "abonnieren", "like", "kommentar",
    # 감탄/칭찬 표현
    "toll", "super", "geil", "krass", "nice", "cool", "mega", "hammer",
}

FR_STOPWORDS = {
    # 일반 불용어
    "merci", "bien", "très", "aussi", "encore", "mais", "ou", "et", "avec",
    "pour", "dans", "sur", "par", "pas", "plus", "moins", "tout", "tous",
    "le", "la", "les", "un", "une", "des", "ce", "cette", "ces",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "est", "sont", "être", "avoir", "fait", "faire",
    # 유튜브 관련
    "vidéo", "chaîne", "abonner", "commentaire",
    # 감탄/칭찬 표현
    "génial", "super", "magnifique", "incroyable", "top", "bravo",
}

ES_STOPWORDS = {
    # 일반 불용어
    "gracias", "bien", "muy", "también", "pero", "más", "menos",
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "con", "por", "para", "sin", "sobre",
    "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas",
    "es", "son", "está", "están", "ser", "estar", "haber", "tener",
    "que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde",
    # 유튜브 관련
    "video", "canal", "suscribir", "comentario",
    # 감탄/칭찬 표현
    "genial", "increíble", "fantástico", "excelente", "bueno", "buena",
}

RU_STOPWORDS = {
    # 일반 불용어
    "спасибо", "очень", "тоже", "также", "но", "и", "или", "а",
    "в", "на", "с", "по", "за", "из", "от", "до", "для", "без",
    "я", "ты", "он", "она", "мы", "вы", "они", "это", "эта", "этот",
    "что", "как", "где", "когда", "почему", "кто",
    "есть", "быть", "было", "будет", "был", "была",
    # 유튜브 관련
    "видео", "канал", "подписка", "комментарий",
    # 감탄/칭찬 표현
    "круто", "класс", "супер", "отлично", "здорово", "молодец",
}

# 범용 불용어 (유튜브 관련, 다국어 공통 인터넷 표현)
UNIVERSAL_STOPWORDS = {
    # 유튜브/인터넷 관련
    "video", "channel", "subscribe", "like", "comment", "watch",
    "youtube", "http", "https", "www", "com", "org", "net",
    # 인터넷 감탄사/이모티콘
    "lol", "omg", "wow", "haha", "hehe", "hihi", "xd", "xdd",
    "nice", "cool", "good", "great", "awesome", "amazing",
    # 숫자/기호
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
}

EN_STOPWORDS = set(ENGLISH_STOP_WORDS) | UNIVERSAL_STOPWORDS

# 전체 불용어 합치기
ALL_STOPWORDS = (
    KO_STOPWORDS | EN_STOPWORDS | JA_STOPWORDS | ZH_STOPWORDS |
    DE_STOPWORDS | FR_STOPWORDS | ES_STOPWORDS | RU_STOPWORDS |
    UNIVERSAL_STOPWORDS
)

# ============================================================
# 2) 언어별 분석기 인스턴스 (지연 로딩)
# ============================================================
_kiwi = None       # 한국어
_spacy_en = None   # 영어
_fugashi = None    # 일본어
_stanza_pipelines = {}  # 기타 언어용 Stanza (언어별 캐싱)

def _get_kiwi():
    """한국어: Kiwi 인스턴스 지연 로딩"""
    global _kiwi
    if _kiwi is None:
        print("[INFO] Kiwi 형태소 분석기 로딩 중...")
        _kiwi = Kiwi()
        print("[SUCCESS] Kiwi 로딩 완료!")
    return _kiwi

def _get_spacy_en():
    """영어: spaCy 인스턴스 지연 로딩"""
    global _spacy_en
    if _spacy_en is None:
        print("[INFO] spaCy 영어 모델 로딩 중...")
        import spacy
        _spacy_en = spacy.load("en_core_web_sm")
        print("[SUCCESS] spaCy 로딩 완료!")
    return _spacy_en

def _get_fugashi():
    """일본어: fugashi 인스턴스 지연 로딩"""
    global _fugashi
    if _fugashi is None:
        print("[INFO] Fugashi 형태소 분석기 로딩 중...")
        import fugashi
        _fugashi = fugashi.Tagger()
        print("[SUCCESS] Fugashi 로딩 완료!")
    return _fugashi

def _get_stanza_pipeline(lang: str):
    """기타 언어: Stanza 파이프라인 지연 로딩 (언어별 캐싱)"""
    global _stanza_pipelines
    
    if lang not in _stanza_pipelines:
        print(f"[INFO] Stanza '{lang}' 모델 로딩 중...")
        import stanza
        
        try:
            # 모델 다운로드 (이미 있으면 스킵)
            stanza.download(lang, verbose=False)
            _stanza_pipelines[lang] = stanza.Pipeline(lang, processors='tokenize,pos', verbose=False)
            print(f"[SUCCESS] Stanza '{lang}' 로딩 완료!")
        except Exception as e:
            print(f"[WARNING] Stanza '{lang}' 로딩 실패: {e}")
            _stanza_pipelines[lang] = None
    
    return _stanza_pipelines.get(lang)

# ============================================================
# 3) Stanza 지원 언어 목록
# ============================================================
# langdetect 코드 → Stanza 코드 매핑 (다른 경우만)
LANGDETECT_TO_STANZA = {
    "zh-cn": "zh-hans",
    "zh-tw": "zh-hant",
}

# Stanza가 지원하는 주요 언어들 (70개+)
STANZA_SUPPORTED = {
    "af", "ar", "be", "bg", "ca", "cs", "da", "de", "el", "es", "et", "eu",
    "fa", "fi", "fr", "ga", "gl", "he", "hi", "hr", "hu", "hy", "id", "it",
    "la", "lt", "lv", "mk", "mt", "nl", "no", "pl", "pt", "ro", "ru", "sk",
    "sl", "sr", "sv", "ta", "te", "th", "tr", "uk", "ur", "vi",
    "zh-hans", "zh-hant",
}

# ============================================================
# 4) 언어별 명사 추출 함수
# ============================================================
def _korean_nouns(text: str) -> List[str]:
    """한국어 명사 추출 (kiwipiepy)"""
    kiwi = _get_kiwi()
    result = kiwi.tokenize(text or "")
    
    out = []
    for token in result:
        word = token.form
        pos = token.tag
        
        # NNG(일반명사), NNP(고유명사)
        if pos in ('NNG', 'NNP'):
            if len(word) >= 2 and word not in KO_STOPWORDS:
                out.append(word)
        # SL(외국어) - 한국어 텍스트 내 영어 단어
        elif pos == 'SL':
            word_lower = word.lower()
            if len(word_lower) >= 2 and word_lower not in EN_STOPWORDS:
                out.append(word_lower)
    
    return out

def _english_nouns(text: str) -> List[str]:
    """영어 명사 추출 (spaCy)"""
    nlp = _get_spacy_en()
    doc = nlp(text or "")
    
    out = []
    for token in doc:
        # NOUN(일반명사), PROPN(고유명사)
        if token.pos_ in ('NOUN', 'PROPN'):
            word = token.text.lower()
            if len(word) >= 2 and word not in EN_STOPWORDS:
                out.append(word)
    
    return out

def _japanese_nouns(text: str) -> List[str]:
    """일본어 명사 추출 (fugashi)"""
    tagger = _get_fugashi()
    
    out = []
    for word in tagger(text or ""):
        # feature: 품사 정보 (명사 = 名詞)
        if word.feature.pos1 == '名詞':
            surface = word.surface
            if len(surface) >= 2 and surface not in JA_STOPWORDS:
                out.append(surface)
    
    return out

def _chinese_nouns(text: str) -> List[str]:
    """중국어 명사 추출 (jieba)"""
    import jieba
    import jieba.posseg as pseg
    
    out = []
    words = pseg.cut(text or "")
    for word, flag in words:
        # n: 명사, nr: 인명, ns: 지명, nt: 기관명, nz: 기타 고유명사
        if flag.startswith('n'):
            if len(word) >= 2 and word not in ZH_STOPWORDS:
                out.append(word)
    
    return out

def _stanza_nouns(text: str, lang: str) -> List[str]:
    """기타 언어 명사 추출 (Stanza)"""
    # langdetect → Stanza 언어 코드 변환
    stanza_lang = LANGDETECT_TO_STANZA.get(lang, lang)
    
    pipeline = _get_stanza_pipeline(stanza_lang)
    if pipeline is None:
        return _fallback_tokenizer(text)
    
    out = []
    try:
        doc = pipeline(text or "")
        for sentence in doc.sentences:
            for word in sentence.words:
                # NOUN(일반명사), PROPN(고유명사)
                if word.upos in ('NOUN', 'PROPN'):
                    w = word.text.lower() if word.text.isascii() else word.text
                    if len(w) >= 2 and w not in ALL_STOPWORDS:
                        out.append(w)
    except Exception as e:
        print(f"[WARNING] Stanza 처리 실패: {e}")
        return _fallback_tokenizer(text)
    
    return out

def _fallback_tokenizer(text: str) -> List[str]:
    """최후의 fallback (정규식 기반)"""
    pattern = re.compile(r'[\w]+', re.UNICODE)
    tokens = pattern.findall(text or "")
    
    out = []
    for t in tokens:
        if t.isdigit():
            continue
        if len(t) >= 2:
            w = t.lower() if t.isascii() else t
            if w not in ALL_STOPWORDS:
                out.append(w)
    
    return out

# ============================================================
# 5) 통합 토크나이저 (언어 감지 → 분기)
# ============================================================
def _multilingual_noun_tokenizer(text: str) -> List[str]:
    """
    다국어 명사 추출 토크나이저
    - langdetect로 언어 감지
    - 언어별 적절한 토크나이저로 분기
    - 지원되지 않는 언어는 Stanza 또는 fallback 사용
    """
    if not text or len(text.strip()) < 3:
        return []
    
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    
    # 언어별 분기 (최적화된 라이브러리 우선)
    if lang == "ko":
        return _korean_nouns(text)
    elif lang == "en":
        return _english_nouns(text)
    elif lang == "ja":
        return _japanese_nouns(text)
    elif lang in ("zh-cn", "zh-tw"):
        return _chinese_nouns(text)
    else:
        # Stanza 지원 언어인지 확인
        stanza_lang = LANGDETECT_TO_STANZA.get(lang, lang)
        if stanza_lang in STANZA_SUPPORTED or lang in STANZA_SUPPORTED:
            return _stanza_nouns(text, lang)
        else:
            # 최후의 fallback
            return _fallback_tokenizer(text)

# ============================================================
# 6) 부분문자열(중복) 제거
# ============================================================
def _dedup_substrings(candidates: List[str], top_n: int) -> List[str]:
    """다른 키워드를 포함하는 긴 항목 제거"""
    final = [
        kw for kw in candidates
        if not any(short != kw and kw.find(short) != -1 for short in candidates)
    ]
    
    # 중복 제거 후 상위 N개 반환
    seen = set()
    return [w for w in final if not (w in seen or seen.add(w))][:top_n]

# ============================================================
# 7) 메인 함수
# ============================================================
def extract_keywords_tfidf(comments: List[str], top_n: int = 10) -> List[str]:
    """
    TF-IDF 상위 키워드 추출 (다국어 지원)
    - comments: 문서 리스트
    - top_n: 상위 몇 개 단어를 반환할지
    - 다국어 명사 추출 + 불용어 제거 + 1~2그램 + 부분중복 제거
    
    지원 언어:
    - 최적화: 한국어, 영어, 일본어, 중국어
    - Stanza: 독일어, 프랑스어, 스페인어, 러시아어 등 70개+
    - Fallback: 기타 모든 언어
    
    반환:
        List[str]: 추출된 키워드 리스트 (AIAnalysisResponse의 keywords 필드)
    """
    if not comments:
        return []

    vec = TfidfVectorizer(
        tokenizer=_multilingual_noun_tokenizer,
        analyzer="word",
        lowercase=False,
        ngram_range=(1, 2),
        max_features=5000,
        token_pattern=None,
    )
    X = vec.fit_transform(comments)

    scores = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    pairs = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)

    # 넉넉히 뽑은 다음(중복 제거 전) 정제
    raw_top = [w for w, _ in pairs[: top_n * 5]]
    
    # 불용어가 포함된 빅그램 정리
    cleaned = []
    for w in raw_top:
        parts = w.split()
        if len(parts) == 2 and all(p in ALL_STOPWORDS or len(p) < 2 for p in parts):
            continue
        cleaned.append(w)

    # 부분문자열 중복 제거
    return _dedup_substrings(cleaned, top_n)


# ============================================================
# 사용 예시 (테스트용 코드)
# ============================================================
if __name__ == "__main__":
    import json
    
    if not VIDEO_KEY:
        print("[ERROR] VIDEO_KEY가 .env 파일에 설정되지 않았습니다.")
        print("[INFO] 테스트 데이터로 실행합니다.\n")
        
        # 테스트 데이터 (다국어)
        test_comments = [
            # 한국어
            "정말 유익한 영상이네요! 감사합니다.",
            "언니가 돌아왔어요! 찰스 최고!",
            # 영어
            "This video is amazing! Great content!",
            "Love the editing style and music choice.",
            # 일본어
            "この動画は本当に面白いです！",
            "編集がとても上手ですね。",
            # 중국어
            "这个视频太棒了！内容很有趣。",
            "剪辑做得很好，音乐也很配。",
            # 독일어 (Stanza)
            "Dieses Video ist fantastisch! Tolle Arbeit!",
            # 프랑스어 (Stanza)
            "Cette vidéo est incroyable! J'adore le contenu.",
            # 스페인어 (Stanza)
            "Este video es increíble! Me encanta.",
            # 러시아어 (Stanza)
            "Это видео потрясающее! Отличная работа!",
        ]
        
        keywords = extract_keywords_tfidf(test_comments, top_n=15)
        
        result = {"keywords": keywords}
        print("=" * 60)
        print("[결과] 키워드 추출 (다국어 테스트 데이터):")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 60)
        
    else:
        if not YOUTUBE_API_KEY:
            print("[ERROR] YOUTUBE_API_KEY가 .env 파일에 설정되지 않았습니다.")
            sys.exit(1)
        
        print(f"[INFO] YouTube 비디오 '{VIDEO_KEY}'에서 댓글 수집 중...")
        
        try:
            youtube_comments = fetch_youtube_comments(
                video_id=VIDEO_KEY,
                api_key=YOUTUBE_API_KEY,
                max_pages=1,
                page_size=100,
                include_replies=False,
                apply_cleaning=True,
            )
            
            print(f"[SUCCESS] {len(youtube_comments)}개 댓글 수집 완료!")
            
            if not youtube_comments:
                print("[WARNING] 수집된 댓글이 없습니다.")
                sys.exit(0)
            
            print(f"[INFO] 키워드 추출 중...\n")
            keywords = extract_keywords_tfidf(youtube_comments, top_n=10)
            
            result = {"keywords": keywords}
            
            print("=" * 60)
            print("[결과] 키워드 추출 (keywords 필드):")
            print("=" * 60)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("=" * 60)
            
        except Exception as e:
            print(f"[ERROR] 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)