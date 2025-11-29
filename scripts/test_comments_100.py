"""
유튜브 댓글 100개 테스트 데이터셋
- 논란 탐지 모델 + 감정 분석 모델 성능 비교용
- (댓글 텍스트, 논란 라벨, 감정 라벨) 형태
- 논란 라벨: True = 논란/사기/뒷광고, False = 일반 댓글
- 감정 라벨: positive, negative, other
"""


# 테스트 데이터: (댓글, 논란 라벨, 감정 라벨)
TEST_COMMENTS = [
    # ===== 긍정적 댓글 (20개) - 논란: False, 감정: positive =====
    ("와 진짜 대박이네요! 감사합니다!", False, "positive"),
    ("너무 유익한 영상이에요 ㅎㅎ", False, "positive"),
    ("최고입니다! 구독 박고 갑니다!", False, "positive"),
    ("오늘도 좋은 영상 감사합니다~", False, "positive"),
    ("정말 도움이 많이 됐어요!", False, "positive"),
    ("설명 진짜 잘하시네요 👍", False, "positive"),
    ("이런 영상 기다렸어요!", False, "positive"),
    ("완전 꿀팁이네요 감사합니다!", False, "positive"),
    ("목소리도 좋으시고 설명도 명쾌하세요", False, "positive"),
    ("앞으로도 좋은 영상 부탁드려요!", False, "positive"),
    ("진심으로 감사드립니다", False, "positive"),
    ("이 채널 알게 돼서 다행이에요", False, "positive"),
    ("영상 퀄리티가 점점 좋아지네요!", False, "positive"),
    ("덕분에 많이 배워갑니다", False, "positive"),
    ("항상 응원합니다!", False, "positive"),
    ("정말 유익한 정보네요", False, "positive"),
    ("이해하기 쉽게 설명해주셔서 감사합니다", False, "positive"),
    ("최고의 영상입니다!", False, "positive"),
    ("구독자 100만 가즈아!", False, "positive"),
    ("사랑합니다 ❤️", False, "positive"),

    # ===== 부정적 댓글 - 비판/불만 (20개) - 논란: False, 감정: negative =====
    ("광고가 너무 길어요...", False, "negative"),
    ("이건 좀 아닌 것 같은데요", False, "negative"),
    ("내용이 너무 빈약하네요", False, "negative"),
    ("기대했는데 실망이에요", False, "negative"),
    ("전문성이 부족한 것 같아요", False, "negative"),
    ("영상 편집 좀 신경 쓰세요", False, "negative"),
    ("사실과 다른 내용이 있네요", False, "negative"),
    ("다른 유튜버가 더 잘 설명하던데", False, "negative"),
    ("왜 이렇게 말을 빙빙 돌리세요?", False, "negative"),
    ("시간만 낭비했네요", False, "negative"),
    ("음질이 너무 안 좋아요", False, "negative"),
    ("제목 낚시 심하네요", False, "negative"),
    ("구독 취소합니다", False, "negative"),
    ("이건 잘못된 정보인데요?", False, "negative"),
    ("댓글 조작 의심됩니다", False, "negative"),
    ("너무 실망스럽네요", False, "negative"),
    ("퀄리티가 떨어졌어요", False, "negative"),
    ("예전만 못하네요", False, "negative"),
    ("억지스러운 부분이 많아요", False, "negative"),
    ("이건 좀 심했다", False, "negative"),

    # ===== 뒷광고/협찬 의심 (15개) - 논란: True, 감정: negative/other =====
    ("뒷광고 아닌가요?", True, "negative"),
    ("협찬 받으셨나요?", True, "other"),  # 단순 질문
    ("돈 받고 홍보하시는 거죠?", True, "negative"),
    ("스폰서십 표기 안 하셨네요", True, "negative"),
    ("광고인지 밝히세요", True, "negative"),
    ("이건 명백한 광고인데요", True, "negative"),
    ("협찬 받고 거짓 리뷰", True, "negative"),
    ("유료 광고 표시 안 하셨네요", True, "negative"),
    ("돈 받고 추천하는 거 맞죠?", True, "negative"),
    ("뒷광고 신고합니다", True, "negative"),
    ("협찬 표시 어디있어요?", True, "negative"),
    ("이거 광고 맞죠?", True, "other"),  # 단순 질문
    ("스폰받고 올리는 건가요?", True, "other"),  # 단순 질문
    ("이런 식으로 속이면 안 되죠", True, "negative"),
    ("광고는 광고라고 밝히세요", True, "negative"),

    # ===== 질문/궁금증 (15개) - 논란: False, 감정: other =====
    ("이거 어디서 살 수 있나요?", False, "other"),
    ("가격이 얼마인가요?", False, "other"),
    ("혹시 링크 있으신가요?", False, "other"),
    ("다음 영상은 언제 올라오나요?", False, "other"),
    ("이거 초보자도 할 수 있을까요?", False, "other"),
    ("어떤 제품 쓰시는 건가요?", False, "other"),
    ("배경음악 제목이 뭔가요?", False, "other"),
    ("몇 시간 걸렸어요?", False, "other"),
    ("이거 무료인가요?", False, "other"),
    ("다른 방법은 없나요?", False, "other"),
    ("이거랑 저거 중에 뭐가 나은가요?", False, "other"),
    ("설명서 같은 거 있나요?", False, "other"),
    ("주의할 점이 있을까요?", False, "other"),
    ("이거 한국에서도 되나요?", False, "other"),
    ("업데이트 예정 있으신가요?", False, "other"),

    # ===== 중립/정보 공유 (10개) - 논란: False, 감정: other =====
    ("참고로 이 방법은 윈도우에서만 됩니다", False, "other"),
    ("저는 다른 방법으로 해결했어요", False, "other"),
    ("2024년 기준으로는 이렇게 바뀌었습니다", False, "other"),
    ("Mac 사용자는 이렇게 하시면 돼요", False, "other"),
    ("추가 정보 공유합니다", False, "other"),
    ("공식 홈페이지 링크 남깁니다", False, "other"),
    ("관련 영상 추천드려요", False, "other"),
    ("업데이트 이후로 달라진 점 있어요", False, "other"),
    ("다른 옵션도 있더라고요", False, "other"),
    ("최신 버전에서는 안 되네요", False, "other"),

    # ===== 농담/밈 (10개) - 논란: False, 감정: positive/negative/other =====
    ("이거 보고 따라했다가 망했어요 ㅋㅋㅋ", False, "positive"),  # 유머
    ("제 통장이 텅장되는 소리가 들리네요", False, "positive"),  # 유머
    ("엄마한테 혼났습니다...", False, "other"),
    ("역시 짤은 살아있다", False, "positive"),  # 유머
    ("누가 보면 프로인 줄 알겠네요 ㅋㅋ", False, "positive"),  # 유머
    ("이거 따라하다 컴퓨터 날렸어요 ㅠㅠ", False, "negative"),  # 실제 불만
    ("내 시간 돌려줘", False, "negative"),  # 불만
    ("전문가의 향기가 나네요", False, "positive"),  # 칭찬
    ("이런 걸 왜 이제 알았을까", False, "positive"),  # 긍정적 놀람
    ("와... 내가 바보였구나", False, "other"),  # 자조

    # ===== 기타/잡담 (10개) - 논란: False, 감정: positive/other =====
    ("몇 번째 시청 중입니다", False, "positive"),  # 좋아서 반복 시청
    ("알고리즘이 저를 여기로", False, "other"),
    ("아 진짜요? 몰랐네요", False, "other"),
    ("오 신기하네요", False, "positive"),  # 긍정적 반응
    ("이게 되네요?", False, "other"),
    ("처음 알았어요", False, "other"),
    ("좋은 정보 감사합니다", False, "positive"),
    ("유용하게 쓸게요", False, "positive"),
    ("나중에 또 볼게요", False, "positive"),
    ("도움 됐습니다", False, "positive"),
]

# 카테고리 인덱스 정보
CATEGORY_INFO = {
    "긍정적 댓글": {"range": (0, 20), "expected_controversy": False, "expected_sentiment": "positive"},
    "부정적 댓글 (비판)": {"range": (20, 40), "expected_controversy": False, "expected_sentiment": "negative"},
    "뒷광고/협찬 의심": {"range": (40, 55), "expected_controversy": True, "expected_sentiment": "negative"},
    "질문/궁금증": {"range": (55, 70), "expected_controversy": False, "expected_sentiment": "other"},
    "중립/정보 공유": {"range": (70, 80), "expected_controversy": False, "expected_sentiment": "other"},
    "농담/밈": {"range": (80, 90), "expected_controversy": False, "expected_sentiment": "mixed"},
    "기타/잡담": {"range": (90, 100), "expected_controversy": False, "expected_sentiment": "mixed"},
}


def get_comments_only():
    """댓글 텍스트만 반환"""
    return [comment for comment, _, _ in TEST_COMMENTS]


def get_controversy_labels():
    """논란 라벨만 반환"""
    return [label for _, label, _ in TEST_COMMENTS]


def get_sentiment_labels():
    """감정 라벨만 반환"""
    return [sentiment for _, _, sentiment in TEST_COMMENTS]


def get_stats():
    """데이터셋 통계"""
    total = len(TEST_COMMENTS)
    
    # 논란 통계
    controversy = sum(1 for _, label, _ in TEST_COMMENTS if label)
    normal = total - controversy
    
    # 감정 통계
    positive = sum(1 for _, _, sent in TEST_COMMENTS if sent == "positive")
    negative = sum(1 for _, _, sent in TEST_COMMENTS if sent == "negative")
    other = sum(1 for _, _, sent in TEST_COMMENTS if sent == "other")
    
    return {
        "total": total,
        # 논란 통계
        "controversy": controversy,
        "normal": normal,
        # 감정 통계
        "positive": positive,
        "negative": negative,
        "other": other,
    }


if __name__ == "__main__":
    stats = get_stats()
    print(f"📊 테스트 데이터셋 통계")
    print(f"  총 댓글: {stats['total']}개")
    print()
    print(f"[논란 탐지용]")
    print(f"  🔴 논란: {stats['controversy']}개")
    print(f"  🟢 정상: {stats['normal']}개")
    print()
    print(f"[감정 분석용]")
    print(f"  😊 긍정 (positive): {stats['positive']}개")
    print(f"  😢 부정 (negative): {stats['negative']}개")
    print(f"  😐 중립 (other): {stats['other']}개")