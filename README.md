# 📍AI 기반 유튜브 댓글 분석 및 사용자 반응 인사이트 도출 시스템

> AI 기반 YouTube 댓글 분석으로 여론의 흐름을 한눈에 파악하세요

---
### 👥 팀원
| **AI** | **FrontEnd** | **BackEnd** | **BackEnd** |
| :------: | :------: | :-----: | :-----: |
| <img src="https://avatars.githubusercontent.com/u/202420613?v=4" width="100px;"/> | <img src="https://avatars.githubusercontent.com/u/102670510?v=4" width="100px;"/> | <img src="https://avatars.githubusercontent.com/u/105547387?v=4" width="100px;"/> | <img src="https://avatars.githubusercontent.com/u/104975763?v=4" width="100px;"/> |
| [이소민](https://github.com/isoxosoi) | [최윤경](https://github.com/yun) | [강예린](https://github.com/kyer5) | [차주혜](https://github.com/Alal11) |

<br></br>

## 📌 프로젝트 요약

**YouTube 댓글 AI 분석 서비스**는 방대한 YouTube 댓글 데이터를 AI로 분석하여 시청자와 크리에이터 모두에게 실질적인 인사이트를 제공하는 웹 서비스입니다.

<br></br>

## 🚨 문제 정의

### 현재의 문제점

#### 1. 편향된 여론 형성
- 유튜브 댓글은 시청자의 실제 의견을 반영하는 중요한 지표
- 하지만 **일부 콘텐츠에서 편향적 반응이나 과도한 비판 여론**이 형성
- 시청자가 균형 잡힌 시각을 갖기 어려운 문제 발생

#### 2. 트렌드 파악의 어려움
- **YouTube 인기 급상승 동영상 기능 폐지**로 실시간 트렌드 파악 불가능
- 어떤 영상이 화제인지, 왜 인기인지 알기 어려움

#### 3. 언어 장벽
- 댓글 내 **신조어, 유행어, 밈** 등으로 인해 맥락 이해 어려움
- 세대 간, 커뮤니티 간 표현 격차로 의미 파악 곤란

#### 4. 기존 도구의 한계
- 기존 분석 도구는 단순한 긍정/부정/중립 분류에 그침
- 시청자 반응의 **전반적 흐름을 파악하는 데 한계** 존재
- 크리에이터는 구체적인 피드백과 개선 방향을 찾기 어려움

### 우리의 해결책

✅ **세분화된 감정 분석**  
7가지 감정 카테고리로 댓글을 분류하여 더 정교한 여론 파악

✅ **시간에 따른 감정 흐름 시각화**  
날짜별 감정 변화를 그래프로 표현하여 논란 발생 시점 및 확산 패턴 신속 탐지

✅ **영상 간 비교 분석**  
유사 콘텐츠 최대 3개를 비교하여 경쟁 채널 대비 자신의 위치 파악

✅ **자체 인기 랭킹 알고리즘**  
검색 빈도, 스크랩 수, 체류 시간을 종합한 **독자적 인기도 점수**로 트렌드 제공

✅ **AI 기반 신조어 해석**  
댓글 단어를 드래그하면 즉시 의미를 설명하여 언어 장벽 제거

<br></br>

## ✨ 주요 기능

### 1. 세분화된 감정 분석 (7가지 감정)
```
😊 기쁨 (JOY)        ❤️ 사랑 (LOVE)       🙏 감사 (GRATITUDE)
😡 화남 (ANGER)      😢 슬픔 (SADNESS)    😨 두려움 (FEAR)
😐 중립 (NEUTRAL)
```
- **SamLowe/roberta-base-go_emotions** 모델 활용
- 단순 긍정/부정을 넘어 복합적인 감정 뉘앙스까지 정교하게 분석
- 댓글 감정 분포를 원형 차트와 막대 그래프로 시각화

### 2. 댓글 반응 흐름 분석
- **시계열 그래프**로 날짜별 감정 변화 추이를 한눈에 확인
- 논란 확산 시점을 자동으로 탐지하고 경고
- 여론이 긍정에서 부정으로 변화하는 패턴 추적

### 3. AI 기반 논란 탐지
- **facebook/bart-large-mnli** 모델의 Zero-shot Classification 활용
- 사기, 뒷광고, 조작 의혹 등 민감한 이슈를 조기 감지
- 2학기 개선: 라벨링 구조 변경 + 임계값 튜닝으로 **오탐지(FP) 감소** 및 정밀도 향상

### 4. AI 댓글 전체 요약
- **OpenAI ChatGPT-4o** 기반으로 수천 개의 댓글을 한 문단으로 요약
- 핵심 의견과 주요 반응만 간추려 시간 절약
- 영상을 보기 전 전체 분위기를 빠르게 파악

### 5. 영상 비교 분석 (최대 3개)
- 유사 콘텐츠 또는 경쟁 채널의 영상을 나란히 비교
- 감정 분포, 댓글 수, 반응 속도 등을 한눈에 비교
- 크리에이터가 자신의 콘텐츠 경쟁력을 객관적으로 평가

### 6. 영상 키워드 검색
- 키워드 기반으로 관련 채널 및 영상을 빠르게 탐색
- 분석하고 싶은 영상을 효율적으로 발견
- 실시간 검색 트렌드 반영

### 7. 자체 인기 랭킹 "지금 핫한"
**인기도 점수 산정 공식:**
```
인기도 = 0.35 × URL 직접 진입
       + 0.25 × 검색 진입
       + 0.35 × 스크랩 수
       + 0.05 × 평균 체류 시간 지수
```
- 매 시간 단위로 자동 업데이트
- YouTube 인기 급상승 기능의 공백을 대체
- **핫한 검색어** 순위도 함께 제공

### 8. 댓글 신조어/유행어 자동 해석
- 댓글 내 단어를 드래그하면 **AI가 즉시 의미를 설명**
- 빠르게 변화하는 온라인 언어 환경에 실시간 대응
- 세대 간, 커뮤니티 간 표현 격차 해소

### 9. 댓글 필터링 및 검색
- 감정별 필터링 (기쁨, 화남 등)
- AI 키워드 기반 댓글 검색
- 좋아요 Top 5 댓글 자동 추출
- 댓글 작성 시간대 분포 차트

### 10. 다국어 댓글 언어 분석
- 댓글의 언어 비율을 자동으로 분석
- 글로벌 시청자 분포 파악
- 다국어 댓글에 대한 통계 제공
 
<br></br>

## 🎥 시연 영상
[🔗 YouTube에서 보기](https://www.youtube.com/watch?v=fYeV95uBCds)

<br></br>

## 🛠️ 기술 스택

### Frontend

&ensp;![Frontend Stack](https://go-skill-icons.vercel.app/api/icons?i=vercel,nextjs,ts,tailwind,axios,yarn)

### Backend

&ensp;![Backend Stack](https://go-skill-icons.vercel.app/api/icons?i=spring,java,mysql,aws,docker)

### AI

 &ensp;![AI Stack](https://go-skill-icons.vercel.app/api/icons?i=python,fastapi,pytorch,huggingface,git)

<br></br>

## 🏗 시스템 아키텍처
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/a655b1be-c8c5-4a4c-ad15-a46764132f9a" />

<br></br>

## 🎨 사용 예시

### 시청자 관점
```
📺 논란 영상을 보기 전에...
   → 댓글 분석으로 미리 분위기 파악
   → 편향된 여론에 휩쓸리지 않고 균형 잡힌 판단

🔍 유행하는 밈이나 신조어가 궁금할 때...
   → 단어 드래그로 즉시 의미 확인
   → 세대 간 표현 격차 해소

📊 여러 영상의 반응을 비교하고 싶을 때...
   → 최대 3개 영상을 동시에 비교 분석
   → 다양한 시각으로 균형 잡힌 정보 습득
```

### 크리에이터 관점
```
📊 내 영상이 어떻게 받아들여지고 있는지...
   → 감정 분포와 시계열 흐름으로 시청자 반응 분석
   → 논란 조기 감지로 빠른 대응 가능

⚔️ 경쟁 채널과 비교하고 싶을 때...
   → 최대 3개 영상의 반응을 나란히 비교
   → 차별화 포인트와 개선 방향 파악

🎯 다음 콘텐츠 기획을 위해...
   → 인기 검색어와 트렌드 분석
   → 댓글 작성 시간대를 고려한 업로드 최적화

💡 시청자 피드백을 구체적으로 파악...
   → 긍정/부정이 아닌 7가지 세부 감정으로 분석
   → "무엇이 좋았고, 무엇이 아쉬웠는지" 명확히 이해
```
-----
# 태그

## 🔹 각 TAG를 언제 쓰는지 예시

| 태그 | 의미 | 예시 커밋 메시지 |
| --- | --- | --- |
| **feat** | 새로운 기능 추가 | `feat: add Question-Answer screen UI` |
| **fix** | 버그 수정 | `fix: wrong API endpoint for food-safety terms` |
| **refact** | 코드 리팩토링 (동작은 같지만 구조 개선) | `refact: simplify fetch hook logic` |
| **comment** | 주석 추가, 오타 수정 (코드 동작 변화 없음) | `comment: add explanation for API params` |
| **docs** | 문서 수정 (README, 위키 등) | `docs: update installation guide` |
| **art** | 이미지, 아이콘, 디자인 리소스 | `art: add logo and app splash image` |
| **merge** | 브랜치 병합 | `merge: feature/api into main` |
| **rename** | 파일/폴더 이름 변경 또는 위치 이동 | `rename: move hooks folder into utils` |
| **chore** | 잡일 (패키지 설치, 설정 변경, 파일정리 및 삭제 등) | `chore: add eslint config and prettier` |

## 📌 정리

1. **브랜치 이름** → 태그/작업이름
    - `feat/login-screen`
    - `fix/api-bug`
    - `docs/readme-update`
2. **커밋 메시지** → 태그: 설명
    - `feat: add login screen UI`
    - `fix: correct API endpoint`
    - `docs: update installation guide`
