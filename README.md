# CSV 기반 RAG 시스템

이 프로젝트는 Streamlit UI를 통해 CSV 파일을 업로드하고, 데이터 전처리(결측치 제거, 키워드 및 AI 기반 필터링)를 수행한 후, 문장 단위로 임베딩하여 Qdrant 또는 ChromaDB에 저장하고, 유사도 검색을 통해 LLM(OpenAI/ollama)과 연동하여 질의 응답을 제공하는 RAG( Retrieval-Augmented Generation) 시스템입니다.

## 프로젝트 구조

```
rag_project/
├── ui.py                # Streamlit UI 애플리케이션
├── preprocess.py        # 데이터 전처리 모듈
├── storage.py           # 벡터 임베딩 및 저장 모듈
├── agent.py             # 검색 및 응답 생성 에이전트 모듈
├── llm_wrapper.py       # LLM 호출 래퍼 모듈
├── requirements.txt     # 프로젝트 의존성
└── README.md            # 프로젝트 설명
```

## 설치 및 실행

1. 가상환경 생성 및 활성화 (옵션)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```

3. Streamlit 앱 실행
   ```bash
   streamlit run ui.py
   ```

4. 브라우저에서 `http://localhost:8501` 접속 후 CSV 업로드 및 기능 사용

## 주요 기능

- CSV 파일 업로드 및 미리보기
- 결측치 비율 설정에 따른 제거
- 키워드 및 AI 기반 광고/홍보 필터링
- 한국어 특화 임베딩 후 Qdrant/ChromaDB 저장
- 유사도 검색 기반 LLM 질의 응답

## 확장 및 커스터마이징

- `ai_filter.py`를 추가하여 자체 파인튜닝 BERT 분류기를 통합 가능
- `VectorStore` 초기화 시 `backend` 옵션으로 Qdrant/ChromaDB 선택
- `RetrievalAgent` 내부 LLM 모델(OpenAI/ollama) 변경 가능
