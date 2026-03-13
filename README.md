# RAG_Agent_System
KMMLU 형사법 객관식 QA를 위한 RAG 기반 에이전트

# Agent System 구조
```
사용자 요청 (query: 문제 + 선택지)
        │
        ▼
  FastAPI 서버 (POST /inference)
        │
        ▼
   RagRuntime
   ├── 1. 쿼리를 임베딩 (text-embedding-3-small)
   ├── 2. train 벡터 인덱스에서 유사 예시 검색 (cosine similarity)
   ├── 3. 유사도 ≥ 0.85인 경우에만 컨텍스트로 활용 (selective RAG)
   └── 4. gpt-4o-mini에 프롬프트 전송 → A/B/C/D 반환
        │
        ▼
  응답: {"answer": "A"}
```
