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

## 초기 구축 방법

`.env` 파일에 `OPENAI_API_KEY=...`를 적어두면 추가 환경변수 설정 없이 바로 실행할 수 있습니다.

```bash
make setup
```

`.env`를 사용하지 않는 경우에는 현재 셸에 환경변수를 설정한 뒤 실행합니다.

```bash
# Linux/macOS
export OPENAI_API_KEY=sk-...
make setup        # uv sync + 벡터 인덱스 생성

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
uv sync
uv run python scripts/build_rag_index.py
```

`make setup`은 의존성 설치(`uv sync`)와 `train.csv` 기반 벡터 인덱스 생성을 순서대로 실행합니다.

## Inference 서버 실행

```bash
make run
# 또는
docker-compose up --build
```

기본 주소: `http://127.0.0.1:8000`
