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

`.env` 파일에 `OPENAI_API_KEY=...`를 설정합니다.

### 로컬 실행 (권장)

#### Linux/macOS

```bash
make setup
make run
```

#### Windows PowerShell

Windows에서는 `make` 대신 `mingw32-make`를 사용합니다.

```powershell
mingw32-make setup
mingw32-make run
```

만약 `mingw32-make`가 없다면 아래 명령으로 동일하게 실행할 수 있습니다.

```powershell
uv sync
uv run python scripts/build_rag_index.py
uv run uvicorn src.server:app --host 0.0.0.0 --port 8000
```

`setup`은 의존성 설치(`uv sync`)와 `train.csv` 기반 벡터 인덱스 생성을 순서대로 수행합니다.

### API 확인

서버 실행 후 아래를 확인합니다.

- Health: `GET /health`
- Inference: `POST /inference`

스모크 테스트:

```powershell
uv run python scripts/smoke_inference.py
```

기본 주소: `http://127.0.0.1:8000`

## Docker 실행

`.env` 파일에 `OPENAI_API_KEY=...`를 설정한 뒤 실행합니다.

```bash
docker compose up --build
```
