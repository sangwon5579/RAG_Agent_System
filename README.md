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
      RAGRuntime
   ├── 1. 쿼리를 임베딩 (text-embedding-3-small)
   ├── 2. train 벡터 인덱스에서 유사 예시 검색 (cosine similarity)
   ├── 3. 유사도 ≥ 0.85인 경우에만 컨텍스트로 활용 (selective RAG)
   └── 4. gpt-4o-mini에 프롬프트 전송 → A/B/C/D 반환
        │
        ▼
  응답: {"answer": "A"}
```

## 실행 방법 (Docker Compose 기준)

`.env` 파일에 `OPENAI_API_KEY=...`를 설정합니다.

### 1) 서버 실행

터미널 A에서 아래 명령으로 서버를 띄웁니다.

```bash
docker compose up --build
```

백그라운드 실행을 원하면:

```bash
docker compose up --build -d
```

### 2) Health check

터미널 B(새 터미널)에서 확인합니다.

```bash
curl http://127.0.0.1:8000/health
```

정상 응답 예시:

```json
{"status":"healthy"}
```

### 3) 단건 추론 확인

터미널 B에서 스모크 테스트를 실행합니다.

```bash
uv run python scripts/smoke_inference.py
```

정상 응답 예시:

```json
{"answer":"A"}
```

### 4) 성능 측정 (dev benchmark)

서버가 켜진 상태에서 터미널 B에서 실행합니다.

```bash
uv run python scripts/benchmark_dev.py
```

출력 예시:

```text
dev_accuracy=0.xxxxxx
correct=...
total=...
```

### 5) 종료

```bash
docker compose down
```

참고:

- Docker 빌드는 `pyproject.toml`과 `uv.lock`을 함께 사용해 의존성을 고정 설치합니다.
- `benchmark_dev.py`는 실행 중인 서버 API(`/inference`)를 호출해 정확도를 계산합니다.

## 1차 실행 결과

- 단건 추론(`/inference`): `answer = A`
- 스모크 테스트(`uv run python scripts/smoke_inference.py`): `{"answer":"A"}`
- dev 벤치마크(`uv run python scripts/benchmark_dev.py`): `dev_accuracy=0.478764`, `correct=124`, `total=259`
