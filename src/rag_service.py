from __future__ import annotations
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from openai import OpenAI
from src.settings import Settings

# 문제/학습 데이터를 담는 자료구조
# 입력 문제를 파싱하고 csv 학습데이터를 읽는 부분
# OpenAI 임베딩 생성기
# RAG 기반으로 정답을 고르는 추론기 RagRuntime

# 학습용 객관식 문제들을 읽음
# 각 문제를 임베딩해서 인덱스 만듬
# 새문제 들어오면 그 문제도 임베딩
# 기존 문제 중 비슷한 문제 몇개 찾음
# 그 예시들을 LLM에 같이 넣어서 abcd중 고르게함
# AI 키 없으면 간단한 fallback으로 작동
# 비슷한 기출 찾아서 문제 푸는 시스템

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
OPTION_PATTERN = re.compile(r"(?im)^\s*([ABCD])[\)\.:]?\s*(.+)$") #객관식 보기 한 줄 찾는 정규식

@dataclass(frozen=True)
class ParsedQuery:
    question: str
    options: dict[str, str]

@dataclass(frozen=True)
class TrainRow:
    question: str
    options: dict[str, str]
    answer: str
    category: str

    # 한 문제를 검색용 텍스트로 변환
    def retrieval_text(self) -> str:
        return "\n".join(
            [
                f"Question: {self.question}",
                f"A: {self.options.get('A', '')}",
                f"B: {self.options.get('B', '')}", 
                f"C: {self.options.get('C', '')}",
                f"D: {self.options.get('D', '')}"
                f"Category: {self.category}",
                f"Correct: {self.answer}"
            ]
        )
    

def parse_query(raw: str) -> ParsedQuery:
    options: dict[str, str] = {}

    for match in OPTION_PATTERN.finditer(raw):
        options[match.group(1)] = match.group(2).strip()

    question_lines: list[str] = []
    for line in raw.splitlines():
        if OPTION_PATTERN.match(line):
            continue
        if line.strip():
            question_lines.append(line.strip())
            
    question = " ".join(question_lines).strip
    return ParsedQuery(question=question, options=options) # type: ignore


# csv 읽어서 TrainRow 객체 리스트로 변환
def load_train_rows(train_path: Path) -> list[TrainRow]:
    rows: list[TrainRow] = []
    with train_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer_raw = str(row.get("answer", "1")).strip()
            answer_num = min(max(int(answer_raw), 1), 4)
            answer = IDX_TO_LETTER[answer_num - 1]
            rows.append(
                TrainRow(
                    question=str(row.get("question", "")).strip(),
                    options={
                        "A": str(row.get("A", "")).strip(),
                        "B": str(row.get("B", "")).strip(),
                        "C": str(row.get("C", "")).strip(),
                        "D": str(row.get("D", "")).strip(),
                    },
                    answer=answer,
                    category=str(row.get("Category", "")).strip(),
                )
            )
    return rows

class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required for embedding")
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timemout=settings.openai_timeout_seconds
        )
        self._model = settings.embedding_model
    def embed_texts(self, texts: list[str], batch_size: int=64) -> np.ndarray:
        # 각 텍스트 임베딩 벡터 담을 리스트
        vectors: list[list[float]] = []
        
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self._model
            )
            return np.asarray(vectors, dtype=np.float32)
    def embed_text(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self._model,
            input=[text]
            return np.asarray(response.data[0].embedding, dtype=np.float32)

# 인덱스 로딩 + retrieval + LLM 선택 + 폴백
class RagRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = Settings
        self._client = OpenAI | None = None
        if settings.openai_api_key:
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds
            )
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds
            )
            self._index_rows: list[TrainRow] = []
            self._index_matrix: np.ndarray | None = None
    
    def load_index(self) -> None:
        index_dir = Path(self.settings.index_dir)
        matrix_path = index_dir / "embeddings.npy"
        rows_path = index_dir / "rows.json"

        if not matrix_path.exists() or not rows_path.exists():
            self._index_matrix = np.load(matrix_path)
            raw_rows = json.loads(rows_path.read_text(encoding="utf-8"))
                        self._index_rows = [
                TrainRow(
                    question=row["question"],
                    options=row["options"],
                    answer=row["answer"],
                    category=row["category"],
                )
                for row in raw_rows
            ]
            return 
        train_rows = load_train_rows(Path("data/train.csv"))
        self._index_rows = train_rows
        self._index_matrix = None

    # fallback
    def _fallback_predict(self, query: ParsedQuery) -> str:
        if not self._index_rows:
            return "A"
        votes = [0, 0, 0, 0]
        q_tokens = set(query.question.lower().split())
        for row in self._index_rows[:160]:
            overlap = len(q_tokens.intersection(set(row.question.lower().split())))
            votes[LETTER_TO_IDX[row.answer]] += overlap
        best_idx = int(np.argmax(np.asarray(votes, dtype=np.int64)))
        return IDX_TO_LETTER[best_idx]
