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
                f"D: {self.options.get('D', '')}",
                f"Category: {self.category}",
                f"Correct: {self.answer}",
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
            
    question = " ".join(question_lines).strip()
    return ParsedQuery(question=question, options=options)


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
            timeout=settings.openai_timeout_seconds,
        )
        self._model = settings.embedding_model

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # 각 텍스트 임베딩 벡터 담을 리스트
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self._model,
            )
            vectors.extend(item.embedding for item in response.data)

        return np.asarray(vectors, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self._model,
            input=[text],
        )
        return np.asarray(response.data[0].embedding, dtype=np.float32)

# 인덱스 로딩 + retrieval + LLM 선택 + 폴백
class RAGRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None
        if settings.openai_api_key:
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
            )

        self._index_rows: list[TrainRow] = []
        self._index_matrix: np.ndarray | None = None
    
    def load_index(self) -> None:
        index_dir = Path(self.settings.index_dir)
        matrix_path = index_dir / "embeddings.npy"
        rows_path = index_dir / "rows.json"

        if matrix_path.exists() and rows_path.exists():
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

    # 질문 임베딩과 가장 비슷한 트레이닝 문제 top k
    def _retrieve(self, query_embedding: np.ndarray, top_k: int) -> list[TrainRow]:
        assert self._index_matrix is not None
        matrix = self._index_matrix
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
        m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        #코사인 유사도
        scores = m @ q
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self._index_rows[int(i)] for i in top_indices]

    # 점수까지
    def _retrieve_with_scores(
        self, query_embedding: np.ndarray, top_k: int
    ) -> tuple[list[TrainRow], list[float]]:
        assert self._index_matrix is not None
        matrix = self._index_matrix
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
        m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        scores = m @ q
        top_indices = np.argsort(scores)[-top_k:][::-1]
        rows = [self._index_rows[int(i)] for i in top_indices]
        sims = [float(scores[int(i)]) for i in top_indices]
        return rows, sims

    # LLM에 예시 문제들과 새 문제 같이 넣어서 고르게 함
    def _llm_choose(self, parsed: ParsedQuery, retrieved: list[TrainRow]) -> str:
        assert self._client is not None

        if retrieved:
            context_parts: list[str] = []
            for i, row in enumerate(retrieved, start=1):
                context_parts.append(
                    f"[예시 {i}]\n"
                    f"문제: {row.question}\n"
                    f"A: {row.options['A']}\n"
                    f"B: {row.options['B']}\n"
                    f"C: {row.options['C']}\n"
                    f"D: {row.options['D']}\n"
                    f"정답: {row.answer}"
                )
            context_block = (
                "아래는 유사한 한국 법률 문제 예시입니다:\n\n"
                + "\n\n".join(context_parts)
                + "\n\n===\n\n"
            )
        else:
            context_block = ""

        # 최종 유저 프롬프트
        prompt = (
            f"{context_block}"
            f"다음 문제의 정답을 선택하세요.\n\n"
            f"문제: {parsed.question}\n"
            f"A: {parsed.options.get('A', '')}\n"
            f"B: {parsed.options.get('B', '')}\n"
            f"C: {parsed.options.get('C', '')}\n"
            f"D: {parsed.options.get('D', '')}\n\n"
            f"정답:"
        )

        response = self._client.chat.completions.create(
            model=self.settings.llm_model,
            # 랜덤성 제거
            temperature=0.0,
            # 최대 1토큰
            max_tokens=1,
            messages=[
                {
                    "role": "system",
                    "content": "한국 형사법 및 법률 전문가. A, B, C, D 중 정답 알파벳 하나만 출력.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = (response.choices[0].message.content or "").strip().upper()
        if raw and raw[0] in ("A", "B", "C", "D"):
            return raw[0]

        for letter in ("A", "B", "C", "D"):
            if letter in raw:
                return letter

        return "A"


    #실행 함수
    # 질문, 선택지, 카테고리 받아서 최종 정답 반환
    def infer_mcq(
        self, question: str, options: dict[str, str], category: str = ""
    ) -> str:
        """Infer MCQ answer given question and options dict."""
        parsed = ParsedQuery(question=question, options=options)

        if set(parsed.options.keys()) != {"A", "B", "C", "D"}:
            return self._fallback_predict(parsed)

        if self._client is None:
            return self._fallback_predict(parsed)

        query_text = "\n".join(
            [
                f"Question: {parsed.question}",
                f"A: {parsed.options['A']}",
                f"B: {parsed.options['B']}",
                f"C: {parsed.options['C']}",
                f"D: {parsed.options['D']}",
            ]
        )
        if category:
            query_text += f"\nCategory: {category}"

        query_embedding = np.asarray(
            self._client.embeddings.create(
                model=self.settings.embedding_model,
                input=[query_text],
            )
            .data[0]
            .embedding,
            dtype=np.float32,
        )

        # 유사도 임계값 이상인 경우에만 컨텍스트 포함 (selective RAG)
        if self._index_matrix is not None and self._index_rows:
            retrieved_rows, sims = self._retrieve_with_scores(
                query_embedding, self.settings.top_k
            )
            threshold = self.settings.retrieval_sim_threshold
            high_quality = [
                row
                for row, s in zip(retrieved_rows, sims, strict=False)
                if s >= threshold
            ]
        else:
            high_quality = []

        return self._llm_choose(parsed, high_quality[:3])

    def infer(self, query: str) -> str:
        """Backward compat: infer from formatted query string."""
        parsed = parse_query(query)
        return self.infer_mcq(parsed.question, parsed.options)


# Backward compatibility for modules that still import RagRuntime.
RagRuntime = RAGRuntime