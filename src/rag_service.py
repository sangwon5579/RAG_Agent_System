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
