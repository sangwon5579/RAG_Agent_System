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

