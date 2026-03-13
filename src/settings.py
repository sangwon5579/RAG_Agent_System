from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass(frozen=True)
# 값 변경 불가능. 설정 객체는 읽기 전용
class Settings:
    openai_api_key: str | None
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    top_k: int = 5
    openai_timeout_seconds: float = 30.0
    index_dir: str = "data/index" # 벡터DB
    retrieval_sim_threshold: float = 0.85 # 유사도 임계값

# 설정 객체 생성 후 반환
def load_settings() -> Settings:
    key = os.getenv("OPENAI_API_KEY")
    timeout_raw = os.getenv("OPENAI_TIMEOUT_SECONDS", "8")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 8.0
    return Settings(
        openai_api_key=key,
        openai_timeout_seconds=timeout
    )