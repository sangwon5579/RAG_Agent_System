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
    top_k: int = 10
    openai_timeout_seconds: float = 30.0
    index_dir: str = "data/index" # 벡터DB
    retrieval_sim_threshold: float = 0.75 # 유사도 임계값
    hybrid_override_top1_sim: float = 0.86
    hybrid_override_margin: float = 0.08

# 설정 객체 생성 후 반환
def load_settings() -> Settings:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    timeout_raw = os.getenv("OPENAI_TIMEOUT_SECONDS", "8")
    top1_sim_raw = os.getenv("HYBRID_OVERRIDE_TOP1_SIM", "0.86")
    margin_raw = os.getenv("HYBRID_OVERRIDE_MARGIN", "0.08")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 8.0

    try:
        top1_sim = float(top1_sim_raw)
    except ValueError:
        top1_sim = 0.86

    try:
        margin = float(margin_raw)
    except ValueError:
        margin = 0.08

    return Settings(
        openai_api_key=key,
        openai_timeout_seconds=timeout,
        hybrid_override_top1_sim=top1_sim,
        hybrid_override_margin=margin,
    )