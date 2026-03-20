from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    query: str

# A,B,C,D 강제
class AgentResponse(BaseModel):
    answer: Literal["A", "B", "C", "D"]
