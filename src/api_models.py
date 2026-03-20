from typing import Literal
from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    query: str = Field(
    ...,
    description="The multiple-choice legal question to be analyzed and answered",
)
    
class AgentResponse(BaseModel):
    answer: Literal["A", "B", "C", "D"] = Field(
    ..., description="The letter of the correct option"
)