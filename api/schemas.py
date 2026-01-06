from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional

#  COMPANY VERIFICATION
class CompanyVerifyRequest(BaseModel):
    url: HttpUrl

class CompanyVerifyResponse(BaseModel):
    valid: bool
    company: Optional[str] = None
    reason: Optional[str] = None

# BASE (VERIFIED COMPANY)
class CompanyBase(BaseModel):
    company: str = Field(..., description="Verified company name")

#  DISCOVERY
class ProductRequest(CompanyBase):
    pass

class PersonaRequest(CompanyBase):
    product: str

class TopicRequest(CompanyBase):
    product: str
    persona: str

#  ANALYSIS (MULTI-LLM)
class AnalysisRequest(BaseModel):
    product: str
    persona: str
    topic: str

    models: List[str] = Field(
        ...,
        description="LLMs to use, e.g. ['openai', 'gemini']"
    )

    num_prompts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Prompts per model"
    )



class ReportRequest(BaseModel):
    brand: str
    product: str
    personas: List[str]
    topics: List[str]
    prompts: List[str]
    models: List[str]   