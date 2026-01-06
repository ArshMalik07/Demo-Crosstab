# discovery/topics.py
from typing import List
from langchain_core.messages import HumanMessage
from llm.llm_factory import get_discovery_llm
import json
import re

def generate_topics(company: str, prompt: str, persona: str, num: int = 6) -> List[str]:
    """
    Generates HIGH-LEVEL discovery topics.
    These are dashboard-style themes, NOT analysis questions.
    """

    prompt = f"""
Generate exactly {num} high-level topic labels
related to the following domain.

Company: {company}
Prompt: {prompt}
persona: {persona}

Rules:
- Topics must be SHORT (3–8 words)
- NO analysis words (no analyze, assess, evaluate, compare)
- NO questions
- NO long sentences
- NO brand names inside topics
- Think: dashboard / report section titles
- Output ONLY a JSON list

Examples:
[
  "5G Adoption in Smartphones",
  "Foldable Phone Market Trends",
  "Smartphone Camera Innovations"
]
"""

    llm = get_discovery_llm() 
    resp = llm.invoke([HumanMessage(content=prompt)])

    raw = resp.content
    if isinstance(raw, list):
        raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))

    match = re.search(r"\[.*\]", str(raw), re.DOTALL)
    return json.loads(match.group(0)) if match else []