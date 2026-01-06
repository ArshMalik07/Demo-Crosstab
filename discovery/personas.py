# discovery/personas.py

from typing import List
from langchain_core.messages import HumanMessage
from llm.llm_factory import get_discovery_llm
import json, re

def generate_personas(company: str, category: str, num: int = 6) -> List[str]:
    """
    Returns high-level analytical personas (roles only).
    No names, no descriptions.
    """

    prompt = f"""
List {num} professional roles who would analyze or influence decisions
in the following domain.

Company: {company}
Category: {category}

Rules:
- Roles only (2–4 words each)
- No personal names
- No explanations
- Business / tech / strategy focused
- Output ONLY a JSON list

Example:
["Product Manager", "Market Research Analyst"]
"""

    llm = get_discovery_llm()  
    resp = llm.invoke([HumanMessage(content=prompt)])

    raw = resp.content
    if isinstance(raw, list):
        raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))

    match = re.search(r"\[.*\]", str(raw), re.DOTALL)
    return json.loads(match.group(0)) if match else []