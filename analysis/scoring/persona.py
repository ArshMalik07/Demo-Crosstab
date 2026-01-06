from typing import Dict, List
import json
import re
from llm.llm_factory import get_llm


def score_personas(corpus: str, personas: List[str]) -> Dict[str, int]:
    if not personas:
        return {}

    llm = get_llm("openai")

    prompt = f"""
You are evaluating how visible each persona is in the following analysis.

---
{corpus[:2500]}
---

Personas to evaluate (IMPORTANT: DO NOT MODIFY THESE NAMES):
{personas}

TASK:
Score ONLY these personas from 40–100 based on how strongly each one appears.

STRICT RULES:
- Do NOT change persona names
- Do NOT create new personas
- Use EXACT strings when returning keys
- Output ONLY JSON

FORMAT:
{{
 "Persona 1": 91,
 "Persona 2": 84
}}
"""

    resp = llm.invoke(prompt)
    text = getattr(resp, "content", str(resp))

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))

        # sort descending
        return dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        return {}