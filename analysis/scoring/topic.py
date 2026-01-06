from typing import Dict, List
import json
import re
from llm.llm_factory import get_llm


def score_topics(topics: List[str], corpus: str) -> Dict[str, int]:
    if not corpus.strip():
        return {}

    llm = get_llm("gemini")

    prompt = f"""
You are scoring how relevant each topic is
based on the following analysis text.

---
{corpus[:2500]}
---

Topics to score:
{topics}

TASK:
Give each topic a relevance score from 40–100.

RULES:
- Higher = more discussed or implied
- NEVER create new topics
- Output ONLY JSON (topic: score)

FORMAT:
{{
 "Topic A": 80,
 "Topic B": 60
}}
"""

    resp = llm.invoke(prompt)
    text = getattr(resp, "content", str(resp))

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}