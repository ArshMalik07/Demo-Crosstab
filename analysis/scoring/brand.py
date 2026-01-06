from typing import Dict
import json
import re
from llm.llm_factory import get_llm


def score_brands(company: str, category: str, corpus: str) -> Dict:
    llm = get_llm("gemini")

    prompt = f"""
You are an expert market analyst.

TASK:
1️⃣ Identify 6 brands relevant to category: "{category}"
2️⃣ ALWAYS include this brand: "{company}"
3️⃣ Score each brand from 40 to 100 based on visibility in this context:

---
{corpus[:2500]}
---

RULES:
- Output ONLY JSON
- No explanations
- Sort from highest to lowest score

FORMAT:
{{
 "brand_visibility": {{"{company}": 78}},
 "brand_mentions": {{
   "BrandA": 92,
   "BrandB": 86,
   "BrandC": 80,
   "BrandD": 74,
   "BrandE": 69
 }}
}}
"""

    response = llm.invoke(prompt)
    text = getattr(response, "content", str(response))

    # try to extract JSON safely
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {"brand_visibility": {}, "brand_mentions": {}}

    try:
        data = json.loads(match.group(0))
        return data
    except Exception:
        return {"brand_visibility": {}, "brand_mentions": {}}