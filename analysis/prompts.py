from typing import List
import json
import re
from langchain_core.messages import HumanMessage

SYSTEM_PROMPTS = """
You generate strategic analytical prompts — not answers.
Each prompt must be a single powerful question.
Avoid brand names, avoid filler, avoid generic fluff.
"""

def generate_prompts(product: str, persona: str, topic: str, num: int, llm) -> List[str]:

    if isinstance(persona, list):
        persona = persona[0]

    prompt = f"""
{SYSTEM_PROMPTS}

Product: {product}
Topic: {topic}
Persona: {persona}

Return exactly {num} prompts in JSON list format ONLY.
Example: ["Prompt 1", "Prompt 2"]
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    raw = str(resp.content).strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    return json.loads(match.group(0)) if match else []