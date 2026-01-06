import json, re
from langchain_core.messages import HumanMessage
from llm.llm_factory import get_discovery_llm
from llm.response_utils import extract_text

def extract_products(company: str):
    llm = get_discovery_llm()

    prompt = f"""
Identify real product categories for company "{company}".
Return ONLY a JSON list.
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    text = extract_text(resp)

    match = re.search(r"\[.*\]", text, re.DOTALL)
    return json.loads(match.group(0)) if match else []