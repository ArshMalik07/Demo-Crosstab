from llm.llm_factory import get_llm
from analysis.report import generate_report

SYSTEM_ANALYSIS = """
You are a senior market and competitive intelligence analyst.
Your task is to reason deeply, write clearly, and avoid generic filler.
Always structure insights, quantify when possible, and avoid hallucinating data.
"""


def run_analysis(payload: dict) -> dict:
    """
    Uses already generated data.
    No discovery logic here.
    """
    return payload
