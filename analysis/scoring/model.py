# analysis/scoring/model.py
from typing import Dict, List

def score_models(models: List[str]) -> Dict[str, int]:
    if not models:
        return {}

    share = round(100 / len(models))
    return {m.upper(): share for m in models}