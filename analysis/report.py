from typing import Dict, List
from collections import defaultdict
from llm.llm_factory import get_llm


# ---------------- Helpers ----------------

def normalize(text):
    if not text:
        return ""
    if isinstance(text, list):
        text = " ".join(map(str, text))
    return str(text).lower()


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


# ---------------- Core Analysis ----------------

def analyze_answers(
    answers: List[str],
    brand: str,
    competitors: List[str],
    personas: List[str],
    topics: List[str]
) -> Dict:

    total_answers = len(answers) or 1

    # ---- Brand setup ----
    all_brands = [brand] + competitors
    brand_counts = {b: 0 for b in all_brands}
    brand_hit_answers = 0

    # ---- Persona / Topic counts ----
    persona_counts = {p: 0 for p in personas}
    topic_counts = {t: 0 for t in topics}

    for ans in answers:
        text = normalize(ans)

        brand_found = False

        # ---- Brand mentions ----
        for b in all_brands:
            bl = b.lower()
            if bl in text:
                count = text.count(bl)
                brand_counts[b] += count
                brand_found = True

        if brand_found:
            brand_hit_answers += 1

        # ---- Persona visibility (heuristic) ----
        for p in personas:
            persona_keywords = [
                p.lower(),
                p.split()[0].lower(),
                "strategy",
                "roadmap",
                "architecture",
                "planning",
                "analysis",
                "decision"
            ]
            if contains_any(text, persona_keywords):
                persona_counts[p] += 1

        # ---- Topic visibility (keyword based) ----
        for t in topics:
            topic_keywords = t.lower().split()[:4]
            if contains_any(text, topic_keywords):
                topic_counts[t] += 1

    return {
        "brand_visibility": int((brand_hit_answers / total_answers) * 100),

        "brand_mentions": dict(
            sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        ),

        "persona_visibility": {
            p: int((c / total_answers) * 100)
            for p, c in persona_counts.items()
            if c > 0
        },

        "topic_visibility": {
            t: int((c / total_answers) * 100)
            for t, c in topic_counts.items()
            if c > 0
        }
    }


# ---------------- Main Report Generator ----------------

def generate_report(payload: Dict) -> Dict:
    """
    Expected payload:
    {
      brand: str,
      competitors: [str],
      personas: [str],
      topics: [str],
      prompts: [{model, prompt}] OR [str],
      models: [str]
    }
    """

    brand = payload["brand"]
    competitors = payload.get("competitors", [])
    personas = payload.get("personas", [])
    topics = payload.get("topics", [])
    prompts = payload.get("prompts", [])
    models = payload.get("models", [])

    # ---- Group prompts by model ----
    prompts_by_model = defaultdict(list)

    for p in prompts:
        # Case 1: model-tagged prompt
        if isinstance(p, dict):
            prompts_by_model[p["model"]].append(p["prompt"])
        else:
            # Case 2: plain prompt → send to ALL models
            for m in models:
                prompts_by_model[m].append(p)

    # ---- Generate answers ----
    answers_by_model = defaultdict(list)

    for model in models:
        llm = get_llm(model)

        for prompt in prompts_by_model.get(model, []):
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            answers_by_model[model].append(content)

    # ---- Per-model analysis ----
    per_model = {}
    combined_answers = []

    for model, answers in answers_by_model.items():
        per_model[model] = analyze_answers(
            answers=answers,
            brand=brand,
            competitors=competitors,
            personas=personas,
            topics=topics
        )
        combined_answers.extend(answers)

    # ---- Combined analysis ----
    combined = analyze_answers(
        answers=combined_answers,
        brand=brand,
        competitors=competitors,
        personas=personas,
        topics=topics
    )

    return {
        "per_model": per_model,
        "combined": combined
    }
