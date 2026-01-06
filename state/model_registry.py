# """
# Stores user-selected LLMs for ANALYSIS phase.
# Discovery phase does NOT use this.
# """

# from typing import List

# # Example: ["openai", "gemini"]
# _SELECTED_MODELS: List[str] = []

# def set_models(models: List[str]):
#     """
#     Called by /select-model API
#     """
#     global _SELECTED_MODELS

#     if not models:
#         raise ValueError("At least one model must be selected")

#     cleaned = [m.lower().strip() for m in models]

#     for m in cleaned:
#         if m not in ("openai", "gemini"):
#             raise ValueError(f"Unsupported model: {m}")

#     _SELECTED_MODELS = cleaned

# def get_models() -> List[str]:
#     """
#     Used by analysis layer
#     """
#     if not _SELECTED_MODELS:
#         # default fallback
#         return ["gemini"]

#     return _SELECTED_MODELS

# state/model_registry.py

_active_model = None


def set_active_model(model_name: str):
    """
    Sets the active LLM model globally.
    Example: 'openai', 'gemini'
    """
    global _active_model
    _active_model = model_name


def get_active_model() -> str:
    """
    Returns currently active model.
    """
    return _active_model
