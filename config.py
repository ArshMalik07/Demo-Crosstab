import os
from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")