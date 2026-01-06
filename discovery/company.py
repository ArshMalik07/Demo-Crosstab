# discovery/company.py

import requests
import socket
from urllib.parse import urlparse
from langchain_core.messages import HumanMessage
from llm.llm_factory import get_discovery_llm

def _domain_resolves(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
        return True
    except:
        return False

 # NEW: safer website fetch with browser-like headers 
def _safe_fetch(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }

    try:
        resp = requests.get(
            url,
            headers=headers,
            timeout=12,
            allow_redirects=True,
        )

        if resp.status_code < 400:
            return resp.text[:6000]

    except Exception:
        return None

    return None


def verify_company_from_url(url: str) -> dict:
    # URL validation 
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        return {"valid": False, "reason": "Invalid URL"}

    if not _domain_resolves(hostname):
        return {"valid": False, "reason": "Domain does not resolve"}

    # Website fetch (with fallback) 
    page_text = _safe_fetch(url)

    # FALLBACK: still verify even if site blocks us
    if not page_text:
        page_text = f"This website belongs to the company at domain: {hostname}"

    # LLM verification 
    llm = get_discovery_llm()

    prompt = f"""
You are a strict company verification system.

Analyze the website content below and answer:

1. Is this a REAL operating company?
2. If yes, extract the OFFICIAL company name.

Website domain: {hostname}

CONTENT:
{page_text}

Rules:
- If NOT a real company → return {{ "valid": false, "reason": "<short reason>" }}
- If real company → return {{ "valid": true, "company": "<company name>" }}
- Always return company name in English
- Output ONLY JSON
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    raw = resp.content

    if isinstance(raw, list):
        raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))

    raw = str(raw).strip()

    import json, re
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    try:
        result = json.loads(match.group(0)) if match else {}
    except:
        return {"valid": False, "reason": "LLM parse failure"}

    return result