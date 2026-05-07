from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import anthropic
import hashlib
import hmac
import json
from datetime import datetime

app = FastAPI(title="Visibly AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Signal(BaseModel):
    model_config = {"populate_by_name": True}

    label: str
    pass_: bool = False
    tip: str

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict) and "pass" in obj:
            obj = {**obj, "pass_": obj.pop("pass")}
        return super().model_validate(obj, *args, **kwargs)


class AnalyzeRequest(BaseModel):
    license_key: str
    content: str
    signals: list[Signal]
    post_url: Optional[str] = None
    gsc_queries: Optional[list[str]] = None


class LicenseValidateRequest(BaseModel):
    license_key: str
    site_url: Optional[str] = None


# ---------------------------------------------------------------------------
# License helpers
# ---------------------------------------------------------------------------

VALID_LICENSES: dict = {}  # In-memory store for MVP — replace with Postgres in Phase 2


def validate_license(license_key: str) -> dict:
    """
    MVP: license keys are stored in the VALID_LICENSES env var as JSON.
    Format: {"va_xxxx": {"plan": "pro", "email": "user@example.com"}}
    """
    licenses_json = os.environ.get("VALID_LICENSES", "{}")
    try:
        licenses = json.loads(licenses_json)
    except json.JSONDecodeError:
        licenses = {}

    if license_key in licenses:
        return licenses[license_key]
    return None


# ---------------------------------------------------------------------------
# Claude prompt
# ---------------------------------------------------------------------------

ANALYZE_SYSTEM_PROMPT = """You are an expert in AEO (Answer Engine Optimization) — optimizing content to be cited by AI answer engines like ChatGPT, Perplexity, and Google AI Overviews.

You analyze WordPress posts and provide specific, copy-pasteable improvements for each failing signal. Be concrete, not generic. The user needs text they can paste directly into their post.

Always respond with valid JSON matching the schema provided."""


def build_analyze_prompt(content: str, signals: list[Signal], post_url: str = None, gsc_queries: list[str] = None) -> str:
    failing = [s for s in signals if not s.pass_]
    signal_list = "\n".join([f"- {s.label}: {s.tip}" for s in failing])

    gsc_section = ""
    if gsc_queries:
        gsc_section = f"\nTop search queries this post ranks for:\n" + "\n".join([f"- {q}" for q in gsc_queries[:10]])

    url_line = f"\nPost URL: {post_url}" if post_url else ""

    return f"""Analyze this WordPress post and provide specific improvements for each failing signal.
{url_line}

Failing signals:
{signal_list}
{gsc_section}

Post content (HTML):
{content[:8000]}

For each failing signal, return a JSON object with:
- "signal": the signal label exactly as provided
- "suggestion": specific text to add or change (copy-pasteable, ready to paste into the post)
- "why": one sentence explaining why this improves AI citability or search ranking

Return a JSON object: {{"suggestions": [...]}}"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "service": "Visibly AI API", "version": "1.0.0"}


@app.post("/v1/license/validate")
def license_validate(req: LicenseValidateRequest):
    license_data = validate_license(req.license_key)
    if not license_data:
        raise HTTPException(status_code=401, detail="Invalid license key")

    return {
        "valid": True,
        "plan": license_data.get("plan", "pro"),
        "email": license_data.get("email", ""),
        "sites_allowed": 1 if license_data.get("plan") == "pro" else 10,
    }


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest):
    # Validate license
    license_data = validate_license(req.license_key)
    if not license_data:
        raise HTTPException(status_code=401, detail="Invalid license key")

    # Check Claude API key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise HTTPException(status_code=500, detail="AI service not configured")

    # Only analyze failing signals
    failing = [s for s in req.signals if not s.pass_]
    if not failing:
        return {"suggestions": [], "message": "All signals are passing — nothing to improve!"}

    # Call Claude
    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        prompt = build_analyze_prompt(req.content, req.signals, req.post_url, req.gsc_queries)

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            system=ANALYZE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text

        # Parse JSON from response
        # Claude may wrap in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")
