from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import anthropic
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
# Usage limits per plan
# ---------------------------------------------------------------------------

PLAN_LIMITS = {
    "pro": 50,
    "agency": 200,
}

# In-memory usage counter for MVP.
# Format: {"va_xxxx": {"count": 12, "month": "2026-05"}}
# Resets automatically when the month changes.
_usage: dict = {}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Signal(BaseModel):
    label: str
    passing: bool = False
    tip: str


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

def get_licenses() -> dict:
    licenses_json = os.environ.get("VALID_LICENSES", "{}")
    try:
        return json.loads(licenses_json)
    except json.JSONDecodeError:
        return {}


def validate_license(license_key: str) -> dict | None:
    licenses = get_licenses()
    return licenses.get(license_key)


def get_usage(license_key: str) -> dict:
    current_month = datetime.utcnow().strftime("%Y-%m")
    entry = _usage.get(license_key)
    if not entry or entry.get("month") != current_month:
        return {"count": 0, "month": current_month}
    return entry


def increment_usage(license_key: str) -> int:
    current_month = datetime.utcnow().strftime("%Y-%m")
    entry = get_usage(license_key)
    entry["count"] += 1
    entry["month"] = current_month
    _usage[license_key] = entry
    return entry["count"]


def check_usage_limit(license_key: str, plan: str) -> tuple[int, int]:
    """Returns (current_count, limit). Raises 429 if over limit."""
    limit = PLAN_LIMITS.get(plan, 50)
    usage = get_usage(license_key)
    count = usage["count"]
    if count >= limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_limit_reached",
                "message": f"You've used all {limit} AI analyses for this month. Resets on the 1st.",
                "used": count,
                "limit": limit,
                "plan": plan,
            }
        )
    return count, limit


# ---------------------------------------------------------------------------
# Claude prompt
# ---------------------------------------------------------------------------

ANALYZE_SYSTEM_PROMPT = """You are an expert in AEO (Answer Engine Optimization) — optimizing content to be cited by AI answer engines like ChatGPT, Perplexity, and Google AI Overviews.

You analyze WordPress posts and provide specific, copy-pasteable improvements for each failing signal. Be concrete, not generic.

CRITICAL FORMATTING RULES for the "suggestion" field:
- Return PLAIN TEXT or simple HTML only (e.g. <h2>, <p>, <strong>)
- NEVER include WordPress block editor comments like <!-- wp:heading --> or <!-- wp:paragraph -->
- The user will paste your suggestion into the WordPress visual editor — write it as human-readable text
- For headings: write just the heading text (e.g. "What Is AI Analysis?") or "<h2>What Is AI Analysis?</h2>"
- For paragraphs: write just the paragraph text
- For FAQ schema: write plain Q&A pairs, not JSON-LD markup

Always respond with valid JSON matching the schema provided."""


def strip_block_comments(content: str) -> str:
    """Remove Gutenberg block editor comments from post content, leaving clean HTML."""
    import re
    # Remove <!-- wp:xxx {...} --> and <!-- /wp:xxx --> style comments
    clean = re.sub(r'<!--\s*/?wp:[^\-]*?-->', '', content)
    # Collapse extra blank lines
    clean = re.sub(r'\n{3,}', '\n\n', clean).strip()
    return clean


def build_analyze_prompt(content: str, signals: list[Signal], post_url: str = None, gsc_queries: list[str] = None) -> str:
    failing = [s for s in signals if not s.passing]
    if not failing:
        return None

    signal_list = "\n".join([f"- {s.label}: {s.tip}" for s in failing])

    gsc_section = ""
    if gsc_queries:
        gsc_section = "\nTop search queries this post ranks for:\n" + "\n".join([f"- {q}" for q in gsc_queries[:10]])

    url_line = f"\nPost URL: {post_url}" if post_url else ""

    # Strip block editor markup so Claude sees clean HTML only
    clean_content = strip_block_comments(content)

    return f"""Analyze this WordPress post and provide specific improvements for each failing signal.
{url_line}

Failing signals:
{signal_list}
{gsc_section}

Post content:
{clean_content[:8000]}

For each failing signal, return a JSON object with:
- "signal": the signal label exactly as provided
- "suggestion": plain text or simple HTML the user can copy-paste into their WordPress post (NO block editor markup). For FAQ signals, format as separate lines: "Q: ...\nA: ...\n\nQ: ...\nA: ..." — not a single run-on sentence.
- "why": one sentence explaining why this improves AI citability

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

    plan = license_data.get("plan", "pro")
    usage = get_usage(req.license_key)
    limit = PLAN_LIMITS.get(plan, 50)

    return {
        "valid": True,
        "plan": plan,
        "email": license_data.get("email", ""),
        "sites_allowed": 1 if plan == "pro" else 10,
        "usage": {
            "used": usage["count"],
            "limit": limit,
            "resets": f"{datetime.utcnow().strftime('%Y-%m')}-01",
        }
    }


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest):
    # Validate license
    license_data = validate_license(req.license_key)
    if not license_data:
        raise HTTPException(status_code=401, detail="Invalid license key")

    plan = license_data.get("plan", "pro")

    # Check monthly usage limit
    used, limit = check_usage_limit(req.license_key, plan)

    # Check Claude API key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise HTTPException(status_code=500, detail="AI service not configured")

    # Only analyze failing signals
    failing = [s for s in req.signals if not s.passing]
    if not failing:
        return {
            "suggestions": [],
            "message": "All signals are passing — nothing to improve!",
            "usage": {"used": used, "limit": limit},
        }

    # Build prompt
    prompt = build_analyze_prompt(req.content, req.signals, req.post_url, req.gsc_queries)

    # Call Claude
    try:
        client = anthropic.Anthropic(api_key=anthropic_key)

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            system=ANALYZE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text

        # Strip markdown code fences if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        # Increment usage only after successful Claude call
        new_count = increment_usage(req.license_key)

        result["usage"] = {"used": new_count, "limit": limit}
        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")
