from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os
import re
import json
import secrets
import anthropic
from datetime import datetime

app = FastAPI(title="Visibly AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PLAN_LIMITS = {"pro": 50, "agency": 200}

DATABASE_URL = os.environ.get("DATABASE_URL")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
EMAIL_FROM = "hello@getvisiblyai.com"

# Map Stripe price IDs → plan (set these env vars in Railway after creating Stripe products)
PRICE_TO_PLAN: dict[str, str] = {
    k: v for k, v in [
        (os.environ.get("STRIPE_PRO_MONTHLY_PRICE_ID", ""), "pro"),
        (os.environ.get("STRIPE_PRO_ANNUAL_PRICE_ID", ""), "pro"),
        (os.environ.get("STRIPE_AGENCY_MONTHLY_PRICE_ID", ""), "agency"),
        (os.environ.get("STRIPE_AGENCY_ANNUAL_PRICE_ID", ""), "agency"),
    ] if k
}

# In-memory usage fallback — only used when DATABASE_URL is not set
_usage_fallback: dict = {}


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------

def _get_db():
    """Open a new Postgres connection. Returns None if DATABASE_URL is not set."""
    if not DATABASE_URL:
        return None
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def _init_db():
    conn = _get_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS licenses (
                    key TEXT PRIMARY KEY,
                    plan TEXT NOT NULL,
                    email TEXT NOT NULL,
                    stripe_customer_id TEXT,
                    stripe_session_id TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    license_key TEXT NOT NULL,
                    month TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (license_key, month)
                )
            """)
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def startup():
    _init_db()


# ---------------------------------------------------------------------------
# License helpers
# ---------------------------------------------------------------------------

def _env_licenses() -> dict:
    """Read VALID_LICENSES env var — used for test keys and local dev."""
    try:
        return json.loads(os.environ.get("VALID_LICENSES", "{}"))
    except json.JSONDecodeError:
        return {}


def validate_license(license_key: str) -> dict | None:
    conn = _get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT plan, email FROM licenses WHERE key = %s", (license_key,))
                row = cur.fetchone()
                if row:
                    return {"plan": row[0], "email": row[1]}
        finally:
            conn.close()
    return _env_licenses().get(license_key)


def _insert_license(key: str, plan: str, email: str, customer_id: str = None, session_id: str = None):
    conn = _get_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO licenses (key, plan, email, stripe_customer_id, stripe_session_id)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (key) DO NOTHING""",
                (key, plan, email, customer_id, session_id),
            )
        conn.commit()
    finally:
        conn.close()


def _lookup_license_by_session(session_id: str) -> tuple[str, str] | None:
    """Returns (key, plan) or None."""
    conn = _get_db()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key, plan FROM licenses WHERE stripe_session_id = %s", (session_id,))
            return cur.fetchone()
    finally:
        conn.close()


def _generate_key(plan: str) -> str:
    return f"va_{plan}_{secrets.token_hex(8)}"


# ---------------------------------------------------------------------------
# Usage helpers
# ---------------------------------------------------------------------------

def get_usage(license_key: str) -> dict:
    current_month = datetime.utcnow().strftime("%Y-%m")
    conn = _get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count FROM usage WHERE license_key = %s AND month = %s",
                    (license_key, current_month),
                )
                row = cur.fetchone()
                return {"count": row[0] if row else 0, "month": current_month}
        finally:
            conn.close()
    entry = _usage_fallback.get(license_key)
    if not entry or entry.get("month") != current_month:
        return {"count": 0, "month": current_month}
    return entry


def increment_usage(license_key: str) -> int:
    current_month = datetime.utcnow().strftime("%Y-%m")
    conn = _get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO usage (license_key, month, count) VALUES (%s, %s, 1)
                       ON CONFLICT (license_key, month) DO UPDATE SET count = usage.count + 1
                       RETURNING count""",
                    (license_key, current_month),
                )
                row = cur.fetchone()
                count = row[0] if row else 1
            conn.commit()
            return count
        finally:
            conn.close()
    entry = get_usage(license_key)
    entry["count"] += 1
    entry["month"] = current_month
    _usage_fallback[license_key] = entry
    return entry["count"]


def check_usage_limit(license_key: str, plan: str) -> tuple[int, int]:
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
            },
        )
    return count, limit


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def _send_license_email(email: str, license_key: str, plan: str):
    if not RESEND_API_KEY:
        return
    import resend
    resend.api_key = RESEND_API_KEY
    plan_label = "Pro" if plan == "pro" else "Agency"
    limit = PLAN_LIMITS.get(plan, 50)
    sites = "1 WordPress site" if plan == "pro" else "up to 10 sites"
    resend.Emails.send({
        "from": f"Visibly AI <{EMAIL_FROM}>",
        "to": [email],
        "subject": f"Your Visibly AI {plan_label} License Key",
        "html": f"""
<div style="font-family:sans-serif;max-width:540px;margin:0 auto;padding:32px 24px;color:#1f2937">
  <h2 style="color:#7c3aed;margin:0 0 8px">Welcome to Visibly AI {plan_label}!</h2>
  <p style="margin:0 0 24px;color:#374151">Here's your license key — paste it into the
  <strong>Pro &amp; License</strong> settings page inside your WordPress plugin.</p>

  <div style="background:#f5f3ff;border:1px solid #ddd6fe;border-radius:8px;padding:20px 24px;margin-bottom:24px">
    <p style="font-size:11px;color:#7c3aed;margin:0 0 6px;text-transform:uppercase;letter-spacing:.06em">Your License Key</p>
    <code style="font-size:17px;font-weight:700;color:#1e1b4b;letter-spacing:.03em;word-break:break-all">{license_key}</code>
  </div>

  <p style="margin:0 0 8px;font-weight:600">What's included:</p>
  <ul style="margin:0 0 24px;padding-left:20px;color:#374151;line-height:1.7">
    <li>{limit} AI analyses per month</li>
    <li>Works on {sites}</li>
    <li>Priority support — just reply to this email</li>
  </ul>

  <p style="color:#6b7280;font-size:14px;margin:0 0 4px">Questions? Reply here or email <a href="mailto:{EMAIL_FROM}" style="color:#7c3aed">{EMAIL_FROM}</a></p>
  <p style="color:#6b7280;font-size:14px;margin:0">— The Visibly AI team</p>
</div>
""",
    })


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
    clean = re.sub(r'<!--\s*/?wp:[^\-]*?-->', '', content)
    clean = re.sub(r'\n{3,}', '\n\n', clean).strip()
    return clean


def build_analyze_prompt(content: str, signals: list[Signal], post_url: str = None, gsc_queries: list[str] = None) -> str | None:
    failing = [s for s in signals if not s.passing]
    if not failing:
        return None

    signal_list = "\n".join([f"- {s.label}: {s.tip}" for s in failing])

    gsc_section = ""
    if gsc_queries:
        gsc_section = "\nTop search queries this post ranks for:\n" + "\n".join([f"- {q}" for q in gsc_queries[:10]])

    url_line = f"\nPost URL: {post_url}" if post_url else ""
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
- "suggestion": plain text or simple HTML the user can copy-paste into their WordPress post (NO block editor markup). For FAQ signals, format as separate lines: "Q: ...\\nA: ...\\n\\nQ: ...\\nA: ..." — not a single run-on sentence.
- "why": one sentence explaining why this improves AI citability

Return a JSON object: {{"suggestions": [...]}}"""


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_PAGE_STYLE = """
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f3ff;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}
  .card{background:#fff;border-radius:16px;padding:40px 36px;max-width:480px;width:100%;box-shadow:0 4px 24px rgba(124,58,237,.12)}
  .logo{font-size:20px;font-weight:700;color:#7c3aed;margin-bottom:28px}
  h1{font-size:22px;font-weight:700;color:#1e1b4b;margin-bottom:8px}
  p{color:#4b5563;line-height:1.6;margin-bottom:16px}
  .key-box{background:#f5f3ff;border:1.5px solid #ddd6fe;border-radius:10px;padding:18px 20px;margin:20px 0 24px}
  .key-label{font-size:11px;color:#7c3aed;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px}
  .key-value{font-family:'Courier New',monospace;font-size:16px;font-weight:700;color:#1e1b4b;word-break:break-all}
  .note{font-size:13px;color:#6b7280;margin-bottom:0}
  .check{color:#059669;font-size:18px;margin-right:6px}
</style>
"""


def _success_html(license_key: str, plan: str) -> str:
    plan_label = "Pro" if plan == "pro" else "Agency"
    limit = PLAN_LIMITS.get(plan, 50)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Payment successful — Visibly AI</title>{_PAGE_STYLE}</head>
<body><div class="card">
  <div class="logo">Visibly AI</div>
  <h1><span class="check">✓</span> You're all set!</h1>
  <p>Your <strong>{plan_label}</strong> plan is active. Copy your license key and paste it into
  the <strong>Pro &amp; License</strong> settings page in your WordPress plugin.</p>
  <div class="key-box">
    <div class="key-label">Your License Key</div>
    <div class="key-value">{license_key}</div>
  </div>
  <p>We also sent this key to your email — check your inbox if you need it later.</p>
  <p class="note">Plan includes {limit} AI analyses per month. Questions? Email
  <a href="mailto:{EMAIL_FROM}" style="color:#7c3aed">{EMAIL_FROM}</a></p>
</div></body></html>"""


def _pending_html(session_id: str) -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Payment received — Visibly AI</title>
<meta http-equiv="refresh" content="4;url=/v1/success?session_id={session_id}">
{_PAGE_STYLE}</head>
<body><div class="card">
  <div class="logo">Visibly AI</div>
  <h1>Payment received!</h1>
  <p>We're generating your license key — this page will refresh automatically in a few seconds.</p>
  <p class="note">Your key will also arrive by email at <strong>{EMAIL_FROM}</strong>.</p>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Routes — core
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "service": "Visibly AI API", "version": "2.0.0"}


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
        },
    }


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest):
    license_data = validate_license(req.license_key)
    if not license_data:
        raise HTTPException(status_code=401, detail="Invalid license key")

    plan = license_data.get("plan", "pro")
    used, limit = check_usage_limit(req.license_key, plan)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise HTTPException(status_code=500, detail="AI service not configured")

    failing = [s for s in req.signals if not s.passing]
    if not failing:
        return {
            "suggestions": [],
            "message": "All signals are passing — nothing to improve!",
            "usage": {"used": used, "limit": limit},
        }

    prompt = build_analyze_prompt(req.content, req.signals, req.post_url, req.gsc_queries)

    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            system=ANALYZE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        new_count = increment_usage(req.license_key)
        result["usage"] = {"used": new_count, "limit": limit}
        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
# Routes — Stripe
# ---------------------------------------------------------------------------

@app.post("/v1/webhook/stripe")
async def stripe_webhook(request: Request):
    if not STRIPE_SECRET_KEY or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    import stripe
    stripe.api_key = STRIPE_SECRET_KEY

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        # stripe.errors.SignatureVerificationError in SDK v8+; stripe.error.* in v7
        if "SignatureVerification" in type(e).__name__ or "signature" in str(e).lower():
            raise HTTPException(status_code=400, detail="Invalid Stripe signature")
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")

    # Stripe SDK v8: use attribute access, not dict-style .get()
    if event.type == "checkout.session.completed":
        session = event.data.object
        session_id = getattr(session, "id", "") or ""
        customer_details = getattr(session, "customer_details", None)
        email = (
            getattr(customer_details, "email", None) if customer_details else None
        ) or getattr(session, "customer_email", "") or ""
        customer_id = getattr(session, "customer", "") or ""

        # Plan: metadata takes priority over price ID mapping
        metadata = getattr(session, "metadata", None) or {}
        plan = metadata.get("plan") if isinstance(metadata, dict) else getattr(metadata, "plan", None)
        if not plan and PRICE_TO_PLAN:
            try:
                items = stripe.checkout.Session.list_line_items(session_id, limit=1)
                if items and items.data:
                    plan = PRICE_TO_PLAN.get(items.data[0].price.id)
            except Exception:
                pass
        plan = plan or "pro"

        license_key = _generate_key(plan)
        _insert_license(license_key, plan, email, customer_id, session_id)
        try:
            _send_license_email(email, license_key, plan)
        except Exception as email_err:
            # Log but don't crash — license is saved, user sees key on success page
            print(f"[email] Failed to send license email: {email_err}")

    return {"received": True}


@app.get("/v1/success", response_class=HTMLResponse)
async def payment_success(session_id: str):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    import stripe
    stripe.api_key = STRIPE_SECRET_KEY

    try:
        stripe.checkout.Session.retrieve(session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    # Look up license by session ID
    conn = _get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT key, plan FROM licenses WHERE stripe_session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if row:
            return HTMLResponse(_success_html(row[0], row[1]))

    # Webhook hasn't fired yet — show pending page with auto-refresh
    return HTMLResponse(_pending_html(session_id))
