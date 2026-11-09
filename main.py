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
from datetime import datetime, timedelta
import urllib.parse
import time
import asyncio

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
EMAIL_FROM = "hello@mail.getvisiblyai.com"

API_BASE_URL = os.environ.get(
    "API_BASE_URL", "https://web-production-8f080.up.railway.app"
)
# Shared secret for /v1/cron/send-sequence. Without it the endpoint refuses to run,
# so a missing env var fails closed rather than exposing a public mailer.
CRON_SECRET = os.environ.get("CRON_SECRET")
# CAN-SPAM requires a physical postal address in commercial email. Set this in Railway
# before the welcome sequence goes anywhere real; emails say so until it is set.
MAILING_ADDRESS = os.environ.get("MAILING_ADDRESS", "")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GSC_REDIRECT_URI = os.environ.get(
    "GSC_REDIRECT_URI",
    "https://web-production-8f080.up.railway.app/v1/gsc/callback",
)
GSC_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"

# In-memory OAuth state store: state_token -> (license_key, expires_at)
_oauth_states: dict[str, tuple[str, float]] = {}

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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gsc_tokens (
                    license_key TEXT PRIMARY KEY,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    token_expiry TIMESTAMPTZ NOT NULL,
                    connected_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscribers (
                    email TEXT PRIMARY KEY,
                    token TEXT NOT NULL UNIQUE,
                    source TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    unsubscribed_at TIMESTAMPTZ,
                    sequence_step INTEGER NOT NULL DEFAULT 0,
                    last_sent_at TIMESTAMPTZ
                )
            """)
        conn.commit()
    finally:
        conn.close()


SEQUENCE_INTERVAL_SECONDS = int(os.environ.get("SEQUENCE_INTERVAL_SECONDS", 3600))


async def _sequence_loop():
    """Runs the welcome sequence on a timer so no external cron is needed.

    Hourly rather than daily: a signup whose inline send failed should not wait a
    day for its welcome, and the day 3 / day 7 steps land closer to the mark. The
    pass is idempotent and advisory-locked, so extra runs cost nothing.
    """
    while True:
        try:
            await asyncio.sleep(SEQUENCE_INTERVAL_SECONDS)
            await asyncio.to_thread(_run_sequence_pass)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never let one bad pass kill the loop for the life of the process.
            pass


# Async on purpose. Starlette runs a sync startup handler directly, but
# asyncio.get_event_loop() is deprecated inside a running loop and its behaviour
# differs across versions; an async handler with create_task is unambiguous.
@app.on_event("startup")
async def startup():
    _init_db()
    app.state.sequence_task = asyncio.create_task(_sequence_loop())


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


class GenerateSchemaRequest(BaseModel):
    license_key: str
    content: str
    schema_type: str  # 'faq' | 'howto'
    post_url: Optional[str] = None


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
    # Reports the deployed commit, not a hardcoded version string. Railway's
    # x-hikari-trace header alternates between edge nodes and is NOT a deployment
    # identifier, so before this there was no way to tell whether a push had
    # actually gone live.
    return {
        "status": "ok",
        "service": "Visibly AI API",
        "commit": (os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "unknown")[:7],
        # Whether the welcome sequence can legally send yet. A boolean, never the
        # address itself: this endpoint is public and the value is a postal address.
        "mailing_address_set": bool(MAILING_ADDRESS),
    }


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
            model="claude-sonnet-5-5",
            max_tokens=2048,
            system=ANALYZE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # Claude 5 returns a thinking block before the text block
        response_text = "".join(
            b.text for b in message.content if getattr(b, "type", None) == "text"
        )

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
# Routes — generate schema
# ---------------------------------------------------------------------------

GENERATE_FAQ_PROMPT = """You are an expert in structured data and Answer Engine Optimization (AEO).

Given a WordPress post, generate 4–6 high-quality FAQ pairs that:
- Answer real questions readers of this post would ask
- Use the post's own content and language (don't invent facts)
- Have concise answers of 2–4 sentences — ideal for Google AI Overviews and Perplexity
- Cover a range of question types: what/why/how/when

Return ONLY valid JSON in this exact format with no other text:
{"faq": [{"question": "...", "answer": "..."}, ...]}"""

GENERATE_HOWTO_PROMPT = """You are an expert in structured data and Answer Engine Optimization (AEO).

Given a WordPress post, generate a HowTo schema object that:
- Extracts the actual steps described in the post (do not invent steps)
- Uses a clear "How to..." title
- Has 3–8 steps, each with a short action name and 1–3 sentence instructions
- Writes a brief description summarising what the reader will achieve

Return ONLY valid JSON in this exact format with no other text:
{"howto_name": "How to ...", "howto_desc": "...", "steps": [{"name": "...", "text": "..."}, ...]}"""


@app.post("/v1/generate-schema")
def generate_schema(req: GenerateSchemaRequest):
    license_data = validate_license(req.license_key)
    if not license_data:
        raise HTTPException(status_code=401, detail="Invalid license key")

    plan = license_data.get("plan", "pro")
    used, limit = check_usage_limit(req.license_key, plan)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise HTTPException(status_code=500, detail="AI service not configured")

    if req.schema_type not in ("faq", "howto"):
        raise HTTPException(status_code=400, detail="schema_type must be 'faq' or 'howto'")

    system_prompt = GENERATE_FAQ_PROMPT if req.schema_type == "faq" else GENERATE_HOWTO_PROMPT
    clean_content = strip_block_comments(req.content)
    url_line = f"\nPost URL: {req.post_url}" if req.post_url else ""
    user_prompt = f"Generate schema markup for this post.{url_line}\n\nPost content:\n{clean_content[:8000]}"

    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        message = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Claude 5 returns a thinking block before the text block
        response_text = "".join(
            b.text for b in message.content if getattr(b, "type", None) == "text"
        )

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
        raise HTTPException(status_code=500, detail=f"Schema generation failed: {str(e)}")


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


# ---------------------------------------------------------------------------
# GSC — DB helpers
# ---------------------------------------------------------------------------

def _save_gsc_tokens(license_key: str, access_token: str, refresh_token: str, expires_in: int):
    conn = _get_db()
    if not conn:
        return
    expiry = datetime.utcnow() + timedelta(seconds=max(0, expires_in - 60))
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO gsc_tokens (license_key, access_token, refresh_token, token_expiry)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (license_key) DO UPDATE SET
                       access_token = EXCLUDED.access_token,
                       refresh_token = EXCLUDED.refresh_token,
                       token_expiry = EXCLUDED.token_expiry""",
                (license_key, access_token, refresh_token, expiry),
            )
        conn.commit()
    finally:
        conn.close()


def _get_gsc_tokens(license_key: str) -> dict | None:
    conn = _get_db()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT access_token, refresh_token, token_expiry FROM gsc_tokens WHERE license_key = %s",
                (license_key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"access_token": row[0], "refresh_token": row[1], "token_expiry": row[2]}
    finally:
        conn.close()


def _delete_gsc_tokens(license_key: str):
    conn = _get_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM gsc_tokens WHERE license_key = %s", (license_key,))
        conn.commit()
    finally:
        conn.close()


def _get_valid_access_token(license_key: str) -> str | None:
    """Returns a valid GSC access token, refreshing via refresh_token if expired."""
    tokens = _get_gsc_tokens(license_key)
    if not tokens:
        return None

    expiry = tokens["token_expiry"]
    now_ts = time.time()
    try:
        import calendar
        if hasattr(expiry, "utcoffset") and expiry.utcoffset() is not None:
            expiry_ts = calendar.timegm(expiry.utctimetuple())
        else:
            expiry_ts = calendar.timegm(expiry.timetuple())
        if now_ts < expiry_ts:
            return tokens["access_token"]
    except Exception:
        pass

    # Token expired — use refresh_token
    try:
        import httpx as _httpx
        resp = _httpx.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "refresh_token": tokens["refresh_token"],
                "grant_type": "refresh_token",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        new_access = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        _save_gsc_tokens(license_key, new_access, tokens["refresh_token"], expires_in)
        return new_access
    except Exception as e:
        print(f"[GSC] Token refresh failed: {e}")
        return None


# ---------------------------------------------------------------------------
# GSC — API helpers
# ---------------------------------------------------------------------------

def _find_matching_site(site_entries: list, page_url: str) -> str | None:
    """Find the GSC property that contains this page URL."""
    parsed = urllib.parse.urlparse(page_url)
    domain = parsed.netloc
    site_urls = [e.get("siteUrl", "") for e in site_entries]

    # Exact domain-property match (sc-domain:)
    for candidate in [f"sc-domain:{domain}", f"sc-domain:{domain.lstrip('www.')}"]:
        if candidate in site_urls:
            return candidate

    # Build www/non-www variants of the page URL to handle mismatches
    # e.g. WP permalink is non-www but GSC property is www (or vice versa)
    url_variants = [page_url]
    if domain.startswith("www."):
        # also try without www
        url_variants.append(page_url.replace(f"://{domain}", f"://{domain[4:]}", 1))
    else:
        # also try with www
        url_variants.append(page_url.replace(f"://{domain}", f"://www.{domain}", 1))

    # URL-prefix match — try all variants, longest wins
    prefix_matches = [
        s for s in site_urls
        if any(variant.startswith(s) for variant in url_variants)
    ]
    if prefix_matches:
        return max(prefix_matches, key=len)

    return None


def _fetch_gsc_queries(access_token: str, page_url: str, days: int = 90) -> tuple[list[dict], str | None]:
    """
    Returns (rows, error_message). rows is a list of query dicts.
    error_message is set when the site property isn't found.
    """
    import httpx as _httpx
    from datetime import date

    headers = {"Authorization": f"Bearer {access_token}"}
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    sites_resp = _httpx.get(
        "https://www.googleapis.com/webmasters/v3/sites",
        headers=headers,
        timeout=10,
    )
    sites_resp.raise_for_status()
    site_url = _find_matching_site(sites_resp.json().get("siteEntry", []), page_url)

    if not site_url:
        return [], "No matching GSC property found for this URL. Make sure your site is verified in Google Search Console."

    # Normalize page_url www-style to match the matched site property.
    # e.g. site_url = "https://www.adoreness.com/" but page_url = "https://adoreness.com/slug/"
    # GSC indexes pages under the property's canonical www/non-www form.
    normalized_url = page_url
    if site_url.startswith("https://www.") or site_url.startswith("http://www."):
        parsed_page = urllib.parse.urlparse(page_url)
        if not parsed_page.netloc.startswith("www."):
            normalized_url = page_url.replace(
                f"://{parsed_page.netloc}", f"://www.{parsed_page.netloc}", 1
            )
    elif site_url.startswith("https://") or site_url.startswith("http://"):
        # site is non-www — strip www from page_url if present
        parsed_page = urllib.parse.urlparse(page_url)
        if parsed_page.netloc.startswith("www."):
            normalized_url = page_url.replace(
                f"://www.{parsed_page.netloc[4:]}", f"://{parsed_page.netloc[4:]}", 1
            )

    body = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "dimensions": ["query"],
        "dimensionFilterGroups": [{
            "filters": [{
                "dimension": "page",
                "operator": "equals",
                "expression": normalized_url,
            }]
        }],
        "rowLimit": 20,
        "orderBy": [{"fieldName": "clicks", "sortOrder": "DESCENDING"}],
    }

    encoded_site = urllib.parse.quote(site_url, safe="")
    analytics_resp = _httpx.post(
        f"https://www.googleapis.com/webmasters/v3/sites/{encoded_site}/searchAnalytics/query",
        headers=headers,
        json=body,
        timeout=15,
    )
    analytics_resp.raise_for_status()

    rows = analytics_resp.json().get("rows", [])
    queries = [
        {
            "query": row["keys"][0],
            "clicks": int(row.get("clicks", 0)),
            "impressions": int(row.get("impressions", 0)),
            "ctr": round(row.get("ctr", 0) * 100, 1),
            "position": round(row.get("position", 0), 1),
        }
        for row in rows
    ]
    return queries, None


# ---------------------------------------------------------------------------
# GSC — HTML helpers
# ---------------------------------------------------------------------------

def _gsc_success_html() -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GSC Connected — Visibly AI</title>{_PAGE_STYLE}
<script>
  if (window.opener) {{
    window.opener.postMessage({{type:'visibly_gsc_connected'}}, '*');
    setTimeout(function(){{ window.close(); }}, 1800);
  }}
</script>
</head><body><div class="card">
  <div class="logo">Visibly AI</div>
  <h1><span class="check">✓</span> Google Search Console connected!</h1>
  <p>Your GSC data will now appear in the Visibly AI sidebar when you edit posts.</p>
  <p class="note">You can close this window and return to WordPress.</p>
</div></body></html>"""


def _gsc_error_html(message: str) -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Connection Error — Visibly AI</title>{_PAGE_STYLE}</head>
<body><div class="card">
  <div class="logo">Visibly AI</div>
  <h1 style="color:#dc2626">Connection failed</h1>
  <p>{message}</p>
  <p class="note">Please close this window and try again from your WordPress settings.</p>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Routes — GSC
# ---------------------------------------------------------------------------

@app.get("/v1/gsc/connect")
def gsc_connect(license_key: str):
    """Validates license then redirects to Google OAuth."""
    from fastapi.responses import RedirectResponse

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=503, detail="GSC OAuth not configured")

    license_data = validate_license(license_key)
    if not license_data:
        raise HTTPException(status_code=401, detail="Invalid license key")

    # Prune expired states
    now = time.time()
    for s in [k for k, v in _oauth_states.items() if v[1] < now]:
        _oauth_states.pop(s, None)

    state = secrets.token_urlsafe(32)
    _oauth_states[state] = (license_key, now + 600)

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GSC_REDIRECT_URI,
        "response_type": "code",
        "scope": GSC_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return RedirectResponse(url=auth_url)


@app.get("/v1/gsc/callback", response_class=HTMLResponse)
def gsc_callback(code: str = None, state: str = None, error: str = None):
    """Google OAuth callback — exchanges code for tokens and stores in DB."""
    import httpx as _httpx

    if error:
        return HTMLResponse(_gsc_error_html(f"Authorization denied: {error}"))
    if not code or not state:
        return HTMLResponse(_gsc_error_html("Missing code or state parameter."))

    state_entry = _oauth_states.pop(state, None)
    if not state_entry:
        return HTMLResponse(_gsc_error_html("Invalid or expired authorization state. Please try connecting again."))

    license_key, expires_at = state_entry
    if time.time() > expires_at:
        return HTMLResponse(_gsc_error_html("Authorization expired. Please try connecting again."))

    try:
        resp = _httpx.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GSC_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        resp.raise_for_status()
        token_data = resp.json()
    except Exception as e:
        return HTMLResponse(_gsc_error_html(f"Failed to exchange authorization code: {e}"))

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expires_in = token_data.get("expires_in", 3600)

    if not access_token or not refresh_token:
        return HTMLResponse(_gsc_error_html("Google did not return the expected tokens. Please try again."))

    _save_gsc_tokens(license_key, access_token, refresh_token, expires_in)
    return HTMLResponse(_gsc_success_html())


@app.get("/v1/gsc/status")
def gsc_status(license_key: str):
    """Returns whether GSC is connected for this license key."""
    license_data = validate_license(license_key)
    if not license_data:
        raise HTTPException(status_code=401, detail="Invalid license key")

    tokens = _get_gsc_tokens(license_key)
    return {"connected": tokens is not None}


@app.get("/v1/gsc/queries")
def gsc_queries(license_key: str, url: str):
    """Returns top GSC queries for the given page URL."""
    import httpx as _httpx

    license_data = validate_license(license_key)
    if not license_data:
        raise HTTPException(status_code=401, detail="Invalid license key")

    access_token = _get_valid_access_token(license_key)
    if not access_token:
        raise HTTPException(
            status_code=403,
            detail={"error": "gsc_not_connected", "message": "GSC not connected. Please connect Google Search Console in your Pro & License settings."},
        )

    try:
        queries, site_error = _fetch_gsc_queries(access_token, url)
    except _httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            raise HTTPException(status_code=403, detail={"error": "gsc_auth_error", "message": "GSC permission denied. Please reconnect."})
        raise HTTPException(status_code=502, detail=f"GSC API error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch GSC data: {e}")

    if site_error:
        return {"connected": True, "queries": [], "near_page_1": [], "low_ctr": [], "query_strings": [], "site_error": site_error}

    near_page_1 = [q for q in queries if 11.0 <= q["position"] <= 20.0]
    low_ctr = [q for q in queries if q["position"] <= 10.0 and q["ctr"] < 3.0 and q["clicks"] > 0]

    return {
        "connected": True,
        "queries": queries,
        "near_page_1": near_page_1,
        "low_ctr": low_ctr,
        "query_strings": [q["query"] for q in queries[:10]],
    }


@app.delete("/v1/gsc/disconnect")
def gsc_disconnect(license_key: str):
    """Removes stored GSC tokens for this license key."""
    license_data = validate_license(license_key)
    if not license_data:
        raise HTTPException(status_code=401, detail="Invalid license key")

    _delete_gsc_tokens(license_key)
    return {"disconnected": True}


# ---------------------------------------------------------------------------
# Newsletter: capture, welcome sequence, unsubscribe
#
# WP.org exposes no way to contact an installer, so this list is the only owned
# channel Visibly AI has. It runs on infrastructure that already exists: the same
# Postgres and the same Resend sending domain used for license delivery.
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]{2,}$")

# Per-IP throttle. An open subscribe endpoint gets scraped and abused, and this
# keeps a single source from stuffing the list. In-memory on purpose: the same
# approach the OAuth state store uses, and a restart losing it is harmless.
_subscribe_hits: dict[str, list[float]] = {}
SUBSCRIBE_MAX_PER_HOUR = 5


def _throttled(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _subscribe_hits.get(ip, []) if now - t < 3600]
    _subscribe_hits[ip] = hits
    if len(hits) >= SUBSCRIBE_MAX_PER_HOUR:
        return True
    hits.append(now)
    return False


def _email_footer(token: str) -> str:
    """Unsubscribe is legally required in every one of these, not optional."""
    addr = (
        f'<p style="margin:8px 0 0">{MAILING_ADDRESS}</p>'
        if MAILING_ADDRESS
        else '<p style="margin:8px 0 0;color:#b91c1c">[MAILING_ADDRESS env var is not '
             'set. CAN-SPAM requires a postal address here.]</p>'
    )
    return f"""
  <hr style="border:none;border-top:1px solid #e5e7eb;margin:32px 0 16px">
  <div style="color:#9ca3af;font-size:12px;line-height:1.6">
    <p style="margin:0">You are getting this because you signed up at getvisiblyai.com.</p>
    <p style="margin:4px 0 0"><a href="{API_BASE_URL}/v1/unsubscribe?token={token}"
      style="color:#9ca3af">Unsubscribe</a></p>
    {addr}
  </div>
"""


def _wrap_email(body_html: str, token: str) -> str:
    return (
        '<div style="font-family:sans-serif;max-width:540px;margin:0 auto;'
        'padding:32px 24px;color:#1f2937">' + body_html + _email_footer(token) + "</div>"
    )


# step -> (days to wait after signup, subject, body html)
SEQUENCE: dict[int, tuple[int, str, str]] = {
    1: (
        0,
        "Thanks for signing up. Start with these three posts",
        """
  <h2 style="color:#7c3aed;margin:0 0 8px">Thanks for opting in</h2>
  <p style="margin:0 0 16px;color:#374151">Here are the three posts people read most, to
  start with:</p>
  <p style="margin:0 0 10px;color:#374151"><a href="https://www.getvisiblyai.com/blog/what-is-aeo" style="color:#7c3aed">What is AEO? Answer Engine Optimization for WordPress Bloggers</a></p>
  <p style="margin:0 0 10px;color:#374151"><a href="https://www.getvisiblyai.com/blog/10-signals-ai-citation" style="color:#7c3aed">10 Signals That Determine If AI Cites Your Content</a></p>
  <p style="margin:0 0 20px;color:#374151"><a href="https://www.getvisiblyai.com/blog/how-to-add-faq-schema-wordpress" style="color:#7c3aed">How to Add FAQ Schema to WordPress (For SEO and AI Search)</a></p>
  <p style="margin:0 0 8px;color:#6b7280;font-size:14px">I'll send a few short tips over the next
  couple of weeks. Unsubscribe any time with the link below.</p>
""",
    ),
    2: (
        3,
        "Check your FAQ schema is actually on the page",
        """
  <h2 style="color:#7c3aed;margin:0 0 8px">Thanks for signing up</h2>
  <p style="margin:0 0 20px;color:#374151">One thing worth doing before anything else,
  and it takes 20 seconds.</p>
  <p style="margin:0 0 12px;color:#374151">If you have built FAQ schema with any WordPress
  plugin, open the published post (not the editor preview), view source, and search for
  <code>FAQPage</code>.</p>
  <p style="margin:0 0 20px;color:#374151">If it is not there, the markup is not there,
  whatever the plugin's panel told you. A scorer and a renderer are different pieces of
  code and nothing guarantees they agree. I know, because mine disagreed for five months.</p>
  <p style="margin:0 0 8px;color:#374151">Worth checking your top 5 posts.</p>
""",
    ),
    3: (
        7,
        "The AEO signal most people skip",
        """
  <h2 style="color:#7c3aed;margin:0 0 8px">The short direct answer</h2>
  <p style="margin:0 0 20px;color:#374151">Of the signals that decide whether an AI engine
  cites you, this is the one most content misses.</p>
  <p style="margin:0 0 12px;color:#374151">Put a 40 to 60 word answer directly under each
  question heading, and make the first sentence answer the question outright. No preamble,
  no "in this article we will look at".</p>
  <p style="margin:0 0 20px;color:#374151">That paragraph is what gets pulled into an
  answer. If your heading asks a question and the next paragraph sets up context instead
  of answering it, there is nothing for the engine to quote.</p>
  <p style="margin:0 0 8px;color:#374151">Question-format headings plus a short direct
  answer underneath does more for citation than anything else on the list.</p>
""",
    ),
    4: (
        10,
        "What Pro actually does",
        """
  <h2 style="color:#7c3aed;margin:0 0 8px">Only if it is useful to you</h2>
  <p style="margin:0 0 16px;color:#374151">The free plugin scores your content against the
  signals and tells you which ones fail. That is genuinely the useful half, and plenty of
  people never need more.</p>
  <p style="margin:0 0 16px;color:#374151">Pro adds two things: Claude reads the post and
  returns specific copy-pasteable fixes for the signals you are failing, and a Google
  Search Console connection makes those suggestions follow your actual search queries
  rather than generic advice.</p>
  <p style="margin:0 0 20px;color:#374151">$19/mo or $149/yr, one site.
  <a href="https://www.getvisiblyai.com" style="color:#7c3aed">getvisiblyai.com</a></p>
  <p style="margin:0 0 8px;color:#6b7280;font-size:14px">That is the last thing I will send
  about Pro. Reply any time if something is broken, it comes straight to me.</p>
""",
    ),
}


def _send_sequence_email(email: str, token: str, step: int) -> bool:
    if not RESEND_API_KEY or step not in SEQUENCE:
        return False
    # Refuse to send commercial email without the postal address CAN-SPAM requires.
    # Returning False leaves sequence_step where it was, so the subscriber is kept and
    # the next cron run delivers it once MAILING_ADDRESS is configured. Better a late
    # welcome than a non-compliant one.
    if not MAILING_ADDRESS:
        return False
    import resend
    resend.api_key = RESEND_API_KEY
    _, subject, body = SEQUENCE[step]
    try:
        resend.Emails.send({
            "from": f"Visibly AI <{EMAIL_FROM}>",
            "to": [email],
            "subject": subject,
            "html": _wrap_email(body, token),
        })
        return True
    except Exception:
        # A send failure must not fail the signup, and must not advance the step,
        # so the next cron run retries this subscriber.
        return False


class SubscribeRequest(BaseModel):
    email: str
    source: Optional[str] = None
    # Honeypot. A real person leaves it empty; most bots fill every field.
    company: Optional[str] = None


@app.post("/v1/subscribe")
def subscribe(req: SubscribeRequest, request: Request):
    # Always the same response shape, so the endpoint cannot be used to find out
    # whether an address is already on the list.
    ok = {"status": "ok"}

    if req.company:
        return ok

    email = (req.email or "").strip().lower()
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")

    ip = request.client.host if request.client else "unknown"
    if _throttled(ip):
        raise HTTPException(status_code=429, detail="Too many signups. Try again later.")

    token = secrets.token_urlsafe(24)
    source = (req.source or "")[:64] or None

    conn = _get_db()
    if not conn:
        raise HTTPException(status_code=503, detail="Signup is unavailable right now.")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO subscribers (email, token, source, sequence_step)
                VALUES (%s, %s, %s, 0)
                ON CONFLICT (email) DO NOTHING
                RETURNING token
                """,
                (email, token, source),
            )
            row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()

    # No row means the address was already subscribed. Say nothing and send nothing:
    # re-signing up should not re-trigger the sequence.
    if not row:
        return ok

    if _send_sequence_email(email, row[0], 1):
        conn = _get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE subscribers SET sequence_step = 1, last_sent_at = NOW() "
                    "WHERE email = %s",
                    (email,),
                )
            conn.commit()
        finally:
            conn.close()

    return ok


@app.get("/v1/unsubscribe", response_class=HTMLResponse)
def unsubscribe(token: str = ""):
    message = "That link is not valid. Nothing was changed."
    if token:
        conn = _get_db()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE subscribers SET unsubscribed_at = NOW() "
                        "WHERE token = %s AND unsubscribed_at IS NULL RETURNING email",
                        (token,),
                    )
                    row = cur.fetchone()
                conn.commit()
            finally:
                conn.close()
            # Already-unsubscribed gets the same wording: the user's intent is served
            # either way and there is no reason to expose which case it was.
            message = "You are unsubscribed. You will not get any more of these."
    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visibly AI</title></head>
<body style="font-family:sans-serif;max-width:480px;margin:80px auto;padding:0 24px;color:#1f2937">
  <h1 style="color:#7c3aed;font-size:20px">Visibly AI</h1>
  <p style="color:#374151">{message}</p>
  <p><a href="https://www.getvisiblyai.com" style="color:#7c3aed">getvisiblyai.com</a></p>
</body></html>""")


# Arbitrary fixed key for the Postgres advisory lock that keeps two workers from
# running a pass at the same time and double-sending.
_SEQUENCE_LOCK_KEY = 8417203


def _run_sequence_pass() -> dict:
    """Sends any sequence emails that are due. Safe to call repeatedly."""
    conn = _get_db()
    if not conn:
        return {"status": "skipped", "reason": "no database", "sent": 0, "failed": 0}

    sent, failed = 0, 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (_SEQUENCE_LOCK_KEY,))
            if not cur.fetchone()[0]:
                return {"status": "skipped", "reason": "another pass is running",
                        "sent": 0, "failed": 0}
        try:
            # Step 1 is attempted inline at signup, but it is included here as the
            # catch-up path: anyone left on step 0 because a send failed (no Resend
            # key, no MAILING_ADDRESS, a transient Resend error) is picked up here.
            for step, (wait_days, _, _) in sorted(SEQUENCE.items()):
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT email, token FROM subscribers
                        WHERE unsubscribed_at IS NULL
                          AND sequence_step = %s
                          AND created_at <= NOW() - (%s * INTERVAL '1 day')
                        LIMIT 200
                        """,
                        (step - 1, wait_days),
                    )
                    due = cur.fetchall()

                for email, token in due:
                    if _send_sequence_email(email, token, step):
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE subscribers SET sequence_step = %s, "
                                "last_sent_at = NOW() WHERE email = %s",
                                (step, email),
                            )
                        conn.commit()
                        sent += 1
                    else:
                        failed += 1
        finally:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_SEQUENCE_LOCK_KEY,))
            conn.commit()
    finally:
        conn.close()

    return {"status": "ok", "sent": sent, "failed": failed}


@app.post("/v1/cron/send-sequence")
def send_sequence(request: Request):
    """Manual or external trigger for the sequence.

    The app runs this on its own schedule too (see _sequence_loop), so this endpoint
    is a convenience, not a requirement. Fails closed without CRON_SECRET.
    """
    if not CRON_SECRET:
        raise HTTPException(status_code=503, detail="CRON_SECRET is not configured.")
    if request.headers.get("x-cron-secret") != CRON_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    return _run_sequence_pass()
