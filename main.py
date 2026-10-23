import os, json, uuid, hashlib, secrets, asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
import httpx
import psycopg
from psycopg.rows import dict_row

APP_NAME = "TDS GA3 A2A Invoice Agent"
A2A_VERSION = "1.0"
PROPOSALS_MIME = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MIME = "application/vnd.ga5.invoice-action-receipts+json"
ALLOWED_ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}

app = FastAPI(title=APP_NAME, version="1.0.0")
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://", 1)
API_TOKEN = os.getenv("AGENT_BEARER_TOKEN", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def compact_sorted(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def digest(value):
    return hashlib.sha256(compact_sorted(value).encode("utf-8")).hexdigest()

def db():
    if not DATABASE_URL:
        raise HTTPException(503, "DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=8)

def init_db():
    if not DATABASE_URL:
        return
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_tasks (
            principal TEXT NOT NULL, task_id TEXT NOT NULL, state JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (principal, task_id)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_idempotency (
            principal TEXT NOT NULL, message_id TEXT NOT NULL, body_hash TEXT NOT NULL,
            task_id TEXT NOT NULL, PRIMARY KEY (principal, message_id)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_decisions (
            package_hash TEXT PRIMARY KEY, decision JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""")

@app.on_event("startup")
def startup():
    if DATABASE_URL:
        init_db()

def auth_principal(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Bearer authorization required")
    token = authorization[7:].strip()
    if API_TOKEN:
        if not secrets.compare_digest(token, API_TOKEN):
            raise HTTPException(401, "Invalid bearer token")
        return "configured-agent-principal"
    # If no configured token, still isolate by token without exposing it.
    if len(token) < 8:
        raise HTTPException(401, "Invalid bearer token")
    return hashlib.sha256(token.encode()).hexdigest()

def public_task(task):
    item = dict(task)
    item.pop("_private", None)
    return item

def task_envelope(task):
    return {"task": public_task(task)}

def task_record(principal, task_id):
    with db() as conn:
        row = conn.execute("SELECT state FROM agent_tasks WHERE principal=%s AND task_id=%s", (principal, task_id)).fetchone()
        return row["state"] if row else None

def save_task(principal, task):
    with db() as conn:
        conn.execute("""
        INSERT INTO agent_tasks(principal,task_id,state,updated_at) VALUES(%s,%s,%s::jsonb,now())
        ON CONFLICT(principal,task_id) DO UPDATE SET state=EXCLUDED.state,updated_at=now()
        """, (principal, task["id"], json.dumps(task)))

def get_task_list(principal):
    with db() as conn:
        rows = conn.execute("SELECT state FROM agent_tasks WHERE principal=%s ORDER BY created_at DESC", (principal,)).fetchall()
        return [r["state"] for r in rows]

def artifact_part(mime, data):
    return {"kind": "data", "data": data, "mediaType": mime}

def text_part(text):
    return {"kind": "text", "text": text}

def make_task(task_id, context_id, state, parts=None, artifacts=None, history=None):
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": {"input-required": "TASK_STATE_INPUT_REQUIRED", "completed": "TASK_STATE_COMPLETED", "canceled": "TASK_STATE_CANCELED", "failed": "TASK_STATE_FAILED"}.get(state, state), "timestamp": now_iso()},
        "history": history or [],
        "artifacts": artifacts or []
    }
    if parts is not None:
        task["status"]["message"] = {"role": "agent", "parts": parts}
    return task

def extract_message(body):
    # A2A SendMessageRequest usually carries "message"; accept the message directly too.
    msg = body.get("message", body)
    if not isinstance(msg, dict):
        raise HTTPException(400, "message must be an object")
    return msg

def extract_parts(message):
    parts = message.get("parts") or []
    out = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        if "data" in p and isinstance(p["data"], dict):
            out.append(p["data"])
        elif "text" in p:
            try:
                obj = json.loads(p["text"])
                if isinstance(obj, dict): out.append(obj)
            except Exception:
                pass
    # Also support message metadata/payload conventions.
    for key in ("data", "payload", "invoiceBatch", "batch"):
        val = message.get(key)
        if isinstance(val, dict): out.append(val)
    return out

def find_batch(message):
    for obj in extract_parts(message):
        if isinstance(obj, dict) and any(k in obj for k in ("packages", "invoices", "batchId")):
            return obj
    for key in ("data", "payload", "invoiceBatch", "batch"):
        val = message.get(key)
        if isinstance(val, dict) and any(k in val for k in ("packages", "invoices", "batchId")):
            return val
    return None

def find_continuation(message):
    for obj in extract_parts(message):
        if isinstance(obj, dict) and any(k in obj for k in ("results", "proposalResults", "receiptResults", "batchId")):
            if any(k in obj for k in ("results", "proposalResults", "receiptResults")):
                return obj
    for key in ("data", "payload", "continuation", "result"):
        val = message.get(key)
        if isinstance(val, dict) and any(k in val for k in ("results", "proposalResults", "receiptResults")):
            return val
    return None

def get_packages(batch):
    packages = batch.get("packages") or batch.get("invoices") or []
    if isinstance(packages, dict):
        packages = list(packages.values())
    return [p for p in packages if isinstance(p, dict)]

async def cached_decision(package):
    key = digest(package)
    if DATABASE_URL:
        with db() as conn:
            row = conn.execute("SELECT decision FROM agent_decisions WHERE package_hash=%s", (key,)).fetchone()
            if row:
                return row["decision"]
    decision = await decide_package(package)
    if DATABASE_URL:
        with db() as conn:
            conn.execute("""
            INSERT INTO agent_decisions(package_hash,decision) VALUES(%s,%s::jsonb)
            ON CONFLICT(package_hash) DO NOTHING
            """, (key, json.dumps(decision)))
            row = conn.execute("SELECT decision FROM agent_decisions WHERE package_hash=%s", (key,)).fetchone()
            if row:
                return row["decision"]
    return decision

async def decide_package(package):
    # Use an LLM when configured; fail closed to an exception when it is not.
    if not LLM_API_KEY:
        return {
            "action": "open_exception",
            "facts": {"reason": "Model API is not configured; manual review required."},
            "evidenceRefs": [],
            "rationale": "No decision was made because the decision model is not configured. Manual review is required."
        }
    prompt = {
        "task": "Classify this invoice package using only the supplied package content and its evidence. Do not invent facts.",
        "allowedActions": sorted(ALLOWED_ACTIONS),
        "rules": [
            "Choose exactly one action.",
            "Cite exactly three evidence references from the decisive evidence paragraph, not cover pages, archives, or training decoys.",
            "evidenceRefs must be copied exactly from source evidence references, including brackets if present.",
            "facts must contain only facts supported by the package.",
            "rationale must be 60 to 1500 characters and cite at least two of the evidenceRefs.",
            "If evidence is insufficient or contradictory, choose hold_invoice or open_exception.",
            "Return JSON only with keys action, facts, evidenceRefs, rationale."
        ],
        "package": package
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": LLM_MODEL,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are a cautious invoice processing agent. Follow all schema and evidence rules exactly."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
                ]
            }
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
    try:
        decision = json.loads(raw)
    except Exception:
        return {"action":"open_exception","facts":{"reason":"Model returned invalid JSON."},"evidenceRefs":[],"rationale":"The model response could not be parsed safely. Manual review is required."}
    if decision.get("action") not in ALLOWED_ACTIONS:
        decision["action"] = "open_exception"
    refs = decision.get("evidenceRefs")
    if not isinstance(refs, list):
        refs = []
    # Do not invent evidence refs. Require three distinct refs or fail closed.
    refs = list(dict.fromkeys(str(x) for x in refs))
    if len(refs) != 3:
        decision["action"] = "open_exception"
        decision["evidenceRefs"] = []
        decision["facts"] = {"reason":"Exactly three decisive evidence references could not be established."}
        decision["rationale"] = "The evidence references did not satisfy the required three-reference rule. Manual review is required."
    else:
        decision["evidenceRefs"] = refs
        rationale = str(decision.get("rationale", ""))
        if len(rationale) < 60 or len(rationale) > 1500 or sum(1 for r in refs if r in rationale) < 2:
            decision["action"] = "open_exception"
            decision["facts"] = {"reason":"Rationale did not meet citation requirements."}
            decision["rationale"] = "The rationale did not satisfy the required length and evidence-citation checks. Manual review is required."
            decision["evidenceRefs"] = []
    if not isinstance(decision.get("facts"), dict):
        decision["facts"] = {"reason":"Facts were not returned as an object."}
    return {k: decision[k] for k in ("action","facts","evidenceRefs","rationale")}

def make_proposal(batch, packages):
    proposals = []
    batch_id = str(batch.get("batchId") or batch.get("id") or uuid.uuid4())
    for i, package in enumerate(packages):
        package_id = str(package.get("packageId") or package.get("id") or package.get("invoiceId") or f"package-{i+1}")
        proposals.append({
            "packageId": package_id,
            "actionId": "act_" + uuid.uuid4().hex[:24],
            "action": "open_exception",
            "facts": {"reason": "Decision pending"},
            "evidenceRefs": [],
            "rationale": "Decision pending."
        })
    return batch_id, proposals

async def create_proposals(batch, packages):
    batch_id = str(batch.get("batchId") or batch.get("id") or uuid.uuid4())
    proposals = []
    for i, package in enumerate(packages):
        decision = await cached_decision(package)
        package_id = str(package.get("packageId") or package.get("id") or package.get("invoiceId") or f"package-{i+1}")
        action_id = "act_" + uuid.uuid4().hex
        proposals.append({
            "packageId": package_id,
            "actionId": action_id,
            "action": decision["action"],
            "facts": decision["facts"],
            "evidenceRefs": decision["evidenceRefs"],
            "rationale": decision["rationale"]
        })
    return batch_id, proposals

def content_type_ok(request):
    ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
    return ct in ("application/a2a+json", "application/json")

@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    base = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return {
        "name": APP_NAME,
        "description": "Evidence-based invoice batch action proposal agent.",
        "url": base,
        "version": "1.0.0",
        "protocolVersion": "1.0",
        "preferredTransport": "JSONRPC",
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "defaultInputModes": ["application/a2a+json", "application/json"],
        "defaultOutputModes": ["application/a2a+json", "application/json"],
        "skills": [{
            "id": "invoice-action-proposals",
            "name": "Invoice action proposals",
            "description": "Proposes evidence-backed invoice actions and waits for grader result continuation.",
            "tags": ["invoice", "evidence", "approval"]
        }],
        "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
        "security": [{"bearerAuth": []}]
    }

@app.post("/message:send")
async def message_send(request: Request, authorization: Optional[str] = Header(default=None), a2a_version: Optional[str] = Header(default=None, alias="A2A-Version")):
    principal = auth_principal(authorization)
    if a2a_version != A2A_VERSION:
        raise HTTPException(400, "A2A-Version: 1.0 required")
    if not content_type_ok(request):
        raise HTTPException(415, "Content-Type must be application/a2a+json")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    message = extract_message(body)
    message_id = str(message.get("messageId") or message.get("id") or "")
    if not message_id:
        raise HTTPException(400, "messageId is required")
    body_hash = digest(message)
    with db() as conn:
        old = conn.execute("SELECT body_hash,task_id FROM agent_idempotency WHERE principal=%s AND message_id=%s", (principal,message_id)).fetchone()
        if old:
            if old["body_hash"] != body_hash:
                raise HTTPException(409, "IDEMPOTENCY_CONFLICT")
            task = task_record(principal, old["task_id"])
            if task:
                return task_envelope(task)
        batch = find_batch(message)
        continuation = find_continuation(message)
        if continuation:
            task_id = str(continuation.get("taskId") or message.get("taskId") or body.get("taskId") or "")
            task = task_record(principal, task_id) if task_id else None
            if not task:
                raise HTTPException(404, "Task not found")
            state = task.get("status", {}).get("state")
            if state in ("TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_CANCELLED", "completed", "failed", "canceled", "cancelled"):
                return task_envelope(task)
            expected = task.get("_private", {})
            batch_id = str(continuation.get("batchId") or "")
            results = continuation.get("results") or continuation.get("proposalResults") or continuation.get("receiptResults") or []
            if batch_id != expected.get("batchId"):
                raise HTTPException(409, "Continuation batchId mismatch")
            if not isinstance(results, list):
                raise HTTPException(400, "Continuation results must be a list")
            proposal_by_id = {p["actionId"]: p for p in expected.get("proposals", [])}
            accepted = []
            seen = set()
            for result in results:
                if not isinstance(result, dict): continue
                action_id = str(result.get("actionId") or "")
                if not action_id or action_id in seen or action_id not in proposal_by_id:
                    continue
                seen.add(action_id)
                status = str(result.get("status") or result.get("result") or "").lower()
                if status in ("accepted", "accept", "success", "approved"):
                    p = proposal_by_id[action_id]
                    accepted.append({
                        "packageId": p["packageId"], "actionId": action_id, "action": p["action"],
                        "status": "accepted", "receiptNonce": secrets.token_urlsafe(24)
                    })
            receipts = {
                "batchId": expected["batchId"],
                "receipts": accepted
            }
            task["status"] = {"state":"TASK_STATE_COMPLETED","timestamp":now_iso()}
            task["artifacts"] = task.get("artifacts", []) + [{"artifactId":"receipt_" + uuid.uuid4().hex, "parts":[artifact_part(RECEIPTS_MIME, receipts)]}]
            task.setdefault("history", []).append({"role":"user","parts":message.get("parts",[])})
            task.pop("_private", None)
            save_task(principal, task)
            with db() as conn:
                conn.execute("INSERT INTO agent_idempotency(principal,message_id,body_hash,task_id) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING", (principal,message_id,body_hash,task_id))
            return task_envelope(task)
        if not batch:
            raise HTTPException(400, "No invoice batch found in message parts")
        packages = get_packages(batch)
        if not packages:
            raise HTTPException(400, "Invoice batch has no packages")
        batch_id, proposals = await create_proposals(batch, packages)
        task_id = str(uuid.uuid4())
        context_id = str(message.get("contextId") or uuid.uuid4())
        task = make_task(task_id, context_id, "input-required", parts=[text_part("Review the proposed invoice actions and continue with grader results.")],
                         artifacts=[{"artifactId":"proposal_" + uuid.uuid4().hex,"parts":[artifact_part(PROPOSALS_MIME, {"batchId":batch_id,"proposals":proposals})]}],
                         history=[message])
        task["_private"] = {"batchId":batch_id,"proposals":proposals}
        save_task(principal, task)
        with db() as conn:
            conn.execute("INSERT INTO agent_idempotency(principal,message_id,body_hash,task_id) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING", (principal,message_id,body_hash,task_id))
        return task_envelope(task)

@app.get("/tasks")
async def list_tasks(authorization: Optional[str] = Header(default=None)):
    principal = auth_principal(authorization)
    tasks = get_task_list(principal)
    # Remove internal fields from public response.
    public = []
    for task in tasks:
        item = dict(task); item.pop("_private", None); public.append(item)
    return {"tasks": public}

@app.get("/tasks/{task_id}")
async def get_task(task_id: str, authorization: Optional[str] = Header(default=None)):
    principal = auth_principal(authorization)
    task = task_record(principal, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    item = dict(task); item.pop("_private", None)
    return task_envelope(item)

@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request, authorization: Optional[str] = Header(default=None), a2a_version: Optional[str] = Header(default=None, alias="A2A-Version")):
    principal = auth_principal(authorization)
    if a2a_version != A2A_VERSION:
        raise HTTPException(400, "A2A-Version: 1.0 required")
    task = task_record(principal, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    state = task.get("status", {}).get("state")
    if state not in ("TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_CANCELLED", "completed", "failed", "canceled", "cancelled"):
        task["status"] = {"state":"TASK_STATE_CANCELED","timestamp":now_iso()}
        task.setdefault("history", []).append({"role":"agent","parts":[text_part("Task canceled; no invoice actions were executed.")]})
        save_task(principal, task)
    item = dict(task); item.pop("_private", None)
    return task_envelope(item)

@app.get("/")
async def root():
    return {"service": APP_NAME, "protocolVersion": A2A_VERSION, "health": "ok"}
