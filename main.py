import os
import json
import uuid
import hashlib
import secrets
import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

APP_NAME = "TDS GA3 A2A Invoice Agent"
A2A_VERSION = "1.0"
BATCH_MIME = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MIME = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MIME = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MIME = "application/vnd.ga5.invoice-action-receipts+json"
A2A_MIME = "application/a2a+json"
ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}
TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
}

app = FastAPI(title=APP_NAME, version="1.0.0")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]
API_TOKEN = os.getenv("AGENT_BEARER_TOKEN", "").strip()
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", "https://tds-ga3-invoice-agent-j2qz.onrender.com/"
).strip()
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://api.groq.com/openai/v1"
).rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def compact_sorted(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(compact_sorted(value).encode("utf-8")).hexdigest()


def principal_for_token(token):
    # Each bearer token gets its own isolated namespace. The configured token is
    # the token to use for normal grader requests; other tokens cannot see its data.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def auth_principal(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer authorization required")
    token = authorization[len("Bearer "):].strip()
    if not token or len(token) < 8:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    # Do not return or log the token. Isolation is based on the token itself.
    return principal_for_token(token)


def db():
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=8)


def init_db():
    if not DATABASE_URL:
        return
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_tasks (
                principal TEXT NOT NULL,
                task_id TEXT NOT NULL,
                state JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (principal, task_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_idempotency (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                body_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (principal, message_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                package_hash TEXT PRIMARY KEY,
                decision JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)


@app.on_event("startup")
def startup():
    if DATABASE_URL:
        init_db()


def json_a2a(data, status_code=200):
    return JSONResponse(content=data, status_code=status_code, media_type=A2A_MIME)


def save_task_conn(conn, principal, task):
    conn.execute("""
        INSERT INTO agent_tasks(principal, task_id, state, updated_at)
        VALUES (%s, %s, %s::jsonb, now())
        ON CONFLICT (principal, task_id)
        DO UPDATE SET state = EXCLUDED.state, updated_at = now()
    """, (principal, task["id"], json.dumps(task, ensure_ascii=False)))


def load_task_conn(conn, principal, task_id, for_update=False):
    suffix = " FOR UPDATE" if for_update else ""
    row = conn.execute(
        "SELECT state FROM agent_tasks WHERE principal=%s AND task_id=%s" + suffix,
        (principal, task_id),
    ).fetchone()
    return row["state"] if row else None


def task_is_terminal(task):
    return task.get("status", {}).get("state") in TERMINAL_STATES


def public_task(task):
    result = json.loads(json.dumps(task))
    result.pop("_private", None)
    return result


def artifact_part(media_type, data):
    return {"kind": "data", "mediaType": media_type, "data": data}


def text_part(text):
    return {"kind": "text", "text": text}


def make_task(task_id, context_id, state, history=None, artifacts=None):
    state_map = {
        "input-required": "TASK_STATE_INPUT_REQUIRED",
        "completed": "TASK_STATE_COMPLETED",
        "canceled": "TASK_STATE_CANCELED",
        "failed": "TASK_STATE_FAILED",
    }
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state_map.get(state, state), "timestamp": now_iso()},
        "history": history or [],
        "artifacts": artifacts or [],
    }


def extract_message(body):
    msg = body.get("message", body) if isinstance(body, dict) else None
    if not isinstance(msg, dict):
        raise HTTPException(status_code=400, detail="message must be an object")
    return msg


def part_data(message, expected_mime=None):
    parts = message.get("parts")
    if not isinstance(parts, list):
        return []
    values = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        mime = part.get("mediaType") or part.get("media_type")
        if expected_mime and mime != expected_mime:
            continue
        data = part.get("data")
        if isinstance(data, dict):
            values.append(data)
        elif isinstance(part.get("text"), str):
            try:
                parsed = json.loads(part["text"])
                if isinstance(parsed, dict):
                    values.append(parsed)
            except (ValueError, TypeError):
                pass
    return values


def find_batch(message):
    candidates = part_data(message, BATCH_MIME)
    if not candidates:
        candidates = part_data(message)
    for item in candidates:
        if isinstance(item.get("packages"), (list, dict)) or isinstance(item.get("invoices"), (list, dict)):
            return item
    return None


def find_continuation(message):
    candidates = part_data(message, RESULTS_MIME)
    if not candidates:
        candidates = part_data(message)
    for item in candidates:
        if isinstance(item.get("results"), list):
            return item
    return None


def get_packages(batch):
    packages = batch.get("packages", batch.get("invoices", []))
    if isinstance(packages, dict):
        packages = list(packages.values())
    if not isinstance(packages, list):
        return []
    return [p for p in packages if isinstance(p, dict)]


def package_id(package, index):
    return str(
        package.get("packageId")
        or package.get("id")
        or package.get("invoiceId")
        or f"package-{index + 1}"
    )


def extract_required_facts(facts):
    if not isinstance(facts, dict):
        return None
    vendor = facts.get("vendorName")
    invoice = facts.get("invoiceNumber")
    amount = facts.get("amountMinor")
    currency = facts.get("currency")
    if not isinstance(vendor, str) or not vendor.strip():
        return None
    if not isinstance(invoice, str) or not invoice.strip():
        return None
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        return None
    if currency != "INR":
        return None
    return {
        "vendorName": vendor.strip(),
        "invoiceNumber": invoice.strip(),
        "amountMinor": amount,
        "currency": "INR",
    }


def fallback_decision(package, reason="Insufficient verified evidence for a safe decision."):
    # Fail closed: never claim an action was authorized without evidence.
    return {
        "action": "open_exception",
        "facts": {
            "vendorName": str(package.get("vendorName") or package.get("vendor") or "Unknown"),
            "invoiceNumber": str(package.get("invoiceNumber") or package.get("invoice_number") or "Unknown"),
            "amountMinor": 0,
            "currency": "INR",
        },
        "evidenceRefs": [],
        "rationale": (
            "Open an exception because the package could not be validated safely. "
            + reason
        )[:1500],
    }


async def call_model_for_packages(packages):
    if not packages or not LLM_API_KEY:
        return [fallback_decision(p, "Decision model is not configured.") for p in packages]

    prompt = {
        "task": (
            "For every supplied invoice package, make exactly one action decision. "
            "Read the actual decisive evidence paragraph, not cover sheets, archive "
            "examples, or training decoys. Never invent evidence or invoice facts."
        ),
        "allowedActions": sorted(ALLOWED_ACTIONS),
        "requiredOutput": {
            "decisions": [{
                "index": "zero-based index matching the input package",
                "action": "one allowed action",
                "facts": {
                    "vendorName": "string",
                    "invoiceNumber": "string",
                    "amountMinor": "integer in paise",
                    "currency": "INR",
                },
                "evidenceRefs": ["exactly three decisive bracketed references"],
                "rationale": "60-1500 characters, names action and cites at least two refs",
            }]
        },
        "decisionRules": [
            "settle_invoice only if valid, reconciled, and within autonomous authority",
            "request_approval if commercially valid but outside delegated authority",
            "hold_invoice if payment must pause until a stated verification completes",
            "reject_duplicate only if the same commercial invoice was already paid",
            "open_exception if material records conflict or evidence is insufficient",
            "Return exactly three distinct exact bracketed refs from the decisive paragraph.",
            "Do not cite cover-sheet, archive, historical example, or training-decoy refs.",
            "amountMinor is an integer in minor currency units (paise for INR).",
            "Do not add markdown. Return valid JSON only.",
        ],
        "packages": packages,
    }

    try:
        async with httpx.AsyncClient(timeout=32) as client:
            response = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a precise invoice decision agent. Return schema-valid JSON only.",
                        },
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                    ],
                },
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        decisions = parsed.get("decisions")
        if not isinstance(decisions, list):
            raise ValueError("No decisions list")
        by_index = {}
        for item in decisions:
            if isinstance(item, dict):
                try:
                    by_index[int(item.get("index"))] = item
                except (TypeError, ValueError):
                    pass

        cleaned = []
        for i, package in enumerate(packages):
            d = by_index.get(i)
            if not isinstance(d, dict):
                cleaned.append(fallback_decision(package, "The model omitted this package."))
                continue
            action = d.get("action")
            facts = extract_required_facts(d.get("facts"))
            refs = d.get("evidenceRefs")
            rationale = d.get("rationale")
            if not isinstance(refs, list):
                refs = []
            refs = [r.strip() for r in refs if isinstance(r, str) and r.strip()]
            refs = list(dict.fromkeys(refs))
            valid_refs = len(refs) == 3 and all(
                r.startswith("[") and r.endswith("]") for r in refs
            )
            rationale = rationale.strip() if isinstance(rationale, str) else ""
            citations = sum(1 for ref in refs if ref in rationale)
            valid = (
                action in ALLOWED_ACTIONS
                and facts is not None
                and valid_refs
                and 60 <= len(rationale) <= 1500
                and citations >= 2
            )
            if not valid:
                cleaned.append(fallback_decision(package, "The model output failed typed-fact or evidence validation."))
            else:
                # Ensure rationale explicitly names the selected action.
                if action not in rationale:
                    rationale = f"{action}: {rationale}"
                cleaned.append({
                    "action": action,
                    "facts": facts,
                    "evidenceRefs": refs,
                    "rationale": rationale[:1500],
                })
        return cleaned
    except Exception:
        return [fallback_decision(p, "The decision service failed validation or was unavailable.") for p in packages]


async def decisions_with_cache(packages):
    results = [None] * len(packages)
    missing_packages = []
    missing_indices = []
    hashes = [digest(p) for p in packages]

    # First read existing semantic decisions.
    if DATABASE_URL:
        with db() as conn:
            for i, key in enumerate(hashes):
                row = conn.execute(
                    "SELECT decision FROM agent_decisions WHERE package_hash=%s",
                    (key,),
                ).fetchone()
                if row:
                    results[i] = row["decision"]

    for i, package in enumerate(packages):
        if results[i] is None:
            missing_indices.append(i)
            missing_packages.append(package)

    # One model request for all uncached packages in this batch.
    if missing_packages:
        new_decisions = await call_model_for_packages(missing_packages)
        for index, decision in zip(missing_indices, new_decisions):
            results[index] = decision
            if DATABASE_URL:
                with db() as conn:
                    conn.execute("""
                        INSERT INTO agent_decisions(package_hash, decision)
                        VALUES (%s, %s::jsonb)
                        ON CONFLICT (package_hash) DO NOTHING
                    """, (hashes[index], json.dumps(decision, ensure_ascii=False)))
                    row = conn.execute(
                        "SELECT decision FROM agent_decisions WHERE package_hash=%s",
                        (hashes[index],),
                    ).fetchone()
                    if row:
                        results[index] = row["decision"]

    return results


async def create_proposals(batch, packages):
    batch_id = str(batch.get("batchId") or batch.get("id") or "")
    if not batch_id:
        raise HTTPException(status_code=400, detail="batchId is required")

    ids = [package_id(p, i) for i, p in enumerate(packages)]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="Package IDs must be unique")

    decisions = await decisions_with_cache(packages)
    proposals = []
    for pkg_id, decision in zip(ids, decisions):
        proposals.append({
            "packageId": pkg_id,
            "actionId": "act_" + uuid.uuid4().hex,
            "action": decision["action"],
            "facts": decision["facts"],
            "evidenceRefs": decision["evidenceRefs"],
            "rationale": decision["rationale"],
        })
    return batch_id, proposals


def check_headers(request, a2a_version):
    if a2a_version != A2A_VERSION:
        raise HTTPException(status_code=400, detail="A2A-Version: 1.0 required")
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != A2A_MIME:
        raise HTTPException(status_code=415, detail="Content-Type must be application/a2a+json")


@app.get("/.well-known/agent-card.json")
async def agent_card():
    base = (PUBLIC_BASE_URL or "https://tds-ga3-invoice-agent-j2qz.onrender.com/").strip()
    if not base.endswith("/"):
        base += "/"
    return JSONResponse({
        "name": APP_NAME,
        "description": "Evidence-based invoice batch action proposal agent.",
        "url": base,
        "version": "1.0.0",
        "supportedInterfaces": [{
            "url": base,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "defaultInputModes": [A2A_MIME, BATCH_MIME, RESULTS_MIME],
        "defaultOutputModes": [A2A_MIME, PROPOSALS_MIME, RECEIPTS_MIME],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Proposes evidence-backed invoice actions and executes only accepted results.",
            "tags": ["invoice", "evidence", "approval"],
        }],
    }, media_type="application/json")


@app.post("/message:send")
async def message_send(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    a2a_version: Optional[str] = Header(default=None, alias="A2A-Version"),
):
    principal = auth_principal(authorization)
    check_headers(request, a2a_version)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    message = extract_message(body)
    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id:
        raise HTTPException(status_code=400, detail="messageId is required")
    if message.get("role") != "ROLE_USER":
        raise HTTPException(status_code=400, detail="role must be ROLE_USER")
    body_hash = digest(message)

    # A transaction-scoped advisory lock serializes same-principal/message retries.
    lock_key = f"{principal}:{message_id}"
    with db() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,))
        old = conn.execute(
            "SELECT body_hash, task_id FROM agent_idempotency WHERE principal=%s AND message_id=%s",
            (principal, message_id),
        ).fetchone()
        if old:
            if old["body_hash"] != body_hash:
                raise HTTPException(status_code=409, detail="IDEMPOTENCY_CONFLICT")
            old_task = load_task_conn(conn, principal, old["task_id"])
            if old_task:
                return json_a2a({"task": public_task(old_task)})

        continuation = find_continuation(message)
        if continuation is not None:
            task_id = message.get("taskId")
            context_id = message.get("contextId")
            if not isinstance(task_id, str) or not task_id:
                raise HTTPException(status_code=400, detail="taskId is required")
            task = load_task_conn(conn, principal, task_id, for_update=True)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            if task.get("contextId") != context_id:
                raise HTTPException(status_code=409, detail="Continuation context mismatch")
            if task_is_terminal(task):
                raise HTTPException(status_code=409, detail="Task is already terminal")
            if task.get("status", {}).get("state") != "TASK_STATE_INPUT_REQUIRED":
                raise HTTPException(status_code=409, detail="Task is not awaiting results")

            private = task.get("_private") or {}
            expected_batch = private.get("batchId")
            if continuation.get("batchId") != expected_batch:
                raise HTTPException(status_code=409, detail="Continuation batchId mismatch")
            results = continuation.get("results")
            if not isinstance(results, list):
                raise HTTPException(status_code=400, detail="results must be a list")

            proposals = private.get("proposals") or []
            by_action_id = {p["actionId"]: p for p in proposals}
            by_package_id = {p["packageId"]: p for p in proposals}
            seen_actions = set()
            seen_packages = set()
            executions = []

            for result in results:
                if not isinstance(result, dict):
                    raise HTTPException(status_code=400, detail="Invalid result entry")
                package_id_value = result.get("packageId")
                action_id = result.get("actionId")
                action = result.get("action")
                outcome = result.get("outcome")
                nonce = result.get("receiptNonce")

                if not all(isinstance(x, str) and x for x in (
                    package_id_value, action_id, action, outcome, nonce
                )):
                    raise HTTPException(status_code=400, detail="Invalid result fields")
                if action_id in seen_actions or package_id_value in seen_packages:
                    raise HTTPException(status_code=409, detail="Duplicate result")
                seen_actions.add(action_id)
                seen_packages.add(package_id_value)

                proposal = by_action_id.get(action_id)
                if not proposal or by_package_id.get(package_id_value) is not proposal:
                    raise HTTPException(status_code=409, detail="Result does not match a stored proposal")
                if proposal["action"] != action:
                    raise HTTPException(status_code=409, detail="Result action mismatch")
                if outcome not in ("ACCEPTED", "REJECTED"):
                    raise HTTPException(status_code=400, detail="Invalid outcome")

                if outcome == "ACCEPTED":
                    executions.append({
                        "packageId": proposal["packageId"],
                        "actionId": proposal["actionId"],
                        "action": proposal["action"],
                        "receiptNonce": nonce,
                        "facts": proposal["facts"],
                        "evidenceRefs": proposal["evidenceRefs"],
                    })

            task.setdefault("history", []).append(message)
            task["artifacts"] = task.get("artifacts", []) + [{
                "artifactId": "receipt_" + uuid.uuid4().hex,
                "parts": [artifact_part(RECEIPTS_MIME, {
                    "batchId": expected_batch,
                    "executions": executions,
                })],
            }]
            task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now_iso()}
            task.pop("_private", None)
            save_task_conn(conn, principal, task)
            conn.execute(
                "INSERT INTO agent_idempotency(principal,message_id,body_hash,task_id) VALUES(%s,%s,%s,%s)",
                (principal, message_id, body_hash, task_id),
            )
            return json_a2a({"task": public_task(task)})

        batch = find_batch(message)
        if batch is None:
            raise HTTPException(status_code=400, detail="No invoice batch found in message parts")
        packages = get_packages(batch)
        if not packages:
            raise HTTPException(status_code=400, detail="Invoice batch has no packages")

        # Keep the advisory lock held while creating and persisting the first Task.
        batch_id, proposals = await create_proposals(batch, packages)
        task_id = str(uuid.uuid4())
        context_id = str(message.get("contextId") or uuid.uuid4())
        task = make_task(
            task_id,
            context_id,
            "input-required",
            history=[message],
            artifacts=[{
                "artifactId": "proposal_" + uuid.uuid4().hex,
                "parts": [artifact_part(PROPOSALS_MIME, {
                    "batchId": batch_id,
                    "proposals": proposals,
                })],
            }],
        )
        task["_private"] = {"batchId": batch_id, "proposals": proposals}
        save_task_conn(conn, principal, task)
        conn.execute(
            "INSERT INTO agent_idempotency(principal,message_id,body_hash,task_id) VALUES(%s,%s,%s,%s)",
            (principal, message_id, body_hash, task_id),
        )
        return json_a2a({"task": public_task(task)})


@app.get("/tasks")
async def list_tasks(authorization: Optional[str] = Header(default=None)):
    principal = auth_principal(authorization)
    with db() as conn:
        rows = conn.execute(
            "SELECT state FROM agent_tasks WHERE principal=%s ORDER BY created_at DESC",
            (principal,),
        ).fetchall()
    return json_a2a({"tasks": [public_task(row["state"]) for row in rows]})


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, authorization: Optional[str] = Header(default=None)):
    principal = auth_principal(authorization)
    with db() as conn:
        task = load_task_conn(conn, principal, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return json_a2a(public_task(task))


@app.post("/tasks/{task_id}:cancel")
async def cancel_task(
    task_id: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    a2a_version: Optional[str] = Header(default=None, alias="A2A-Version"),
):
    principal = auth_principal(authorization)
    if a2a_version != A2A_VERSION:
        raise HTTPException(status_code=400, detail="A2A-Version: 1.0 required")
    with db() as conn:
        task = load_task_conn(conn, principal, task_id, for_update=True)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task_is_terminal(task):
            raise HTTPException(status_code=409, detail="Task is already terminal")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now_iso()}
        task.setdefault("history", []).append({
            "role": "ROLE_AGENT",
            "parts": [text_part("Task canceled; no invoice actions were executed.")],
        })
        task.pop("_private", None)
        save_task_conn(conn, principal, task)
    return json_a2a(public_task(task))


@app.get("/")
async def root():
    return JSONResponse({
        "service": APP_NAME,
        "protocolVersion": A2A_VERSION,
        "health": "ok",
    }, media_type="application/json")
