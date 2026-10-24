from fastapi import FastAPI, HTTPException
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import base64
import hashlib
import json

app = FastAPI()
STORAGE = {}

def compute_input_digest(dossiers) -> str:
    sorted_json = json.dumps(dossiers, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(sorted_json.encode("utf-8")).hexdigest()

@app.post("/")
async def handle_mailroom(payload: dict):
    operation = payload.get("operation")
    if operation == "propose":
        return handle_propose(payload)
    elif operation == "commit":
        return handle_commit(payload)
    else:
        raise HTTPException(status_code=400, detail="Invalid operation")

def handle_propose(data: dict):
    eval_id = data.get("evaluationId")
    dossiers = data.get("dossiers", [])
    verifier = data.get("receiptVerifier")
    input_digest = compute_input_digest(dossiers)
    
    if eval_id in STORAGE:
        if STORAGE[eval_id]["input_digest"] != input_digest:
            raise HTTPException(status_code=409, detail="IDEMPOTENCY_CONFLICT")
        return STORAGE[eval_id]["propose_response"]

    proposals = []
    for dossier in dossiers:
        dossier_id = dossier["dossierId"]
        call_id = hashlib.sha256(f"{eval_id}-{dossier_id}".encode()).hexdigest()[:20]
        
        proposals.append({
            "dossierId": dossier_id,
            "callId": call_id,
            "action": "no_action",
            "target": None,
            "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "ref_123"},
            "evidence": [dossier["sources"][0]["lines"][0]["lineId"]] if dossier["sources"] and dossier["sources"][0]["lines"] else []
        })

    response_body = {
        "profile": "ga5-mailroom-action-gate/v2",
        "evaluationId": eval_id,
        "status": "awaiting_receipts",
        "inputDigest": input_digest,
        "proposals": proposals
    }

    STORAGE[eval_id] = {
        "input_digest": input_digest,
        "verifier": verifier,
        "proposals": {p["callId"]: p for p in proposals},
        "propose_response": response_body
    }
    return response_body

def handle_commit(data: dict):
    eval_id = data.get("evaluationId")
    input_digest = data.get("inputDigest")
    receipts = data.get("receipts", [])

    if eval_id not in STORAGE:
        raise HTTPException(status_code=400, detail="Unknown evaluation ID")
    
    stored = STORAGE[eval_id]
    if stored["input_digest"] != input_digest:
        raise HTTPException(status_code=409, detail="Input digest mismatch")

    pub_key_jwk = stored["verifier"]["publicKeyJwk"]
    outcomes = []
    
    for r in receipts:
        call_id = r["callId"]
        dossier_id = r["dossierId"]
        action = r["action"]
        accepted = r["accepted"]
        receipt_id = r["receiptId"]
        proposal_digest = r["proposalDigest"]
        sig_b64 = r["receiptSignature"]

        inner_receipt = {
            "dossierId": dossier_id, "callId": call_id, "action": action,
            "accepted": accepted, "proposalDigest": proposal_digest, "receiptId": receipt_id
        }
        verification_payload = {
            "profile": "ga5-mailroom-action-gate/v2",
            "evaluationId": eval_id, "inputDigest": input_digest, "receipt": inner_receipt
        }
        
        msg_bytes = json.dumps(verification_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        
        try:
            x_bytes = base64.urlsafe_b64decode(pub_key_jwk["x"] + "===")
            public_key = Ed25519PublicKey.from_public_bytes(x_bytes)
            public_key.verify(base64.b64decode(sig_b64), msg_bytes)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid receipt signature")

        outcomes.append({
            "dossierId": dossier_id, "callId": call_id, "action": action,
            "proposalDigest": proposal_digest, "receiptId": receipt_id,
            "status": "executed" if accepted else "rejected"
        })

    return {
        "profile": "ga5-mailroom-action-gate/v2",
        "evaluationId": eval_id,
        "status": "completed",
        "inputDigest": input_digest,
        "outcomes": outcomes
    }
