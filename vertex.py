#!/usr/bin/env python3
"""Minimal Vertex AI client: service-account JWT -> access token -> generateContent.

siraj/ shelled out to `gcloud auth print-access-token`. There is no gcloud here and
no google-auth, so the JWT-bearer exchange is done directly with pyjwt. Same result,
one less dependency.

Two hard-won details carried over from siraj/README.md:
  - gemini-3.x 404s on the regional host. Use the GLOBAL endpoint.
  - the model sometimes returns a JSON array despite an object schema; coerce it.
"""
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import jwt

SA_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".secrets/service_account.json"),
)
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_lock = threading.Lock()
_tok = {"v": None, "exp": 0.0}


def _sa():
    with open(SA_PATH, encoding="utf-8") as f:
        return json.load(f)


def project_id():
    return os.environ.get("HTR_PROJECT") or _sa()["project_id"]


def token():
    """Cached OAuth2 access token via the JWT-bearer grant."""
    with _lock:
        if _tok["v"] and time.time() < _tok["exp"] - 60:
            return _tok["v"]
        sa = _sa()
        now = int(time.time())
        assertion = jwt.encode(
            {
                "iss": sa["client_email"],
                "scope": SCOPE,
                "aud": sa["token_uri"],
                "iat": now,
                "exp": now + 3600,
            },
            sa["private_key"],
            algorithm="RS256",
            headers={"kid": sa.get("private_key_id")},
        )
        body = urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        }).encode()
        req = urllib.request.Request(
            sa["token_uri"], data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        _tok["v"] = d["access_token"]
        _tok["exp"] = time.time() + int(d.get("expires_in", 3600))
        return _tok["v"]


def _endpoint(model):
    # gemini-3.x is global-only; 2.5 lives in a region.
    if model.startswith("gemini-3"):
        host, loc = "aiplatform.googleapis.com", "global"
    else:
        loc = os.environ.get("VERTEX_LOCATION", "us-central1")
        host = f"{loc}-aiplatform.googleapis.com"
    return (f"https://{host}/v1/projects/{project_id()}/locations/{loc}"
            f"/publishers/google/models/{model}:generateContent")


REVOKED = (
    "HTR MODEL ACCESS IS OFF BY DEFAULT.\n"
    "\n"
    "This is a hard guard rather than a comment, because a comment only warns a reader who is already\n"
    "looking. Any project using this module for real transcription work will eventually want to revoke\n"
    "model access -- after a paid-API budget cap, after a run of reader artifacts traced to one model,\n"
    "or simply between sessions -- and a guard living in one module is one import away from being\n"
    "bypassed if it is not enforced HERE, at the single call site every reader-script must go through.\n"
    "\n"
    "If your project is append-only over its gold reads (an OCR/HTR candidate table a human later\n"
    "adjudicates), the right discipline is: existing reads stay, because their value includes being the\n"
    "record of what the instrument used to say at the time; what a revocation forbids is ADDING new ones.\n"
    "Whether that applies to your project is your call, not this library's.\n"
    "\n"
    "Set HTR_MODEL_ACCESS=1 to enable. Edit or remove this guard to fit your own project's policy --\n"
    "it is a template, not a fixed rule."
)


def generate(parts, model=None, system=None, max_tokens=8192,
             json_out=False, retries=3, temperature=None, thinking=None):
    """One generateContent call. `parts` is a list of Gemini content parts.

    REFUSES TO RUN unless HTR_MODEL_ACCESS=1. See REVOKED above.

    `thinking`: "minimal"/"low"/"medium"/"high" for gemini-3.x. This matters more
    than it looks -- on a transcription task 3.x spends its output budget on
    thinking tokens and returns finishReason=MAX_TOKENS with a truncated page while
    candidatesTokenCount is still small. A truncated read looks like a clean read
    that simply omitted the tail, so it scores as a false omission (siraj hit the
    same trap and wrote tarzi_fix_trunc.py for it). Always check `finish`.
    """
    if os.environ.get("HTR_MODEL_ACCESS") != "1":
        raise PermissionError(REVOKED)
    # Probed 2026-09-04 on this project: gemini-3.8-flash, gemini-3.6-flash,
    # gemini-3.1-pro-preview and gemini-2.5-flash all resolve. The `-preview`
    # and `gemini-flash-latest` aliases 404.
    model = model or os.environ.get("HTR_MODEL", "gemini-3.8-flash")
    cfg = {"maxOutputTokens": max_tokens}
    if json_out:
        cfg["responseMimeType"] = "application/json"
    if temperature is not None:
        cfg["temperature"] = temperature
    if thinking and model.startswith("gemini-3"):
        cfg["thinkingConfig"] = {"thinkingLevel": thinking}
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    data = json.dumps(body).encode()

    last = None
    for a in range(retries + 1):
        try:
            req = urllib.request.Request(_endpoint(model), data=data, headers={
                "Authorization": f"Bearer {token()}",
                "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            c = (d.get("candidates") or [{}])[0]
            txt = "".join(p.get("text", "")
                          for p in (c.get("content") or {}).get("parts") or [])
            return {"text": txt, "finish": c.get("finishReason"),
                    "usage": d.get("usageMetadata", {})}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
            if e.code in (400, 401, 403, 404):     # not transient; do not retry
                break
        except Exception as e:                      # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(2 ** a, 20))
    return {"error": last}


def image_part(path):
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return {"inlineData": {"mimeType": mime,
                               "data": base64.b64encode(f.read()).decode()}}


def coerce_json(txt):
    """The judge occasionally wraps its object in an array. siraj lost 168 paid
    calls to this; coerce rather than crash."""
    v = json.loads(txt)
    if isinstance(v, list):
        v = next((x for x in v if isinstance(x, dict)), None)
    return v


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else None
    print("project:", project_id())
    print("token:", "ok" if token() else "FAILED")
    r = generate([{"text": "Reply with exactly: pong"}], model=m, max_tokens=32)
    print("probe:", json.dumps(r, ensure_ascii=False)[:400])
