"""
Run the Rafiq server on YOUR OWN PC ($0, no card).  One-time setup:
    py -m pip install -r requirements-local.txt
Every time:
    window 1:  py run_local.py
    window 2:  cloudflared tunnel --url http://localhost:8000      (free, no account)
Copy the https://....trycloudflare.com address that window 2 prints into the app: ⚙ → رابط الخادم,
and the access token window 1 prints into ⚙ → رمز الدخول.
(The address changes each time you start the tunnel; the PC must stay on while people use it.)
"""
import os
import secrets
import sys
import traceback

os.environ.setdefault("RAFIQ_THREADS", str(max(2, (os.cpu_count() or 4) - 1)))
# v14.6: the tunnel makes this server reachable from the internet, so it is NOT left open. Unless you provide your own
# RAFIQ_AUTH_SECRET, a fresh one is generated for this run and a 30-day token is printed for you to paste in the app.
_GENERATED_SECRET = not os.environ.get("RAFIQ_AUTH_SECRET", "").strip()
if _GENERATED_SECRET:
    os.environ["RAFIQ_AUTH_SECRET"] = secrets.token_urlsafe(48)

import uvicorn  # noqa: E402

import modal_app as m  # noqa: E402  (works without Modal installed)

print("Loading the model (the first run downloads about 2.3 GB)...")
try:
    m.RUNTIME.ensure_loaded()
except Exception:
    m.RUNTIME.load_error = traceback.format_exc()[-1500:]
    print(m.RUNTIME.load_error)
    sys.exit(1)
print("Ready (precision: %s). Speed test (real-time factor, lower is better):" % ("int8" if m.RUNTIME.int8 else "fp32"), m.RUNTIME.measure_rtf())
print("Now start the tunnel in a second window:  cloudflared tunnel --url http://localhost:8000")
if _GENERATED_SECRET:
    print("Access token for the app (⚙ -> رمز الدخول), valid 30 days, changes at every restart of this script:")
    print("   " + m.issue_token(os.environ["RAFIQ_AUTH_SECRET"], "local", 30 * 86400))
    print("(Set RAFIQ_AUTH_SECRET yourself to keep the same tokens across restarts.)")
uvicorn.run(m.build_fastapi_app(m.RUNTIME), **m.uvicorn_kwargs(8000))   # no access log, websocket compression off
