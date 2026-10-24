"""
Run the Rafiq server on YOUR OWN PC ($0, no card).  One-time setup:
    py -m pip install -r requirements-local.txt
Every time:
    window 1:  py run_local.py
    window 2:  cloudflared tunnel --url http://localhost:8000      (free, no account)
Copy the https://....trycloudflare.com address that window 2 prints into the app: ⚙ → رابط الخادم.
(The address changes each time you start the tunnel; the PC must stay on while people use it.)
"""
import os
import sys
import traceback

os.environ.setdefault("RAFIQ_THREADS", str(max(2, (os.cpu_count() or 4) - 1)))

import uvicorn  # noqa: E402

import modal_app as m  # noqa: E402  (works without Modal installed)

print("Loading the model (the first run downloads about 2.3 GB)...")
try:
    m.RUNTIME.ensure_loaded()
except Exception:
    m.RUNTIME.load_error = traceback.format_exc()[-1500:]
    print(m.RUNTIME.load_error)
    sys.exit(1)
print("Ready. Speed test (real-time factor, lower is better):", m.RUNTIME.bench_rtf)
print("Now start the tunnel in a second window:  cloudflared tunnel --url http://localhost:8000")
uvicorn.run(m.build_fastapi_app(m.RUNTIME), **m.uvicorn_kwargs(8000))   # no access log, websocket compression off
