"""
Security tests for the Rafiq backend (v14.6).  Needs only numpy.  Run:  python3 dev/test_security.py

What this verifies: the access-control / validation / limit / error-hygiene LOGIC, by driving the real
`ws_live` coroutine and the real route functions through a stub of FastAPI (FastAPI / Starlette / uvicorn are not
installed in the build sandbox). What it does NOT verify: the behaviour of real Starlette (e.g. that closing before
accept() becomes an HTTP 403 handshake rejection) or of Modal. Those are listed as BLOCKED in the report.
"""
import asyncio, json, os, re, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.pop("RAFIQ_ALLOW_ANONYMOUS", None)
for _k in ("RAFIQ_AUTH_SECRET", "RAFIQ_ADMIN_TOKEN", "RAFIQ_DEBUG_ENDPOINTS", "RAFIQ_CORS_ORIGINS"):
    os.environ.pop(_k, None)

import sim
import modal_app as M

FAILS = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + "security: " + name + ((("   -> " + str(detail)) if (detail and not cond) else "")))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- stub of FastAPI (records routes and constructor kwargs)
APP_KW = {}
MIDDLEWARE = []


class _FastAPI:
    def __init__(self, **kw):
        self.routes = {}
        APP_KW.clear(); APP_KW.update(kw); MIDDLEWARE.clear()

    def add_middleware(self, cls, **k):
        MIDDLEWARE.append((cls, k))

    def __getattr__(self, kind):
        if kind in ("get", "post", "websocket"):
            return lambda path: (lambda f: self.routes.__setitem__((kind, path), f) or f)
        raise AttributeError(kind)


class _JR(dict):
    def __init__(self, x, status_code=200):
        super().__init__(x)
        self.status_code = status_code


fa = types.ModuleType("fastapi"); fa.FastAPI = _FastAPI; fa.WebSocket = object; fa.Request = object
mw = types.ModuleType("fastapi.middleware"); cors = types.ModuleType("fastapi.middleware.cors"); cors.CORSMiddleware = "CORS"
resp = types.ModuleType("fastapi.responses"); resp.JSONResponse = _JR
sys.modules.update({"fastapi": fa, "fastapi.middleware": mw, "fastapi.middleware.cors": cors, "fastapi.responses": resp})

SECRET = "s" * 40
ADMIN = "a" * 40
ref = sim.build_ref()
PAUSES = {6, 11, 18, 25, 31, 39, 43, 46}
BASE = [ref.waqf[k] if (k in PAUSES and ref.waqf[k]) else ref.wasl[k] for k in range(ref.n)]


class FakeRuntime:
    int8 = False
    device = "cpu"

    def __init__(self, speech, status=None, fail_recognize=None, fail_ref=None):
        self.speech = speech
        self.ab = ((speech.audio * 32767).astype("<i2").astype(np.float32) / 32768.0).astype(np.float32).tobytes()
        self.calls = {"ensure_loaded": 0, "get_ref": 0, "recognize": 0}
        self._status = status or {"status": "ok"}
        self.fail_recognize, self.fail_ref = fail_recognize, fail_ref

    def ensure_loaded(self):
        self.calls["ensure_loaded"] += 1

    def get_ref(self, s, a):
        self.calls["get_ref"] += 1
        if self.fail_ref:
            raise self.fail_ref
        return ref

    def status(self):
        return dict(self._status)

    def recognize(self, wave, ref_):
        self.calls["recognize"] += 1
        if self.fail_recognize:
            raise self.fail_recognize
        off = self.ab.find(wave[:800].tobytes()) // 4
        sel = [ch for (tm, ch, _w) in self.speech.chars if off / M.SR <= tm < (off + len(wave)) / M.SR]
        return M.Recognition("".join(sel), [0.97] * len(sel), None)


class FakeWS:
    def __init__(self, qp=None, headers=None):
        self.query_params = qp if qp is not None else {"sura_idx": "2", "aya_idx": "255", "strictness": "balanced"}
        self.headers = headers if headers is not None else {}
        self.q = asyncio.Queue(); self.sent = []; self.closed = False; self.accepted = False; self.close_code = None

    async def accept(self):
        self.accepted = True

    async def send_text(self, t):
        self.sent.append(json.loads(t))

    async def receive(self):
        return await self.q.get()

    async def close(self, code=1000):
        if self.close_code is None:
            self.close_code = code
        if not self.closed:
            self.closed = True
            await self.q.put({"type": "websocket.disconnect"})

    def types(self):
        return [m.get("type") for m in self.sent]


SPEECH = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)


def pcm(t0, t1):
    return (SPEECH.audio[int(t0 * M.SR): int(t1 * M.SR)] * 32767).astype("<i2").tobytes()


def tok(sub="alice", ttl=3600, secret=SECRET, **kw):
    return M.issue_token(secret, sub, ttl, **kw)


def auth(sub="alice"):
    return {"authorization": "Bearer " + tok(sub)}


def handler_of(rt):
    app = M.build_fastapi_app(rt)
    return app, app.routes[("websocket", "/ws/live-recitation")]


async def short_session(handler, ws, seconds=2.5):
    task = asyncio.create_task(handler(ws))
    t = 0.0
    while t < seconds and not ws.closed:
        await ws.q.put({"type": "websocket.receive", "bytes": pcm(t, t + 0.1)})
        t += 0.1
        await asyncio.sleep(0.003)
    if not ws.closed:
        await ws.q.put({"type": "websocket.receive", "text": json.dumps({"type": "stop"})})
    await asyncio.wait_for(task, 60)
    return ws


# ============================================================================== 1. tokens (pure functions)
t0 = 1_800_000_000
good = M.issue_token(SECRET, "alice", 3600, now=t0)
claims = M.verify_token(SECRET, good, now=t0 + 10)
check("token round trip returns subject, scope and expiry", claims["sub"] == "alice" and claims["scope"] == "live" and claims["exp"] == t0 + 3600, claims)


def reason(token, secret=SECRET, now=t0 + 10, scope="live"):
    try:
        M.verify_token(secret, token, now=now, scope=scope)
        return "accepted"
    except M.AuthError as e:
        return e.reason
    except Exception as e:  # anything else is a bug: untrusted input must only ever raise AuthError
        return "CRASH:" + type(e).__name__


p = good.split(".")
flip = lambda s: s[:-1] + ("A" if s[-1] != "A" else "B")
check("tampered signature is rejected", reason(".".join([p[0], p[1], flip(p[2])])) == "signature")
forged = M.issue_token(SECRET, "bob", 3600, now=t0).split(".")
check("claims of another subject re-signed with a different key are rejected (forgery)", reason(".".join([forged[0], forged[1], p[2]])) == "signature")
check("token signed with another secret is rejected", reason(M.issue_token("x" * 40, "alice", 3600, now=t0)) == "signature")
check("expired token is rejected (beyond the 60 s clock-skew allowance)", reason(good, now=t0 + 3600 + 61) == "expired")
check("a token that expired seconds ago is still tolerated (clock skew)", reason(good, now=t0 + 3600 + 30) == "accepted")
check("a token issued far in the future is rejected", reason(M.issue_token(SECRET, "alice", 3600, now=t0 + 10_000)) == "malformed")
check("scope mismatch is rejected", reason(M.issue_token(SECRET, "alice", 3600, now=t0, scope="admin")) == "scope")
bad_inputs = ["", "x", "a.b", "rq1..", "rq1.a.", "rq1..sig", "rq1.e30.AAAA", "rq2." + p[1] + "." + p[2], "rq1." + p[1],
              good + ".extra", "rq1.é." + p[2], "A" * 600, None, 12345, b"rq1.a.b", "rq1.%s.%s" % ("A" * 10, "B" * 10)]
outs = [reason(b) for b in bad_inputs]
check("malformed / hostile tokens only ever produce a clean rejection (never an exception, never accepted)",
      all(o in ("malformed", "signature") for o in outs), outs)
check("an unsigned token (empty signature, 'alg none' style) is rejected", reason("rq1." + p[1] + ".") in ("signature", "malformed"))
junk_body = M._b64e(b"{not json")
check("the signature is checked BEFORE the claims are parsed (garbage body with a wrong signature -> 'signature', not 'malformed')",
      reason("rq1." + junk_body + "." + p[2]) == "signature")
signed_junk = "rq1." + junk_body + "." + M._sign(SECRET, junk_body)
check("a correctly signed but non-JSON body is rejected as malformed", reason(signed_junk) == "malformed")
for sub in ("../etc", "a b", "a\nb", "é", "", "x" * 65, "a;b", "a/b"):
    try:
        M.issue_token(SECRET, sub, 60); ok = False
    except ValueError:
        ok = True
    check("subject %r cannot be issued (path / header / log injection)" % sub, ok)
for ttl in (0, -5, M.MAX_TOKEN_TTL_S + 1):
    try:
        M.issue_token(SECRET, "alice", ttl); ok = False
    except ValueError:
        ok = True
    check("ttl %d is refused" % ttl, ok)
try:
    M.issue_token("short", "alice", 60); ok = False
except ValueError:
    ok = True
check("a secret shorter than 32 characters cannot sign tokens", ok)

B = M.bearer_token
check("bearer parsing: 'Bearer x' / 'bearer x' accepted, other schemes / missing / non-strings ignored",
      B({"authorization": "Bearer abc"}) == "abc" and B({"authorization": "bearer abc"}) == "abc" and B({"authorization": "Basic abc"}) is None
      and B({}) is None and B({"authorization": "Bearer "}) is None and B(None) is None and B({"authorization": 5}) is None)

# ============================================================================== 2. policy (fail closed)
os.environ.pop("RAFIQ_AUTH_SECRET", None); os.environ.pop("RAFIQ_ALLOW_ANONYMOUS", None)
check("no secret configured and no anonymous switch -> refuse everything (fail closed)", M.authenticate({"authorization": "Bearer " + good}) == (None, "auth_not_configured") and M.AccessPolicy.mode() == "closed")
os.environ["RAFIQ_AUTH_SECRET"] = "short"
check("a too-short configured secret counts as 'not configured' (still closed)", M.AccessPolicy.secret() is None and M.AccessPolicy.mode() == "closed")
os.environ["RAFIQ_AUTH_SECRET"] = SECRET; os.environ["RAFIQ_ALLOW_ANONYMOUS"] = "1"
check("the anonymous switch is ignored once a secret is set (a token is still required)",
      M.authenticate({}) == (None, "missing_token") and M.AccessPolicy.mode() == "token")
check("valid token authenticates and yields its subject", M.authenticate({"authorization": "Bearer " + tok("carol")}) == ("carol", "ok"))
check("an expired token is refused by authenticate()", M.authenticate({"authorization": "Bearer " + M.issue_token(SECRET, "carol", 5, now=1000)})[0] is None)
os.environ.pop("RAFIQ_AUTH_SECRET"); os.environ["RAFIQ_ALLOW_ANONYMOUS"] = "1"
check("explicit local-dev anonymous mode works only without a secret", M.authenticate({}) == ("anonymous", "anonymous") and M.AccessPolicy.mode() == "anonymous")
os.environ.pop("RAFIQ_ALLOW_ANONYMOUS")
os.environ["RAFIQ_AUTH_SECRET"] = SECRET

# ============================================================================== 3. request validation
check("verse table: 114 suras, 6236 verses, spot checks (2:286, 3:200, 9:129, 112:4, 114:6)",
      len(M.AYA_COUNTS) == 114 and sum(M.AYA_COUNTS) == 6236 and M.AYA_COUNTS[1] == 286 and M.AYA_COUNTS[2] == 200
      and M.AYA_COUNTS[8] == 129 and M.AYA_COUNTS[111] == 4 and M.AYA_COUNTS[113] == 6, (len(M.AYA_COUNTS), sum(M.AYA_COUNTS)))
P = lambda **q: M.parse_session_params(q)
check("defaults to 2:255 balanced", P() == ({"sura": 2, "aya": 255, "strictness": "balanced"}, "ok"))
check("last verse of a sura is accepted, one past it is not", P(sura_idx="2", aya_idx="286")[0] and P(sura_idx="2", aya_idx="287")[0] is None
      and P(sura_idx="112", aya_idx="4")[0] and P(sura_idx="112", aya_idx="5")[0] is None)
for bad in ("0", "115", "-1", "abc", "1e3", "２", "١", "2 ; drop", "", "9" * 50, "0x10", "2.5", "+2", "٢٥٥"):
    check("sura_idx %r is rejected" % bad, P(sura_idx=bad)[0] is None)
for bad in ("0", "-3", "abc", "٢٥٥", "255; x", "1" * 20):
    check("aya_idx %r is rejected" % bad, P(sura_idx="2", aya_idx=bad)[0] is None)
check("strictness is a whitelist (unknown / injection strings rejected, case-insensitive match allowed)",
      P(strictness="STRICT")[0]["strictness"] == "strict" and all(P(strictness=s)[0] is None for s in ("ultra", "../x", "balanced;", "")))

# ============================================================================== 4. WebSocket handler
async def ws_tests():
    # --- unauthenticated: refused during the handshake, no model work at all
    rt = FakeRuntime(SPEECH); app, h = handler_of(rt)
    ws = FakeWS(); await asyncio.wait_for(h(ws), 5)
    check("no token -> closed with 1008 BEFORE accept; the model is never touched",
          ws.close_code == 1008 and not ws.accepted and not ws.sent and rt.calls == {"ensure_loaded": 0, "get_ref": 0, "recognize": 0}, (ws.close_code, ws.accepted, rt.calls))
    ws = FakeWS(headers={"authorization": "Bearer " + flip(tok())}); await asyncio.wait_for(h(ws), 5)
    check("forged token -> refused before accept", ws.close_code == 1008 and not ws.accepted and rt.calls["ensure_loaded"] == 0)
    ws = FakeWS(qp={"sura_idx": "2", "aya_idx": "255", "strictness": "balanced", "token": tok()}); await asyncio.wait_for(h(ws), 5)
    check("a token in the query string is NOT honoured (it would be written to proxy logs)", ws.close_code == 1008 and not ws.accepted)
    ws = FakeWS(headers={"authorization": "Bearer " + M.issue_token(SECRET, "alice", 5, now=1000)}); await asyncio.wait_for(h(ws), 5)
    check("expired token -> refused before accept", ws.close_code == 1008 and not ws.accepted)
    os.environ.pop("RAFIQ_AUTH_SECRET")
    ws = FakeWS(headers=auth()); await asyncio.wait_for(h(ws), 5)
    check("server with no secret configured refuses even a well-formed token (fail closed)", ws.close_code == 1008 and not ws.accepted)
    os.environ["RAFIQ_AUTH_SECRET"] = SECRET

    # --- authorised happy path
    rt = FakeRuntime(SPEECH); app, h = handler_of(rt)
    ws = await short_session(h, FakeWS(headers=auth()), 2.5)
    check("valid token -> accepted, 'ready' first, session ends with a final and a normal close",
          ws.accepted and ws.types()[0] == "ready" and "final" in ws.types() and "error" not in ws.types() and ws.close_code == 1000, ws.types()[:6])

    # --- invalid parameters
    rt = FakeRuntime(SPEECH); app, h = handler_of(rt)
    for qp in ({"sura_idx": "999"}, {"sura_idx": "2", "aya_idx": "9999"}, {"sura_idx": "x"}, {"strictness": "nope"}):
        ws = FakeWS(qp=qp, headers=auth()); await asyncio.wait_for(h(ws), 5)
        e = ws.sent[0] if ws.sent else {}
        check("bad query %s -> error code bad_request, close 1008, nothing cached or computed" % qp,
              e.get("code") == "bad_request" and ws.close_code == 1008 and rt.calls["get_ref"] == 0 and rt.calls["ensure_loaded"] == 0, (e, rt.calls))

    # --- per-subject concurrency cap (held-open sockets), release on disconnect
    rt = FakeRuntime(SPEECH); app, h = handler_of(rt)
    a1, a2, a3 = FakeWS(headers=auth("alice")), FakeWS(headers=auth("alice")), FakeWS(headers=auth("alice"))
    t1, t2 = asyncio.create_task(h(a1)), asyncio.create_task(h(a2))
    await asyncio.sleep(0.4)
    await asyncio.wait_for(h(a3), 5)
    check("third concurrent session of one subject is refused (too_many_sessions, 1013)",
          a1.types()[:1] == ["ready"] and a2.types()[:1] == ["ready"] and a3.sent[0].get("code") == "too_many_sessions" and a3.close_code == 1013, [x.sent[:1] for x in (a1, a2, a3)])
    b1 = FakeWS(headers=auth("bob")); tb = asyncio.create_task(h(b1)); await asyncio.sleep(0.3)
    check("another subject is unaffected by that cap", b1.types()[:1] == ["ready"])
    await a1.q.put({"type": "websocket.disconnect"}); await asyncio.wait_for(t1, 10)
    a4 = FakeWS(headers=auth("alice")); t4 = asyncio.create_task(h(a4)); await asyncio.sleep(0.3)
    check("the slot is released after an abrupt disconnect", a4.types()[:1] == ["ready"], a4.sent[:1])
    for w, t in ((a2, t2), (b1, tb), (a4, t4)):
        await w.q.put({"type": "websocket.disconnect"}); await asyncio.wait_for(t, 10)

    # --- total cap
    rt = FakeRuntime(SPEECH); app, h = handler_of(rt)
    old_total = M.MAX_SESSIONS_TOTAL
    M.MAX_SESSIONS_TOTAL = 1
    app, h = handler_of(rt)          # limiter reads the constant at construction time
    x1, x2 = FakeWS(headers=auth("u1")), FakeWS(headers=auth("u2"))
    tx = asyncio.create_task(h(x1)); await asyncio.sleep(0.3); await asyncio.wait_for(h(x2), 5)
    check("global session cap: the next caller gets 'busy' (1013)", x2.sent and x2.sent[0].get("code") == "busy" and x2.close_code == 1013)
    await x1.q.put({"type": "websocket.disconnect"}); await asyncio.wait_for(tx, 10)
    M.MAX_SESSIONS_TOTAL = old_total

    # --- frame / rate limits
    rt = FakeRuntime(SPEECH); app, h = handler_of(rt)
    ws = FakeWS(headers=auth()); task = asyncio.create_task(h(ws)); await asyncio.sleep(0.2)
    await ws.q.put({"type": "websocket.receive", "bytes": b"\x00" * (M.MAX_WS_BINARY_BYTES + 2)}); await asyncio.wait_for(task, 10)
    check("oversized audio frame -> too_large, close 1009, no final", any(m.get("code") == "too_large" for m in ws.sent) and ws.close_code == 1009 and "final" not in ws.types(), (ws.types(), ws.close_code))
    ws = FakeWS(headers=auth()); task = asyncio.create_task(h(ws)); await asyncio.sleep(0.2)
    await ws.q.put({"type": "websocket.receive", "text": "x" * (M.MAX_WS_TEXT_BYTES + 1)}); await asyncio.wait_for(task, 10)
    check("oversized control frame -> too_large, close 1009", any(m.get("code") == "too_large" for m in ws.sent) and ws.close_code == 1009)
    ws = FakeWS(headers=auth()); task = asyncio.create_task(h(ws)); await asyncio.sleep(0.2)
    for _ in range(M.MAX_CONTROL_PER_S + 5):
        await ws.q.put({"type": "websocket.receive", "text": json.dumps({"type": "pause", "duration_ms": 600})})
    await asyncio.wait_for(task, 10)
    check("control-frame flood -> rate_limited, close 1008", any(m.get("code") == "rate_limited" for m in ws.sent) and ws.close_code == 1008, (ws.types()[-3:], ws.close_code))

    # --- hostile but small control payloads must not crash the session
    rt = FakeRuntime(SPEECH); app, h = handler_of(rt)
    ws = FakeWS(headers=auth()); task = asyncio.create_task(h(ws)); await asyncio.sleep(0.2)
    for payload in ('{"type":"pause","duration_ms":"abc"}', "[1,2,3]", '{"type":"resume","word_idx":1e99}', '{"type":"resume","word_idx":null}',
                    '{"type":"pause","duration_ms":-5,"speech_tail_padding_ms":{}}', "null", '"stop"', "{", '{"type":["stop"]}'):
        await ws.q.put({"type": "websocket.receive", "text": payload})
    for t_ in (0.0, 0.1, 0.2, 0.3):
        await ws.q.put({"type": "websocket.receive", "bytes": pcm(t_, t_ + 0.1)})
    await ws.q.put({"type": "websocket.receive", "text": json.dumps({"type": "stop"})}); await asyncio.wait_for(task, 30)
    check("malformed control payloads never raise: no error frame, session ends with a final", "error" not in ws.types() and "final" in ws.types(), ws.types())

    # --- error hygiene: nothing from an exception reaches the client
    leak = "TOP-SECRET /srv/model/path token=abc123"
    rt = FakeRuntime(SPEECH, fail_recognize=RuntimeError(leak)); app, h = handler_of(rt)
    ws = await short_session(h, FakeWS(headers=auth()), 2.0)
    blob = json.dumps(ws.sent, ensure_ascii=False)
    errs = [m for m in ws.sent if m.get("type") == "error"]
    check("an inference exception is reported only as a generic code + reference id (no class name, text or path)",
          errs and all(e.get("code") == "internal" and e.get("ref") for e in errs) and "TOP-SECRET" not in blob and "RuntimeError" not in blob and "/srv" not in blob, errs[:1])
    rt = FakeRuntime(SPEECH, fail_ref=ValueError(leak)); app, h = handler_of(rt)
    ws = FakeWS(headers=auth()); await asyncio.wait_for(h(ws), 10)
    blob = json.dumps(ws.sent, ensure_ascii=False)
    check("a failure while building the session is also generic (and the slot is released)",
          ws.sent and ws.sent[0].get("code") == "internal" and "TOP-SECRET" not in blob and "ValueError" not in blob, ws.sent[:1])

    # --- wall-clock cap: a trickle of tiny frames cannot hold a slot forever
    rt = FakeRuntime(SPEECH); app, h = handler_of(rt)
    old_wall = M.MAX_SOCKET_WALL_S; M.MAX_SOCKET_WALL_S = 0.6
    ws = FakeWS(headers=auth()); task = asyncio.create_task(h(ws)); t_start = asyncio.get_event_loop().time()
    while not task.done() and asyncio.get_event_loop().time() - t_start < 8:
        await ws.q.put({"type": "websocket.receive", "bytes": b"\x00\x00" * 160}); await asyncio.sleep(0.1)
    ended = task.done(); await asyncio.sleep(0)
    if not ended:
        task.cancel()
    M.MAX_SOCKET_WALL_S = old_wall
    check("a socket that keeps sending tiny frames is ended by the wall-clock cap", ended and ws.closed, (ended, ws.types()[-2:]))

asyncio.run(ws_tests())

# ============================================================================== 5. HTTP routes: health / admin / debug
async def route_tests():
    detail = {"status": "ok", "engine": M.ENGINE_NAME, "model": "obadx/secret-model-id", "diag": {"x": 1}, "load_error": "Traceback /srv/app.py line 3", "quant": {"q": 1}}
    rt = FakeRuntime(SPEECH, status=detail)
    app = M.build_fastapi_app(rt); R = app.routes
    Req = lambda h=None: types.SimpleNamespace(headers=h or {})
    pub = await R[("get", "/health")]()
    check("/health is public but minimal: status, engine, auth mode only (no model id, diag, traceback)",
          set(pub) == {"status", "engine", "auth"} and "Traceback" not in json.dumps(pub) and "secret-model-id" not in json.dumps(pub), pub)
    os.environ["RAFIQ_ADMIN_TOKEN"] = ADMIN
    r = await R[("get", "/admin/health")](Req())
    check("/admin/health without the admin bearer answers 404 (does not advertise itself)", getattr(r, "status_code", 0) == 404)
    r = await R[("get", "/admin/health")](Req({"authorization": "Bearer " + "b" * 40}))
    check("/admin/health with a wrong bearer answers 404", getattr(r, "status_code", 0) == 404)
    r = await R[("get", "/admin/health")](Req({"authorization": "Bearer " + ADMIN}))
    check("/admin/health with the admin bearer returns the detailed status", getattr(r, "status_code", 0) == 200 and r.get("diag") == {"x": 1})
    check("a user access token is not an admin credential", getattr(await R[("get", "/admin/health")](Req(auth())), "status_code", 0) == 404)

    # a session has run -> its trace must not be reachable
    before = len(M.SESSIONS)
    h = R[("websocket", "/ws/live-recitation")]
    ws = await short_session(h, FakeWS(headers=auth("victim")), 2.0)
    check("a finished session leaves no trace in the global debug store when debug is off", len(M.SESSIONS) == before)
    r = await R[("get", "/debug/last")](Req({"authorization": "Bearer " + ADMIN}), 30)
    check("/debug/last answers 404 by default, even with the admin bearer (switch is off)", getattr(r, "status_code", 0) == 404)
    r = await R[("get", "/debug/last")](Req(), 30)
    check("/debug/last answers 404 to anonymous callers", getattr(r, "status_code", 0) == 404)
    r = await R[("get", "/debug/ref")](Req({"authorization": "Bearer " + ADMIN}), 2, 255)
    check("/debug/ref answers 404 by default", getattr(r, "status_code", 0) == 404)
    check("the final message to the learner carries no server internals while debug is off",
          "diag" not in json.dumps([m for m in ws.sent if m.get("type") == "final"]) and "secret-model-id" not in json.dumps(ws.sent))

    os.environ["RAFIQ_DEBUG_ENDPOINTS"] = "1"
    r = await R[("get", "/debug/last")](Req(), 30)
    check("debug switch on but no bearer -> still 404", getattr(r, "status_code", 0) == 404)
    r = await R[("get", "/debug/last")](Req(auth()), 30)
    check("debug switch on, a normal user token -> still 404", getattr(r, "status_code", 0) == 404)
    ws = await short_session(h, FakeWS(headers=auth("victim")), 2.0)
    r = await R[("get", "/debug/last")](Req({"authorization": "Bearer " + ADMIN}), 30)
    check("debug switch on + admin bearer -> the trace is returned", getattr(r, "status_code", 0) == 200 and ("steps" in r or "info" in r), list(r)[:5])
    n_ref = rt.calls["get_ref"]
    r = await R[("get", "/debug/ref")](Req({"authorization": "Bearer " + ADMIN}), 999, 1)
    check("/debug/ref validates the verse (422) before any phonetiser work or caching", getattr(r, "status_code", 0) == 422 and rt.calls["get_ref"] == n_ref, r)
    os.environ.pop("RAFIQ_ADMIN_TOKEN")
    r = await R[("get", "/debug/last")](Req({"authorization": "Bearer " + ADMIN}), 30)
    check("debug switch on but NO admin token configured -> 404 (never open by default)", getattr(r, "status_code", 0) == 404)
    os.environ.pop("RAFIQ_DEBUG_ENDPOINTS")

    check("interactive docs / OpenAPI schema are disabled", APP_KW.get("docs_url") is None and APP_KW.get("redoc_url") is None and APP_KW.get("openapi_url") is None, APP_KW)
    check("no CORS middleware by default (native app needs none)", MIDDLEWARE == [], MIDDLEWARE)
    os.environ["RAFIQ_CORS_ORIGINS"] = "*, https://app.example.org"
    M.build_fastapi_app(rt)
    check("CORS is opt-in, never the wildcard", len(MIDDLEWARE) == 1 and MIDDLEWARE[0][1]["allow_origins"] == ["https://app.example.org"], MIDDLEWARE)
    os.environ.pop("RAFIQ_CORS_ORIGINS")

asyncio.run(route_tests())

# ============================================================================== 6. repository hygiene
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
pat = re.compile(r"(AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|xox[abp]-[A-Za-z0-9-]{10,}|(?i:secret|password|api[_-]?key|token)\s*[:=]\s*[\"'](?![A-Z0-9_]+[\"'])[A-Za-z0-9+/_\-]{20,}[\"'])")
hits = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in (".git", "build", ".gradle", "__pycache__")]
    for f in fn:
        if f.endswith((".keystore", ".jks", ".png", ".zip", ".onnx", ".pyc")):
            continue
        fp = os.path.join(dp, f)
        try:
            txt = open(fp, encoding="utf-8").read()
        except Exception:
            continue
        for m in pat.finditer(txt):
            line = txt[: m.start()].count("\n") + 1
            if os.path.relpath(fp, ROOT).replace("\\", "/") == "backend/dev/test_security.py":
                continue                     # this file builds fake secrets on purpose
            hits.append((os.path.relpath(fp, ROOT), line, m.group(0)[:30]))
check("no hard-coded secrets / private keys / tokens in any text file of the project", not hits, hits)
mf = open(os.path.join(ROOT, "android", "app", "src", "main", "AndroidManifest.xml"), encoding="utf-8").read()
check("Android: cleartext (http/ws) traffic is not allowed", 'usesCleartextTraffic="false"' in mf, "usesCleartextTraffic")
check("Android: app data is excluded from cloud backup / adb backup", 'allowBackup="false"' in mf)

print("\n%d security check(s) failed" % len(FAILS) if FAILS else "\nSECURITY CHECKS PASSED")
sys.exit(1 if FAILS else 0)
