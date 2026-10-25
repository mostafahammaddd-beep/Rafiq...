# Security and privacy — v14.6

Scope: `backend/modal_app.py` (WebSocket + HTTP routes), the Android app's networking/storage, repository hygiene.
"Tested" means covered by `backend/dev/test_security.py` (97 checks) against the real handler through a **stub of FastAPI**.
Real Starlette/uvicorn/Modal behaviour was **not** exercised (not installed / no network) — see "Not verified".

## Findings in v14.5 and what changed

| # | Finding in v14.5 (read from the code) | Fix in v14.6 | Tested |
|---|---|---|---|
| 1 | `/ws/live-recitation` accepted **any anonymous caller**; each session burns the owner's CPU-seconds (denial-of-wallet) and runs the model on strangers' audio | HMAC-SHA256 signed, expiring access tokens in the `Authorization: Bearer` header (never the query string). Refused **during the handshake** (before `accept`, before any model work). **Fail closed**: no secret configured ⇒ every session refused | yes |
| 2 | `/debug/last` returned the **last learner's trace** (words, predicted phonemes, timings) to anyone — cross-user data exposure via a global `SESSIONS` deque | Off by default (404). Needs `RAFIQ_DEBUG_ENDPOINTS=1` **and** the admin bearer. When off, traces are not even kept server-side | yes |
| 3 | Unvalidated `sura_idx` / `aya_idx` / `strictness`: `int()` ran outside any `try` (crash after `accept`), impossible verses reached the phonetiser, and `get_ref` **cached** every distinct pair forever (CPU + memory growth from the outside) | Strict ASCII-digit parsing, sura 1–114, aya within that sura's true verse count (table self-checked: sum 6236), strictness whitelist; checked before any model/phonetiser work or caching | yes |
| 4 | Exception text (`ClassName: message`, paths) echoed to clients | Clients get `{code, generic Arabic message, ref}`; details stay in the server log under `ref` | yes |
| 5 | No limits on frame size, control-frame rate, concurrent sessions, or socket lifetime; malformed control JSON (`"duration_ms":"abc"`, a JSON list, `1e99`) could raise | 128 KiB audio-frame cap, 1 KiB control-frame cap, 20 control frames/s, 2 sessions per subject, 16 total, wall-clock cap per socket, all numeric fields clamped | yes |
| 6 | `/health` exposed model id, quantiser details, diagnostics and the last 1500 characters of a **load traceback** | `/health` = `{status, engine, auth}` only; details at `/admin/health` (admin bearer, else 404) | yes |
| 7 | CORS `*`, interactive docs and OpenAPI schema public | No CORS by default (native app needs none; opt-in list, never `*`); docs/OpenAPI disabled | yes |
| 8 | Android: `usesCleartextTraffic="true"`, backups allowed | `usesCleartextTraffic="false"`, `allowBackup="false"` | manifest text only |
| 9 | `run_local.py` published the server through a public tunnel with no protection | Generates a fresh secret per run, prints a 30-day token | not run (needs torch) |

No secret is in the source. A scan of every text file for key/token patterns is part of the suite (the committed
`android/app/debug.keystore` is the conventional Android *debug* key with the public default password; it only signs debug
APKs so that updates install over each other — never ship a Play-Store build signed with it).

## Operating it

```bash
cd backend
python3 tools/issue_token.py --new-secret                      # run twice: signing secret, admin token
modal secret create rafiq-auth RAFIQ_AUTH_SECRET=<secret1> RAFIQ_ADMIN_TOKEN=<secret2>
py -m modal deploy modal_app.py
export RAFIQ_AUTH_SECRET=<secret1>
python3 tools/issue_token.py --sub student-017 --days 30       # give the token to that learner/teacher
```
In the app: ⚙ → رمز الدخول → paste → حفظ. A 401/403 at the handshake shows "invalid or expired token" and stops retrying.
Revocation: rotate `RAFIQ_AUTH_SECRET` (invalidates **all** tokens). There is **no per-token revocation list** and no
login/account system; tokens are bearer credentials stored unencrypted in the app's private preferences. Use short
lifetimes for children's devices.

## Privacy (what the code does today — verified by reading it)

* The server keeps audio **in memory only** (≤ 360 s per socket) and never writes audio, text or results to disk; the
  Android app persists only preferences (server URL, strictness, token) and no audio. Platform-level logging by Modal or
  an intermediary was not reviewed.
* Audio does leave the device (it is streamed to the server over TLS). Upper bound by arithmetic: 16 kHz × 16-bit =
  32 kB/s of speech, ≈ 2.1 MB for 65 s, less with silence suppression (not measured on a network).
* Tokens carry a pseudonymous `sub` — no names are needed.

## Not verified / not done (human work needed)

* Real Starlette behaviour (close-before-accept → HTTP 403, `Request` injection) and real Modal deployment: **BLOCKED**.
* **Consent flow for recording, especially children's voices, a privacy notice, parental-consent mechanism, retention
  statement and data-subject procedures: not implemented.** What the law requires depends on where learners are; get legal
  advice before releasing to minors. The app should show the consent screen before the first microphone use.
* The app must also say, prominently, that automatic Tajweed feedback can be wrong and does not replace a teacher — required
  by the Zipformer licence if it is ever adopted, and the right behaviour regardless. Not added (no Android build to test UI).
* Token storage uses plain private SharedPreferences; EncryptedSharedPreferences/Keystore would add a dependency I could not
  compile-test.
* Rate limiting is per container (state is in memory); several containers each enforce their own caps.
* Reference audio is hot-linked from everyayah.com (`VerseData.kt`); its terms of use were not checked.
