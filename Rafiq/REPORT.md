# Rafiq v14.6 — engineering report

Date: 2026-10-25. Base: `Rafiq-v14.5.zip` (22 files). This report separates **implemented**, **verified**, **proposed** and
**blocked**. Nothing here was measured on a phone, on real recordings, or with the real neural model.

## 1. Bottom line
* The brief's three goals cannot all be met with what is legally and technically available today (details:
  `docs/ARCHITECTURE_DECISION.md`). The one on-device-capable, streaming Tajweed-oriented model found (Quran-Lab Zipformer
  v3 / v3.1, 65.5 M parameters) is licensed **NPL-1.2 — never charge for any feature it powers**, is gated, is phoneme-only,
  and its own card says it transcribes the rule rather than the deviation and is 2–3× worse on children. The MIT-licensed
  Muaalem v3.2 (~600 M parameters, sifat heads, 75.8 % Tajweed F1 on a 159-clip single-verse benchmark) cannot run on-device
  or stream statefully.
* **Decision:** keep the existing architecture (reference-conditioned CTC posteriors → alignment → explicit judge →
  abstention, Muaalem v3.2 on cloud CPU) and harden it. Zipformer stays a *benchmark* candidate with written adoption conditions.
  TajweedFormer could not be found.
* The most valuable verified work this session: closing real security holes (an open, billable WebSocket; a public
  endpoint leaking the previous learner's trace; unvalidated input cached forever) and building an evaluation toolkit that
  refuses to call a target met without enough samples.
* **No accuracy target is verified. No training was run. No Android build was run. No stateful streaming was implemented.**

## 2. Implemented (and how it was checked)
| Area | Change | Evidence |
|---|---|---|
| Auth | HMAC-signed expiring tokens, handshake-time refusal, fail-closed, admin bearer | `dev/test_security.py` |
| Input safety | strict verse validation (6236-verse table), frame / rate / session / wall-clock limits, clamped control fields | same |
| Info leaks | `/health` minimal, `/admin/health`, `/debug/*` off by default and admin-only, generic client errors, docs/CORS off | same |
| Ops | `tools/issue_token.py`; `run_local.py` self-generates a secret; Modal secret attached from `rafiq-auth` | CLI smoke-run; config check with a fake Modal |
| Android | token field + `Authorization` header, 401/403 → clear message and no retry loop, cleartext off, backup off, versionCode 19 | **static** Kotlin check only |
| Evaluation | `eval/eval_stats.py`, `run_eval.py`, `bench_engine_overhead.py` | `eval/test_eval_stats.py` |
| Docs | architecture decision, security/privacy, licences, low-end devices, evaluation protocol | — |
| Judge / engine | **unchanged** | `mutation_test.py` output byte-identical to v14.5 |

## 3. Test and build results
| Item | Result |
|---|---|
| `python3 backend/dev/run_all_tests.py` | **PASS** — 347 checks, 0 failed: 216 original engine checks (simulator, not real audio) + 1 Modal-secret check + 2 suite wrappers + 97 security + 31 evaluation-statistics |
| `python3 backend/dev/test_security.py` | **PASS** 97/97 |
| `python3 backend/eval/test_eval_stats.py` | **PASS** 31/31 (verifies the maths with synthetic numbers; says nothing about any model) |
| `backend/dev/mutation_test.py` | **PASS / unchanged** vs v14.5 (simulator) |
| `backend/dev/static_check_kt.py` | **PASS** (a custom static check — brace balance, member existence, JSON-key parity — **not a compiler**) |
| `backend/dev/cost_proof.py` | runs; its own verdict is **FAIL** against the project's cost target (≈ $0.00145 per 65 s session modelled vs $0.00025 target; the INT8 ×1.8 speed-up and 3 s snapshot restore are assumptions) — unchanged from v14.5 |
| Gradle / APK build (`assembleDebug`) | **BLOCKED** — no Android SDK / Gradle / network in the sandbox. Kotlin edits are small and mirror existing patterns, but **compile before trusting them** (the repo's GitHub workflow does this) |
| Real FastAPI / Starlette / uvicorn behaviour | **BLOCKED** — not installed; tests use a stub (e.g. "close before accept ⇒ HTTP 403" is assumed, standard but unverified here) |
| `modal deploy` | **BLOCKED** |
| Model loading / inference with real weights | **BLOCKED** — no weights, no network to model hosts, no GPU |
| Tests on Android devices (low / mid / high) | **BLOCKED** — no devices |

## 4. Benchmarks actually measured
Only one, and it is deliberately narrow: **Rafiq's own non-neural engine code** (alignment, judge, gate) driven by the
simulator's fake recogniser — `python3 backend/eval/bench_engine_overhead.py --sessions 12`:

| Item | Value |
|---|---|
| Hardware | Intel Xeon (cloud VM), **1 core**, 4 GB, Python 3.12.3, numpy 2.4.4 — **not a phone** |
| Sessions / steps | 12 simulated sessions (37.9 s audio mean), 643 `step()` calls |
| `step()` wall time | mean 0.61 ms, p50 0.60 ms, p95 1.01 ms, max 2.88 ms |
| Engine-only RTF | p50 0.0009, p95 0.0010, max 0.0010 (n = 12 — below the 20-span rule, so the harness itself marks it UNVERIFIED as a release figure) |
| Peak RSS | 105 MiB (whole process including numpy) |
| Meaning | outside the neural model the engine costs about a thousandth of real time on a server core; **the model is the budget**. This says nothing about model latency, phone performance, battery, thermals or accuracy |

Not measured (BLOCKED): model latency / RTF, peak RAM of the model, load time, INT8-vs-fp32 accuracy, chunk-size trade-offs,
CPU/XNNPACK/QNN, battery, thermals, network latency, per-device-class results, per-Tajweed-category precision/recall/FPR.
**Per-category and per-device-class result tables are therefore empty by necessity**, not omitted.

## 5. Targets (all UNVERIFIED)
| Proposed target | Status | Why |
|---|---|---|
| ≥ 95 % precision of high-confidence corrections | **UNVERIFIED** | no human-reviewed recordings; the only published Tajweed-level figure (F1 75.8 %, 159 clips, one verse) is not a precision and does not support it; needs ≥ 73 predicted units per category even in the best case |
| FPR < 1 % on clean recitation | **UNVERIFIED** | needs ≥ 299 clean units per category with zero false positives |
| Per-category recall | **UNVERIFIED** | no data |
| RTF p95 < 1 on weakest supported device | **UNVERIFIED** | no device, no model run, no weakest device defined |

Also noted: the engine's README admits CTC posteriors "are ~0.99–1.0 for almost everything", i.e. the raw confidence is not
calibrated. The judge's thresholds (`relaxed / balanced / strict`) were left untouched because retuning them without real
labelled audio would be guessing; `eval_stats.reliability` and `threshold_for_precision` are the tools to do it properly.

## 6. Product features (brief §7)
The app is a **single-verse demo** (Ayat al-Kursi, hard-coded words); the backend accepts any verse. **Not implemented:**
interactive full Uthmani Quran and tafsir, hands-free memorisation, adaptive revision, teacher/student management, mastery
criteria, age-appropriate modes, consent flow. Reasons: they need data sources with checked licences, an account/database
backend with child-data safeguards, and an Android build to test any UI — none available — and half-built, untested UI in an
app that currently works would be a regression risk. Existing functionality is preserved.

## 7. Outstanding work (needs resources I do not have)
1. Compile and run the Android app (CI workflow or Android Studio); test token entry, 401/403 handling, low-end devices.
2. Create the `rafiq-auth` Modal secret, deploy, issue tokens (`docs/SECURITY.md`).
3. Legal review: Zipformer NPL-1.2 vs Rafiq's business model; Quran text and EveryAyah audio terms; children's-data consent.
4. Collect and human-review a speaker-disjoint test set (protocol in `docs/EVALUATION.md`); run `run_eval.py` on the current server.
5. Head-to-head Muaalem vs Zipformer v3 / v3.1 (licence permitting) incl. INT8, chunk sizes, execution providers, on real phones.
6. Consent screen, "feedback can be wrong" notice, retention policy, per-token revocation, encrypted token storage.
7. Product features of §6.

## 8. Run it
```bash
python3 backend/dev/run_all_tests.py          # needs only numpy; ~20 s; everything above marked PASS
python3 backend/eval/bench_engine_overhead.py --sessions 12
python3 backend/eval/run_eval.py --records units.jsonl --splits splits.json   # once you have reviewed data
```
Deploy / tokens: `README.md` and `docs/SECURITY.md`. Real server: `py -m modal deploy backend/modal_app.py` after creating the secret.
