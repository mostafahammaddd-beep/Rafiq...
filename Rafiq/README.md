# رَفِيق — Rafiq v14.6

Recite Ayat al-Kursi → every word turns **green** (correct), **yellow** (tashkeel), **orange** (tajweed)
or **red** (letters / skipped word). Tap a coloured word for a short Arabic note.

**v14.5 is live-only.** The record-then-upload mode, the `/analyze` endpoint and the mode selector are gone;
everything runs over one WebSocket (`/ws/live-recitation`).

## v14.6 — access control (read first)
The live server now **refuses every session without a signed access token** (it used to accept anyone, which let strangers spend your
Modal credits). One-time setup, details in `docs/SECURITY.md`:
```
cd backend
python3 tools/issue_token.py --new-secret          # run it twice: a signing secret and an admin token
modal secret create rafiq-auth RAFIQ_AUTH_SECRET=<first> RAFIQ_ADMIN_TOKEN=<second>
py -m modal deploy modal_app.py
set RAFIQ_AUTH_SECRET=<first>                       # (export on Linux/macOS) same value as in Modal
python3 tools/issue_token.py --sub student-017 --days 30   # prints a token: paste it in the app ⚙ → رمز الدخول
```
`/debug/*` is off (404) unless `RAFIQ_DEBUG_ENDPOINTS=1` **and** the admin token is sent; `/health` is minimal; details at `/admin/health`.
`run_local.py` generates its own secret and prints a token. Full report and what is still unverified: `REPORT.md`.

## Update
1. `cd backend` → `py -m modal deploy modal_app.py` (after the one-time secret above).
   *The server is now a class (`Fastapi.asgi_app`) so the memory snapshot can be taken in `@modal.enter(snap=True)`.
   Modal derives the hostname from the names; `Fastapi` + `asgi_app` is meant to keep the v14.4 hostname
   (`…-rafiq-quran-muaalem-backend-fastapi-asgi-app…`) — if the printed URL differs, paste it into the app: ⚙ → رابط الخادم.*
2. Push this folder to GitHub (the *Build APK* workflow builds the app). Delete any other file in
   `.github/workflows` besides `build.yml`.

## How the live engine works (`backend/modal_app.py`)
| piece | behaviour |
|---|---|
| **Dual-stage lock** | *soft:* `tracked` (the word being recited) moves the instant the alignment enters it. *hard:* a word is judged and locked only after ≥ 2 phonemes of the next word are heard with posterior ≥ 0.7, or a pause arrives. Every window starts 300 ms before the first unlocked word. |
| **Elastic tail** | The model sees only the uncommitted audio: ≤ 2.2 s, 4 s while a long madd is held or after a stalled pass, ≤ 6 s catching up, 10 s last resort. Words are never skipped into "unchecked" while the lock advances. |
| **Alignment** | `local` (the words the window can contain) → `wide` (26 words) → `restart` (phrase start, the reciter's aborted first attempt is skipped as junk) — all on the *same* logits, no extra neural pass. Substitution table is pre-compiled to integers. |
| **Int16 gate** | `memoryview(bytes).cast('h')` → RMS, zero-crossing rate, first difference per 10 ms, no float conversion. While a high-confidence (> 0.95, twice) vowel is steady the engine may skip **one** neural hop, never two; a formant / energy change wakes it. |
| **Madd muttasil** (شَاءَ, w=39) | 4..6 harakat in **every** preset. Decoded count is exact; in addition the vowel is measured on the audio (CTC frame span + 10 ms voiced envelope, pre-hamza closure trimmed, tempo calibrated on the reciter's own madd tabii, clamped 85–185 ms) — see *Limits*. |
| **Sifat** | Matched to phonemes by base letter; each of the 10 heads may slide on its own and is re-aligned **only on evidence**; a contrast is judged only where it exists (tafashie→ش, tikraar→ر, istitala→ض, safeer→ص ز س …); probabilities outside [0,1] (the live report's `conf: 2, 3`) are treated as "no evidence"; clean ultra-short words (فِي) ignore voicing bleed. |
| **Stop at the last word** | w=49 is always a stop target: a vowel on the final letter is `waqf_vowel`; madd 'arid ≥ 2. |
| **Rival pairs** | ط/ت ص/س ض/د ظ/ذ/ز ق/ك ح/ه ع/ء ر/غ use a halved confidence gate (and the relative odds when the runner-up posterior is captured). |

### WebSocket protocol
client → server: binary PCM16 16 kHz mono (~160 ms) · `{"type":"pause","duration_ms":N,"speech_tail_padding_ms":500}` ·
`{"type":"resume","word_idx":k}` (first frame after a reconnect) · `{"type":"stop"}` · `{"type":"reset"}`
server → client: `ready` · `progress` (`tracked`, `committed`, `lock_ms`, `audio_ms`, `updates`) · `final` · `paused_idle` (`resume_word_idx`, `audio_ms`) · `error`.

The phone keeps 250 ms before speech and 500 ms after it, sends nothing after > 500 ms of silence (one `pause` frame instead),
and a socket silent for 8 s is closed with `paused_idle`; the app reconnects when speech resumes and sends `resume`.
The app reacts to the record button on **touch-down**, so the WebSocket handshake starts ~100–200 ms earlier; it no longer wakes
the server when merely opened (nothing is billed while browsing).

## Cost — read `python3 backend/dev/cost_proof.py` (prints the full breakdown)
Rates: $0.0000131/core-s, $0.00000222/GiB-s. Modelled per 65 s session; **the INT8 speed-up (×1.8) and the 3 s snapshot restore are assumptions, not measurements**.
| one Ayat al-Kursi session | cost |
|---|---|
| v14.4 live (4 cores, 4 GB, measured RTF 0.352) | ≈ $0.0051 |
| v14.4 record-then-analyze (the removed default) | ≈ $0.0023 |
| **v14.5 live, model estimate** (0.125-core reservation, 1.5 GiB, INT8, CPU-bound cadence) | **≈ $0.0015** |
| reservation + RAM alone (no compute) | ≈ $0.0002 |
Compute dominates: one INT8 pass over 55 s of speech is already ≈ $0.0006, and the hop setting multiplies it
(`RAFIQ_HOP_S`: 0.16 s → ×8.0 redundancy, 0.32 s → ×4.7, 0.64 s → ×2.9, 1.0 s → ×2.2, all 50/50 words judged in the simulator).
A model of 600 M parameters does not fit in 256–768 MiB.

## Hafs 'an 'Asim policy used by the judge (see `madd_kind` / `madd_policy` in `backend/modal_app.py`)
madd muttasil **4–6 in every preset** · madd munfasil / silah kubra: 2 (qasr) to 5 in balanced / relaxed, **4–5 in strict** ·
lazim 6 · 'arid and lin at a stop 2/4/6 · tabii, silah sughra, badal, 'iwad exactly 2. A final vowel dropped at a stop,
ta marbuta → ha, tanween → alif at a stop, a word started fresh after a stop are valid *when the audio really contains a stop*
(a silence, or a `pause` frame from the phone). The expected pronunciations come from `quran-transcript`.

## Limits (measured, not hoped)
* The audio channel for madd muttasil measures a perfect 4-harakat madd as 3.56 ± 0.18 harakat (simulator): it overrides the decoded
  count only for a *clearly* short (≲ 2.5) or long (≳ 7.5) vowel. 3 vs 4 harakat is decided by the decoded count.
  It abstains until two madd tabii runs were measured, and when the reciter's tempo is outside the 85–185 ms clamp.
* The Kotlin sources were **statically checked** (`backend/dev/static_check_kt.py`), not compiled in this environment.
* The 216 engine checks and `mutation_test.py` run on a **simulator that fabricates phoneme strings**, not on real recordings: they
  test the judge's logic, not Tajweed accuracy. No precision / false-positive figure for real audio exists yet (`docs/EVALUATION.md`).
* INT8 quantisation (encoder Linear layers only; heads, projections and adapter stay fp32, guarded by a logit-cosine check),
  the memory snapshot and the Modal class deployment were written but **not run** here (no torch / Modal / network).

## Telling me what is wrong
Tap a green word → "⚠ فيها خطأ لم يُكتشف"; tap a coloured word → "✔ نطقتها صحيحة"; then ⚙ → "نسخ تقرير فني"
and paste it. Free fallback: `backend/run_local.py` (see the file).
Self-tests: `python3 backend/dev/run_all_tests.py` (347 checks: 216 engine checks on a *simulator*, 97 security, 31 evaluation-statistics), `backend/dev/mutation_test.py` (detection / false-alarm rates,
identical to v14.4), `backend/dev/static_check_kt.py`, `backend/dev/cost_proof.py`.
