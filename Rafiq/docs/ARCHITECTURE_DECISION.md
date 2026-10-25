# Architecture decision — v14.6

Sources were read on 2026-10-25 (model cards, the Muaalem paper abstract on arXiv, dataset cards, discussion threads).
Everything below is labelled **verified** (read in a primary source this session), **reported** (stated by a third party or a
self-reported number I could not reproduce) or **not found**. No model weights were downloaded and no audio was processed:
the build sandbox has no network access to model hosts, no GPU and no Android device.

## 1. What was found

| Candidate | Exists? | What it is | Published evidence | Licence | On-device? |
|---|---|---|---|---|---|
| **Quran Muaalem `obadx/muaalem-model-v3_2`** | verified | wav2vec2-BERT 2.0 encoder, multi-level CTC: phonemes **plus 10 sifat heads** (the project's own README puts it at ~600 M parameters) | Paper arXiv 2509.00094 (verified abstract): PER 0.21 % on its test set, 1.94 % on qdat_bench, **Tajweed F1 75.8 %** on qdat_bench (159 clips, madd / ghunna / qalqala only) | **MIT** (HF tag; base model w2v-BERT 2.0 reported MIT) | No. A third party's INT8 ONNX export is **876 MB** (verified size) |
| **Quran-Lab Zipformer `zipformer_p-arabic-v3` / v3.1** ("Zipformer Quran 3.1") | verified (card readable, files gated) | 65.5 M-parameter cache-aware **streaming** Zipformer2 + CTC, 251-symbol phonetic alphabet, sherpa-onnx compatible ONNX / INT8 ONNX, 63 MB CoreML. v3.1 = madd fine-tune of v3 | Self-reported PER 1.43 % (studio reciters), 3.65 % (phone recordings), 9.10 % (unseen professional reciter). qdat_bench madd RMSE (harakat): alif 0.210 vs Muaalem 0.449; munfasil 0.887 vs 0.687; aared 1.458 | **Quran-Lab No-Profit License 1.2 (NPL-1.2)**: *never charge for the model, access to it, or any feature it powers*; do not present output as an authoritative ruling; state in the app that feedback can be wrong. LICENSE file is behind a login, **not read** | **Yes** (ONNX / CoreML). Android latency **not found / not measured by me** |
| **TajweedFormer** | **not found** (two searches, incl. an exact-phrase one: no paper, repo or model with that name) | — | — | — | — |
| `sysofwan/hifzguide-muaalem-mini` | verified (card) | Distilled student of Muaalem for on-device use | Card gives no accuracy table I could verify; **phoneme identity only, no sifat heads**; trained on Tadabur audio; 19 downloads at the time | Card: MIT lineage; dataset licence not checked | Intended |
| TajweedAI (qalqalah classifier, NeurIPS-workshop abstract) | verified (abstract) | Binary qalqalah classifier on top of an ASR aligner | 100 % on its internal set, **57.14 % external** | not checked | — |

## 2. What the evidence says about the brief's targets

* **"Phoneme-recognition accuracy ≠ Tajweed-detection accuracy."** Every low PER above is a phoneme error rate on
  (mostly correct) recitation. The only published Tajweed-level number found is Muaalem's **75.8 % F1**, on 159 clips of
  **one verse** (Al-Ma'idah 5:109), covering **three** rule families. It is an F1, not a precision.
* The Zipformer card states its own limits: it is *"trained almost entirely on correct recitation"*, so for a deviating
  reciter it *"tends to transcribe the rule rather than the deviation"* (on qdat_bench it reported full ikhfaa on 81 of 94
  clips where annotators heard a plain noon); free-choice madd lengths are unreliable in the token stream; **under-12
  reciters score 2–3× worse than adults.** Children are Rafiq's primary audience.
* A third-party field report on the Muaalem model page (anecdotal, small sample, not verified by me) says letter
  substitutions were detected inconsistently (2 of 5), insertion detection suffers from recording noise, and a dropped
  shadda can raise a false omission on the neighbouring word.
* **Consequence:** the proposed ≥ 95 % precision and < 1 % false-positive targets are **UNVERIFIED** for every candidate and
  there is no published evidence that any of them meets them across categories. Reaching them will require
  per-category abstention (implemented in the judge, **not tuned on real data here**) and a human-reviewed test set
  (see `docs/EVALUATION.md`).
* **No dataset found supports training a Tajweed-error detector.** Muaalem's 848 h / 286 K utterances are labelled by a
  98 %-automated pipeline on correct recitation; qdat_bench has real learner errors but only 159 clips of one verse (a good
  *test* set for three rule families, far too small to train on, licence not stated on the card); the Zipformer corpus is
  professional/phone recitation with canonical labels. Therefore **no training or fine-tuning was run and no training
  pipeline is claimed.**

## 3. Decision (ONE architecture)

> **Reference-conditioned multi-level CTC posteriors → alignment to the exact Quranic phonetic reference →
> explicit per-rule judge with abstention → server-side CPU inference. Acoustic model: Muaalem v3.2.**

This is the architecture the project already implements; v14.6 keeps it and hardens it rather than swapping it.

Why Muaalem v3.2 remains the production acoustic model **today**:

1. It is the only candidate that is **both** permissively licensed (MIT) **and** exposes the sifat heads the judge consumes
   (ghunna, qalqala, tafkhim, hams/jahr, shidda/rakhawa, itbaq, safeer, tikraar, tafashie, istitala). The Zipformer family
   has a single CTC head over a 251-symbol alphabet and no separate sifat heads (per its card), so it cannot replace the per-attribute sifat checks without a different judging design.
2. **Legal fit.** NPL-1.2 forbids charging for any feature the model powers. Rafiq's business model is unknown to me;
   if any feature that uses the model will ever be paid (subscription, teacher tier, in-app purchase, and — until a lawyer
   says otherwise — ad-supported features), the Zipformer is **not usable** in that feature.
3. It is integrated, and the 216 pre-existing engine checks (simulator, not real audio) still pass unchanged.

Rejected, and why:

* **Zipformer v3 / v3.1 as production dependency** — licence (above), phoneme-only, rule-bias limitation, weak on children.
  **Kept as the benchmark candidate** you asked for, and as the most credible route to an **on-device** first pass
  (progress tracking + forced-alignment timing) — see conditions below.
* **TajweedFormer** — could not be verified to exist.
* **hifzguide-muaalem-mini** — no sifat heads, thin evidence, unreviewed dataset licence. Benchmark-only.
* **TajweedAI** — single rule, 57 % external accuracy.
* **Third-party 876 MB INT8 Muaalem ONNX** — too large for low-end phones; unverified publisher (supply-chain risk).

### Conditions to adopt the Zipformer (written down so the decision can be revisited)
1. The full LICENSE read by someone qualified, plus written confirmation (or a separate licence) that Rafiq's monetisation
   never charges for a feature it powers.
2. A head-to-head on **the same human-reviewed recordings** (children included): Muaalem v3.2 vs Zipformer v3 vs v3.1, per
   category, with `eval/run_eval.py`.
3. Android measurements (peak RAM, load time, RTF p95, battery, thermals) on ≥ 3 device classes, per `docs/LOW_END_DEVICES.md`.
4. Duration (madd) judged from forced-alignment timing, not symbol run-length, as the card itself recommends.

## 4. Streaming — what is and is not true

* **Implemented in the repository (unchanged, tested only on the simulator):** the server re-encodes the *uncommitted tail*
  (≤ 2.2 s normally) every 160 ms hop. The README's own modelled redundancy at this hop is ×8 (×2.2 at a 1 s hop).
* **Not possible with this model:** state carried between chunks. A wav2vec2-BERT/Conformer encoder with full attention is
  non-causal (my assessment from the architecture, not a measurement), so "genuinely stateful streaming" would need a
  cache-aware model — i.e. the Zipformer family (the card documents a 48-frame chunk contract with cache tensors).
  **No stateful streaming was implemented**, because the only weights that support it are gated behind a licence I could not
  read and I could not download them. Writing a runner against an unseen tensor layout would have been guesswork.
* Chunk-size / INT8 / execution-provider (CPU, XNNPACK, QNN) benchmarks: **BLOCKED** (no weights, no phone). Retaining INT8
  requires the A/B accuracy test in `docs/EVALUATION.md`; the existing server code only guards INT8 with a logit-cosine
  check (never run here).

## 5. Where cloud vs device sits in the final design

| Work | Where | Why |
|---|---|---|
| Microphone capture, 250 ms pre-roll, silence gate, pause frames | phone (existing) | cheap; avoids sending silence |
| Acoustic model (Muaalem v3.2, ~600 M params) | cloud CPU (Modal), authenticated | does not fit low-end phones; the only MIT-licensed model with sifat heads |
| Alignment, judge, abstention | cloud (measured: ~0.6 ms per step on one Xeon core, engine-only RTF ≈ 0.001 — see REPORT.md) | negligible next to the model |
| Routine **on-device** assessment | **not available** | needs a license-cleared streaming model; see conditions |

Honest consequence: the brief's "on-device first" goal is **not met** by a legally clear, Tajweed-capable model that exists
today. v14.6 documents this rather than papering over it.
