# Model, dataset and dependency licence notes (read 2026-10-25)

**This is engineering due diligence, not legal advice.** "Verified" = seen on the page/card this session.

| Component | Licence | Status | Notes / obligations |
|---|---|---|---|
| Muaalem model `obadx/muaalem-model-v3_2` | MIT | verified (HF tag) | Keep the licence notice; model card documents no bias evaluation |
| `facebook/w2v-bert-2.0` (Muaalem's base) | MIT | reported by a downstream card; not independently opened | |
| `quran-muaalem`, `quran-transcript` libraries | MIT | reported by third-party cards; repos not opened | They produce the phonetic references (QPS) |
| **Quran-Lab `zipformer_p-arabic-v3` / `v3.1`** | **NPL-1.2** (one search snippet showed 1.1: the terms changed between versions — check the version attached to the files you download) | card verified, **LICENSE file not readable (login-gated)** | *(1) never charge for the model, access to it, or any feature it powers; (2) never present output as an authoritative religious ruling; (3) state in any app that automatic feedback can be wrong and is no substitute for a teacher.* Incompatible with any paid feature; ad-supported use needs a legal opinion. **Not integrated.** Access requires the user's own acceptance of the terms |
| `hifzguide-muaalem-mini` | card says MIT lineage | card only | trained on FaisaI/tadabur — dataset licence not checked. Not integrated |
| sherpa-onnx (only relevant if an on-device Zipformer is adopted) | Apache-2.0 | reported by package-index pages | each pre-trained model has its own licence |
| obadx/qdat_bench (159 clips, 5:109 only) | **none stated on the card** | verified (card) | built on QDAT (Sharief et al.) — original terms not checked. Fine for internal evaluation; confirm before redistributing |
| obadx/muaalem-annotated-v3, Tadabur, Quran-Lab benchmark, EveryAyah | not inspected | — | needed only for training/benchmarking; check before use |
| Quran text in `VerseData.kt` and the one in `quran-transcript` | source not stated in the repo | **unresolved** | If the text is Tanzil's, its licence (reported CC BY 3.0) requires attribution and forbids altering the text |
| EveryAyah Minshawi audio (`AUDIO_URL`) | site terms not checked | **unresolved** | hot-linked third-party media |
| Android dependencies (Compose BOM, AndroidX, OkHttp 4.12) | Apache-2.0 | general knowledge, **not re-checked** | |
| Modal | commercial service | terms not reviewed | |

Unresolved before any paid release: (a) the Quran text and reference-audio sources; (b) whether Zipformer can ever be used;
(c) dataset licences if any training is attempted; (d) the debug keystore must not sign a store release.
