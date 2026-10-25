# Scientific validation — how to produce the numbers this project cannot yet claim

**Status: no accuracy figure for Rafiq is VERIFIED.** The existing "detected 50/50" numbers (`dev/mutation_test.py`) come from a
**text-level simulator** that fabricates phoneme strings; they test the judge's logic, not acoustic accuracy.

## 1. Tools that exist and are tested
* `eval/eval_stats.py` — intervals, per-category confusion, VERIFIED / UNVERIFIED / FAIL verdicts, RTF report, leakage check,
  reliability (ECE), confidence-threshold selection. 31 checks in `eval/test_eval_stats.py` (they verify the mathematics with
  synthetic numbers, **not** any model).
* `eval/run_eval.py` — Markdown report from a records file (format in its docstring).
* `eval/bench_engine_overhead.py` — time spent in Rafiq's own non-neural code (REPORT.md has the measurement).

## 2. Definitions (fixed in `eval_stats.py`)
Unit = one (clip, category) opportunity. Precision = TP/(TP+FP) over predicted units; recall = TP/(TP+FN) over human-marked
errors (an abstention on an error is a miss); FPR = FP/(FP+TN) over clean units; coverage = share of units not abstained;
RTF = processing s / audio s (p95, nearest rank). A target is **VERIFIED only if the 95 % confidence bound meets it**.

Sample sizes implied by the targets (computed, see tests):
* FPR < 1 % can be shown with **zero** false positives only from **299 clean units per category** (one-sided 95 %).
* Precision ≥ 95 % (Wilson lower bound) needs **≥ 73 predicted units** even if every one is correct; with a few errors, many more.
* RTF p95 needs ≥ 20 measured spans **and** the hardware stated.

## 3. Data protocol (what a valid test set needs)
1. **Material:** correct recitations *and* realistic learner errors by the real audience (children, adults, accents, phones, noise).
   Public sets are insufficient: qdat_bench is one verse (5:109), 159 clips, three rule families, licence unstated.
2. **Labels:** two independent qualified human reviewers per unit, adjudicated by a third; report inter-rater agreement. Labels
   must say *which category* of error and where. Reference labels produced by a model are not ground truth.
3. **Partitions:** train / val / test, **speaker-disjoint**, no duplicate audio, fixed before any tuning. Verify with
   `leakage_report` (speaker, clip id and audio hash overlap). Tune thresholds on val only; report once on test.
4. **Children:** parental consent and a privacy review before recording anyone under 18 (see `SECURITY.md`).
5. **Report** per category and per recording condition (device class, noise, speed, age group), never only pooled.

## 4. Experiments to run once data and hardware exist
| Question | Design |
|---|---|
| Does Muaalem v3.2 meet the targets, per category? | run the current server on the test set → `run_eval.py` |
| Does abstention buy precision? | `threshold_for_precision` per category on val; report coverage lost on test |
| Is INT8 acceptable? | fp32 vs INT8 on the **same** test set; keep INT8 only if per-category recall/precision deltas are inside the CI |
| Is Zipformer v3 / v3.1 better or worse (licence permitting)? | same test set, same judge; madd from forced-alignment timing |
| Which chunk size? | Zipformer only: 8 / 16 / 24 frames (the sizes it was trained on) — accuracy vs latency |
| Existing vs new system | same recordings, same labels, same metrics |

## 5. Training
Not run and not scaffolded: no legally usable, error-annotated dataset of adequate size was found, and the sandbox has no GPU or
network. If such data is collected under the protocol above, fine-tuning is a separate project with its own licence review
(base model w2v-BERT 2.0 and Muaalem are MIT; Zipformer is NPL-1.2).
