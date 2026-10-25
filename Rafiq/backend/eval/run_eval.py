"""
Build the per-category evaluation report from HUMAN-REVIEWED records.

    python3 eval/run_eval.py --records units.jsonl [--splits splits.json] [--rtf rtf.json] [--out report.md]

units.jsonl   one JSON object per (clip, category) opportunity:
              {"clip_id": "c17", "speaker_id": "spk-0042", "category": "madd_muttasil",
               "truth": true,        # a qualified human reviewer marked an error of this category here
               "predicted": true,    # the system reported a correction of this category here
               "abstained": false,   # optional: the system declined to judge this unit
               "confidence": 0.93,   # optional: system confidence of the prediction (enables calibration + thresholds)
               "split": "test"}      # optional: train | val | test
splits.json   {"train": [{"clip_id":..,"speaker_id":..,"audio_sha256":..}, ...], "val": [...], "test": [...]}
rtf.json      {"hardware": "Pixel 6a, Android 14, ORT 1.x CPU x4", "process_s": [...], "audio_s": [...]}

Nothing here produces a number by itself: the report is exactly as good as the records. A target is VERIFIED only when the
95 % confidence BOUND meets it; otherwise it says UNVERIFIED (and how many samples would be needed) or FAIL.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_stats as E  # noqa: E402


def fmt(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--splits")
    ap.add_argument("--rtf")
    ap.add_argument("--out")
    ap.add_argument("--only-split", default="test", help="report this partition (default test); 'all' uses every record")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.records, encoding="utf-8") if l.strip()]
    sel = recs if a.only_split == "all" else [r for r in recs if r.get("split", "test") == a.only_split]
    lines = ["# Rafiq evaluation report", "",
             "Records: %d total, %d in partition `%s`. %s" % (len(recs), len(sel), a.only_split, E.CATEGORIES_NOTE), ""]
    if a.splits:
        probs = E.leakage_report(json.load(open(a.splits, encoding="utf-8")))
        lines += ["## Data hygiene", ""] + (["- **LEAKAGE / HYGIENE FINDINGS** (results below are NOT valid until fixed):"] + ["  - " + p for p in probs]
                                            if probs else ["- No speaker / clip-id / audio-hash overlap between partitions."]) + [""]
    else:
        lines += ["## Data hygiene", "", "- **UNVERIFIED**: no splits file given, so speaker-disjoint evaluation is not demonstrated.", ""]
    conf = E.confusion(sel)
    lines += ["## Per-category results (95 % intervals; precision/recall Wilson, FPR exact one-sided upper bound)", "",
              "| category | units | pred. | TP | FP | FN | TN | abst. | precision (CI) | recall (CI) | FPR upper95 | coverage |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(conf, key=lambda s: (s == "ALL", s)):
        c = conf[k]
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %d | %s (%s-%s) | %s (%s-%s) | %s | %s |" % (
            k, c["n"], c["predicted"], c["tp"], c["fp"], c["fn"], c["tn"], c["abstain"], fmt(c["precision"]), fmt(c["precision_ci95"][0]),
            fmt(c["precision_ci95"][1]), fmt(c["recall"]), fmt(c["recall_ci95"][0]), fmt(c["recall_ci95"][1]), fmt(c["fpr_upper95"], 4), fmt(c["coverage"])))
    lines += ["", "## Verdicts against the proposed release targets", ""]
    for k in sorted(conf, key=lambda s: (s == "ALL", s)):
        for t, v in E.verdict(conf[k]).items():
            lines.append("- `%s` %s: **%s**" % (k, t, v))
    withc = [r for r in sel if r.get("confidence") is not None and r.get("predicted") and not r.get("abstained")]
    if withc:
        lines += ["", "## Calibration of reported confidence (predicted units only)", ""]
        rel = E.reliability([r["confidence"] for r in withc], [bool(r["truth"]) for r in withc])
        lines.append("ECE %s over %d predictions (biased upward for small n)." % (fmt(rel["ece"]), rel["n"]))
        lines += ["", "| bin | n | mean confidence | observed precision |", "|---|---|---|---|"] + [
            "| %s | %d | %s | %s |" % (b["bin"], b["n"], fmt(b["mean_confidence"]), fmt(b["accuracy"])) for b in rel["bins"]]
        lines += ["", "Confidence threshold that meets 95 % precision (Wilson lower bound) per category - select on `val`, report on `test`:", ""]
        for cat in sorted({r["category"] for r in withc}):
            rr = [r for r in withc if r["category"] == cat]
            t = E.threshold_for_precision([r["confidence"] for r in rr], [bool(r["truth"]) for r in rr])
            lines.append("- `%s`: %s" % (cat, "**none - this category must abstain or needs more data**" if t is None else "%.3f" % t))
    if a.rtf:
        j = json.load(open(a.rtf, encoding="utf-8"))
        rep = E.rtf_report(j["process_s"], j["audio_s"], j.get("hardware"))
        lines += ["", "## Real-time factor", "", "- hardware: %s" % (rep["hardware"] or "NOT SPECIFIED"), "- spans: %d, p50 %s, p95 %s, max %s" % (
            rep["n"], fmt(rep["p50"]), fmt(rep["p95"]), fmt(rep["max"])), "- verdict: **%s**" % rep["verdict"]]
    text = "\n".join(lines) + "\n"
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
