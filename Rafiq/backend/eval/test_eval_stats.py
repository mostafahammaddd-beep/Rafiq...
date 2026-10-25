"""Tests of the evaluation statistics. They verify the MATHEMATICS and the gating logic with synthetic numbers.
They say nothing about how accurate any Quran model is."""
import math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_stats as E

FAILS = []
def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + "eval: " + name + ((("   -> " + str(detail)) if (detail and not cond) else "")))
    if not cond: FAILS.append(name)

near = lambda a, b, tol=1e-3: abs(a - b) <= tol

# intervals against published reference values
lo, hi = E.wilson(8, 10); check("Wilson 8/10 = (0.4902, 0.9433)", near(lo, 0.4902) and near(hi, 0.9433), (lo, hi))
lo, hi = E.wilson(0, 20); check("Wilson 0/20 lower bound is 0, upper about 0.161", lo == 0.0 and near(hi, 0.1611, 2e-3), (lo, hi))
check("Wilson with n=0 carries no information: (0, 1)", E.wilson(0, 0) == (0.0, 1.0))
lo, hi = E.wilson(50, 50); check("Wilson 50/50 lower bound about 0.929 (a perfect small sample does not prove 95 %)", near(lo, 0.9287, 2e-3) and hi == 1.0, (lo, hi))
check("rule of three: 299 clean units with 0 false positives is the minimum that demonstrates FPR < 1 % (298 is not)",
      E.min_clean_units_for_fpr() == 299 and E.upper_bound_zero_events(299) < 0.01 <= E.upper_bound_zero_events(298) + 1e-12)
check("Clopper-Pearson one-sided 95 % upper bound for 3/100 is 0.0757", near(E.upper_bound_binomial(3, 100), 0.0757, 5e-4), E.upper_bound_binomial(3, 100))
check("upper bound is monotone in k and never below the observed rate", all(E.upper_bound_binomial(k, 200) >= k / 200 for k in range(0, 40)) and
      all(E.upper_bound_binomial(k, 200) <= E.upper_bound_binomial(k + 1, 200) for k in range(0, 39)))
check("a perfect record needs 73 predictions before the Wilson lower bound reaches 95 %", E.min_predicted_units_for_precision() == 73)

# confusion and verdicts
def recs(category, tp, fp, fn, tn, abstain_pos=0, abstain_neg=0):
    out = []
    out += [dict(clip_id="c", speaker_id="s", category=category, truth=True, predicted=True)] * tp
    out += [dict(clip_id="c", speaker_id="s", category=category, truth=False, predicted=True)] * fp
    out += [dict(clip_id="c", speaker_id="s", category=category, truth=True, predicted=False)] * fn
    out += [dict(clip_id="c", speaker_id="s", category=category, truth=False, predicted=False)] * tn
    out += [dict(clip_id="c", speaker_id="s", category=category, truth=True, predicted=False, abstained=True)] * abstain_pos
    out += [dict(clip_id="c", speaker_id="s", category=category, truth=False, predicted=False, abstained=True)] * abstain_neg
    return out

c = E.confusion(recs("madd", 40, 2, 10, 300))["madd"]
check("counts, precision, recall, FPR", (c["tp"], c["fp"], c["fn"], c["tn"]) == (40, 2, 10, 300) and near(c["precision"], 40 / 42) and near(c["recall"], 0.8) and near(c["fpr"], 2 / 302), c)
v = E.verdict(c)
check("95 % precision is UNVERIFIED for 40/42 (point 0.952 but the lower bound is far below 0.95)", v["precision>=95%"].startswith("UNVERIFIED"), v)
check("FPR 2/302 is UNVERIFIED, not failed and not verified", v["FPR<1%"].startswith("UNVERIFIED"), v)
c = E.confusion(recs("ghunna", 300, 0, 5, 400))["ghunna"]; v = E.verdict(c)
check("300/300 correct predictions: precision VERIFIED (lower bound 0.987)", v["precision>=95%"].startswith("VERIFIED"), v)
check("0 false positives in 400 clean units: FPR VERIFIED; with 299 exactly it is too, with 250 it is not",
      v["FPR<1%"].startswith("VERIFIED") and E.verdict(E.confusion(recs("g", 10, 0, 0, 299))["g"])["FPR<1%"].startswith("VERIFIED")
      and E.verdict(E.confusion(recs("g", 10, 0, 0, 250))["g"])["FPR<1%"].startswith("UNVERIFIED"))
c = E.confusion(recs("qalqala", 20, 80, 0, 400))["qalqala"]; v = E.verdict(c)
check("precision 20/100 is a FAIL (even the upper bound is under 95 %)", v["precision>=95%"].startswith("FAIL"), v)
c = E.confusion(recs("qalqala", 5, 60, 0, 400))["qalqala"]
check("FPR 60/460 is a FAIL (the lower bound is above 1 %)", E.verdict(c)["FPR<1%"].startswith("FAIL"), E.verdict(c))
c = E.confusion(recs("x", 0, 0, 5, 50))["x"]
check("no predictions at all: precision UNVERIFIED, recall 0", E.verdict(c)["precision>=95%"].startswith("UNVERIFIED") and c["recall"] == 0.0)
c = E.confusion(recs("idgham", 10, 0, 0, 0, abstain_pos=10, abstain_neg=30))["idgham"]
check("abstaining on a real error counts as a miss (recall 0.5), on clean audio it costs nothing; coverage is reported",
      near(c["recall"], 0.5) and c["abstain"] == 40 and near(c["coverage"], 10 / 50) and c["fp"] == 0, c)
allc = E.confusion(recs("a", 5, 1, 1, 10) + recs("b", 3, 0, 2, 20))
check("categories stay separate and an ALL row pools them", allc["a"]["tp"] == 5 and allc["b"]["tp"] == 3 and allc["ALL"]["tp"] == 8 and allc["ALL"]["n"] == (5 + 1 + 1 + 10) + (3 + 0 + 2 + 20))

# latency
r = E.rtf_report([0.5] * 30, [1.0] * 30, "Pixel 6a, Android 14, ORT 1.x CPU 4 threads")
check("RTF 0.5 on 30 spans with hardware stated -> VERIFIED for that hardware only", r["verdict"].startswith("VERIFIED on this hardware only") and r["p95"] == 0.5, r)
check("RTF without a hardware description is UNVERIFIED", E.rtf_report([0.5] * 30, [1.0] * 30, None)["verdict"].startswith("UNVERIFIED"))
check("RTF from 5 spans is UNVERIFIED", E.rtf_report([0.5] * 5, [1.0] * 5, "x")["verdict"].startswith("UNVERIFIED"))
check("RTF p95 >= 1 is a FAIL", E.rtf_report([0.4] * 28 + [1.5, 1.6], [1.0] * 30, "x")["verdict"].startswith("FAIL"))
check("p95 uses nearest rank (19 of 20 values at or below it)", E.percentile_nearest_rank(list(range(1, 21)), 95) == 19)

# leakage
ok = {"train": [dict(clip_id="1", speaker_id="A", audio_sha256="h1"), dict(clip_id="2", speaker_id="B", audio_sha256="h2")],
      "test": [dict(clip_id="3", speaker_id="C", audio_sha256="h3")]}
check("disjoint partitions produce no leakage findings", E.leakage_report(ok) == [])
bad = {"train": [dict(clip_id="1", speaker_id="A", audio_sha256="h1")], "test": [dict(clip_id="9", speaker_id="A", audio_sha256="h9"), dict(clip_id="1", speaker_id="Z", audio_sha256="h1")]}
rep = E.leakage_report(bad)
check("same speaker, same clip id and identical audio across partitions are all detected", len(rep) == 3 and any("speaker" in x for x in rep) and any("clip id" in x for x in rep) and any("identical audio" in x for x in rep), rep)
check("missing speaker ids are reported (speaker-disjointness cannot be claimed)", any("no speaker_id" in x for x in E.leakage_report({"a": [dict(clip_id="1")], "b": [dict(clip_id="2")]})))

# calibration
rng = random.Random(0)
conf = [rng.random() for _ in range(20000)]
cal = [rng.random() < c for c in conf]
check("a perfectly calibrated synthetic scorer has ECE near 0", E.reliability(conf, cal)["ece"] < 0.02, E.reliability(conf, cal)["ece"])
over = [min(1.0, c + 0.3) for c in conf]
check("an over-confident scorer (raw 'posteriors' near 1.0 for everything) shows a large ECE", E.reliability(over, cal)["ece"] > 0.15, E.reliability(over, cal)["ece"])
flat = [0.995 + 0.004 * rng.random() for _ in range(5000)]
flat_ok = [rng.random() < 0.7 for _ in flat]
check("threshold selection refuses (None) when confidence carries no information and precision is 0.7: the category must abstain",
      E.threshold_for_precision(flat, flat_ok) is None)
rng2 = random.Random(1)
conf2 = [rng2.random() for _ in range(6000)]
cor2 = [rng2.random() < (0.5 + 0.5 * c) ** 3 * 0.5 + 0.5 * (c > 0.9) * 0.99 or c > 0.97 for c in conf2]
t = E.threshold_for_precision(conf2, cor2)
sel = [k for c_, k in zip(conf2, cor2) if t is not None and c_ >= t]
check("a threshold is returned only if the Wilson LOWER bound of the selected predictions meets 95 %",
      t is not None and E.wilson(sum(sel), len(sel))[0] >= 0.95 and len(sel) > 0, (t, len(sel)))

# CLI smoke test: the report must say UNVERIFIED for a small sample and flag missing splits
import json, subprocess, tempfile
with tempfile.TemporaryDirectory() as d:
    f = os.path.join(d, "u.jsonl")
    with open(f, "w", encoding="utf-8") as fh:
        for i in range(30):
            fh.write(json.dumps(dict(clip_id="c%d" % i, speaker_id="s%d" % (i % 5), category="madd_muttasil", truth=(i % 3 == 0), predicted=(i % 3 == 0), confidence=0.9, split="test")) + "\n")
    out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_eval.py"), "--records", f], capture_output=True, text=True)
    check("run_eval.py renders a report; 10/10 correct predictions are still UNVERIFIED for 95 % precision; missing splits flagged",
          out.returncode == 0 and "UNVERIFIED" in out.stdout and "no splits file" in out.stdout and "madd_muttasil" in out.stdout and "VERIFIED (95% Wilson" not in out.stdout, out.stdout[-500:] + out.stderr[-300:])

print("\n%d eval check(s) failed" % len(FAILS) if FAILS else "\nEVAL STATS CHECKS PASSED")
sys.exit(1 if FAILS else 0)
