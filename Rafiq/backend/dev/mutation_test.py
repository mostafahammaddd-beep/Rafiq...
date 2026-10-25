"""Injects ONE deliberate mistake into each word in turn and reports detection / false-alarm rates."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, modal_app as M

def mutate(kind, form):
    if kind == "vowel":
        for i, ch in enumerate(form):
            if ch in M.CORE_VOWELS:
                other = {M.FATHA: M.KASRA, M.DAMMA: M.FATHA, M.KASRA: M.DAMMA}[ch]
                return form[:i] + other + form[i + 1:], "tashkeel"
    if kind == "letter":
        letters = [i for i, ch in enumerate(form) if ch in "بتثجحخدذرزسشصضطظعغفقكلمنهـ" ]
        if letters:
            i = letters[len(letters) // 2]
            swap = {"س": "ص", "ص": "س", "ت": "ط", "ط": "ت", "د": "ض", "ض": "د", "ك": "ق", "ق": "ك", "ح": "ه", "ه": "ح"}.get(form[i], "ز" if form[i] != "ز" else "ر")
            return form[:i] + swap + form[i + 1:], "letters"
    if kind == "drop":
        letters = [i for i, ch in enumerate(form) if ch in "بتثجحخدذرزسشصضطظعغفقكلمنه"]
        if len(letters) >= 3:
            i = letters[len(letters) // 2]
            return form[:i] + form[i + 1:], "letters"
    return None, None

def run(r, kind, preset):
    stats = collections.Counter(); misses = []; falses = []
    pauses = {6, 11, 18, 25, 31, 39, 43, 46}
    base = [r.waqf[k] if (k in pauses and r.waqf[k]) else r.wasl[k] for k in range(r.n)]
    for k in range(r.n):
        spoken = list(base)
        m, expect = mutate(kind, spoken[k])
        if m is None or m == spoken[k]:
            continue
        spoken[k] = m
        eng, log, final, sp = sim.run_stream(r, sim.Speech(r, spoken, pauses=pauses, seed=100 + k), preset=preset)
        got = {w["i"]: w for w in final["words"] if w["s"] != "ok"}
        stats["tested"] += 1
        if k in got:
            stats["detected"] += 1
            if got[k]["s"] == expect: stats["right_colour"] += 1
        else:
            misses.append((k, sim.WORDS[k]))
        fp = [w for w in got if w != k]
        stats["false_alarm_words"] += len(fp)
        if fp: falses.append((k, fp))
    return stats, misses, falses

for label, lib in (("exact segmentation", sim.FakeLib()), ("fallback 'aligned' segmentation", sim.FakeLib(spaced="none"))):
    r = M.build_ref_aya(2, 255, lib)
    print("\n=== %s (%s)" % (label, r.info["segmentation"]))
    for preset in ("strict", "balanced"):
        for kind in ("vowel", "letter", "drop"):
            st, miss, fals = run(r, kind, preset)
            print("  %-8s %-6s detected %2d/%2d (right colour %2d) | false-alarm words: %d %s%s" % (
                preset, kind, st["detected"], st["tested"], st["right_colour"], st["false_alarm_words"],
                ("| missed: " + str(miss[:4])) if miss else "", (" | FP in runs: " + str(fals[:3])) if fals else ""))
