"""
Self-checking simulation suite for the Rafiq engine (no model, no GPU, no internet needed).
Run:  py dev/run_all_tests.py      (needs only numpy)
A fake QPS phonemizer + a fake recitation model (which returns the phonemes inside whatever audio
window the engine hands it, including cut-off words at the edges) exercise the real engine code.
"""
import asyncio, json, os, sys, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
import modal_app as M
# v14.6: the server fails closed. These engine tests call the WebSocket handler directly without a token, so they run in the
# explicit local-dev anonymous mode; dev/test_security.py (run at the end of this file) covers authentication itself.
os.environ["RAFIQ_ALLOW_ANONYMOUS"] = "1"
os.environ.pop("RAFIQ_AUTH_SECRET", None)

FAILS = []
def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (("   -> " + str(detail)) if (detail and not cond) else ""))
    if not cond: FAILS.append(name)

ref = sim.build_ref()
PAUSES = {6, 11, 18, 25, 31, 39, 43, 46}
BASE = [ref.waqf[k] if (k in PAUSES and ref.waqf[k]) else ref.wasl[k] for k in range(ref.n)]
WASL = list(ref.wasl)

def flagged(final): return {w["i"]: w for w in final["words"] if w["s"] != "ok"}
def kinds(w): return [i["k"] for i in w.get("issues", [])]

# ---------------------------------------------------------------- reference building
check("reference: 50 words, stop sign ignored, exact segmentation", ref.n == 50 and not ref.approx)
check("reference: stop (waqf) forms computed for words that end in a vowel/tanween",
      ref.info["waqf_forms"] >= 20 and ref.waqf[9] and ref.waqf[11], ref.info.get("waqf_forms"))

# ---------------------------------------------------------------- clean recitation
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, BASE, pauses=PAUSES, seed=3))
check("clean: all 50 judged correct", final["summary"]["ok"] == 50 and final["summary"]["complete"], final["summary"])
lags = sim.lag_stats(log, sp)
check("clean: tracking lag <= 3 words (before client smoothing)", max(lags) <= 3, max(lags))
check("clean: the model only sees the uncommitted tail (<= 4.5 s; v14.3 sent up to 6 s, v14.1 the whole recitation)",
      max(s["wave_s"] for s in eng.debug["steps"]) <= 4.5, max(s["wave_s"] for s in eng.debug["steps"]))
check("clean: monotonic completeness - every word below the lock has a verdict",
      sorted(eng.results) == list(range(eng.cursor)) and eng.cursor == 50)

# ---------------------------------------------------------------- injected mistakes
def inject(spoken):
    s = list(spoken); exp = {}
    s[3] = s[3].replace("ل", "ن", 1); exp[3] = ("letters", "replace")
    s[14] = s[14].replace(M.KASRA, M.FATHA); exp[14] = ("tashkeel", "vowel_swap")
    i = s[15].index("اا"); s[15] = s[15][:i] + s[15][i + 2:]; exp[15] = ("tajweed", "madd_dropped")
    s[21] = s[21].replace("لل", "ل", 1); exp[21] = ("tajweed", "shaddah_dropped")
    s[27] = ""; exp[27] = ("letters", "skipped")
    i = s[33].index("ح"); s[33] = s[33][:i] + s[33][i + 1:]; exp[33] = ("letters", "missing")
    return s, exp
for preset in ("balanced", "strict"):
    spoken, exp = inject(BASE)
    eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=PAUSES, seed=5), preset=preset)
    got = flagged(final)
    ok = all(w in got and got[w]["s"] == e[0] and e[1] in kinds(got[w]) for w, e in exp.items())
    check("errors/%s: all 6 injected mistakes found with the right colour and kind" % preset, ok,
          {w: (got[w]["s"], kinds(got[w])) if w in got else None for w in exp})
    check("errors/%s: zero false positives" % preset, set(got) == set(exp), set(got) - set(exp))
spoken, exp = inject(BASE)
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=PAUSES, seed=5), preset="relaxed")
bal_, rel_, str_ = M.get_preset("balanced"), M.get_preset("relaxed"), M.get_preset("strict")
def mk(exp, seg, p): return [i.k for i in M.judge_word([exp], seg, None, p)[0]]
nat = "سَمَااوَاتِ"
check("madd natural: exact 2 ok everywhere", all(mk(nat, nat, p) == [] for p in (bal_, rel_, str_)))
check("madd natural shortened to 1: balanced+strict flag, relaxed tolerates",
      mk(nat, nat.replace("اا", "ا", 1), bal_) == ["madd_short"] and mk(nat, nat.replace("اا", "ا", 1), str_) == ["madd_short"]
      and mk(nat, nat.replace("اا", "ا", 1), rel_) == [])
check("madd natural stretched to 3: only strict flags; to 4: balanced flags too; relaxed tolerates 4",
      mk(nat, nat.replace("اا", "ااا", 1), bal_) == [] and mk(nat, nat.replace("اا", "ااا", 1), str_) == ["madd_long"]
      and mk(nat, nat.replace("اا", "اااا", 1), bal_) == ["madd_long"] and mk(nat, nat.replace("اا", "اااا", 1), rel_) == [])
check("madd dropped entirely is flagged in every preset", all(mk(nat, nat.replace("اا", "", 1), p) == ["madd_dropped"] for p in (bal_, rel_, str_)))
mut = "شَااااءَ"
check("madd muttasil: 4 ok, 2 flagged (4 harakat needed)", mk(mut, mut, bal_) == [] and mk(mut, "شَااءَ", bal_) == ["madd_short"])

# ---- the two real mistakes seen in the live log (QPS uses ۦ for kasra-madd): must read as ONE tashkeel issue
iss = M.judge_word(["لمِهِۦۦۦۦ"], "لمِهَاا", None, str_)[0]
check("live log w36 (هِۦ said as هَا): a single yellow tashkeel issue, no confusing tajweed/letters note",
      [i.t for i in iss] == ["tashkeel"] and iss[0].k == "vowel_swap", [(i.t, i.k) for i in iss])
iss = M.judge_word(["اشَااااءَ"], "اشِۦۦۦۦءَ", None, str_)[0]
check("live log w39 (شَاءَ said as شِيء): single tashkeel issue", [i.t for i in iss] == ["tashkeel"], [(i.t, i.k) for i in iss])
iss = M.judge_word(["بِمَاا"], "بِمَۦۦ", None, str_)[0]
check("madd letter swapped alone (ا -> ۦ) is reported as tashkeel 'madd_vowel_swap' with Arabic names",
      [(i.t, i.k) for i in iss] == [("tashkeel", "madd_vowel_swap")] and "الياء" in iss[0].n and "الألف" in iss[0].n, [(i.t, i.k) for i in iss])
iss = M.judge_word(["بِمَاا"], "بِمَيي", None, str_)[0]
check("madd alif said as a real ya is a letters error", [(i.t, i.k) for i in iss] == [("letters", "replace")], [(i.t, i.k) for i in iss])

# ---- segmentation ladder: the library's spaced output may misbehave in several ways
def bounds(r): return r.starts
for mode in ("none", "waqf_all", "merge"):
    r2 = M.build_ref_aya(2, 255, sim.FakeLib(spaced=mode))
    drift = max(abs(a - b) for a, b in zip(bounds(r2), bounds(ref)))
    want = ("spaced+aligned",) if mode == "merge" else ("aligned",)
    check("ladder/%s: spaced output unusable/merged -> %s segmentation, boundaries within 2 chars" % (mode, want[0]),
          r2.info["segmentation"] in want and drift <= 2 and r2.n == 50, (r2.info.get("segmentation"), drift))
    check("ladder/%s: no word starts with a vowel mark or silah letter (leaked-vowel artefact)" % mode,
          all(not M.is_mark(f[0]) and f[0] not in "ۥۦ" for f in r2.wasl), [f[:3] for f in r2.wasl if M.is_mark(f[0]) or f[0] in "ۥۦ"])
    check("ladder/%s: stop forms still found" % mode, r2.info.get("waqf_forms", 0) >= 15, r2.info.get("waqf_forms"))
    eng, log, final, sp = sim.run_stream(r2, sim.Speech(ref, BASE, pauses=PAUSES, seed=3), preset="strict")
    check("ladder/%s: clean recitation -> 50 ok, none flagged (strict)" % mode,
          final["summary"]["ok"] == 50 and not flagged(final), (final["summary"], list(flagged(final))[:6]))
    spoken, exp = inject(BASE)
    eng, log, final, sp = sim.run_stream(r2, sim.Speech(ref, spoken, pauses=PAUSES, seed=5), preset="strict")
    got = flagged(final)
    check("ladder/%s: the 6 injected mistakes are still found (strict), no extras" % mode,
          set(got) == set(exp) and all(got[w]["s"] == e[0] for w, e in exp.items()),
          {w: got[w]["s"] if w in got else None for w in exp} | {"extra": sorted(set(got) - set(exp))})
r_adopt = M.build_ref_aya(2, 255, sim.FakeLib(spaced="ok"))
check("ladder/ok: exact spaced segmentation is preferred", r_adopt.info["segmentation"] == "spaced-exact")

# ---- stop-form without a stop: silence in the audio decides
def waqf_case(preset, pause):
    spoken = list(BASE); spoken[9] = ref.waqf[9]            # سِنَةٌ said in stop form
    p = set(PAUSES) | ({9} if pause else set()); p = p - ({9} if not pause else set())
    eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=p, seed=7), preset=preset)
    return flagged(final)
g = waqf_case("strict", pause=False)
check("waqf without a pause: strict flags the dropped final vowel (yellow)",
      list(g) == [9] and g[9]["s"] == "tashkeel" and kinds(g[9]) == ["waqf_no_pause"], {w: (g[w]["s"], kinds(g[w])) for w in g})
check("waqf WITH a real pause: strict does not flag", waqf_case("strict", pause=True) == {})
check("waqf without a pause: balanced does not flag (default stays conservative)", waqf_case("balanced", pause=False) == {})

# ---------------------------------------------------------------- robustness (all must be clean)
cases = {
    "fast reciter": dict(scale=0.6, pauses=PAUSES, seed=11),
    "slow reciter": dict(scale=1.7, pauses=PAUSES, seed=12),
    "no pauses (all wasl)": dict(pauses=set(), seed=13, spoken=WASL),
    "waqf form on every word": dict(pauses=set(range(50)), seed=14, spoken=[ref.waqf[k] or ref.wasl[k] for k in range(50)]),
    "opening du'a before the verse": dict(pauses=PAUSES, seed=15, junk=sim.fake_ph("أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ")),
    "model noise (3 %, low confidence)": dict(pauses=PAUSES, seed=17, noise_p=0.03),
    "3-second pause mid-verse": dict(pauses=PAUSES | {29}, seed=19),
}
for name, kw in cases.items():
    spoken = kw.pop("spoken", BASE)
    eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, spoken, **kw))
    check("robust: " + name, final["summary"]["complete"] and not flagged(final), (final["summary"], list(flagged(final))[:5]))
st = list(BASE); st[20] = st[19] + st[20]
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, st, pauses=PAUSES, seed=18))
check("robust: stutter (previous word repeated)", final["summary"]["complete"] and not flagged(final))
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, BASE, pauses=PAUSES, seed=16, stop_after=24, trail=0.8))
check("stop halfway: ~25 words judged, none flagged, not marked complete",
      20 <= final["summary"]["judged"] <= 26 and not flagged(final) and not final["summary"]["complete"], final["summary"])
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, BASE, pauses=PAUSES, seed=20), rtf=0.6)
check("slow server (rtf 0.6): still complete and clean", final["summary"]["complete"] and not flagged(final))
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, BASE, pauses=PAUSES, seed=20), rtf=1.3)
check("overloaded server (rtf 1.3): completes, window capped at 10 s",
      final["summary"]["complete"] and max(s["wave_s"] for s in eng.debug["steps"]) <= 10.01)

# ---------------------------------------------------------------- burst replay (reconnect)
spoken, _ = inject(BASE)
sp = sim.Speech(ref, spoken, pauses=PAUSES, seed=5); holder = {}
eng = M.SessionEngine(ref, sp.recognizer(holder), M.get_preset("balanced")); holder["engine"] = eng
eng.add_audio(sp.audio)
while eng.ready_to_run() and not eng.complete: eng.step()
final = eng.flush()
check("burst replay (whole recording at once): same 6 mistakes, nothing else", set(flagged(final)) == {3, 14, 15, 21, 27, 33}, list(flagged(final)))

# ---------------------------------------------------------------- silence
eng = M.SessionEngine(ref, lambda w: M.Recognition("", None, None), M.get_preset("balanced"))
rng = np.random.default_rng(0)
for _ in range(100):
    eng.add_audio((rng.standard_normal(1600) * 0.0008).astype(np.float32))
    if eng.ready_to_run(): eng.step()
check("silence only: nothing is judged", eng.seq == 0 and eng.tracked == -1)

# ---------------------------------------------------------------- sifat (qalqalah / tafkheem)
def sifat_run(preset, flip):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); holder = {}; plain = sp.recognizer(holder)
    def rec(wave):
        r = plain(wave); items = []
        for g in sim.fake_groups(r.text):
            letter = next((c for c in g.phonemes if not M.is_mark(c)), ""); d = {"phonemes_group": g.phonemes}
            for lv in ("qalqla", "tafkheem_or_taqeeq", "ghonna"):
                d[lv] = types.SimpleNamespace(text=flip(lv, letter, getattr(g, lv)), prob=0.97)
            items.append(d)
        return M.Recognition(r.text, r.probs, M.convert_sifat(items))
    eng = M.SessionEngine(ref, rec, M.get_preset(preset)); holder["engine"] = eng
    t = 0.0
    while t < sp.total and not eng.complete:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.1) * M.SR)]); t += 0.1
        if eng.ready_to_run(): eng.step()
    return flagged(eng.flush())
g = sifat_run("balanced", lambda lv, l, lab: lab)
check("sifat: faithful sifat -> no flags", not g)
g = sifat_run("balanced", lambda lv, l, lab: "not_moqalqal" if (lv == "qalqla" and l == "ط") else lab)
check("sifat: missing qalqalah on ط is reported (orange)", list(g) == [33] and g[33]["s"] == "tajweed" and kinds(g[33]) == ["qalqla"], g)
g = sifat_run("balanced", lambda lv, l, lab: "moraqaq" if (lv == "tafkheem_or_taqeeq" and l == "ر") else lab)
check("sifat: light ر reported as tafkheem error", set(g) == {18, 41, 43}, list(g))
g = sifat_run("relaxed", lambda lv, l, lab: "moraqaq" if (lv == "tafkheem_or_taqeeq" and l == "ر") else lab)
check("sifat: relaxed ignores tafkheem", not g)


# ---- regression tests built from the REAL strings in the live report
real_groups = ["ءَللَااهُ", "لَاااا", "ءِلَااهَ", "ءِللَاا", "هُوَ", "لحَييُ", "لقَييُۥۥمُ", "لَاا", "تَءخُذُهُۥۥ", "سِنَتُوووَلَاا"]
real_F = "".join(real_groups)
gs, acc = [], 0
for gr in real_groups[:-1]:
    acc += len(gr); gs.append(acc)
est = [0] + [g + (1 if i % 2 == 0 else -1) for i, g in enumerate(gs)] + [len(real_F)]    # +-1 estimation noise
snapped, exact = M._snap_to_library_groups(est, gs)
check("real log: estimated boundaries snap to the library's exact group edges",
      snapped[1:-1] == gs and all(exact[1:-1]), (snapped, gs))
leak = "كُرسِييُهُ" + "لسسَمَااوَاتِ"
b = [0, len("كُرسِييُه"), len(leak)]                    # boundary placed BEFORE the final damma (real artefact)
check("real log: a final vowel mark never leaks into the next word", M.refine_bounds(leak, b)[1] == len("كُرسِييُهُ"), M.refine_bounds(leak, b))
leak2 = "هُوَ" + "لعَلِييُ"
check("real log: silah/madd letters stay with the previous word",
      M.refine_bounds("هُو" + "ۥۥ" + "لحَييُ", [0, 3, 9])[1] == 5)

# ---- shaddah / ghunna length strictness (junction 'ووو' = idgham with ghunna)
junction = "سِنَتُوووَلَاا"
def jk(seg, p): return [i.k for i in M.judge_word([junction], seg, None, M.get_preset(p))[0]]
check("ghunna/shaddah run: exact length ok everywhere", all(jk(junction, p) == [] for p in ("relaxed", "balanced", "strict")))
check("ghunna/shaddah run: one unit short -> strict flags, balanced+relaxed tolerate",
      jk("سِنَتُووَلَاا", "strict") == ["shaddah_dropped"] and jk("سِنَتُووَلَاا", "balanced") == [] and jk("سِنَتُووَلَاا", "relaxed") == [])
check("ghunna/shaddah run: nearly gone -> balanced and strict flag",
      jk("سِنَتُوَلَاا", "balanced") == ["shaddah_dropped"] and jk("سِنَتُوَلَاا", "strict") == ["shaddah_dropped"])

# ---- tolerant sifat mapping (22 % of real windows had pred_mapped=false)
grp = [{"g": "ءَ"}, {"g": "لل"}, {"g": "َا"}]
m = M.tolerant_group_map(grp, "ءَللَِا")
check("sifat mapping: survives a doubled vowel mark at a window edge", m == [0, 0, 1, 1, 2, 2, 2], m)
check("sifat mapping: refuses unrelated text", M.tolerant_group_map(grp, "بتثجحخدذرزسشص") is None)

# ---- [PAD] labels (prob 0.0, seen in the live report) must never produce a flag
def sifat_pad(preset):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); holder = {}; plain = sp.recognizer(holder)
    def rec(wave):
        r = plain(wave); items = []
        for g in sim.fake_groups(r.text):
            items.append({"phonemes_group": g.phonemes,
                          "tafkheem_or_taqeeq": types.SimpleNamespace(text="[PAD]", prob=0.0),
                          "qalqla": types.SimpleNamespace(text="[PAD]", prob=0.0),
                          "ghonna": types.SimpleNamespace(text="[PAD]", prob=0.0)})
        return M.Recognition(r.text, r.probs, M.convert_sifat(items))
    eng = M.SessionEngine(ref, rec, M.get_preset(preset)); holder["engine"] = eng
    t = 0.0
    while t < sp.total and not eng.complete:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.1) * M.SR)]); t += 0.1
        if eng.ready_to_run(): eng.step()
    return flagged(eng.flush())
check("sifat: [PAD] / prob 0.0 predictions never flag a word", sifat_pad("strict") == {})


# ====================================================================== v14.3: regression tests from the live report
import re as _re
# ---- the live failure: a recitation recorded while the server was cold-starting arrives as one big backlog
def backlog_run(preset="strict", burst_s=50.0, chunk_s=10.0, spoken=None):
    """~50 s of audio recorded while the server was cold-starting arrives at once, then the rest trickles in."""
    sp = sim.Speech(ref, spoken or BASE, scale=2.0, pauses=PAUSES, seed=21, lead=10.0, pause_s=1.2)
    holder = {}; eng = M.SessionEngine(ref, sp.recognizer(holder), M.get_preset(preset)); holder["engine"] = eng
    eng.add_audio(sp.audio[: int(burst_s * M.SR)]); t = burst_s
    if eng.ready_to_run(): eng.step()
    while t < sp.total:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + chunk_s) * M.SR)]); t += chunk_s
        if eng.ready_to_run(): eng.step()                 # ONE window per arrival -> the backlog stays large
    return eng, eng.flush(), sp
eng, fin, sp = backlog_run()
check("live failure: a %.0f s recitation delivered as a growing backlog is judged completely (50/50)" % sp.total,
      fin["summary"]["judged"] == 50 and fin["summary"]["complete"] and not flagged(fin), fin["summary"])
saved = M.MAX_BACKLOG_S
M.MAX_BACKLOG_S = 45.0                                    # the v14.2 limit: this must FAIL (proves the test is meaningful)
eng_old, fin_old, _sp = backlog_run()
M.MAX_BACKLOG_S = saved
check("live failure: the old 45 s limit really did strand words (test is meaningful)", fin_old["summary"]["judged"] < 40, fin_old["summary"]["judged"])

# ---- adjacent swapped sifat labels (the live report: meem loses ghunna, next lam gains it) are not mistakes
def sifat_custom(mutate, preset="strict"):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); holder = {}; plain = sp.recognizer(holder)
    def rec(wave):
        r = plain(wave); groups = sim.fake_groups(r.text); items = []
        for g in groups:
            items.append({"phonemes_group": g.phonemes,
                          "qalqla": types.SimpleNamespace(text=g.qalqla, prob=0.97),
                          "tafkheem_or_taqeeq": types.SimpleNamespace(text=g.tafkheem_or_taqeeq, prob=0.97),
                          "ghonna": types.SimpleNamespace(text=g.ghonna, prob=0.97)})
        mutate(items)
        return M.Recognition(r.text, r.probs, M.convert_sifat(items))
    eng = M.SessionEngine(ref, rec, M.get_preset(preset)); holder["engine"] = eng
    t = 0.0
    while t < sp.total and not eng.complete:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.1) * M.SR)]); t += 0.1
        if eng.ready_to_run(): eng.step()
    return flagged(eng.flush())
def lbl(items, i, key, text): items[i][key] = types.SimpleNamespace(text=text, prob=0.97)
APPLIED = {"n": 0}
def first_letter(g): return next((c for c in g if not M.is_mark(c)), "")
def candidates(items):
    """(i, i+1): a noon/meem group followed by a group of a letter that can never carry ghunna"""
    return [i for i in range(len(items) - 1)
            if first_letter(items[i]["phonemes_group"]) in "نم"
            and first_letter(items[i + 1]["phonemes_group"]) not in "نمويى" and first_letter(items[i + 1]["phonemes_group"])]
def swap_pair(items):
    for i in candidates(items):
        lbl(items, i, "ghonna", "not_maghnoon"); lbl(items, i + 1, "ghonna", "maghnoon"); APPLIED["n"] += 1
def only_first(items):
    for i in candidates(items):
        lbl(items, i, "ghonna", "not_maghnoon"); APPLIED["n"] += 1
def extra_on_plain(items):
    for i in range(len(items)):
        if first_letter(items[i]["phonemes_group"]) in "لتبفكس":
            lbl(items, i, "ghonna", "maghnoon"); APPLIED["n"] += 1
APPLIED["n"] = 0
g_swap = sifat_custom(swap_pair); n_swap = APPLIED["n"]
check("sifat: swapped neighbouring ghunna labels (live report w6/w7) are not flagged (mutation applied %d times)" % n_swap,
      n_swap > 10 and g_swap == {}, (n_swap, list(g_swap)))
APPLIED["n"] = 0
g_one = sifat_custom(only_first); n_one = APPLIED["n"]
check("sifat: a genuinely missing ghunna (no neighbour swap) is still flagged (applied %d times)" % n_one,
      n_one > 10 and any("ghonna" in kinds(w) for w in g_one.values()), (n_one, list(g_one)))
APPLIED["n"] = 0
g_extra = sifat_custom(extra_on_plain); n_extra = APPLIED["n"]
check("sifat: 'ghunna' on ل/ت/ب/ف/ك/س is impossible and never flagged (applied %d times)" % n_extra,
      n_extra > 10 and g_extra == {}, (n_extra, list(g_extra)))
# prove the guards are what make those pass: with the guards disabled the same mutations DO flag words
_sg, _sh = M.GHUNNA_LETTERS, M.SessionEngine._label_shift
M.GHUNNA_LETTERS = set("ءابتثجحخدذرزسشصضطظعغفقكلمنهوي"); M.SessionEngine._label_shift = lambda *a, **k: False
g_nog = sifat_custom(swap_pair); g_nog2 = sifat_custom(extra_on_plain)
M.GHUNNA_LETTERS, M.SessionEngine._label_shift = _sg, _sh
check("sifat: with the guards disabled those same mutations are flagged (the tests have teeth)", len(g_nog) > 0 and len(g_nog2) > 0, (len(g_nog), len(g_nog2)))

# ---- a final vowel kept while the stop form is cheaper: report the VOWEL mistake, not 'vowel on a sukun'
iss = M.judge_word(["تَءخُذُهُۥۥ", "تَءخُذُه"], "تَءخُذُهَاا", None, M.get_preset("strict"))[0]
check("live log w8 (تَأْخُذُهُ said ـهَا): one clear vowel_swap on ه (not 'vowel on a sukun')",
      [(i.t, i.k) for i in iss] == [("tashkeel", "vowel_swap")] and "«ه»" in iss[0].n, [(i.t, i.k) for i in iss])

# ---- client silence gate: dead air is dropped, but pauses/word edges/madd are untouched
for lbl_, kw in (("long 3 s pauses + 4 s lead/trail", dict(pause_s=3.0, lead=4.0, trail=4.0)),
                 ("normal pauses", dict(pause_s=0.6, lead=1.0, trail=1.5))):
    sp0 = sim.Speech(ref, BASE, pauses=PAUSES, seed=3, **kw)
    sp1, pf = sim.client_trim(sp0)
    saved_pct = 100.0 * (1 - len(sp1.audio) / len(sp0.audio))
    eng, log, fin, _ = sim.run_stream(ref, sp1, preset="strict", pauses=pf)
    check("silence gate (%s): saves %.0f%% of the audio and the verdict is unchanged (50 ok)" % (lbl_, saved_pct),
          fin["summary"]["ok"] == 50 and not flagged(fin) and (saved_pct > 25 if "long" in lbl_ else True), (fin["summary"], saved_pct))
spoken, exp = inject(BASE)
sp1, pf = sim.client_trim(sim.Speech(ref, spoken, pauses=PAUSES, seed=5, pause_s=3.0, lead=4.0, trail=4.0))
eng, log, fin, _ = sim.run_stream(ref, sp1, preset="strict", pauses=pf); got = flagged(fin)
check("silence gate: all 6 injected mistakes still found, nothing extra", set(got) == set(exp), (sorted(got), sorted(exp)))
for pause, expect in ((True, set()), (False, {9})):
    spoken9 = list(BASE); spoken9[9] = ref.waqf[9]
    p = (set(PAUSES) | {9}) if pause else (set(PAUSES) - {9})
    sp1, pf = sim.client_trim(sim.Speech(ref, spoken9, pauses=p, seed=7, pause_s=3.0))
    eng, log, fin, _ = sim.run_stream(ref, sp1, preset="strict", pauses=pf)
    check("silence gate keeps the stop-form logic exact (%s a real pause)" % ("with" if pause else "without"),
          set(flagged(fin)) == expect, list(flagged(fin)))

# ---- Android <-> backend contract audit
HERE = os.path.dirname(os.path.abspath(__file__))
kt_dir = os.path.join(HERE, "..", "..", "android", "app", "src", "main", "java", "com", "rafiq", "app")
kt = open(os.path.join(kt_dir, "RecitationSession.kt"), encoding="utf-8").read()
cfg_kt = open(os.path.join(kt_dir, "RecitationConfig.kt"), encoding="utf-8").read()
kotlin_keys = set(_re.findall(r'opt(?:String|Int|Long|Boolean|JSONArray|JSONObject)\("([A-Za-z_]+)"', kt))
spoken, exp = inject(BASE)
eng, log, fin, _ = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=PAUSES, seed=5), preset="strict")
msgs = [m for _t, m in log] + [fin]
fin_ws = None
def all_keys(o, acc):
    if isinstance(o, dict):
        for k, v in o.items(): acc.add(k); all_keys(v, acc)
    elif isinstance(o, list):
        for v in o: all_keys(v, acc)
    return acc
server_keys = set()
for m_ in msgs: all_keys(m_, server_keys)
server_keys |= {"words", "type", "resume_word_idx"}   # 'ready' carries the word list, 'paused_idle' the checkpoint (both exercised by the websocket tests)
server_keys |= {"reason"}   # key of the JSON body of the HTTP 403 handshake refusal (asserted in dev/test_security.py section 4b)
final_msg = dict(eng.final_message()); final_msg["debug"] = M.debug_snapshot(eng, sim_runtime := types.SimpleNamespace(status=lambda: {}))
all_keys(final_msg, server_keys)
missing = sorted(k for k in kotlin_keys if k not in server_keys and k != "message")
check("contract: every JSON field the Android app reads exists in the server messages", not missing, missing)
check("contract: server sends the 'message' field of error messages", '"message"' in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read())
check("contract: app sends {\"type\":\"stop\"} and the server understands it",
      '{\\"type\\":\\"stop\\"}' in kt and 'kind == "stop"' in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read())
check("contract: websocket query parameters match", all(k in cfg_kt for k in ("sura_idx=", "aya_idx=", "strictness="))
      and all(('qp.get("%s"' % k) in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read() for k in ("sura_idx", "aya_idx", "strictness")))
check("contract: audio is 16 kHz mono PCM16 little-endian on both sides",
      "SAMPLE_RATE: Int = 16000" in cfg_kt and "SR = 16000" in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read()
      and "(s shr 8) and 0xFF" in kt and 'cast("h")' in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read()
      and "dtype=np.int16" in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read())


# ====================================================================== v14.3b: Hafs policy, ghunna symbol, ibtida', sub-phonemic
S_, B_, R_ = M.get_preset("strict"), M.get_preset("balanced"), M.get_preset("relaxed")
def mkk(exp, seg, p): return [i.k for i in M.judge_word([exp], seg, None, p)[0]]
# madd munfasil / silah kubra: balanced + relaxed accept 2 (qasr) .. 5 (Hafs 'an 'Asim); v14.5 STRICT follows the
# reference's configured length (4 or 5) and flags qasr / 3 harakat (live report #2 w1 and w36)
munf = "لَاااا"
check("v14.5 strict: madd munfasil with 2 harakat (qasr) is flagged madd_short; balanced and relaxed still accept it",
      mkk(munf, "لَاا", S_) == ["madd_short"] and mkk(munf, "لَاا", B_) == [] and mkk(munf, "لَاا", R_) == [])
check("v14.5 strict: madd munfasil with 3 harakat is flagged; 4 and 5 are valid", mkk(munf, "لَاا" + "ا", S_) == ["madd_short"]
      and all(mkk(munf, "لَ" + "ا" * n, S_) == [] for n in (4, 5)))
check("hafs: madd munfasil dropped to 1 harakah is an error", mkk(munf, "لَا", S_) == ["madd_short"])
check("hafs: madd munfasil stretched past 5: strict flags 6, balanced tolerates 6 but flags 7",
      mkk(munf, "لَ" + "ا" * 6, S_) == ["madd_long"] and mkk(munf, "لَ" + "ا" * 6, B_) == []
      and mkk(munf, "لَ" + "ا" * 7, B_) == ["madd_long"])
silah = "عِںںںدَهُۥۥۥۥ"
check("v14.5 strict: silah kubra with 2 harakat is flagged (live report #2 w36 pattern); balanced and relaxed accept it",
      mkk(silah, "عِںںںدَهُۥۥ", S_) == ["madd_short"] and mkk(silah, "عِںںںدَهُۥۥ", B_) == [] and mkk(silah, "عِںںںدَهُۥۥ", R_) == [])
check("hafs: madd muttasil still needs 4 (2 is an error)", mkk("شَااااءَ", "شَااءَ", S_) == ["madd_short"] and mkk("شَااااءَ", "شَااااءَ", S_) == [])
arid = "لعَظِۦۦۦۦم"
check("hafs: madd 'arid at a stop is valid at 2, 4 and 6 (7 flagged)",
      all(mkk(arid, "لعَظِ" + "ۦ" * n + "م", S_) == [] for n in (2, 4, 6)) and mkk(arid, "لعَظِ" + "ۦ" * 7 + "م", S_) == ["madd_long"])
# ghunna symbol runs (ikhfaa): ںںں in the live output
ikh = "مَںںںذَ"
check("ghunna ikhfaa: exact length ok; one short -> strict flags; two short -> balanced+strict flag; relaxed tolerates",
      all(mkk(ikh, ikh, p) == [] for p in (S_, B_, R_))
      and mkk(ikh, "مَںںذَ", S_) == ["ghunna_short"] and mkk(ikh, "مَںںذَ", B_) == []
      and mkk(ikh, "مَںذَ", B_) == ["ghunna_short"] and mkk(ikh, "مَںذَ", S_) == ["ghunna_short"] and mkk(ikh, "مَںذَ", R_) == [])
check("ghunna ikhfaa: ikhfaa not applied at all (noon instead of the ghunna symbol) is a tajweed 'rule' error",
      mkk(ikh, "مَنذَ", S_) == ["rule"])
# liin letter stretched like a madd (ya of بَيْنَ)
check("liin letter: بَيْنَ with an elongated ya -> strict flags, balanced tolerates 2 but flags 3",
      mkk("بَينَ", "بَينَ", S_) == [] and mkk("بَينَ", "بَيينَ", S_) == ["lin_extended"]
      and mkk("بَينَ", "بَيينَ", B_) == [] and mkk("بَينَ", "بَييينَ", B_) == ["lin_extended"])

# ---- vowel on a different letter is alignment drift (stutter after a stop), a real swap on the same letter is not
check("drift: a heard vowel that belongs to another letter is not a vowel_swap",
      M.judge_word(["لءَرضِ"], "لءَرضمَن", None, S_, lead=0, trail=3)[0] == []
      and [i.k for i in M.judge_word(["لءَرضِ"], "لءَرضَ", None, S_)[0]] == ["vowel_swap"])
# the same alignment-drift case at window level: reciter stops, then repeats the next word
st_ = list(BASE); st_[20] = st_[19] + st_[20]
eng_, log_, fin_, sp_ = sim.run_stream(ref, sim.Speech(ref, st_, pauses=PAUSES, seed=18), preset="strict")
check("drift: stop + repeating the next word is judged clean (50 ok, strict)", fin_["summary"]["ok"] == 50 and not flagged(fin_), {w: kinds(x) for w, x in flagged(fin_).items()})

# ---- ibtida' (start) forms: after a pause the word starts fresh (live report w11->w12: tanween idgham not applied)
check("ibtida': start forms were computed (hamzat al-wasl / no assimilation)", ref.info.get("start_forms", 0) >= 6, ref.info.get("start_forms"))
check("ibtida': لَهُ after a paused نَوْمٌ has a distinct start form (single lam, not the idgham 'لل')",
      ref.start_forms[12] and ref.start_forms[12] != ref.wasl[12] and ref.wasl[12].startswith("لل") and not ref.start_forms[12].startswith("لل"),
      (ref.wasl[12], ref.start_forms[12]))
def with_ibtida(r, pauses):
    out = []
    for k in range(r.n):
        after, at = (k - 1) in pauses, k in pauses
        if after and at: f = r.start_waqf[k] or r.start_forms[k] or r.waqf[k] or r.wasl[k]
        elif after: f = r.start_forms[k] or r.wasl[k]
        elif at: f = r.waqf[k] or r.wasl[k]
        else: f = r.wasl[k]
        out.append(f)
    return out
IB_PAUSES = PAUSES | {9, 11}
spoken_ib = with_ibtida(ref, IB_PAUSES)
eng, log, fin, sp = sim.run_stream(ref, sim.Speech(ref, spoken_ib, pauses=IB_PAUSES, seed=31), preset="strict")
check("ibtida': a recitation that restarts words fresh after every pause is 50/50 correct (strict)",
      fin["summary"]["ok"] == 50 and not flagged(fin), (fin["summary"], {w: kinds(x) for w, x in flagged(fin).items()}))
saved_sf, saved_sw = list(ref.start_forms), list(ref.start_waqf)
ref.start_forms, ref.start_waqf = [None] * ref.n, [None] * ref.n
eng, log, fin_no, sp = sim.run_stream(ref, sim.Speech(ref, spoken_ib, pauses=IB_PAUSES, seed=31), preset="strict")
ref.start_forms, ref.start_waqf = saved_sf, saved_sw
check("ibtida': without start forms the same correct recitation IS falsely flagged (the test has teeth)", len(flagged(fin_no)) > 0, len(flagged(fin_no)))


# ---- a fresh start without a stop: hamzat al-wasl pronounced / idgham not applied in continuous reading
def fresh_start_case(preset, k, pause_before):
    spoken = list(BASE); spoken[k] = ref.start_forms[k]
    p = set(PAUSES); p = (p | {k - 1}) if pause_before else (p - {k - 1})
    if not pause_before:
        spoken[k - 1] = ref.wasl[k - 1]          # no stop: the previous word is spoken in its continuing form
    eng, log, fin, sp = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=p, seed=41), preset=preset)
    return flagged(fin)
g1 = fresh_start_case("strict", 5, pause_before=False)                      # الْحَيُّ said as ءَلحَيّ right after هُوَ
check("hamzat al-wasl pronounced in continuous reading (strict): flagged as a tajweed 'start_no_pause'",
      list(g1) == [5] and g1[5]["s"] == "tajweed" and kinds(g1[5]) == ["start_no_pause"], {w: (x["s"], kinds(x)) for w, x in g1.items()})
check("the same fresh start AFTER a real stop is correct (no flag)", fresh_start_case("strict", 5, pause_before=True) == {})
check("balanced does not apply the start-without-pause check (conservative default)", fresh_start_case("balanced", 5, pause_before=False) == {})
g2 = fresh_start_case("strict", 12, pause_before=False)                     # لَهُ said with a single lam although نَوْمٌ was not stopped on
check("idgham not applied after نَوْمٌ in continuous reading (strict): flagged", 12 in g2 and "start_no_pause" in kinds(g2[12]), {w: kinds(x) for w, x in g2.items()})

# ---- strict: sub-phonemic mistakes (right phoneme, wrong articulation) via the other sifat levels
def sifat_generic(preset, letter, key, new_label):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); holder = {}; plain = sp.recognizer(holder)
    def rec(wave):
        r = plain(wave); items = []
        for g in sim.fake_groups(r.text):
            L = next((c for c in g.phonemes if not M.is_mark(c)), "")
            d = {"phonemes_group": g.phonemes}
            for lv in ("qalqla", "tafkheem_or_taqeeq", "ghonna", "hams_or_jahr"):
                lab = getattr(g, lv)
                if lv == key and L == letter: lab = new_label
                d[lv] = types.SimpleNamespace(text=lab, prob=0.95)
            items.append(d)
        return M.Recognition(r.text, r.probs, M.convert_sifat(items))
    eng = M.SessionEngine(ref, rec, M.get_preset(preset)); holder["engine"] = eng
    t = 0.0
    while t < sp.total and not eng.complete:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.1) * M.SR)]); t += 0.1
        if eng.ready_to_run(): eng.step()
    return flagged(eng.flush())
g_s = sifat_generic("strict", "ب", "hams_or_jahr", "hams")
g_b = sifat_generic("balanced", "ب", "hams_or_jahr", "hams")
check("strict: ب articulated whispered (hams instead of jahr) is flagged as a sifa mistake; balanced ignores it",
      len(g_s) > 0 and all(any(k.startswith("sifa:") for k in kinds(w)) for w in g_s.values()) and g_b == {}, (len(g_s), len(g_b)))


# ---------------------------------------------------------------- real WebSocket handler with a fake socket
fa = types.ModuleType("fastapi")
class FastAPI:
    def __init__(self, **kw): self.routes = {}
    def add_middleware(self, *a, **k): pass
    def __getattr__(self, kind):
        if kind in ("get", "post", "websocket"):
            return lambda path: (lambda f: self.routes.__setitem__((kind, path), f) or f)
        raise AttributeError(kind)
fa.FastAPI = FastAPI; fa.WebSocket = object; fa.Request = object
mw = types.ModuleType("fastapi.middleware"); cors = types.ModuleType("fastapi.middleware.cors"); cors.CORSMiddleware = object
class _JR(dict):
    def __init__(self, x, status_code=200): super().__init__(x); self.status_code = status_code
resp = types.ModuleType("fastapi.responses"); resp.JSONResponse = _JR
sys.modules.update({"fastapi": fa, "fastapi.middleware": mw, "fastapi.middleware.cors": cors, "fastapi.responses": resp})

class FakeRuntime:
    int8 = False; device = "cpu"
    def __init__(self, speech):
        self.speech = speech
        self.ab = ((speech.audio * 32767).astype("<i2").astype(np.float32) / 32768.0).astype(np.float32).tobytes()
    def ensure_loaded(self): pass
    def get_ref(self, s, a): return ref
    def status(self): return {"status": "ok"}
    def recognize(self, wave, ref_):
        off = self.ab.find(wave[:800].tobytes()) // 4
        sel = [ch for (tm, ch, _w) in self.speech.chars if off / M.SR <= tm < (off + len(wave)) / M.SR]
        return M.Recognition("".join(sel), [0.97] * len(sel), None)

class FakeWS:
    def __init__(self, qp): self.query_params = qp; self.q = asyncio.Queue(); self.sent = []; self.closed = False
    async def accept(self): pass
    async def send_text(self, t): self.sent.append(json.loads(t))
    async def receive(self): return await self.q.get()
    async def close(self, code=1000):
        self.close_code = code
        if not self.closed: self.closed = True; await self.q.put({"type": "websocket.disconnect"})

async def ws_case(stop_at=None):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    handler = M.build_fastapi_app(FakeRuntime(sp)).routes[("websocket", "/ws/live-recitation")]
    ws = FakeWS({"sura_idx": "2", "aya_idx": "255", "strictness": "balanced"})
    task = asyncio.create_task(handler(ws)); end = sp.total if stop_at is None else stop_at; t = 0.0
    while t < end and not ws.closed:
        await ws.q.put({"type": "websocket.receive", "bytes": (sp.audio[int(t*M.SR): int((t+0.1)*M.SR)] * 32767).astype("<i2").tobytes()})
        t += 0.1; await asyncio.sleep(0.004)
    if not ws.closed: await ws.q.put({"type": "websocket.receive", "text": json.dumps({"type": "stop"})})
    await asyncio.wait_for(task, 30)
    return ws, [m for m in ws.sent if m["type"] == "final"]
async def ws_abrupt():
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    handler = M.build_fastapi_app(FakeRuntime(sp)).routes[("websocket", "/ws/live-recitation")]
    ws = FakeWS({}); task = asyncio.create_task(handler(ws))
    for i in range(30):
        await ws.q.put({"type": "websocket.receive", "bytes": (sp.audio[i*1600:(i+1)*1600] * 32767).astype("<i2").tobytes()}); await asyncio.sleep(0.003)
    await ws.q.put({"type": "websocket.disconnect"}); await asyncio.wait_for(task, 10)
async def ws_burst():
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    handler = M.build_fastapi_app(FakeRuntime(sp)).routes[("websocket", "/ws/live-recitation")]
    ws = FakeWS({"sura_idx": "2", "aya_idx": "255", "strictness": "strict"}); task = asyncio.create_task(handler(ws))
    pcm = (sp.audio * 32767).astype("<i2").tobytes()
    for i in range(0, len(pcm), 3200 * 10):
        await ws.q.put({"type": "websocket.receive", "bytes": pcm[i:i + 3200 * 10]})
    await ws.q.put({"type": "websocket.receive", "text": json.dumps({"type": "stop"})})
    await asyncio.wait_for(task, 60)
    return ws, [m for m in ws.sent if m["type"] == "final"]
async def ws_all():
    ws, fin = await ws_case(None)
    kinds_ = [m["type"] for m in ws.sent]
    check("websocket: ready -> progress... -> final(50 ok, complete) -> closed, no errors",
          kinds_[0] == "ready" and len(fin) == 1 and fin[0]["summary"]["ok"] == 50 and ws.closed and "error" not in kinds_, fin[:1])
    ws, fin = await ws_case(14.0)
    check("websocket: the final message carries the debug trace (server may scale to zero afterwards)",
          "debug" in fin[0] and "ref_info" in fin[0]["debug"] and len(fin[0]["debug"]["steps"]) > 0)
    ws, fin = await ws_case(14.0)
    check("websocket: stop button -> final with judged words, then closed", len(fin) == 1 and 15 <= fin[0]["summary"]["judged"] <= 24 and ws.closed, fin[:1])
    ws, fin = await ws_burst()
    prog = [m for m in ws.sent if m["type"] == "progress"]
    check("websocket: whole recording sent at once then stop -> progress keeps streaming while catching up, final is complete",
          len(prog) >= 4 and len(fin) == 1 and fin[0]["summary"]["judged"] == 50, (len(prog), fin[:1]))
    await ws_abrupt(); check("websocket: abrupt network drop ends the handler cleanly", True)
asyncio.run(ws_all())

# ======================================================================================================
# v14.5 regression tests  (live reports #1 and #2)
# ======================================================================================================
import copy, importlib.util, re as _re2, subprocess
PRESETS3 = ("strict", "balanced", "relaxed")
SHAA = ref.words.index("شَاءَ")

def run_spoken(spoken, preset="strict", ref_=None, **kw):
    r = ref_ or ref
    sp = sim.Speech(r, spoken, pauses=kw.pop("pauses", PAUSES), seed=kw.pop("seed", 5), **kw)
    return sim.run_stream(r, sp, preset=preset)

# ---------------------------------------------------------------- D6.1  madd muttasil on شَاءَ: 4..6 harakat in EVERY preset
check("index note: شَاءَ is w=39 in the 50-word 0-based array (#40 1-based); w=26 is يَعْلَمُ (not muttasil) - the rule is keyed on the phoneme pattern",
      SHAA == 39 and ref.words[26] == "يَعْلَمُ" and ref.wasl[SHAA] == "شَااااءَ"
      and [k for k in range(ref.n) for c, s0, e0 in M.split_runs(ref.wasl[k]) if c in "اۥۦ" and e0 - s0 >= 3
           and M.madd_kind(ref.wasl[k], s0, e0) == "muttasil"] == [SHAA])
def expect_madd(n):
    return ["madd_short"] if n < 4 else (["madd_long"] if n > 6 else [])
for form_exp, tail in (("شَااااءَ", "ءَ"), ("شَااااء", "ء")):               # wasl and waqf
    ok = all(mkk(form_exp, "شَ" + "ا" * n + tail, M.get_preset(p)) == expect_madd(n) for p in PRESETS3 for n in range(1, 9))
    check("madd muttasil %s: 1..3 harakat -> madd_short, 4..6 ok, 7..8 -> madd_long in strict, balanced AND relaxed" % ("wasl" if tail == "ءَ" else "waqf"), ok,
          {p: [(n, mkk(form_exp, "شَ" + "ا" * n + tail, M.get_preset(p))) for n in range(1, 9)] for p in PRESETS3})
for p in PRESETS3:                                                              # full ayah, w=39, 2..8 harakat
    got = {}
    for n in range(2, 9):
        spoken = list(BASE); spoken[SHAA] = "شَ" + "ا" * n + "ءَ"
        eng, log, fin, _ = run_spoken(spoken, p)
        got[n] = (sorted(flagged(fin)), kinds(flagged(fin).get(SHAA, {})))
    check("full ayah w=39 [%s]: 2,3 -> madd_short; 4,5,6 -> ok; 7,8 -> madd_long; no other word flagged" % p,
          all((got[n][0] == ([SHAA] if expect_madd(n) else []) and got[n][1] == expect_madd(n)) for n in got), got)
class SubLib(sim.FakeLib):                                                       # "word #26" convention: a reduced reference
    def get_uthmani(self, s, a): return " ".join(sim.WORDS[13:41])
ref26 = M.build_ref_aya(2, 255, SubLib(with_stop_token=False))
check("reduced reference: شَاءَ sits at index 26 (the spec-test indexing), followed by one more word", ref26.n == 28 and ref26.words[26] == "شَاءَ" and ref26.wasl[26] == "شَااااءَ")
for p in PRESETS3:
    got = {}
    for n in range(2, 9):
        spoken = [ref26.wasl[k] for k in range(ref26.n)]; spoken[26] = "شَ" + "ا" * n + "ءَ"
        eng, log, fin, _ = run_spoken(spoken, p, ref_=ref26, pauses=set())
        got[n] = (sorted(flagged(fin)), kinds(flagged(fin).get(26, {})))
    check("reduced ayah #26 [%s]: <4 -> madd_short, 4..6 ok, >6 -> madd_long (same rule, other index)" % p,
          all((got[n][0] == ([26] if expect_madd(n) else []) and got[n][1] == expect_madd(n)) for n in got), got)

# ---- audio-based measurement (CTC frames + 10 ms voiced envelope, pre-hamza dip trimmed)
def synth_env(n, mph=120, dip_ms=80, lead_ms=60):
    e = [0.05] * (lead_ms // 10) + [0.2 * (1 + 0.1 * np.sin(i)) for i in range(int(n * mph / 10))]
    e += [0.004] * (dip_ms // 10) + [0.12] * 6 + [0.002] * 20
    t0 = lead_ms / 1000.0; t_cap = (lead_ms + n * mph + dip_ms) / 1000.0
    return np.array(e), t0, t_cap
errs, dips = [], []
for n in range(2, 9):
    env, t0, t_cap = synth_env(n)
    h = M.harakat_from_audio(env, t0, t_cap, mph_ms=120)
    dur, trimmed = M.measure_madd_voiced(env, t0, t_cap)
    errs.append(abs(h - n)); dips.append(trimmed)
check("acoustic meter: 2..8 harakat measured within 0.2 harakah; the 80 ms silent pre-hamza dip is trimmed, not counted",
      max(errs) < 0.2 and all(abs(d - 80) <= 10 for d in dips), (errs, dips))
env, t0, t_cap = synth_env(4, dip_ms=0)
check("acoustic meter: the tempo is clamped to [85, 185] ms per harakah", M.clamp_mph(40) == 85 and M.clamp_mph(400) == 185 and M.clamp_mph(120) == 120)
ok = True; rows = {}
for p in PRESETS3:
    for n in range(2, 9):
        env, t0, t_cap = synth_env(n)
        h = M.harakat_from_audio(env, t0, t_cap, 120)
        got = [i.k for i in M.judge_word(["شَااااءَ"], "شَااااءَ", None, M.get_preset(p), acoustic=lambda a, b, h=h: h)[0]]
        want = ["madd_short"] if n <= 2 else (["madd_long"] if n >= 8 else [])
        rows[(p, n)] = got; ok = ok and got == want
check("audio vs decoded count: decoded says 4 alifs; a clearly short vowel (2 harakat) or clearly long one (8) overrides it, "
      "3..7 are inside the +-1 harakah measurement margin (measured 3.56 +- 0.18 for a perfect 4) so the decoded count stands (all presets)", ok, rows)
check("the decoded count alone is still exact: 3 -> madd_short, 7 -> madd_long, 4..6 ok when the model decodes them (all presets, no audio)",
      all(mkk("شَااااءَ", "شَ" + "ا" * n + "ءَ", M.get_preset(p)) == expect_madd(n) for p in PRESETS3 for n in range(1, 9)))

# ---- live report #2 w=39: the model decodes 4 alifs but the reciter held the madd only ~2 harakat
def mute(sp, w, ch, skip, pad=0.042):
    ts = [tm for tm, c, ww in sp.chars if ww == w and c == ch]
    sp.audio[int((ts[skip] - pad) * M.SR): int((ts[-1] + pad) * M.SR)] *= 0.02
for p in PRESETS3:
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    mute(sp, SHAA, "ا", 2)
    eng, log, fin, _ = sim.run_stream(ref, sp, preset=p)
    got = flagged(fin)
    check("live report #2 w=39 [%s]: decoded 4 alifs, audio only ~2 harakat -> madd_short (was ok); nothing else flagged" % p,
          sorted(got) == [SHAA] and kinds(got[SHAA]) == ["madd_short"], {w: kinds(x) for w, x in got.items()})
    info = eng.word_log[SHAA].get("madd_audio", [{}])[0]
    check("   ...the report carries the measurement (voiced ms, tempo, harakat)", info.get("harakat", 9) < 3.25 and info.get("mph", 0) >= 85, info)
eng, log, fin, _ = run_spoken(BASE, "strict")
check("honest 4-alif madd with calibrated tempo stays ok", not flagged(fin) and eng.mph_trusted() is not None and 85 <= eng.mph_trusted() <= 185, eng.local_mph())
def tail_speech(sp, t_from):
    """What the phone replays after a reconnect: only the audio from the checkpoint on."""
    out = copy.copy(sp); i0 = int(t_from * M.SR)
    out.audio = sp.audio[i0:].copy(); out.chars = [(tm - t_from, ch, w) for tm, ch, w in sp.chars if tm >= t_from]
    out.spans = [(a - t_from, b - t_from) for a, b in sp.spans]; out.total = len(out.audio) / M.SR
    return out
def resumed_engine(k):
    def f(r, rec, preset):
        e = M.SessionEngine(r, rec, preset); e.resume_from(k); return e
    return f
sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); mute(sp, SHAA, "ا", 2)
sp = tail_speech(sp, sp.spans[38][0] - 0.3)
eng, log, fin, _ = sim.run_stream(ref, sp, preset="strict", engine_factory=resumed_engine(38))
check("uncalibrated tempo (resumed at w=38: < 2 tabii runs measured when w=39 is judged): the audio channel abstains, the decoded count is judged",
      not eng.word_log[SHAA].get("madd_audio") and SHAA not in flagged(fin), (eng.word_log[SHAA].get("madd_audio"), eng.mph_samples))

# ---------------------------------------------------------------- D6.2  sifat_cmp: shifted tiles, SingleUnit idx/prob, domain gating, bleed
LV10 = ("hams_or_jahr", "shidda_or_rakhawa", "tafkheem_or_taqeeq", "itbaq", "safeer", "qalqla", "tikraar", "tafashie", "istitala", "ghonna")
def su(text, prob=0.97, idx=1): return types.SimpleNamespace(text=text, prob=prob, idx=idx)
def sprobe(w, mutate=None, preset="strict", clean=True, ref_=None):
    r = ref_ or ref
    eng = M.SessionEngine(r, lambda x: M.Recognition("", None, None), M.get_preset(preset))
    pred = r.wasl[w]
    objs = []
    for it in sim.fake_groups(pred):
        o = types.SimpleNamespace(phonemes=it.phonemes)
        for lv in LV10: setattr(o, lv, su(getattr(it, lv)))
        objs.append(o)
    if mutate: mutate(objs)
    ps = M.convert_sifat(objs)
    gmap = M.group_index_map(ps, pred) or M.tolerant_group_map(ps, pred)
    iss, tr = eng._sifat_issues(w, M.align(pred, pred), r.starts[w], pred, pred, gmap, ps, clean)
    return iss, tr
def slide(objs, d):                       # group i carries the labels of group i+d (the model's label sequence slides)
    snap = [{lv: copy.copy(getattr(o, lv)) for lv in LV10} for o in objs]
    for i, o in enumerate(objs):
        if 0 <= i + d < len(objs):
            for lv in LV10: setattr(o, lv, snap[i + d][lv])
def set_on(letter, **labels):
    def m(objs):
        for o in objs:
            if o.phonemes.startswith(letter):
                for lv, v in labels.items(): setattr(o, lv, v if not isinstance(v, str) else su(v))
    return m
def both(*fs):
    return lambda objs: [f(objs) for f in fs]
check("sifat: clean words produce no sifa issue (w14, w39, w42, w43, w49)", all(not sprobe(w)[0] for w in (14, 39, 42, 43, 49)))
iss, tr = sprobe(43, lambda o: slide(o, +1))
check("sifat w=43 (وَلءَرضَ, report #2): every property one group to the left -> no ل/ء/ر/ض hallucination (was shidda, tikraar, istitala, itbaq)",
      not iss and any(t[1].startswith("label_shift") or "label_shift" in t[1] for t in tr), [(i.k, i.exp, i.pred) for i in iss])
iss, tr = sprobe(42, lambda o: slide(o, +1))
check("sifat w=42/15 (سسَمَااوَااتِ, report #1 w15 pattern): shifted tiles -> م is not accused of hams / safeer / ghonna belonging to س", not iss, [(i.k, i.exp, i.pred) for i in iss])
iss, tr = sprobe(39, set_on("ش", tafashie=su("not_motafashie", 0.999)) if False else both(set_on("ش", tafashie=su("not_motafashie", 0.999)), set_on("ء", tafashie=su("motafashie", 0.998))))
check("sifat w=39 (شَاءَ, report #1): ش 'not_motafashie' + ء 'motafashie' (tafashie head slid 2 groups) -> neither letter accused (was both)",
      not iss and any("label_shift:tafashie" in t[1] for t in tr), [(i.k, i.exp, i.pred, i.focus) for i in iss])
iss, _ = sprobe(39, set_on("ء", tafashie=su("motafashie", 0.998)))
check("domain gating: tafashie on ء, tikraar on ء, istitala on ر are never evaluated (they exist only on ش / ر / ض)",
      not iss and not sprobe(43, set_on("ء", tikraar=su("mokarar", 0.99)))[0] and not sprobe(43, set_on("ر", istitala=su("mostateel", 0.99)))[0])
check("domain gating: a REAL mistake on the right letter is still found (ش loses tafashie, ض loses itbaq, ر loses tikraar)",
      [i.k for i in sprobe(39, set_on("ش", tafashie=su("not_motafashie")))[0]] == ["sifa:tafashie"]
      and [i.k for i in sprobe(43, set_on("ض", itbaq=su("monfateh")))[0]] == ["sifa:itbaq"]
      and [i.k for i in sprobe(43, set_on("ر", tikraar=su("not_mokarar")))[0]] == ["sifa:tikraar"])
# SingleUnit idx in the probability slot (report #2 w=42: conf 2 and 3)
iss, tr = sprobe(42, both(set_on("س", shidda_or_rakhawa=su("between", prob=2, idx=2)), set_on("ت", shidda_or_rakhawa=su("rikhw", prob=3, idx=3))))
confs = [i.conf for i in iss] + [t[4] for t in tr]
check("SingleUnit idx/prob: class index 2 / 3 in the probability slot is never a confidence - no issue, every conf in [0, 1]",
      not iss and all(0.0 <= c <= 1.0 for c in confs), (confs, [(i.k, i.conf) for i in iss]))
u = [M._unwrap_unit(v) for v in (su("between", 2, 2), su("rikhw", 3.0, 3), su("jahr", 0.9309921264648438, 2), su("x", float("nan")), su("x", -0.2),
                                 {"text": "y", "probs": [0.1, 0.7, 0.2], "idx": 1}, types.SimpleNamespace(text="z", prob=None, probs=None))]
check("_unwrap_unit: prob 2 / 3 / NaN / negative -> 0.0 (no evidence); a real probability and a distribution+idx are read BY NAME",
      [x[1] for x in u[:5]] == [0.0, 0.0, 0.9309921264648438, 0.0, 0.0] and u[5] == ("y", 0.7) and u[6] == ("z", None), u)
check("Issue.conf is clamped into [0, 1] whatever is passed in", [M.Issue("t", "k", "n", c).conf for c in (3, -1, 0.5, float("nan"), float("inf"))] == [1.0, 0.0, 0.5, 0.0, 0.0])
# w=14 فِي
f_jahr = set_on("ف", hams_or_jahr=su("jahr", 0.957))
check("w=14 (فِي, report #1): clean short function word, ف decoded cleanly, adjacent voiced kasrah bleeds -> no false hams_or_jahr (was flagged 0.957)",
      not sprobe(14, f_jahr)[0] and any("bleed-ignored" in t[1] for t in sprobe(14, f_jahr)[1]))
check("w=17 (second فِي): same", not sprobe(17, f_jahr)[0])
check("bleed guard is narrow: unclean decode of فِي, or ف inside a longer word (حِفظُهُمَا), is still judged",
      [i.k for i in sprobe(14, f_jahr, clean=False)[0]] == ["sifa:hams_or_jahr"] and [i.k for i in sprobe(46, f_jahr)[0]] == ["sifa:hams_or_jahr"])
check("bleed guard: balanced / relaxed never flag hams_or_jahr at all", all(not sprobe(46, f_jahr, preset=p)[0] for p in ("balanced", "relaxed")))
# strict: sifat of ظ on the last word
check("w=49 (الْعَظِيمُ): ظ tafkheem / itbaq are validated (strict: itbaq; balanced: tafkheem)",
      [i.k for i in sprobe(49, set_on("ظ", itbaq=su("monfateh", 0.98)))[0]] == ["sifa:itbaq"]
      and [i.k for i in sprobe(49, set_on("ظ", tafkheem_or_taqeeq=su("moraqaq", 0.98)), preset="balanced")[0]] == ["tafkheem"])
# end to end through the engine (real group matching)
def sifat_engine(mutate, preset="strict"):
    holder = {}; base_rec = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    rec0 = base_rec.recognizer(holder)
    def rec(wave):
        r = rec0(wave)
        items = sim.fake_groups(r.text); objs = []
        for it in items:
            o = types.SimpleNamespace(phonemes=it.phonemes)
            for lv in LV10: setattr(o, lv, su(getattr(it, lv)))
            objs.append(o)
        mutate(objs)
        return M.Recognition(r.text, r.probs, M.convert_sifat(objs), r.frames)
    holder_eng = M.SessionEngine(ref, rec, M.get_preset(preset)); holder["engine"] = holder_eng
    t = 0.0
    while t < base_rec.total:
        holder_eng.add_audio(base_rec.audio[int(t * M.SR): int((t + 0.16) * M.SR)]); t += 0.16
        if holder_eng.ready_to_run(): holder_eng.step()
    return flagged(holder_eng.flush())
def slid_tafashie(objs):                  # the report's pattern: wherever ش is followed (2 groups later) by ء, the head slid
    for i, o in enumerate(objs):
        if o.phonemes.startswith("ش") and i + 2 < len(objs) and objs[i + 2].phonemes.startswith("ء"):
            o.tafashie = su("not_motafashie", 0.99); objs[i + 2].tafashie = su("motafashie", 0.99)
got = sifat_engine(slid_tafashie)
check("end-to-end (strict, group matching through the engine): the report's tafashie slide on شَاءَ is judged ok (hallucinated sifa suppressed), nothing else flagged",
      not got, {w: kinds(x) for w, x in got.items()})
got = sifat_engine(set_on("ش", tafashie=su("not_motafashie", 0.99)))
check("end-to-end control: ش really losing tafashie (no slide pattern) IS still flagged", SHAA in got and "sifa:tafashie" in kinds(got[SHAA]), {w: kinds(x) for w, x in got.items()})

# ---------------------------------------------------------------- D6.3  w=49 الْعَظِيمُ: waqf form + trailing vowel
spoken = list(BASE); eng, log, fin, _ = run_spoken(spoken, "strict")
check("w=49 is reported as form 'waqf' (the ayah end is always a stop); other wasl words stay 'wasl'",
      eng.word_log[49]["form"] == "waqf" and eng.word_log[48]["form"] == "wasl" and ref.stop_target[49] and not any(ref.stop_target[:49]) and not flagged(fin))
for vowel, name in ((M.FATHA, "fatha"), (M.DAMMA, "damma"), (M.KASRA, "kasra")):
    row = {}
    for p in PRESETS3:
        spoken = list(BASE); spoken[49] = ref.wasl[49] + vowel
        eng, log, fin, _ = run_spoken(spoken, p)
        row[p] = (sorted(flagged(fin)), kinds(flagged(fin).get(49, {})))
    check("w=49: a trailing %s on the stopped word (لعَظِۦۦۦۦمَ vs لعَظِۦۦۦۦم) is flagged tashkeel/waqf_vowel in every preset, nothing else" % name,
          all(v == ([49], ["waqf_vowel"]) for v in row.values()), row)
check("w=49 judge level: madd 'arid at the stop needs >= 2 harakat (1 -> short, 2..6 ok)",
      mkk("لعَظِييم", "لعَظِيم", S_) == ["madd_short"] and all(mkk("لعَظِييم", "لعَظِ" + "ي" * n + "م", S_) == [] for n in range(2, 7)))
check("w=49 judge level: stop_target is what makes the final vowel an error (the same seg is fine on a non-final word)",
      [i.k for i in M.judge_word(["لعَظِييم"], "لعَظِييمَ", None, S_, trail=1, stop_target=True)[0]] == ["waqf_vowel"]
      and not M.judge_word(["لعَظِييم"], "لعَظِييمَ", None, S_, trail=1, stop_target=False)[0])

# ---------------------------------------------------------------- D6.4  swallowed substitutions, truncations, bleed
check("rival pairs are the specified ones (ط/ت ص/س ض/د ظ/ذ/ز ق/ك ح/ه ع/ء ر/غ)",
      all((a, b) in M.RIVAL_PAIRS and (b, a) in M.RIVAL_PAIRS for a, b in [("ط", "ت"), ("ص", "س"), ("ض", "د"), ("ظ", "ذ"), ("ظ", "ز"), ("ذ", "ز"), ("ق", "ك"), ("ح", "ه"), ("ع", "ء"), ("ر", "غ")]))
for p in PRESETS3:
    pr = M.get_preset(p)
    got = [i.k for i in M.judge_word(["لءَرض"], "لءَغض", [.9, .9, .9, .45, .9], pr)[0]]
    ctl = [i.k for i in M.judge_word(["لءَبض"], "لءَنض", [.9, .9, .9, .45, .9], pr)[0]]
    check("w=18 (الْأَرْضِ) ر->غ at posterior 0.45 [%s]: flagged letters/replace (rival gate); the same posterior on a non-rival swap is still gated" % p,
          got == ["replace"] and ctl == [], (got, ctl))
check("w=18: relative odds p(غ)/(p(غ)+p(ر)) rescue a very low posterior inside the rival pair",
      [i.k for i in M.judge_word(["لءَرض"], "لءَغض", [.9, .9, .9, .20, .9], S_, alts=[None, None, None, {"ر": 0.15}, None])[0]] == ["replace"]
      and not M.judge_word(["لءَرض"], "لءَغض", [.9, .9, .9, .20, .9], S_)[0])
def with_probs(sp, holder, fn):
    rec0 = sp.recognizer(holder)
    def rec(wave):
        r = rec0(wave); probs = list(r.probs); fn(r.text, probs); return M.Recognition(r.text, probs, None, r.frames)
    return rec
for p in PRESETS3:
    spoken = list(BASE); spoken[18] = spoken[18].replace("ر", "غ")
    sp = sim.Speech(ref, spoken, pauses=PAUSES, seed=5); holder = {}
    eng = M.SessionEngine(ref, with_probs(sp, holder, lambda t, pr: [pr.__setitem__(i, 0.45) for i, ch in enumerate(t) if ch == "غ"]), M.get_preset(p)); holder["engine"] = eng
    t = 0.0
    while t < sp.total: eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.16) * M.SR)]); t += 0.16; eng.ready_to_run() and eng.step()
    fin = eng.flush(); got = flagged(fin)
    check("w=18 end-to-end [%s]: ر pronounced غ with a split posterior is flagged letters/replace (was s:ok, issues:[])" % p,
          18 in got and got[18]["s"] == "letters" and "replace" in kinds(got[18]) and set(got) == {18}, {w: kinds(x) for w, x in got.items()})
for p in PRESETS3:
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); holder = {}
    rec0 = sp.recognizer(holder)
    def rec(wave, rec0=rec0, sp=sp):
        r = rec0(wave); keep = []
        w9 = {i for i, (tm, ch, w) in enumerate(sp.chars) if w == 9}
        # drop every phoneme of word 9 except the final ت (and give it a low posterior)
        a, b = holder["engine"].last_window
        sel = [(tm, ch, i) for i, (tm, ch, w) in enumerate(sp.chars) if a / M.SR <= tm < b / M.SR]
        txt, pr = [], []
        for (tm, ch, i), p_ in zip(sel, r.probs):
            if i in w9 and ch != "ت": continue
            txt.append(ch); pr.append(0.3 if i in w9 else p_)
        return M.Recognition("".join(txt), pr, None, None)
    eng = M.SessionEngine(ref, rec, M.get_preset(p)); holder["engine"] = eng
    t = 0.0
    while t < sp.total: eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.16) * M.SR)]); t += 0.16; eng.ready_to_run() and eng.step()
    fin = eng.flush(); got = flagged(fin)
    check("w=9 (سِنَةٌ) heard as just 'ت' [%s]: never ok when most consonants are missing (was s:ok, seg 'ت')" % p,
          9 in got and got[9]["s"] == "letters", {w: (x["s"], kinds(x)) for w, x in got.items()})
check("judge level: > 50% of a word's consonants missing is 'unclear' even when every confidence is low",
      [i.k for i in M.judge_word(["سِنَتُ"], "ت", [0.3], S_)[0]] and M.judge_word(["سِنَتُ"], "ت", [0.05], B_)[0][0].t == "letters")
check("judge level: a word that is merely clipped by one consonant is NOT promoted to unclear", not M.judge_word(["سِنَتُ"], "سِنَتُ", None, S_)[0])
# cross-word bleed in the report
spoken = list(BASE); spoken[23] = ref.wasl[23][:-4] + "ااا"             # the reciter's silah leaves a stray alif in front of w=24
eng, log, fin, _ = run_spoken(spoken, "strict")
check("cross-word bleed: a stray madd letter before the next word is not shown as part of it (w24 seg is exactly its own phonemes)",
      eng.word_log[24]["seg"] == ref.wasl[24], eng.word_log[24]["seg"])
eng, log, fin, _ = run_spoken(BASE, "strict")
leak = [w for w in range(50) if eng.word_log[w]["seg"] != eng.word_log[w]["exp"]]
check("clean run: every word's shown segment equals its expected form (no neighbour letters in seg: w13, w24, w32, w41, w46 ...)", not leak, leak)

# ---------------------------------------------------------------- engine behaviour: dual-stage lock, elastic tail, escape hatch, completeness
scripted = {"text": "", "probs": None}
def scripted_rec(wave): return M.Recognition(scripted["text"], scripted["probs"] or [0.97] * len(scripted["text"]), None)
def scripted_engine(): 
    e = M.SessionEngine(ref, scripted_rec, M.get_preset("strict"))
    rng = np.random.default_rng(1); e.add_audio((0.1 * rng.standard_normal(int(3.0 * M.SR))).astype(np.float32)); return e
eng = scripted_engine(); w0, w1, w2 = ref.wasl[0], ref.wasl[1], ref.wasl[2]
n01 = len(w0) + len(w1)
scripted["text"] = w0 + w1 + w2[:3]                                        # ء ِ ل : two letters of word 2, the 2nd with posterior 0.5
scripted["probs"] = [0.97] * (n01 + 1) + [0.5, 0.5]
msg = eng.step()
check("stage 1 (soft): the tracked word moves the instant the alignment enters word 2 (>= 40% of its letters)", eng.tracked == 2 and msg is not None and msg["tracked"] == 2, (eng.tracked, msg and msg["tracked"]))
check("stage 2 (hard): ...but a low-posterior phoneme (0.5) is not 'established': word 0 is locked, word 1 is NOT yet", eng.cursor == 1 and 0 in eng.results and 1 not in eng.results, (eng.cursor, sorted(eng.results)))
scripted["probs"] = [0.97] * (n01 + 3); eng.last_run_end = 0
msg = eng.step()
check("stage 2 (hard): once >= 2 phonemes of word 2 are heard with a solid posterior, word 1 is judged and locked", eng.cursor == 2 and 1 in eng.results, (eng.cursor, sorted(eng.results)))
eng = scripted_engine(); scripted["text"] = w0 + w1 + w2[:1]; scripted["probs"] = None
eng.step()
check("stage 2 (hard): a single phoneme of word 2 never reaches it (tracked stays on word 1); only word 0 is locked", eng.tracked == 1 and eng.cursor == 1, (eng.tracked, eng.cursor))
eng = scripted_engine(); scripted["text"] = w0 + w1 + w2[:1]; scripted["probs"] = None
eng.step(); eng.on_pause(600); eng.last_run_end = 0; eng.step()
check("a pause frame locks the last heard word at once (no second phoneme needed): word 1 is judged", eng.cursor >= 2 and 1 in eng.results, (eng.cursor, sorted(eng.results)))
eng = scripted_engine(); eng.resume_from(7); scripted["probs"] = None
W = [ref.wasl[k] for k in range(7, 11)]
scripted["text"] = W[0] + W[1] + W[2] + W[0] + W[1] + W[2] + W[3][:3]
eng.step()
check("escape hatch (scripted, same logits): the reciter's aborted first copy of words 7..9 is skipped as junk (stage 'restart'), words 7,8 are judged ok from the repeat",
      eng.debug["steps"][-1].get("stage") in ("restart", "wide", "local") and eng.cursor >= 9 and all(eng.results[w][0] == "ok" for w in (7, 8)),
      (eng.debug["steps"][-1].get("stage"), eng.cursor, {w: eng.results[w][0] for w in eng.results}))
eng, log, fin, sp_ = run_spoken(BASE, "strict")
bad = []
for st_ in eng.debug["steps"]:
    c_ = st_.get("cursor_out")
    if st_.get("next_win_start_s") is not None and c_ is not None and c_ < 50:
        t_first = min(tm for tm, ch, w in sp_.chars if w == c_)
        if st_["next_win_start_s"] > t_first - 0.30 + 0.06: bad.append((st_["seq"], st_["next_win_start_s"], round(t_first, 2)))
check("every window starts >= 300 ms (+-1 CTC frame) before the first unlocked word: strict left-context overlap", not bad and len(eng.debug["steps"]) > 20, bad[:3])

# monotonic completeness on every kind of run
for name, kw in {"fast": dict(scale=0.7), "slow x2.5": dict(scale=2.5), "3 s pause": dict(pause_s=3.0, seed=5)}.items():
    eng, log, fin, _ = run_spoken(BASE, "strict", **kw) if name != "3 s pause" else run_spoken(BASE, "strict", pause_s=3.0)
    caps = sorted({s.get("cap_s") for s in eng.debug["steps"]})
    check("%s: complete, no 'unchecked', every word has a verdict in order, windows <= 10 s (caps used: %s)" % (name, caps),
          fin["summary"]["complete"] and fin["summary"]["unchecked"] == 0 and sorted(eng.results) == list(range(50))
          and max(s["wave_s"] for s in eng.debug["steps"]) <= 10.0 and fin["summary"]["ok"] == 50, fin["summary"])
eng, log, fin, _ = run_spoken(BASE, "strict", scale=2.5)
check("elastic tail: a slow tahqiq recitation stretches the tail up to 4 s when needed (never truncating word beginnings)",
      any(s.get("cap_s", 0) >= 4.0 for s in eng.debug["steps"]) or max(s["wave_s"] for s in eng.debug["steps"]) <= 4.0)
# phrase restart (report #1 steps 2-3 and 8-13): the reciter repeats words 7..9 from the start of the phrase
spoken = list(BASE); spoken[7] = BASE[7] + BASE[8] + BASE[9] + BASE[7]
eng, log, fin, _ = run_spoken(spoken, "strict")
check("phrase restart: repeating the phrase (7,8,9, 7,8,9) never leaves words unchecked, the verse completes, no corrupted downstream words",
      fin["summary"]["complete"] and fin["summary"]["unchecked"] == 0 and not any(w > 12 for w in flagged(fin)), {w: kinds(x) for w, x in flagged(fin).items()})
spoken = list(BASE); spoken[20] = BASE[19] + BASE[20]
eng, log, fin, _ = run_spoken(spoken, "strict")
check("stutter of a single word is still absorbed", fin["summary"]["complete"] and fin["summary"]["unchecked"] == 0)

# ---------------------------------------------------------------- zero-copy Int16 gate + interleaved neural gating
def sine(f, n, amp=0.3): return (amp * np.sin(2 * np.pi * f * np.arange(n) / M.SR)).astype(np.float32)
g = M.PcmGate(); g.feed_f32(sine(140, M.SR // 2))
check("gate: a steady vowel (140 Hz) is reported steady after 160 ms", g.steady_ms() >= 160, g.steady_ms())
g.feed_f32(sine(2400, M.SR // 10, 0.12))
check("gate: a formant / spectral shift (ZCR + first difference change) ends the steady state immediately", g.steady_ms() == 0, g.steady_ms())
g2 = M.PcmGate(); g2.feed_f32((0.2 * np.random.default_rng(2).standard_normal(M.SR // 2)).astype(np.float32))
check("gate: noise is never steady", g2.steady_ms() == 0)
g3 = M.PcmGate(); g3.feed_f32(sine(140, M.SR // 4)); g3.feed_f32(np.zeros(M.SR // 10, dtype=np.float32))
check("gate: a glottal closure / silence ends the steady state", g3.steady_ms() == 0)
raw = (sine(140, 4000) * 32767).astype("<i2").tobytes()
e1, e2 = M.SessionEngine(ref, lambda w: M.Recognition(""), M.get_preset("strict")), M.SessionEngine(ref, lambda w: M.Recognition(""), M.get_preset("strict"))
e1.add_pcm16(raw + b"\x01"); e2.add_audio(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
check("add_pcm16: zero-copy int16 path == float path (odd trailing byte ignored); the gate saw the int16 samples directly",
      e1.buf.n == e2.buf.n == 4000 and np.allclose(e1.buf.snapshot(0, 4000), e2.buf.snapshot(0, 4000)) and len(e1.gate.rms) == 25)
class GateRec:
    def __init__(self): self.calls = 0
    def __call__(self, wave):
        self.calls += 1; txt = ref.wasl[0][:4] + "ا" * 4
        return M.Recognition(txt, [0.99] * len(txt), None)
rec = GateRec(); eng = M.SessionEngine(ref, rec, M.get_preset("strict")); skipped_row = 0; max_row = 0
tone = sine(140, int(8 * M.SR)); t = 0
while t < len(tone):
    eng.add_audio(tone[t:t + int(0.16 * M.SR)]); t += int(0.16 * M.SR)
    if eng.ready_to_run():
        before = eng.skips; eng.step()
        skipped_row = skipped_row + 1 if eng.skips > before else 0; max_row = max(max_row, skipped_row)
check("interleaved gating: during a steady, high-confidence madd the engine skips neural hops (%d skipped of %d) but never two in a row" % (eng.skips, eng.skips + rec.calls),
      eng.skips >= 3 and max_row <= 1, (eng.skips, rec.calls, max_row))
eng2 = M.SessionEngine(ref, GateRec(), M.get_preset("strict"))
for blk in (sine(140, int(1.0 * M.SR)),):
    for i in range(0, len(blk), int(0.16 * M.SR)):
        eng2.add_audio(blk[i:i + int(0.16 * M.SR)])
        if eng2.ready_to_run(): eng2.step()
eng2.add_audio(sine(2400, int(0.2 * M.SR), 0.1))
check("interleaved gating: a formant shift wakes the neural pass (no skip allowed right after it)", not eng2._gate_may_skip())

# ---------------------------------------------------------------- pause control frames
for p, expect in (("strict", True),):
    spoken = list(BASE); spoken[9] = ref.waqf[9]
    sp = sim.Speech(ref, spoken, pauses=set(), seed=7, pause_s=0.0)
    eng, log, fin, _ = sim.run_stream(ref, sp, preset=p)
    no_frame = flagged(fin)
    check("pause frame: stop form with NO silence in the audio and no pause frame -> waqf_no_pause (control)", 9 in no_frame and "waqf_no_pause" in kinds(no_frame[9]), {w: kinds(x) for w, x in no_frame.items()})
    sp = sim.Speech(ref, spoken, pauses=set(), seed=7, pause_s=0.0)
    eng, log, fin, _ = sim.run_stream(ref, sp, preset=p, pauses=[(sp.spans[9][1] + 0.05, 600)])
    check("pause frame: the same audio WITH a {type:pause} frame at the boundary -> no false waqf_no_pause", 9 not in flagged(fin) and len(eng.pauses) == 1, {w: kinds(x) for w, x in flagged(fin).items()})
spoken = list(BASE); spoken[10] = ref.start_forms[10] or ref.wasl[10]
eng = M.SessionEngine(ref, lambda w: M.Recognition(""), M.get_preset("strict"))
rng = np.random.default_rng(4); eng.add_audio((0.1 * rng.standard_normal(M.SR)).astype(np.float32))
eng.add_audio(np.zeros(M.SR // 2, dtype=np.float32) + 0.05 * rng.standard_normal(M.SR // 2).astype(np.float32))    # 500 ms hangover with room noise
eng.on_pause(600, 500)
check("hangover padding is decoupled from the voicing offset: the 500 ms of room noise before the pause frame never counts as voicing",
      abs(eng.pauses[-1]["voice_end"] - M.SR) <= 2 * M.FRAME, eng.pauses[-1])
env_noise = np.concatenate([np.full(40, 0.2), np.full(50, 0.05)])
check("   ...and the audio meter cannot run past that offset (terminal madd is not inflated)",
      abs(eng._cap_by_pause(1.2, 0.0, 0) - eng.pauses[-1]["voice_end"] / M.SR) < 1e-9 and eng._cap_by_pause(0.5, 0.0, 0) == 0.5)
eng = M.SessionEngine(ref, lambda w: M.Recognition(""), M.get_preset("strict"))
eng.add_audio(np.zeros(M.SR, dtype=np.float32)); eng.on_pause(700)
check("a pause announced during leading silence (no speech yet) is only evidence: it can never settle a half-heard first word", eng.pauses[-1]["done"] is True and eng._pending_pause(0) is None)

# ---------------------------------------------------------------- stateless checkpoint: resume
for k in (20, 40):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    sp = tail_speech(sp, sp.spans[k][0] - 0.3)
    eng, log, fin, _ = sim.run_stream(ref, sp, preset="strict", engine_factory=resumed_engine(k))
    check("resume at word %d: only words %d..49 are judged, all ok, nothing before the checkpoint is re-judged, complete" % (k, k),
          sorted(eng.results) == list(range(k, 50)) and fin["summary"]["ok"] == 50 - k and fin["summary"]["complete"], (sorted(eng.results)[:3], fin["summary"]))
e = M.SessionEngine(ref, lambda w: M.Recognition(""), M.get_preset("strict")); e.resume_from(50)
check("resume past the last word completes the session at once", e.complete and e.cursor == 50)

# ---------------------------------------------------------------- websocket: pause / resume / idle zombie killer / state-diff / instant shutdown
async def ws_feed(sp, hop=0.16, until=None, extra=None, qp=None, first=None, wait=30):
    handler = M.build_fastapi_app(FakeRuntime(sp)).routes[("websocket", "/ws/live-recitation")]
    ws = FakeWS(qp or {"sura_idx": "2", "aya_idx": "255", "strictness": "balanced"})
    task = asyncio.create_task(handler(ws)); t = 0.0
    for m_ in (first or []): await ws.q.put({"type": "websocket.receive", "text": json.dumps(m_)})
    end = sp.total if until is None else until
    while t < end and not ws.closed:
        await ws.q.put({"type": "websocket.receive", "bytes": (sp.audio[int(t * M.SR): int((t + hop) * M.SR)] * 32767).astype("<i2").tobytes()})
        t += hop; await asyncio.sleep(0.003)
    return ws, task, t
async def ws_v145():
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    ws, task, t_sent = await ws_feed(sp)                                       # runs to the end of the recording, never sends stop
    await asyncio.wait_for(task, 30)
    fin = [m for m in ws.sent if m["type"] == "final"]
    check("instant shutdown: the final payload is emitted the moment word 50 is judged (before the trailing silence is even sent) and the socket closes with 1000",
          len(fin) == 1 and fin[0]["summary"]["ok"] == 50 and ws.closed and ws.close_code == 1000 and t_sent < sp.total - 0.4, (t_sent, sp.total))
    prog = [m for m in ws.sent if m["type"] == "progress"]
    sigs = [(m["tracked"], m["committed"], tuple((u["i"], u["s"]) for u in m["updates"])) for m in prog]
    check("state-diff emission: no two consecutive identical progress frames", all(a != b for a, b in zip(sigs, sigs[1:])) and len(prog) > 20, len(prog))
    check("progress frames carry the soft tracked word AND the lock position (resume checkpoint)",
          all("tracked" in m and "committed" in m and "lock_ms" in m and "audio_ms" in m for m in prog) and any(m["tracked"] > m["committed"] for m in prog))
    # pause frame over the socket
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    ws, task, _ = await ws_feed(sp, until=12.0)
    await ws.q.put({"type": "websocket.receive", "text": json.dumps({"type": "pause", "duration_ms": 640, "speech_tail_padding_ms": 500})})
    await ws.q.put({"type": "websocket.receive", "text": json.dumps({"type": "stop"})}); await asyncio.wait_for(task, 30)
    fin = [m for m in ws.sent if m["type"] == "final"]
    check("websocket: a {type:pause} frame is parsed (counted in the final debug) and the session still ends with one final",
          len(fin) == 1 and fin[0]["debug"]["engine_stats"]["pauses"] == 1, fin[0]["debug"]["engine_stats"] if fin else None)
    # idle zombie killer
    keep = M.IDLE_KILL_S; M.IDLE_KILL_S = 0.6
    try:
        sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
        ws, task, _ = await ws_feed(sp, until=14.0)
        t0 = asyncio.get_event_loop().time()
        await asyncio.wait_for(task, 10)
        waited = asyncio.get_event_loop().time() - t0
    finally:
        M.IDLE_KILL_S = keep
    idle = [m for m in ws.sent if m["type"] == "paused_idle"]
    check("idle zombie killer: a socket that goes silent is answered with {type:paused_idle, resume_word_idx:k} and closed (no final)",
          len(idle) == 1 and idle[0]["resume_word_idx"] >= 10 and not [m for m in ws.sent if m["type"] == "final"] and ws.closed and ws.close_code == 1000 and waited < 5,
          (idle, waited))
    check("   ...the checkpoint is sound: every word below resume_word_idx was already judged and sent to the app",
          sorted({u["i"] for m in ws.sent if m["type"] == "progress" for u in m["updates"]}) == list(range(idle[0]["resume_word_idx"])), idle)
    check("   ...the default idle limit is 8 s", keep == 8.0)
    # reconnect with resume
    k = idle[0]["resume_word_idx"]
    sp2 = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    t_from = max(0.0, sp2.spans[k][0] - 0.3)
    handler = M.build_fastapi_app(FakeRuntime(sp2)).routes[("websocket", "/ws/live-recitation")]
    ws2 = FakeWS({"sura_idx": "2", "aya_idx": "255", "strictness": "balanced"}); task = asyncio.create_task(handler(ws2))
    await ws2.q.put({"type": "websocket.receive", "text": json.dumps({"type": "resume", "word_idx": k})})
    t = t_from
    while t < sp2.total and not ws2.closed:
        await ws2.q.put({"type": "websocket.receive", "bytes": (sp2.audio[int(t * M.SR): int((t + 0.16) * M.SR)] * 32767).astype("<i2").tobytes()}); t += 0.16; await asyncio.sleep(0.003)
    await asyncio.wait_for(task, 30)
    fin = [m for m in ws2.sent if m["type"] == "final"]
    done_words = sorted(w["i"] for w in fin[0]["words"]) if fin else []
    check("reconnect with {type:resume, word_idx:k}: the new server session judges exactly words k..49 (all ok) and finishes",
          len(fin) == 1 and done_words == list(range(k, 50)) and fin[0]["summary"]["ok"] == 50 - k, (k, done_words[:3], fin[0]["summary"] if fin else None))
asyncio.run(ws_v145())

# ---------------------------------------------------------------- batch eradication, versions, static checks, Modal configuration
py_src = open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read()
py_code = py_src.split('"""', 2)[2]                                  # code after the module docstring (which documents the removal)
check("batch mode is gone from the backend: no /analyze, no upload parsing, no offline runner, no batch helpers",
      not any(tok in py_code for tok in ("/analyze", "decode_audio_bytes", "run_offline", "UploadFile", "multipart", "api.post(")))
res = subprocess.run([sys.executable, os.path.join(HERE, "static_check_kt.py")], capture_output=True, text=True)
check("kotlin static check: balanced, no batch symbols (uploadBatch, buildWav, KEY_MODE, getMode, saveMode), every used member exists, listener parity, JSON key parity with Python",
      res.returncode == 0, res.stdout[-600:])
# ---- client reconnect policy + no warm-up traffic (Kotlin cannot be compiled here: these are SOURCE assertions)
ctrl_kt = open(os.path.join(kt_dir, "RafiqController.kt"), encoding="utf-8").read()
main_kt = open(os.path.join(kt_dir, "MainActivity.kt"), encoding="utf-8").read()
def kconst(name):
    mm = _re.search(r"const val %s\s*=\s*([0-9_]+)L?" % name, kt)
    return int(mm.group(1).replace("_", "")) if mm else None
MAXR, BASE_MS = kconst("MAX_RETRIES"), kconst("RETRY_BASE_MS")
schedule = [BASE_MS << (n - 1) for n in range(1, (MAXR or 0) + 1)]           # the exact expression onSocketGone uses: RETRY_BASE_MS shl (attempts - 1)
check("client: at most 3 reconnect attempts with exponential back-off 1 s, 2 s, 4 s (constants parsed from RecitationSession.kt)",
      MAXR == 3 and schedule == [1000, 2000, 4000] and "RETRY_BASE_MS shl (attempts - 1)" in kt and "attempts > MAX_RETRIES" in kt, (MAXR, BASE_MS, schedule))
check("client: the whole back-off (%d ms) fits inside the server's scaledown_window (%d s): a retry meets the same warm container" % (sum(schedule), M.MODAL_SCALEDOWN_S),
      sum(schedule) < M.MODAL_SCALEDOWN_S * 1000, (schedule, M.MODAL_SCALEDOWN_S))
ready_branch = kt.split('"ready" -> {', 1)[1].split('"progress" ->', 1)[0]
check("client: receiving 'ready' does NOT reset the retry budget (a server that says ready and then drops used to loop forever)",
      "attempts = 0" not in ready_branch and "readyAtMs = SystemClock.elapsedRealtime()" in ready_branch)
sgone = kt.split("private fun onSocketGone", 1)[1].split("/** يرسل إطار resume", 1)[0]
check("client: the budget is renewed only after a connection that stayed up STABLE_MS, and the session has an absolute retry ceiling",
      "STABLE_MS" in sgone and "attempts = 0" in sgone and "MAX_RETRIES_PER_SESSION" in sgone and (kconst("MAX_RETRIES_PER_SESSION") or 99) <= 10 and "attempts > 8" not in kt)
fail_cb = kt.split("override fun onFailure", 1)[1].split("private fun denialReason", 1)[0]
check("client: a handshake 401/403 or other 4xx is fatal at once (no retry); the reason code in the 403 body selects the message",
      "http == 401 || http == 403" in fail_cb and "fatal(authMessage(why))" in fail_cb and "400..499" in fail_cb and "denialReason(response)" in fail_cb
      and all(r_ in kt for r_ in ("token_missing", "token_expired", "token_invalid", "server_auth_unconfigured")))
check("client: the reason codes the app understands are exactly the ones the server sends", set(M.REJECT_PUBLIC.values()) == {"token_missing", "token_expired", "token_invalid", "server_auth_unconfigured"})
check("client: close codes 1008 / 1009 are not retried; callbacks of superseded sockets are ignored (generation counter)",
      "code == 1008 || code == 1009" in kt and kt.count("gen != connGen.get()") >= 5 and "connGen.incrementAndGet()" in kt.split("fun abort()", 1)[1].split("fun voicedSinceMs", 1)[0])
check("client: with no saved token the metered server is not contacted at all (it would only start a container to be refused)",
      "token.isEmpty() && RecitationConfig.isMetered(appContext)" in kt.split("private fun connect()", 1)[1].split("val url =", 1)[0])
check("client: no warm-up traffic - the /health polling loop is gone; nothing calls the server before the record button is pressed",
      "fun warmUp" not in ctrl_kt and "healthUrl" not in ctrl_kt and "tries < 120" not in ctrl_kt and "warmUp" not in main_kt and "retryServer" not in (ctrl_kt + main_kt)
      and "markSleeping()" in main_kt)
gradle = open(os.path.join(HERE, "..", "..", "android", "app", "build.gradle.kts"), encoding="utf-8").read()
check("versions: engine rafiq-v14.6, versionCode 19, versionName 14.6",
      M.ENGINE_NAME == "rafiq-v14.6" and M.VERSION == "14.6" and "versionCode = 19" in gradle and 'versionName = "14.6"' in gradle and "14.4" not in py_src.split("class QuranLib")[0])
check("the live server reports engine rafiq-v14.6 in /health", M.ModelRuntime().status()["engine"] == "rafiq-v14.6")

class _Chain:
    def __init__(self): self.calls = []
    def __getattr__(self, name):
        def f(*a, **k): self.calls.append((name, a, k)); return self
        return f
def load_with_fake_modal():
    rec = {"cls_kwargs": None, "concurrent": None, "enter": [], "serve": None, "image": _Chain()}
    fm = types.ModuleType("modal")
    class App:
        def __init__(self, name): rec["app"] = name
        def cls(self, **kw):
            rec["cls_kwargs"] = kw
            return lambda c: c
        def function(self, **kw): return lambda f: f
    fm.App = App
    fm.Image = types.SimpleNamespace(debian_slim=lambda **kw: rec["image"].__getattr__("debian_slim")(**kw))
    fm.concurrent = lambda **kw: (rec.__setitem__("concurrent", kw) or (lambda c: c))
    fm.enter = lambda **kw: (rec["enter"].append(kw) or (lambda f: f))
    fm.web_server = lambda port, **kw: (rec.__setitem__("serve", ("web_server", port, kw)) or (lambda f: f))
    fm.asgi_app = lambda **kw: (rec.__setitem__("serve", ("asgi_app",)) or (lambda f: f))
    fm.is_local = lambda: True
    fm.Secret = types.SimpleNamespace(from_name=lambda name, **k: ("secret", name))
    saved = sys.modules.get("modal"); sys.modules["modal"] = fm
    try:
        spec = importlib.util.spec_from_file_location("modal_app_fakemodal", os.path.join(HERE, "..", "modal_app.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    finally:
        if saved is None: sys.modules.pop("modal", None)
        else: sys.modules["modal"] = saved
    return mod, rec
mod, rec = load_with_fake_modal(); kw = rec["cls_kwargs"]
check("modal config: cpu=(0.125, 2.0), memory tuple, scaledown_window=20, memory snapshot on, timeout set",
      kw["cpu"] == (0.125, 2.0) and isinstance(kw["memory"], tuple) and kw["scaledown_window"] == 20 and kw["enable_memory_snapshot"] is True, kw)
check("modal config: idle cost is controlled ONLY by the short scaledown_window - no keep-warm / minimum / buffer containers",
      not any(k in kw for k in ("min_containers", "keep_warm", "buffer_containers", "max_containers")) and kw["scaledown_window"] <= 30, kw)
_saved_sd = os.environ.get("RAFIQ_SCALEDOWN_S")
try:
    os.environ["RAFIQ_SCALEDOWN_S"] = "0"; low = load_with_fake_modal()[1]["cls_kwargs"]["scaledown_window"]
    os.environ["RAFIQ_SCALEDOWN_S"] = "99999"; high = load_with_fake_modal()[1]["cls_kwargs"]["scaledown_window"]
finally:
    if _saved_sd is None: os.environ.pop("RAFIQ_SCALEDOWN_S", None)
    else: os.environ["RAFIQ_SCALEDOWN_S"] = _saved_sd
check("modal config: RAFIQ_SCALEDOWN_S override is clamped to Modal's sane range (2 .. 600 s)", (low, high) == (2, 600), (low, high))
rf = [c for c in rec["image"].calls if c[0] == "run_function"]
check("image build: the weights step runs as run_function with enough memory for fp32 + INT8 copies (>= 6 GiB) and a generous timeout",
      len(rf) == 1 and rf[0][1][0].__name__ == "download_muaalem_weights" and rf[0][2].get("memory", 0) >= 6144 and rf[0][2].get("timeout", 0) >= 900, rf)
check("modal config: default (pre-emptible) capacity and NO region pinning (no 3x / 1.75x surcharge): neither `nonpreemptible` nor `region` is passed",
      "nonpreemptible" not in kw and "region" not in kw and "gpu" not in kw)
check("modal config: @modal.concurrent(max_inputs=16) with one live session per container target", rec["concurrent"]["max_inputs"] == 16 and rec["concurrent"]["target_inputs"] == 1, rec["concurrent"])
check("modal config: the snapshot hook is @modal.enter(snap=True) and the server is a web_server on :8000 (uvicorn: compression off, no access log)",
      rec["enter"] == [{"snap": True}] and rec["serve"][0] == "web_server" and rec["serve"][1] == 8000)
uk = mod.uvicorn_kwargs()
check("uvicorn: permessage-deflate disabled, access log disabled", uk["ws_per_message_deflate"] is False and uk["access_log"] is False, uk)
env_calls = [c for c in rec["image"].calls if c[0] == "env"]
envd = env_calls[0][1][0] if env_calls else {}
check("image env: passive OpenMP wait, no KMP spin, malloc trim threshold 65536 (zero-spin, kernel memory reclamation)",
      envd.get("OMP_WAIT_POLICY") == "PASSIVE" and envd.get("KMP_BLOCKTIME") == "0" and envd.get("MALLOC_TRIM_THRESHOLD_") == "65536", envd)
check("process env is pinned before numpy/torch import (PASSIVE, trim threshold)", os.environ.get("OMP_WAIT_POLICY") == "PASSIVE" and os.environ.get("MALLOC_TRIM_THRESHOLD_") == "65536")
order = []
rt = mod.RUNTIME
rt.ensure_loaded = lambda: order.append("load"); rt.warmup = lambda: order.append("warmup"); rt.snapshot_prepare = lambda: order.append("freeze")
mod.Fastapi.load(mod.Fastapi()) if hasattr(mod.Fastapi, "load") else None
check("modal config: the auth secret comes from a Modal secret (never from the source) and is attached to the class",
      kw.get("secrets") == [("secret", "rafiq-auth")], kw.get("secrets"))
check("snapshot hook order: load -> dummy warm-up inference -> gc/malloc_trim/gc.freeze right before the snapshot", order == ["load", "warmup", "freeze"], order)
import gc as _gc
frozen_before = _gc.get_freeze_count(); real_rt = mod.ModelRuntime(); real_rt.snapshot_prepare()
check("snapshot_prepare really freezes the heap (gc.freeze) and survives a missing libc.so.6", _gc.get_freeze_count() >= frozen_before)
_gc.unfreeze()

# ---------------------------------------------------------------- start-up never quantises; INT8 is built ONCE at image build
import contextlib, io, shutil, tempfile
from unittest import mock


class _Model:                                   # stands in for a torch module
    def __init__(self, tag): self.tag, self.training = tag, True
    def eval(self): self.training = False; return self


class _FakeEnv:
    """Injects fake torch / quran_muaalem / quran_transcript so the REAL ModelRuntime and build code run without them.
    `calls` collects everything that would be quantisation or benchmarking."""
    def __init__(self, torch_version="2.9.9+cpu", cuda=False, load_raises=None):
        self.calls, self.loaded, self.saved, self.muaalem = [], [], [], None
        env = self
        t = types.ModuleType("torch"); t.__version__ = torch_version
        t.set_num_threads = lambda n: None; t.set_num_interop_threads = lambda n: None
        t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        t.float32, t.bfloat16, t.qint8 = "float32", "bfloat16", "qint8"
        def _quantize_dynamic(*a, **k): env.calls.append("torch.quantize_dynamic"); return _Model("q")
        t.ao = types.SimpleNamespace(quantization=types.SimpleNamespace(quantize_dynamic=_quantize_dynamic)); t.quantization = t.ao.quantization
        def _load(path, **kw):
            env.loaded.append((path, kw))
            if load_raises: raise load_raises
            return _Model("int8-from-disk")
        def _save(obj, path):
            env.saved.append(path)
            with open(path, "wb") as fh: fh.write(b"Q" * 64)
        t.load, t.save = _load, _save
        mu = types.ModuleType("quran_muaalem")
        class Muaalem:
            def __init__(s, model_name_or_path=None, device="cpu", dtype=None):
                s.model, s.device = _Model("fp32"), device; env.muaalem = s
        mu.Muaalem = Muaalem
        qt = types.ModuleType("quran_transcript")
        qt.MoshafAttributes = lambda **k: k
        qt.quran_phonetizer = lambda *a, **k: None
        qt.Aya = lambda su, ay: types.SimpleNamespace(get=lambda: types.SimpleNamespace(uthmani="x"))
        self.mods = {"torch": t, "quran_muaalem": mu, "quran_transcript": qt}
    def __enter__(self):
        self._old = {k: sys.modules.get(k) for k in self.mods}; sys.modules.update(self.mods); return self
    def __exit__(self, *a):
        for k, v in self._old.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


def start_runtime(env, qdir, precision="auto"):
    """Runs the real ModelRuntime.ensure_loaded(); any call to a quantise / benchmark helper is recorded in env.calls."""
    trap = lambda name: (lambda *a, **k: (env.calls.append(name), a[-1] if a else None)[1])   # records the call, hands the model back unchanged
    with contextlib.redirect_stdout(io.StringIO()), env, mock.patch.object(M, "QUANT_DIR", qdir), mock.patch.object(M, "PRECISION", precision), \
            mock.patch.object(M.ModelRuntime, "_probe_internals", lambda self, m: {}), \
            mock.patch.object(M.ModelRuntime, "get_ref", lambda self, su, ay: None), \
            mock.patch.object(M.ModelRuntime, "_quantize_encoder_only", trap("_quantize_encoder_only")), \
            mock.patch.object(M.ModelRuntime, "_quant_close", trap("_quant_close")), \
            mock.patch.object(M.ModelRuntime, "_time_inference", trap("_time_inference")):
        rt = M.ModelRuntime(); rt.ensure_loaded()
    return rt


def write_artifact(qdir, **over):
    meta = {"format": M.QUANT_FORMAT, "model_id": M.MODEL_ID, "torch": "2.9.9+cpu", "int8": True, "bytes": 64,
            "scope": "encoder-linear", "layers": 24, "min_logit_cosine": 0.999, "cosine_checked": True}
    meta.update(over)
    with open(os.path.join(qdir, M.QUANT_META_FILE), "w", encoding="utf-8") as fh: json.dump(meta, fh)
    with open(os.path.join(qdir, M.QUANT_MODEL_FILE), "wb") as fh: fh.write(b"Q" * 64)


def run_builder(qdir, cos_seq, rtfs, precision="auto", quant_raises=None):
    env, cos, rt_it = _FakeEnv(), iter(cos_seq), iter(rtfs)
    def fake_quantize(self, model):
        if quant_raises: raise quant_raises
        self.quant_info = {"scope": "encoder-linear", "layers": 24}; return _Model("q")
    with contextlib.redirect_stdout(io.StringIO()), env, mock.patch.object(M, "PRECISION", precision), mock.patch.object(M, "QuranLib", lambda: object()), \
            mock.patch.object(M.ModelRuntime, "get_ref", lambda self, su, ay: "REF"), \
            mock.patch.object(M.ModelRuntime, "_quantize_encoder_only", fake_quantize), \
            mock.patch.object(M.ModelRuntime, "_quant_close", lambda self, m, a, b: next(cos)), \
            mock.patch.object(M.ModelRuntime, "_time_inference", lambda self, m, ref, seconds=3.0: next(rt_it)):
        m = types.SimpleNamespace(model=_Model("fp32")); fp32 = m.model
        meta = M.build_prequantized_artifact(m, qdir)
    return meta, m, fp32, env


_tmp = tempfile.mkdtemp(prefix="rafiq_quant_")
try:
    # --- start-up with a valid image artifact: loads it, and quantises / benchmarks NOTHING
    d = os.path.join(_tmp, "ok"); os.makedirs(d); write_artifact(d)
    env = _FakeEnv(); rt_ = start_runtime(env, d)
    check("start-up with the image's INT8 artifact: the quantised model is swapped in, flagged int8, source=image-artifact",
          rt_.int8 is True and rt_.muaalem.model.tag == "int8-from-disk" and rt_.muaalem.model.training is False
          and rt_.quant_info.get("source") == "image-artifact" and rt_.quant_info.get("layers") == 24, (rt_.int8, rt_.quant_info))
    check("start-up NEVER quantises or benchmarks (no quantize_dynamic, no logit-cosine check, no fp32-vs-int8 timing) - this was the 75 s",
          env.calls == [] and len(env.loaded) == 1 and env.loaded[0][0].endswith(M.QUANT_MODEL_FILE) and env.loaded[0][1].get("weights_only") is False, (env.calls, env.loaded))
    check("start-up records per-stage timings (fp32 load, INT8 load, total) for the log and /admin/health",
          {"fp32_load_s", "int8_load_s", "total_s", "probe_s"} <= set(rt_.timings) and rt_.status()["quant"]["source"] == "image-artifact", rt_.timings)

    # --- every way the artifact can be unusable: fp32, no crash, still no quantisation, the reason is recorded
    cases = [
        ("no artifact in the image", None, {}, "no pre-quantised artifact"),
        ("build rejected int8", "w", {"int8": False, "reason": "logits drifted"}, "logits drifted"),
        ("built with another torch", "w", {"torch": "1.0.0"}, "built with torch"),
        ("built for another model", "w", {"model_id": "someone/else"}, "built for model"),
        ("old artifact format", "w", {"format": 0}, "format"),
        ("truncated file (size != meta)", "w", {"bytes": 999}, "truncated"),
    ]
    for label, mode, over, why in cases:
        d = os.path.join(_tmp, label.replace(" ", "_")); os.makedirs(d)
        if mode: write_artifact(d, **over)
        env = _FakeEnv(); rt_ = start_runtime(env, d)
        check("artifact unusable (%s): stays fp32, server still comes up, reason recorded, nothing quantised at start-up" % label,
              rt_.muaalem is not None and rt_.int8 is False and rt_.muaalem.model.tag == "fp32" and why in rt_.quant_info.get("reason", "")
              and env.calls == [] and env.loaded == [], (rt_.quant_info, env.calls, env.loaded))
    d = os.path.join(_tmp, "corrupt"); os.makedirs(d); write_artifact(d)
    env = _FakeEnv(load_raises=RuntimeError("unpickling failed")); rt_ = start_runtime(env, d)
    check("artifact that fails to deserialise: falls back to fp32 instead of crashing the container", rt_.int8 is False and rt_.muaalem.model.tag == "fp32"
          and "unpickling failed" in rt_.quant_info.get("error", "") and env.calls == [], rt_.quant_info)
    d = os.path.join(_tmp, "p32"); os.makedirs(d); write_artifact(d)
    env = _FakeEnv(); rt_ = start_runtime(env, d, precision="fp32")
    check("PRECISION='fp32': the artifact is ignored", rt_.int8 is False and env.loaded == [] and rt_.quant_info.get("reason") == "PRECISION=fp32")
    d = os.path.join(_tmp, "gpu"); os.makedirs(d); write_artifact(d)
    env = _FakeEnv(cuda=True); rt_ = start_runtime(env, d)
    check("GPU: dynamic INT8 is CPU-only, so the artifact is not used", rt_.device == "cuda" and rt_.int8 is False and env.loaded == [])
    import ast as _ast, inspect, textwrap
    def _code_names(fn):                          # identifiers actually used in the CODE (docstrings / comments excluded)
        tree = _ast.parse(textwrap.dedent(inspect.getsource(fn)))
        return {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)} | {n.attr for n in _ast.walk(tree) if isinstance(n, _ast.Attribute)}
    used_ = _code_names(M.ModelRuntime.ensure_loaded) | _code_names(M.ModelRuntime._use_prequantized)
    bad_ = used_ & {"quantize_dynamic", "_quantize_encoder_only", "_tune_precision", "_time_inference", "_quant_close", "measure_rtf"}
    check("source guard: the code of ensure_loaded / _use_prequantized uses no quantize_dynamic, _quantize_encoder_only, _tune_precision or timing helper",
          not bad_ and "_use_prequantized" in used_, bad_)

    # --- image-build step: what it keeps and what it rejects
    d = os.path.join(_tmp, "b_ok"); meta, m_, fp32, benv = run_builder(d, [0.995, 0.99999], [0.50, 0.30])
    on_disk = json.load(open(os.path.join(d, M.QUANT_META_FILE)))
    check("build: good logits + faster + faithful reload -> INT8 file and meta written, bench recorded, fp32 model left on m",
          meta["int8"] is True and os.path.getsize(os.path.join(d, M.QUANT_MODEL_FILE)) == meta["bytes"] == 64 and on_disk == meta
          and meta["bench"]["int8_rtf"] < meta["bench"]["fp32_rtf"] and m_.model is fp32 and meta["min_logit_cosine"] == 0.995, meta)
    env = _FakeEnv(); rt_ = start_runtime(env, d)
    check("build -> start-up contract: the artifact the build step wrote is accepted by the loader (schema, torch version, size)",
          rt_.int8 is True and rt_.quant_info["source"] == "image-artifact" and env.calls == [])
    d = os.path.join(_tmp, "b_drift"); meta, *_ = run_builder(d, [0.90], [0.5])
    check("build: logits drift (cosine 0.90 < 0.985) -> fp32 kept, no model file, reason written",
          meta["int8"] is False and "drifted" in meta["reason"] and not os.path.exists(os.path.join(d, M.QUANT_MODEL_FILE)) and os.path.exists(os.path.join(d, M.QUANT_META_FILE)), meta)
    d = os.path.join(_tmp, "b_slow"); meta, *_ = run_builder(d, [0.995], [0.50, 0.70])
    check("build: INT8 slower than fp32 on the build host -> fp32 kept", meta["int8"] is False and "slower" in meta["reason"] and not os.path.exists(os.path.join(d, M.QUANT_MODEL_FILE)), meta)
    d = os.path.join(_tmp, "b_reload"); meta, *_ = run_builder(d, [0.995, 0.5], [0.5, 0.3])
    check("build: a saved file that does not reproduce the in-memory model is deleted and rejected",
          meta["int8"] is False and "does not reproduce" in meta["reason"] and not os.path.exists(os.path.join(d, M.QUANT_MODEL_FILE)), meta)
    d = os.path.join(_tmp, "b_exc"); meta, *_ = run_builder(d, [], [0.5], quant_raises=RuntimeError("no fbgemm"))
    check("build: an exception never fails the image build; meta says why", meta["int8"] is False and "no fbgemm" in meta["reason"] and os.path.exists(os.path.join(d, M.QUANT_META_FILE)), meta)
    d = os.path.join(_tmp, "b_unverifiable"); meta, *_ = run_builder(d, [None, None], [0.5, 0.3])
    check("build: if the logit check cannot run (library internals changed) the artifact is kept but honestly marked cosine_checked=false",
          meta["int8"] is True and meta["cosine_checked"] is False and meta["min_logit_cosine"] is None, meta)
    d = os.path.join(_tmp, "b_stale"); os.makedirs(d); write_artifact(d)
    meta, *_ = run_builder(d, [], [], precision="fp32")
    check("build: a stale artifact from an earlier layer is removed before anything else (PRECISION='fp32' -> no model file)",
          meta["int8"] is False and meta["reason"] == "PRECISION=fp32" and not os.path.exists(os.path.join(d, M.QUANT_MODEL_FILE)))

    # --- the Modal image function really calls the build step (CPU only)
    calls_ = []
    for cuda_ in (False, True):
        with _FakeEnv(cuda=cuda_) as fe, mock.patch.object(mod, "build_prequantized_artifact", lambda m, *a, **k: calls_.append((cuda_, m))):
            mod.download_muaalem_weights()
    check("image function download_muaalem_weights: loads the model once, calls the pre-quantisation step on CPU and skips it on a GPU image",
          len(calls_) == 1 and calls_[0][0] is False and calls_[0][1].model.tag == "fp32", calls_)
finally:
    shutil.rmtree(_tmp, ignore_errors=True)

res = subprocess.run([sys.executable, os.path.join(HERE, "cost_proof.py"), "--redundancy", "2.2"], capture_output=True, text=True)
check("cost script runs and prints the breakdown (reservation, burst, RAM) and a verdict", res.returncode == 0 and "CPU reserv." in res.stdout and "VERDICT" in res.stdout and "CPU burst" in res.stdout, res.stderr[-300:])


res = subprocess.run([sys.executable, os.path.join(HERE, "test_security.py")], capture_output=True, text=True)
print(res.stdout.rstrip())
check("security suite (dev/test_security.py): authentication, validation, limits, debug gating, error hygiene all pass",
      res.returncode == 0, res.stdout[-800:] + res.stderr[-400:])
res = subprocess.run([sys.executable, os.path.join(HERE, "..", "eval", "test_eval_stats.py")], capture_output=True, text=True)
print(res.stdout.rstrip())
check("evaluation statistics suite (eval/test_eval_stats.py) passes", res.returncode == 0, res.stdout[-800:] + res.stderr[-400:])


print("\n%d check(s) failed" % len(FAILS) if FAILS else "\nALL CHECKS PASSED")
sys.exit(1 if FAILS else 0)
