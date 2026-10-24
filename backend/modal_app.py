"""
Rafiq v14.5 - Real-time Quran recitation corrector (Modal.com backend).

Engine: `quran-muaalem` (obadx/muaalem-model-v3_2) + `quran-transcript` (QPS phonemes).

v14.5 = LIVE ONLY.  The record-then-upload (batch) mode and its /analyze endpoint are gone.

  * Word-synchronous tracking with a DUAL-STAGE lock:
      stage 1 (soft)  a progress message carries `tracked` as soon as the alignment enters word k;
      stage 2 (hard)  word k is judged and locked only after >= 2 stable phonemes of word k+1 were heard
                      (or a pause arrived); the next model window starts LEFT_CTX_S (300 ms) before the first
                      unlocked word, so cross-word rules and right-edge context are preserved.
  * The model only ever sees the uncommitted tail (<= 2.2 s, stretching to 4 s while a long madd is sustained,
    widening further only if the tail cannot make progress).
  * A zero-copy Int16 gate (RMS / zero-crossing rate / first difference) lets the engine skip at most one
    neural hop in a row while a vowel is steady; any formant/energy change wakes the neural pass.
  * The client drops silence (after 500 ms) and sends {"type":"pause"}; the server keeps real acoustic pauses
    for waqf / ibtida' decisions and cuts a socket that stays idle for IDLE_KILL_S (paused_idle).
  * Madd muttasil (4..6 harakat) is judged on the audio too (CTC frame spans + 10 ms voiced envelope).

WebSocket protocol  (`/ws/live-recitation?sura_idx=2&aya_idx=255&strictness=balanced`)
  client -> server : binary = 16 kHz mono int16 LE PCM (~160 ms per frame)
                     text   {"type":"stop"} | {"type":"reset"} | {"type":"pause","duration_ms":N[,"speech_tail_padding_ms":500]}
                            {"type":"resume","word_idx":k}   (first frame after a reconnect)
  server -> client : {"type":"ready"|"progress"|"final"|"paused_idle"|"error", ...}   (see SessionEngine)

Other endpoints: GET /health, GET /debug/last, GET /debug/ref.
"""

import asyncio
import bisect
import gc
import json
import math
import os
import re
import threading
import time
import traceback
import unicodedata
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

VERSION = "14.5"
ENGINE_NAME = "rafiq-v14.5"

# Thread pools / allocator behaviour must be pinned BEFORE numpy / torch are imported.
# Spin-waiting is disabled (PASSIVE) because Modal bills CPU that is burned while "waiting".
NUM_THREADS = int(os.environ.get("RAFIQ_THREADS", "2"))   # matches cpu=(0.125, 2.0) in the Modal section
for _k, _v in (
    ("OMP_NUM_THREADS", str(NUM_THREADS)), ("MKL_NUM_THREADS", str(NUM_THREADS)),
    ("OPENBLAS_NUM_THREADS", str(NUM_THREADS)), ("OMP_WAIT_POLICY", "PASSIVE"), ("KMP_BLOCKTIME", "0"),
    ("GOMP_SPINCOUNT", "0"), ("MALLOC_TRIM_THRESHOLD_", "65536"), ("TOKENIZERS_PARALLELISM", "false"),
):
    os.environ.setdefault(_k, _v)

import numpy as np

try:  # allows importing the pure-python engine in local tests without Modal installed
    import modal
except ImportError:  # pragma: no cover
    modal = None

try:  # faster JSON for the websocket frames (optional)
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None

# ==============================================================================
# Deployment switches (the only things you may want to change)
# ==============================================================================
USE_MEMORY_SNAPSHOT = True   # Modal restores a snapshot of the loaded model: near-instant cold start
USE_GPU = False          # set True ONLY after you add a card to Modal (T4 GPU = ~10x faster)
PRECISION = "int8"       # "auto": int8 only if fp32 is far too slow (rtf > AUTO_INT8_ABOVE_RTF) | "fp32" | "int8"
AUTO_INT8_ABOVE_RTF = 0.60
MODEL_ID = "obadx/muaalem-model-v3_2"
APP_NAME = "rafiq-quran-muaalem-backend"
DEFAULT_SURA, DEFAULT_AYA = 2, 255

# ==============================================================================
# Streaming-engine constants
# ==============================================================================
SR = 16000
FRAME = 320                       # 20 ms analysis frame
HOP_S = float(os.environ.get("RAFIQ_HOP_S", "0.32"))   # run the model when this much new audio arrived (160 ms; see dev/cost_proof.py)
HOP_SLOW_S = max(HOP_S, 0.20)                           # ...and 200 ms when the server is slow (rtf > 0.5)
MIN_VOICED_S = 0.30               # ...and the window contains at least this much speech
MIN_NEW_VOICED_S = 0.10           # ...of which this much is new
LEFT_CTX_S = 0.30                 # strict left-context overlap of every window (cross-word rules, right-edge attention)
TAIL_BASE_S = 2.2                 # uncommitted tail sent to the model
TAIL_ELASTIC_S = 4.0              # ...while a long madd / slow tahqiq is being sustained, or after a stalled pass
MAX_WINDOW_S = 10.0               # last-resort cap (hopeless stall / burst replay)
BURST_WINDOW_S = 6.0              # window used while catching up on a large backlog
MAX_BACKLOG_S = 240.0             # backlog (cold-start replay, bad network) is judged in order; skipped only beyond this
PRE_PAD_S = 0.25                  # speech onset padding (low-energy phonemes, plosive closures)
POST_PAD_S = 0.50                 # speech offset padding (hangover)
TAIL_SILENCE_S = 0.35             # silence at the end of the audio = the reciter paused (waqf)
PAUSE_MIN_MS = 500                # the client sends a pause frame after this much continuous silence
IDLE_KILL_S = 8.0                 # a socket with no frame at all for this long is closed (paused_idle)
CTX_WORDS = 2                     # already-judged words kept in the reference as free context
LOCAL_AHEAD_WORDS = 3             # default alignment look-ahead beyond the tracked word (k-1, k, k+1 + slack)
REGION_AHEAD_WORDS = 26           # widened look-ahead (escape hatch)
PHRASE_BACK_MAX = 12              # how far back the restart stage may reach
LOCK_STABLE_PHONEMES = 2          # phonemes of word k+1 that must be established to lock word k
LOCK_STABLE_PROB = 0.70           # ...each matched with at least this posterior
MAX_SESSION_S = 360               # memory guard
QUALITY_MAX_COST = 0.62           # window rejected when alignment cost per reference char > this
MIN_MATCHED_LETTERS = 3           # ...or when fewer letters than this were matched
WORD_REACH_COV = 0.40             # share of a word's letters that must match to count as "reached"
WORD_SKIP_COV = 0.25              # below this a word that was passed over counts as skipped
LAST_WORD_SETTLE_COV = 0.60       # last word is judged on a pause only if this much was matched
UNCLEAR_ISSUES = 3                # >= this many letter issues in one word -> one "unclear word" note
DEBUG_KEEP_STEPS = 80
# ---- spectral-flux / steady-vowel gate
GATE_HIGH_CONF = 0.95             # decoded tail vowel/madd must have at least this posterior (2 passes in a row)
GATE_MAX_SKIP_IN_ROW = 1          # never skip two neural hops in a row
# ---- acoustic madd measurement
ENV_HOP_S = 0.010                 # 10 ms RMS envelope
MPH_MIN_MS, MPH_MAX_MS, MPH_DEFAULT_MS = 85.0, 185.0, 120.0   # one harakah, clamped / deterministic fallback
MPH_MIN_SAMPLES = 2               # tabii runs needed before the reciter's own tempo is trusted
ACOUSTIC_MAX_DISAGREE = 4         # |acoustic - decoded| harakat above which the measurement is treated as a failure
ACOUSTIC_MARGIN_H = 1.0           # the audio only ADDS a madd flag when it is this far (harakat) outside the allowed range.
                                  # Measured on clean 4-harakat madds (dev/sim.py): 3.56 +- 0.18, min 3.06 - the CTC token sits
                                  # inside the phoneme (-0.44 bias) and frames are 40 ms. The audio therefore catches CLEAR
                                  # under/over-extension (<= ~2.5 / >= ~7.5 harakat); 3 vs 4 is decided by the decoded count.
MUTTASIL_LO, MUTTASIL_HI = 4, 6
RIVAL_GATE_FACTOR = 0.5           # confidence gate multiplier for substitutions inside a rival confusion pair

SIFAT_LEVELS: Tuple[str, ...] = (
    "ghonna", "qalqla", "tafkheem_or_taqeeq", "hams_or_jahr", "shidda_or_rakhawa",
    "itbaq", "safeer", "tikraar", "tafashie", "istitala",
)


@dataclass(frozen=True)
class Preset:
    """Strictness profile. 'balanced' is the default.
    NOTE: CTC posteriors of emitted tokens are ~0.99-1.0 for almost everything (see live log), so the
    confidence gates only remove genuinely uncertain emissions; the tolerance rules do the real work."""
    name: str
    conf_letter: float          # min model confidence to report a letter / tajweed issue
    conf_vowel: float           # min model confidence to report a vowel (tashkeel) issue
    conf_sifat: float           # min confidence to report a qalqala / tafkheem / ghunna mismatch
    short_slack: int            # madd may be this many harakat shorter than the lower bound
    long_slack: int             # madd may be this many harakat longer than the upper bound
    flag_missing_vowel: bool
    flag_extra_vowel: bool
    flag_extra_letter: bool
    flag_extra_shaddah: bool
    flag_waqf_no_pause: bool    # word-final vowel dropped although the reciter did not stop
    sifat_levels: Tuple[str, ...]


PRESETS: Dict[str, Preset] = {
    "relaxed": Preset("relaxed", conf_letter=0.85, conf_vowel=0.85, conf_sifat=0.92,
                      short_slack=1, long_slack=2, flag_missing_vowel=False, flag_extra_vowel=False,
                      flag_extra_letter=False, flag_extra_shaddah=False, flag_waqf_no_pause=False,
                      sifat_levels=("qalqla", "ghonna")),
    "balanced": Preset("balanced", conf_letter=0.70, conf_vowel=0.70, conf_sifat=0.82,
                       short_slack=0, long_slack=1, flag_missing_vowel=True, flag_extra_vowel=True,
                       flag_extra_letter=False, flag_extra_shaddah=False, flag_waqf_no_pause=False,
                       sifat_levels=("qalqla", "ghonna", "tafkheem_or_taqeeq")),
    "strict": Preset("strict", conf_letter=0.50, conf_vowel=0.50, conf_sifat=0.65,
                     short_slack=0, long_slack=0, flag_missing_vowel=True, flag_extra_vowel=True,
                     flag_extra_letter=True, flag_extra_shaddah=True, flag_waqf_no_pause=True,
                     sifat_levels=("qalqla", "ghonna", "tafkheem_or_taqeeq", "hams_or_jahr", "shidda_or_rakhawa",
                                   "itbaq", "safeer", "tikraar", "tafashie", "istitala")),
}


def get_preset(name: Optional[str]) -> Preset:
    return PRESETS.get((name or "balanced").strip().lower(), PRESETS["balanced"])


# ==============================================================================
# Character classes (structural, NOT based on guessed QPS symbols)
# ==============================================================================
FATHA, DAMMA, KASRA, SHADDA = "\u064e", "\u064f", "\u0650", "\u0651"
CORE_VOWELS: Dict[str, str] = {FATHA: "الفتحة", DAMMA: "الضمة", KASRA: "الكسرة"}
VOWEL_HOW: Dict[str, str] = {
    FATHA: "افتح فمك قليلًا وانطق الحرف بصوت «أَ».",
    DAMMA: "اضمم شفتيك مستديرتين وانطق الحرف بصوت «أُ».",
    KASRA: "اخفض فكّك وافتح شفتيك للجانبين وانطق الحرف بصوت «إِ».",
}
BASIC_LETTERS = set("ءابتثجحخدذرزسشصضطظعغفقكلمنهويىةأإؤئآ")
MADD_LETTERS = set("اويىۥۦ")
ALWAYS_MADD = set("اۥۦ")

LETTER_NAMES: Dict[str, str] = {
    "ء": "الهمزة", "ا": "الألف", "ب": "الباء", "ت": "التاء", "ث": "الثاء", "ج": "الجيم",
    "ح": "الحاء", "خ": "الخاء", "د": "الدال", "ذ": "الذال", "ر": "الراء", "ز": "الزاي",
    "س": "السين", "ش": "الشين", "ص": "الصاد", "ض": "الضاد", "ط": "الطاء", "ظ": "الظاء",
    "ع": "العين", "غ": "الغين", "ف": "الفاء", "ق": "القاف", "ك": "الكاف", "ل": "اللام",
    "م": "الميم", "ن": "النون", "ه": "الهاء", "و": "الواو", "ي": "الياء", "ى": "الياء",
    "ۥ": "الواو", "ۦ": "الياء", "ة": "التاء المربوطة",
}
MAKHRAJ: Dict[str, str] = {
    "ء": "يخرج من أقصى الحلق", "ه": "يخرج من أقصى الحلق",
    "ع": "يخرج من وسط الحلق", "ح": "يخرج من وسط الحلق",
    "غ": "يخرج من أدنى الحلق", "خ": "يخرج من أدنى الحلق",
    "ق": "يخرج من أقصى اللسان مع الحنك الأعلى", "ك": "يخرج من أقصى اللسان أسفل مخرج القاف قليلًا",
    "ج": "يخرج من وسط اللسان مع الحنك الأعلى", "ش": "يخرج من وسط اللسان مع الحنك الأعلى",
    "ي": "يخرج من وسط اللسان مع الحنك الأعلى",
    "ض": "يخرج من حافة اللسان مع الأضراس العليا",
    "ل": "يخرج من حافة اللسان مع اللثة العليا", "ن": "يخرج من طرف اللسان مع اللثة العليا",
    "ر": "يخرج من طرف اللسان مع تكرار خفيف",
    "ط": "يخرج من طرف اللسان مع أصول الثنايا العليا", "د": "يخرج من طرف اللسان مع أصول الثنايا العليا",
    "ت": "يخرج من طرف اللسان مع أصول الثنايا العليا",
    "ص": "يخرج من طرف اللسان مع أطراف الثنايا السفلى (حرف صفير)",
    "س": "يخرج من طرف اللسان مع أطراف الثنايا السفلى (حرف صفير)",
    "ز": "يخرج من طرف اللسان مع أطراف الثنايا السفلى (حرف صفير)",
    "ظ": "يخرج من طرف اللسان مع أطراف الثنايا العليا", "ذ": "يخرج من طرف اللسان مع أطراف الثنايا العليا",
    "ث": "يخرج من طرف اللسان مع أطراف الثنايا العليا",
    "ف": "يخرج من باطن الشفة السفلى مع أطراف الثنايا العليا",
    "ب": "يخرج من انطباق الشفتين", "م": "يخرج من انطباق الشفتين مع غنة",
    "و": "يخرج من استدارة الشفتين",
}


GHUNNA_SYMBOLS = set("ں")                # QPS symbol for ikhfaa / idgham ghunna (seen in live output: مَںںںذَ)
GHUNNA_LETTERS = set("نمويى")          # only these can carry ghunna (noon, meem, idgham bi-ghunna)
TAFKHEEM_LETTERS = set("رلصضطظقغخ")      # letters whose heaviness is a real tajweed question

# Phonetic domain of the rare sifat: a contrast is only evaluated on letters where it is phonetically valid.
# (tafashie exists only on ش, tikraar only on ر, istitala only on ض, safeer only on ص ز س, ...)
SIFA_DOMAIN: Dict[str, str] = {
    "tafashie": "ش", "tikraar": "ر", "istitala": "ض", "safeer": "صزس", "qalqla": "قطبجد", "itbaq": "صضطظ",
}
# The label that marks the property as present (everything else is the default / absent label).
SIFA_RARE: Dict[str, str] = {
    "tafashie": "motafashie", "tikraar": "mokarar", "istitala": "mostateel", "safeer": "safeer",
    "qalqla": "moqalqal", "ghonna": "maghnoon", "itbaq": "motbaq", "tafkheem_or_taqeeq": "mofakham",
    "hams_or_jahr": "hams",
}
# Voicing / manner contrasts on a consonant that is squeezed between voiced vowels are unreliable on very short
# function words (the 40 ms CTC frame of ف in فِي is dominated by the neighbouring kasrah / yaa).
SIFA_BLEED_LEVELS = ("hams_or_jahr", "shidda_or_rakhawa")
BLEED_MAX_CORE = 2
# Rival confusion pairs (tajweed): the model often splits its probability mass inside these pairs, so a
# substitution inside one is judged with a lower confidence gate (and with the relative odds when available).
_RIVALS = [("ط", "ت"), ("ص", "س"), ("ض", "د"), ("ظ", "ذ"), ("ظ", "ز"), ("ذ", "ز"), ("ق", "ك"),
           ("ح", "ه"), ("ع", "ء"), ("ر", "غ")]
RIVAL_PAIRS = frozenset(_RIVALS + [(b, a) for a, b in _RIVALS])


def is_mark(c: str) -> bool:
    return unicodedata.category(c) == "Mn"


def lname(c: str) -> str:
    return LETTER_NAMES.get(c, "")


def clean_phonemes(text: str) -> str:
    """Drops whitespace and any <special-token> the tokenizer may emit."""
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", "", str(text))
    return "".join(ch for ch in text if not ch.isspace())


def letters_only(s: str) -> List[str]:
    return [c for c in s if not is_mark(c)]


def collapse_runs(chars: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in chars:
        if not out or out[-1] != c:
            out.append(c)
    return out


def split_runs(s: str) -> List[Tuple[str, int, int]]:
    runs: List[Tuple[str, int, int]] = []
    i, n = 0, len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        runs.append((s[i], i, j))
        i = j
    return runs


def prev_letter(s: str, idx: int) -> str:
    """Nearest letter (non-mark) at or before idx — the consonant a vowel mark belongs to."""
    k = idx
    while k >= 0:
        if not is_mark(s[k]):
            return s[k]
        k -= 1
    return ""


# ==============================================================================
# Arabic notes (what the user sees when tapping a coloured word)
# ==============================================================================
def hint(letter: str) -> str:
    h = MAKHRAJ.get(letter)
    return f" ({h})" if h else ""


def _disp(c: str) -> str:
    """How a phoneme is shown to the user (QPS madd marks are shown by name, never as raw symbols)."""
    if c in ALWAYS_MADD:
        return {"ا": "مدّ الألف", "ۥ": "مدّ الواو", "ۦ": "مدّ الياء"}[c]
    return f"«{c}»"


def note_replace(e: str, p: str) -> str:
    return f"نطقتَ {_disp(p)} بدل {_disp(e)}. الصواب أن تنطق {_disp(e)}{hint(e)}."


def note_missing_letter(e: str) -> str:
    return f"لم يُسمع حرف «{e}» في هذه الكلمة. انطقه بوضوح{hint(e)}."


def note_extra_letter(p: str) -> str:
    return f"زدتَ حرفًا غير موجود في الكلمة (سُمع «{p}»). اقرأ الكلمة كما هي دون زيادة."


NOTE_SKIPPED = "لم تقرأ هذه الكلمة (تجاوزتها). ارجع واقرأها قبل أن تنتقل لما بعدها."
NOTE_UNCLEAR = "لم تُنطق هذه الكلمة بوضوح كافٍ، أو فيها أكثر من خطأ في الحروف. أعد قراءتها ببطء ووضوح."
NOTE_RULE = ("لم يُطبَّق حكم التجويد المطلوب في هذا الموضع (كالإخفاء أو الإدغام أو الغنة أو القلب). "
             "راجع حكم هذه الكلمة ثم أعد قراءتها.")


def _on_letter(letter: str) -> str:
    return f" على حرف «{letter}»" if letter in BASIC_LETTERS else ""


def note_vowel_swap(letter: str, e: str, p: str) -> str:
    return f"المطلوب{_on_letter(letter)} {CORE_VOWELS[e]} ولكنك نطقتَ {CORE_VOWELS[p]}. {VOWEL_HOW[e]}"


def note_vowel_missing(letter: str, e: str) -> str:
    return f"لم تنطق {CORE_VOWELS[e]}{_on_letter(letter)}. انطق الحرف بحركته كاملة. {VOWEL_HOW[e]}"


def note_vowel_extra(letter: str, p: str) -> str:
    return f"نطقتَ {CORE_VOWELS[p]}{_on_letter(letter)} وهو ساكن في الكلمة. اقرأه ساكنًا دون حركة."


def madd_name(c: str) -> str:
    return {"ا": "الألف", "و": "الواو", "ۥ": "الواو", "ي": "الياء", "ى": "الياء", "ۦ": "الياء"}.get(c, "حرف المد")


MADD_VOWEL = {"ا": "الفتحة", "ۥ": "الضمة", "ۦ": "الكسرة"}


def note_madd_vowel_swap(c: str, p: str) -> str:
    return (f"حرف المدّ: المطلوب {_disp(c)} (من {MADD_VOWEL.get(c, '')}) ولكنك مددتَ {_disp(p)} "
            f"(من {MADD_VOWEL.get(p, '')}). انتبه لحركة الحرف قبل المدّ.")


def note_waqf_no_pause(vowel: str) -> str:
    name = CORE_VOWELS.get(vowel, "حركة")
    how = VOWEL_HOW.get(vowel, "")
    return (f"آخر هذه الكلمة: المطلوب {name} (أو تنوين) لأنك تصل القراءة بما بعدها، "
            f"لكنك سكّنتَ آخرها دون أن تقف. التسكين يكون عند الوقف فقط. {how}").strip()


def note_madd_short(c: str, need_text: str) -> str:
    return (f"المدّ ({madd_name(c)}): قصّرتَ في مدّه. حقّه {need_text}؛ "
            "مُدّ الصوت بهدوء، ومقدار الحركة كقبض الإصبع وبسطه.")


def note_waqf_vowel(letter: str, vowel: str) -> str:
    name = CORE_VOWELS.get(vowel, "حركة")
    return (f"هذه آخر كلمة في الآية فيُوقف عليها بالسكون، لكنك نطقتَ {name}{_on_letter(letter)} في آخرها. "
            "اقف على آخر حرف ساكنًا دون حركة.")


def note_madd_dropped(c: str, need_text: str) -> str:
    return f"المدّ ({madd_name(c)}): لم تمدّ الصوت إطلاقًا. حقّه {need_text}؛ مُدّه بهدوء وبمقدار ثابت."


def note_madd_long(c: str, e: int) -> str:
    return f"المدّ ({madd_name(c)}): أطلتَ فيه أكثر من اللازم. اقتصر على نحو {e} حركات."


def note_shaddah_dropped(letter: str) -> str:
    extra = " مع الغنة" if letter in "نم" else ""
    return (f"التشديد: حرف «{letter}» مشدَّد ولم تُشدّده. اضغط عليه لحظة ثم أطلقه{extra}.")


def note_shaddah_extra(letter: str) -> str:
    return f"شدّدتَ حرف «{letter}» وهو غير مشدَّد في هذا الموضع. انطقه مخفّفًا."


def note_qalqla(letter: str) -> str:
    where = f" الحرف «{letter}»" if letter in BASIC_LETTERS else " هذا الحرف"
    return ("القلقلة:" + where + " ساكن أو موقوف عليه ويلزمه قلقلة؛ أي نبرة خفيفة مرتدّة تُسمع عند "
            "إطلاقه (حروفها: ق ط ب ج د).")


def note_ghonna(letter: str, expected_ghonna: bool) -> str:
    l = f"«{letter}»" if letter in BASIC_LETTERS else "هذا الحرف"
    if expected_ghonna:
        return f"الغنة: ينقص صوت الغنة في {l}. أخرج الغنة من الخيشوم بمقدار حركتين."
    return f"أظهرتَ غنة في {l} وليس فيه غنة هنا. انطقه دون غنة."


def note_tafkheem(letter: str, expected_heavy: bool) -> str:
    l = f"«{letter}»" if letter in BASIC_LETTERS else "هذا الحرف"
    if expected_heavy:
        return f"التفخيم: حقّ {l} التفخيم هنا؛ فخّمه بتغليظ الصوت مع رفع مؤخرة اللسان نحو الحنك."
    return f"الترقيق: حقّ {l} الترقيق هنا؛ رقّقه بإخراج الصوت رقيقًا مع خفض مؤخرة اللسان."


SIFA_AR = {
    "hams_or_jahr": "الهمس والجهر", "shidda_or_rakhawa": "الشدة والرخاوة", "itbaq": "الإطباق والانفتاح",
    "safeer": "الصفير", "tikraar": "التكرار", "tafashie": "التفشّي", "istitala": "الاستطالة",
}


def note_sifa(level: str, letter: str) -> str:
    return (f"صفة «{SIFA_AR.get(level, 'الحرف')}» في «{letter}» لم تظهر كما ينبغي (تقدير دقيق من النموذج، "
            f"قد يتأثر بالميكروفون). استمع للقارئ ثم أعد الكلمة بوضوح{hint(letter)}.")


def note_ghunna_short(E: int) -> str:
    return ("الغنة: قصّرتَ في غنّة الإخفاء/الإدغام. أخرج الصوت من الخيشوم بمقدار ثابت (نحو حركتين) "
            "ولا تقطعه مبكرًا.")


def note_ghunna_long() -> str:
    return "الغنة: أطلتَ الغنة أكثر من اللازم. اقتصر على نحو حركتين."


def note_start_no_pause() -> str:
    return ("لا وقف قبل هذه الكلمة، فالقراءة وصلٌ بما قبلها: لا تبدأها من جديد (بنطق همزة الوصل، أو بترك "
            "الإدغام/الإخفاء مع الكلمة السابقة). صِلها بما قبلها دون توقف.")


def note_lin_extended(c: str) -> str:
    return (f"حرف اللين «{c}» في هذه الكلمة لا مدّ فيه؛ مددتَه كأنه حرف مدّ. انطقه سريعًا بلا إطالة.")


def note_sifa_generic(letter: str) -> str:
    l = f"«{letter}»" if letter in BASIC_LETTERS else "هذا الحرف"
    return f"صفة {l} لم تظهر كما ينبغي. استمع للقارئ ثم أعد قراءة الكلمة."


# ==============================================================================
# Phoneme aligner (edit distance with run-aware gap costs)
# ==============================================================================
@lru_cache(maxsize=16384)
def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    ma, mb = is_mark(a), is_mark(b)
    if ma and mb:
        return 0.6 if (a in CORE_VOWELS and b in CORE_VOWELS) else 0.8
    if ma != mb:
        return 1.7                      # never turn a vowel mark into a letter
    if a in MADD_LETTERS and b in MADD_LETTERS:
        return 0.9
    return 1.0


# ---- pre-compiled integer substitution table (built once at import; extended lazily for unseen symbols)
_TAB_LOCK = threading.Lock()
_CHARS: List[str] = []
_CODE: Dict[str, int] = {}
_TAB: List[List[float]] = []


def _code_of(ch: str) -> int:
    k = _CODE.get(ch)
    if k is not None:
        return k
    with _TAB_LOCK:
        k = _CODE.get(ch)
        if k is None:
            k = len(_CHARS)
            _CHARS.append(ch)
            for r in range(k):
                _TAB[r].append(sub_cost(_CHARS[r], ch))
            _TAB.append([sub_cost(ch, c2) for c2 in _CHARS])
            _CODE[ch] = k
    return k


for _ch in sorted(BASIC_LETTERS | MADD_LETTERS | GHUNNA_SYMBOLS | set("ٱ\u064b\u064c\u064d\u0651\u0652\u06e1\u0670")
                  | set(CORE_VOWELS)):
    _code_of(_ch)


def codes_of(s: str) -> List[int]:
    return [_code_of(ch) for ch in s]


def gap_costs(s: str) -> List[float]:
    """Per-character gap cost; a repeated character (madd / shaddah length) is cheap to add or drop."""
    out: List[float] = []
    prev = ""
    for c in s:
        if is_mark(c):
            out.append(0.7)
        elif c == prev:
            out.append(0.25)
        else:
            out.append(1.0)
        prev = c
    return out


@dataclass
class Alignment:
    pairs: List[Tuple[int, int]]       # (ref_idx, pred_idx); -1 = gap on that side
    i_end: int                         # reference chars consumed
    j_end: int                         # predicted chars consumed
    cost: float


def align(
    ref: str,
    pred: str,
    *,
    lead_free: int = 0,
    lead_pen: float = 0.0,
    pred_lead_free: int = 0,
    pred_lead_pen: float = 0.08,
    tail_free_ref: bool = True,
    tail_ins_chars: int = 0,
    tail_ins_pen: float = 0.08,
) -> Alignment:
    """
    Anchored-start alignment.
      lead_free / lead_pen   : the first `lead_free` reference chars may be skipped at a tiny cost
                               (the context words already judged, or a few words after an overload)
      pred_lead_free         : the first predicted chars may be inserted cheaply (opening du'a, basmala, an aborted
                               first attempt of a repeated phrase) - before any reference char AND right after the free context
      tail_free_ref          : the reference tail is free (the reciter has not reached it yet)
      tail_ins_chars         : when the reference is fully consumed, up to this many trailing
                               predicted chars may be ignored cheaply (e.g. "sadaqa Allah...")
    """
    M, L = len(ref), len(pred)
    ins = gap_costs(pred)
    dele = gap_costs(ref)
    pred_mark = [is_mark(ch) for ch in pred]
    pcodes = codes_of(pred)
    dp = [[0.0] * (L + 1) for _ in range(M + 1)]
    bt = [[0] * (L + 1) for _ in range(M + 1)]
    for j in range(1, L + 1):
        dp[0][j] = dp[0][j - 1] + (pred_lead_pen if j <= pred_lead_free else ins[j - 1])
        bt[0][j] = 2
    for i in range(1, M + 1):
        dp[i][0] = dp[i - 1][0] + (lead_pen if i <= lead_free else dele[i - 1])
        bt[i][0] = 1
        rc = ref[i - 1]
        di = dele[i - 1]
        prev, cur, btr = dp[i - 1], dp[i], bt[i]
        rm = is_mark(rc)
        trow = _TAB[_code_of(rc)]
        lead_row = lead_free > 0 and i == lead_free      # where the free context ends, junk may still be skipped cheaply
        for j in range(1, L + 1):
            c = trow[pcodes[j - 1]]
            if rm and pred_mark[j - 1] and i >= 2 and j >= 2 and ref[i - 2] != pred[j - 2]:
                c += 0.4                    # a vowel belongs to its letter: pair marks that follow the same letter
            d = prev[j - 1] + c
            u = prev[j] + di
            v = cur[j - 1] + (pred_lead_pen if (lead_row and j <= pred_lead_free) else ins[j - 1])
            if d <= u and d <= v:
                cur[j] = d
                btr[j] = 0
            elif u <= v:
                cur[j] = u
                btr[j] = 1
            else:
                cur[j] = v
                btr[j] = 2

    best_c, best_i, best_j = float("inf"), M, L
    if tail_free_ref:
        for i in range(M + 1):
            c = dp[i][L]
            if c < best_c - 1e-9:
                best_c, best_i, best_j = c, i, L
    else:
        best_c, best_i, best_j = dp[M][L], M, L
    if tail_ins_chars > 0:
        for j in range(max(0, L - tail_ins_chars), L):
            c = dp[M][j] + tail_ins_pen * (L - j)
            if c < best_c - 1e-9:
                best_c, best_i, best_j = c, M, j

    pairs: List[Tuple[int, int]] = []
    i, j = best_i, best_j
    while i > 0 and j > 0:
        t = bt[i][j]
        if t == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif t == 1:
            pairs.append((i - 1, -1))
            i -= 1
        else:
            pairs.append((-1, j - 1))
            j -= 1
    while j > 0:
        pairs.append((-1, j - 1))
        j -= 1
    while i > 0:
        if i > lead_free:
            pairs.append((i - 1, -1))
        i -= 1
    pairs.reverse()
    return Alignment(pairs, best_i, best_j, float(best_c))


# ==============================================================================
# Sifat helpers (per-letter properties predicted by the model)
# ==============================================================================
def _sane_prob(prob: Any, idx: Any = None) -> Optional[float]:
    """None = no probability supplied. A value outside [0, 1] is CORRUPT (the live report showed the integer
    class index 2 / 3 of the `between` / `rikhw` labels arriving in the probability slot): it becomes 0.0, i.e.
    'no evidence', so it can never pass a confidence gate and never be printed as a confidence of 2 or 3."""
    if prob is None:
        return None
    try:
        p = float(prob)
    except Exception:
        return 0.0
    if not math.isfinite(p) or p < 0.0 or p > 1.0 + 1e-6:
        return 0.0
    return min(1.0, p)


def _field(v: Any, *names: str) -> Any:
    for nm in names:
        val = v.get(nm) if isinstance(v, dict) else getattr(v, nm, None)
        if val is not None:
            return val
    return None


def _unwrap_unit(v: Any) -> Tuple[Optional[str], Optional[float]]:
    """SingleUnit(text, prob, idx) -> (label, probability).  Reads fields BY NAME (never by position), takes the
    probability of the chosen class when a whole distribution is supplied, and sanitises the result."""
    if v is None:
        return None, None
    if isinstance(v, dict) or hasattr(v, "text"):
        label = _field(v, "text")
    else:
        label = getattr(v, "value", v)
    prob = _field(v, "prob")
    dist = _field(v, "probs") if prob is None else None
    idx = _field(v, "idx", "index", "id")
    if prob is None and dist is not None:
        try:
            if hasattr(dist, "detach"):
                dist = dist.detach().float().cpu().numpy()
            arr = np.asarray(dist, dtype=np.float64).reshape(-1)
            if arr.size:
                ok_idx = isinstance(idx, (int, np.integer)) and 0 <= int(idx) < arr.size
                prob = float(arr[int(idx)]) if ok_idx else float(arr.max())
        except Exception:
            prob = None
    elif prob is not None:
        try:
            if hasattr(prob, "detach"):
                prob = prob.detach().float().cpu().numpy()
            arr = np.asarray(prob, dtype=np.float64).reshape(-1)
            prob = float(arr.max()) if arr.size else None
        except Exception:
            prob = None
    return (str(label) if label is not None else None), _sane_prob(prob, idx)


def convert_sifat(items: Any) -> Optional[List[Dict[str, Any]]]:
    """-> [{'g': group_text, 'L': {level: (label, prob)}}] or None when unavailable."""
    if not items:
        return None
    out: List[Dict[str, Any]] = []
    try:
        for it in items:
            g = ""
            for name in ("phonemes_group", "phonemes", "group"):
                val = it.get(name) if isinstance(it, dict) else getattr(it, name, None)
                if isinstance(val, str) and val:
                    g = val
                    break
            levels: Dict[str, Tuple[Optional[str], Optional[float]]] = {}
            for lv in SIFAT_LEVELS:
                raw = it.get(lv) if isinstance(it, dict) else getattr(it, lv, None)
                levels[lv] = _unwrap_unit(raw)
            out.append({"g": clean_phonemes(g), "L": levels})
    except Exception:
        return None
    return out or None


def group_index_map(groups: Optional[List[Dict[str, Any]]], text: str) -> Optional[List[int]]:
    """char index -> sifat group index; only if the groups tile the phoneme string exactly."""
    if not groups:
        return None
    if "".join(g["g"] for g in groups) != text:
        return None
    m: List[int] = []
    for gi, g in enumerate(groups):
        m.extend([gi] * len(g["g"]))
    return m


def tolerant_group_map(groups: Optional[List[Dict[str, Any]]], text: str) -> Optional[List[int]]:
    """Like group_index_map, but survives small differences (e.g. a doubled vowel mark at a window
    edge) by aligning the concatenated group text to the phoneme string."""
    if not groups or not text:
        return None
    joined = "".join(g["g"] for g in groups)
    if not joined:
        return None
    al = align(joined, text, tail_free_ref=False)
    if al.cost > 0.15 * max(len(joined), len(text)):
        return None
    owner: List[int] = []
    for gi, g in enumerate(groups):
        owner.extend([gi] * len(g["g"]))
    out = [-1] * len(text)
    for ri, pj in al.pairs:
        if ri >= 0 and pj >= 0:
            out[pj] = owner[ri]
    last = 0
    for j in range(len(out)):                  # unmatched chars inherit the previous group
        if out[j] < 0:
            out[j] = last
        else:
            last = out[j]
    return out


# ==============================================================================
# Reference (one ayah): words, wasl/waqf phoneme forms, expected sifat
# ==============================================================================
_BASE_RE = re.compile("[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]")


def split_words(uthmani: str) -> Tuple[List[str], List[str], List[int]]:
    """Splits on spaces and drops stop-sign-only tokens (they are not words)."""
    raw = uthmani.split()
    words: List[str] = []
    idx: List[int] = []
    for i, t in enumerate(raw):
        if _BASE_RE.search(t):
            words.append(t)
            idx.append(i)
    return raw, words, idx


def split_groups(spaced_phonemes: str) -> List[str]:
    return [clean_phonemes(g) for g in str(spaced_phonemes).split(" ") if clean_phonemes(g)]


def proportional_groups(words: List[str], phonemes: str) -> List[str]:
    weights = [max(1, len(_BASE_RE.findall(w))) for w in words]
    total = float(sum(weights))
    n = len(phonemes)
    out: List[str] = []
    acc, prev = 0.0, 0
    for k, w in enumerate(weights):
        acc += w
        cut = n if k == len(weights) - 1 else int(round(acc / total * n))
        cut = max(prev, min(n, cut))
        out.append(phonemes[prev:cut])
        prev = cut
    return out


class RefAya:
    def __init__(self, sura: int, aya: int, words: List[str], wasl: List[str],
                 waqf: List[Optional[str]], approx: bool, full_out: Any,
                 exp_groups: Optional[List[Dict[str, Any]]], uthmani: str = "",
                 info: Optional[Dict[str, Any]] = None,
                 word_exact: Optional[List[bool]] = None):
        self.sura, self.aya = sura, aya
        self.info: Dict[str, Any] = info or {}
        self.word_approx: List[bool] = [not e for e in (word_exact or [not approx] * len(words))]
        approx = any(self.word_approx)
        self.words = words
        self.wasl = wasl
        self.waqf = waqf
        self.approx = approx
        self.full_out = full_out
        self.uthmani = uthmani
        self.n = len(words)
        self.full_ph = "".join(wasl)
        self.starts: List[int] = [0]
        self.word_of_char: List[int] = []
        for k, f in enumerate(wasl):
            self.starts.append(self.starts[-1] + len(f))
            self.word_of_char.extend([k] * len(f))
        self.stop_target: List[bool] = [k == len(words) - 1 for k in range(len(words))]   # the last word of the ayah is always stopped on
        self.start_forms: List[Optional[str]] = [None] * len(words)     # word said fresh after a pause (ibtida')
        self.start_waqf: List[Optional[str]] = [None] * len(words)
        self.exp_groups = exp_groups
        self.group_of_char = group_index_map(exp_groups, self.full_ph)
        self.core: List[List[str]] = [collapse_runs(letters_only(f)) for f in wasl]

    def forms(self, k: int) -> List[str]:
        f = [self.wasl[k]]
        if self.waqf[k] and self.waqf[k] != self.wasl[k]:
            f.append(self.waqf[k])
        return f

    def forms_ext(self, k: int) -> Tuple[List[str], Dict[str, str]]:
        """All acceptable pronunciations of word k: continuing (wasl) or stopping (waqf) at its end, each
        with or without the start-of-speech (ibtida') beginning; plus waqf->wasl partners for re-selection."""
        forms = [self.wasl[k]]
        partner: Dict[str, str] = {}
        w, s, sw = self.waqf[k], self.start_forms[k], self.start_waqf[k]
        if w and w not in forms:
            forms.append(w)
            partner[w] = self.wasl[k]
        if s and s not in forms:
            forms.append(s)
        if sw and sw not in forms:
            forms.append(sw)
            partner[sw] = s or self.wasl[k]
        return forms, partner

    def describe(self) -> Dict[str, Any]:
        return {
            "sura": self.sura, "aya": self.aya, "n_words": self.n, "approx_segmentation": self.approx,
            "sifat_available": self.group_of_char is not None, "info": self.info,
            "words": [{"i": k, "text": self.words[k], "wasl": self.wasl[k], "waqf": self.waqf[k],
                       "start": self.start_forms[k]}
                      for k in range(self.n)],
        }


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def refine_bounds(full_ph: str, bounds: List[int]) -> List[int]:
    """A word cannot start with a vowel mark or a silah/madd letter: those stay with the previous word
    (the live log showed the previous word's final vowel leaking into the next word's expected form)."""
    out = list(bounds)
    n = len(full_ph)
    for k in range(1, len(out) - 1):
        b = out[k]
        while b < n - 1 and b < out[k + 1] - 1 and (is_mark(full_ph[b]) or full_ph[b] in "ۥۦ"):
            b += 1
        out[k] = b
    return out


def _segment_by_isolated_words(lib: Any, words: List[str], full_ph: str,
                               info: Dict[str, Any]) -> Optional[List[int]]:
    """
    Word boundaries that do not rely on the library's spaced output: phonetize every word on its own,
    align the concatenation against the full phoneme string, read the boundaries off the alignment.
    """
    iso: List[str] = []
    for w in words:
        try:
            iso.append(clean_phonemes(lib.phonetize(w, True).phonemes))
        except Exception as exc:
            info["iso_error"] = repr(exc)[:160]
            return None
    if any(not s for s in iso):
        info["iso_error"] = "empty isolated phonemes"
        return None
    S = "".join(iso)
    cum = [0]
    for s in iso:
        cum.append(cum[-1] + len(s))
    al = align(S, full_ph, tail_free_ref=False)
    s2f = {ri: pj for ri, pj in al.pairs if ri >= 0 and pj >= 0}
    info["iso_align_cost_per_char"] = round(al.cost / max(1, len(full_ph)), 4)
    bounds = [0]
    for k in range(1, len(words)):
        j = None
        for r in range(cum[k], len(S)):
            if r in s2f:
                j = s2f[r]
                break
        if j is None:
            return None
        bounds.append(j)
    bounds.append(len(full_ph))
    if any(bounds[i + 1] <= bounds[i] for i in range(len(bounds) - 1)):
        info["iso_error"] = "non-monotonic boundaries"
        return None
    return refine_bounds(full_ph, bounds)


def _snap_to_library_groups(bounds: List[int], group_starts: List[int],
                            tol: int = 3) -> Tuple[List[int], List[bool]]:
    """Replace estimated boundaries by the library's own (exact) group boundaries where one is close."""
    out = list(bounds)
    exact = [False] * len(bounds)
    exact[0] = exact[-1] = True
    used = set()
    for k in range(1, len(bounds) - 1):
        cands = [g for g in group_starts if abs(g - bounds[k]) <= tol and g not in used]
        if cands:
            g = min(cands, key=lambda x: abs(x - bounds[k]))
            if out[k - 1] < g:
                out[k], exact[k] = g, True
                used.add(g)
    for k in range(1, len(out)):                 # keep strictly increasing
        if out[k] <= out[k - 1]:
            out[k], exact[k] = out[k - 1] + 1, False
    return out, exact


def build_ref_aya(sura: int, aya: int, lib: Any, with_waqf: bool = True) -> RefAya:
    """
    `lib` must provide:  get_uthmani(sura, aya) -> str
                         phonetize(text, remove_spaces) -> object with .phonemes (str) [and .sifat]

    Word segmentation ladder (first that validates wins; the winner is recorded in `info`):
      spaced-exact    the library's own spaced output matches the no-space output exactly
      spaced-adopted  same number of words, <= 2 % different: use the spaced words as the reference
      aligned         isolated-word phonetization aligned to the full string (about +-1 char at edges)
      proportional    last resort (flagged approximate)
    """
    uth = lib.get_uthmani(sura, aya)
    raw, words, tok_idx = split_words(uth)
    n = len(words)
    full = lib.phonetize(uth, True)
    F = clean_phonemes(full.phonemes)
    info: Dict[str, Any] = {"n_words": n, "full_len": len(F), "full_head": F[:90]}

    groups: Optional[List[str]] = None
    strategy = ""
    word_exact: Optional[List[bool]] = None
    lib_groups: Optional[List[str]] = None
    try:
        spaced_raw = str(lib.phonetize(uth, False).phonemes)
        g = split_groups(spaced_raw)
        joined = "".join(g)
        info.update({"spaced_head": spaced_raw[:90], "n_groups": len(g), "join_equal": joined == F})
        if joined != F:
            info["first_diff_at"] = _lcp(joined, F)
        else:
            lib_groups = g
        if len(g) == n:
            if joined == F:
                groups, strategy = g, "spaced-exact"
            else:
                diff = align(joined, F, tail_free_ref=False).cost / max(1, len(F))
                info["spaced_diff"] = round(diff, 4)
                if diff <= 0.02:
                    groups, strategy = g, "spaced-adopted"
    except Exception as exc:
        info["spaced_error"] = repr(exc)[:200]
    if groups is None:
        bounds = _segment_by_isolated_words(lib, words, F, info)
        if bounds is not None:
            strategy = "aligned"
            exact = [False] * len(bounds)
            exact[0] = exact[-1] = True
            if lib_groups:                      # merged words in the library output: snap to its exact edges
                gs, acc = [], 0
                for gr in lib_groups[:-1]:
                    acc += len(gr)
                    gs.append(acc)
                bounds, exact = _snap_to_library_groups(bounds, gs)
                info["snapped_boundaries"] = sum(1 for e in exact[1:-1] if e)
                if info["snapped_boundaries"]:
                    strategy = "spaced+aligned"
            groups = [F[bounds[i]:bounds[i + 1]] for i in range(n)]
            word_exact = [exact[i] and exact[i + 1] for i in range(n)]
    if groups is None:
        groups, strategy = proportional_groups(words, F), "proportional"
    info["segmentation"] = strategy
    if word_exact is None:
        word_exact = [strategy in ("spaced-exact", "spaced-adopted")] * n
    approx = not all(word_exact)
    info["inexact_words"] = sum(1 for e in word_exact if not e)

    # ---- waqf (stop) form of every word: tail of the phonemes of "text up to this word"
    waqf: List[Optional[str]] = [None] * n
    if with_waqf and strategy != "proportional":
        starts = [0]
        for gr in groups:
            starts.append(starts[-1] + len(gr))
        ok_cnt = 0
        for k in range(n - 1):
            try:
                P = clean_phonemes(lib.phonetize(" ".join(raw[: tok_idx[k] + 1]), True).phonemes)
            except Exception:
                continue
            start = starts[k]
            if _lcp(P, F) < start - 1:
                continue
            tail = P[start:]
            if 1 <= len(tail) <= len(groups[k]) + 6 and tail != groups[k]:
                waqf[k] = tail
                ok_cnt += 1
        info["waqf_forms"] = ok_cnt

    # ---- start (ibtida') form: how the word sounds when the reciter begins fresh at it (after a pause):
    #      hamzat al-wasl pronounced, no assimilation of the previous word's tanween / noon into it.
    start_forms: List[Optional[str]] = [None] * n
    start_waqf: List[Optional[str]] = [None] * n
    if with_waqf and strategy != "proportional":
        starts_ = [0]
        for gr in groups:
            starts_.append(starts_[-1] + len(gr))
        cnt = 0
        for k in range(1, n):
            try:
                P = clean_phonemes(lib.phonetize(" ".join(raw[tok_idx[k]:]), True).phonemes)
            except Exception:
                continue
            rest = F[starts_[k + 1]:]
            lcs = 0
            while lcs < min(len(P), len(rest)) and P[len(P) - 1 - lcs] == rest[len(rest) - 1 - lcs]:
                lcs += 1
            if lcs < len(rest) - 1:
                continue
            sf = P[: len(P) - len(rest)] if rest else P
            if sf and sf != groups[k] and abs(len(sf) - len(groups[k])) <= 6:
                start_forms[k] = sf
                cnt += 1
                wq = waqf[k]
                if wq:
                    c0 = _lcp(groups[k], wq)
                    tail_wasl, tail_waqf = groups[k][c0:], wq[c0:]
                    if (not tail_wasl or sf.endswith(tail_wasl)) and tail_waqf != tail_wasl:
                        start_waqf[k] = (sf[: len(sf) - len(tail_wasl)] if tail_wasl else sf) + tail_waqf
        info["start_forms"] = cnt

    raw_sifat = getattr(full, "sifat", None)
    exp_groups = convert_sifat(raw_sifat)
    try:
        info["exp_sifat_n"] = len(raw_sifat) if raw_sifat else 0
        if raw_sifat:
            info["exp_sifat_sample"] = repr(raw_sifat[0])[:300]
    except Exception:
        pass
    ref = RefAya(sura, aya, words, groups, waqf, approx, full, exp_groups, uth, info, word_exact)
    ref.start_forms, ref.start_waqf = start_forms, start_waqf
    info["sifat_tiles_reference"] = ref.group_of_char is not None
    return ref


# ==============================================================================
# Word judge
# ==============================================================================
@dataclass
class Issue:
    t: str                 # "letters" | "tashkeel" | "tajweed"
    k: str                 # kind (machine readable)
    n: str                 # Arabic note
    conf: float
    pos: int = 0           # position inside the expected form (debug / boundary filter)
    exp: str = ""
    pred: str = ""
    focus: str = ""

    def __post_init__(self) -> None:
        try:                                     # a confidence is a probability: never outside [0, 1]
            c = float(self.conf)
            self.conf = 0.0 if not math.isfinite(c) else min(1.0, max(0.0, c))
        except Exception:
            self.conf = 0.0

    def to_client(self) -> Dict[str, Any]:
        d = {"t": self.t, "k": self.k, "n": self.n}
        if self.focus:
            d["f"] = self.focus
        return d

    def to_debug(self) -> Dict[str, Any]:
        return {"t": self.t, "k": self.k, "conf": round(self.conf, 3), "pos": self.pos,
                "exp": self.exp, "pred": self.pred, "note": self.n}


SEVERITY = {"letters": 3, "tashkeel": 2, "tajweed": 1}


def status_of(issues: List[Issue]) -> str:
    if not issues:
        return "ok"
    return max(issues, key=lambda i: SEVERITY[i.t]).t


def _conf(probs: Optional[List[float]], idxs: Sequence[int], default: float = 0.92) -> float:
    if probs is None:
        return default
    vals = [probs[i] for i in idxs if 0 <= i < len(probs)]
    return float(sum(vals) / len(vals)) if vals else default


def _madd_is_run(exp: str, s: int, e: int) -> bool:
    c = exp[s]
    if c in ALWAYS_MADD:
        return True
    if c in "ويى":
        nxt = exp[e] if e < len(exp) else ""
        return nxt not in CORE_VOWELS      # a waw/ya followed by a vowel is a doubled consonant
    return False


def _need_text(lo: int, e: int) -> str:
    if lo >= 4:
        return "أربع حركات على الأقل"
    return "حركتان" if e <= 2 else f"نحو {e} حركات"


def madd_kind(exp: str, s: int, e: int) -> str:
    """Which madd the run exp[s:e] is: lazim | muttasil | munfasil (incl. silah kubra) | arid | inner | tabii."""
    E = e - s
    after = [ch for ch in exp[e:] if not is_mark(ch)]
    if E >= 6:
        return "lazim"
    if after and after[0] == "ء" and E >= 3:
        return "muttasil"                       # hamza in the SAME word right after the madd letter (شَاءَ)
    if not after:
        return "munfasil" if E >= 3 else "tabii"
    if len(after) == 1:
        return "arid"
    return "inner" if E >= 3 else "tabii"


_MADD_RANGE = {"lazim": (5, 7), "muttasil": (MUTTASIL_LO, MUTTASIL_HI), "munfasil": (2, 5), "tabii": (2, 2),
               "arid": (2, 6), "inner": (3, 6)}


def madd_bounds(exp: str, s: int, e: int) -> Tuple[int, int]:
    """
    Accepted number of harakat (lo, hi) for the madd run exp[s:e], following Hafs 'an 'Asim:
      madd munfasil & silah kubra : 2 (qasr, Tayyibat an-Nashr) .. 5 (tawassut/fuwayq, Shatibiyyah)
      madd muttasil               : 4 .. 6      madd lazim : 6        madd 'arid / lin at a stop : 2, 4 or 6
      madd tabii / silah sughra / badal / 'iwad : exactly 2
    (`madd_policy` adds the preset rules on top.)
    """
    return _MADD_RANGE[madd_kind(exp, s, e)]


def madd_policy(kind: str, lo: int, hi: int, preset: Preset) -> Tuple[int, int, int, int]:
    """-> (lo, hi, short_slack, long_slack) actually enforced for this preset.
      * madd muttasil is 4..6 harakat in EVERY preset: the preset slacks never apply to it.
      * madd munfasil / silah kubra accept 2 (qasr) .. 5 harakat in EVERY preset, strict included."""
    if kind == "muttasil":
        return MUTTASIL_LO, MUTTASIL_HI, 0, 0
    return lo, hi, preset.short_slack, preset.long_slack


def _replace_conf(probs: Optional[List[float]], alts: Optional[Sequence[Optional[Dict[str, float]]]],
                  pj: int, exp_c: str, pred_c: str) -> float:
    """Confidence of a letter substitution. When the full distribution of the token was captured (`alts`: the
    runner-up letters with their posteriors) a substitution inside a rival pair is scored by the RELATIVE odds
    p(pred) / (p(pred) + p(expected)), so a probability mass split between rivals cannot hide the mistake."""
    base = _conf(probs, [pj])
    if alts is not None and 0 <= pj < len(alts) and alts[pj]:
        pe = alts[pj].get(exp_c)
        if pe is not None:
            pp = probs[pj] if (probs is not None and 0 <= pj < len(probs)) else alts[pj].get(pred_c, base)
            tot = pp + pe
            if tot > 0:
                return max(base, pp / tot)
    return base


def judge_word(
    forms: List[str],
    seg: str,
    probs: Optional[List[float]],
    preset: Preset,
    lead: int = 0,
    trail: int = 0,
    approx: bool = False,
    ctx_chars: str = "",
    partner: Optional[Dict[str, str]] = None,
    stop_target: bool = False,
    acoustic: Optional[Callable[[int, int], Optional[float]]] = None,
    alts: Optional[Sequence[Optional[Dict[str, float]]]] = None,
) -> Tuple[List[Issue], Optional[Alignment], str, Dict[str, float]]:
    """
    Compares ONE spoken word (`seg` = its predicted phonemes, with up to `lead`/`trail` chars of
    unassigned neighbours) against its expected form(s). Returns (issues, alignment, chosen_form).
    Only errors that survive the tolerance rules and the confidence gate are returned.
      stop_target : the word is stopped on (last word of the ayah): a vowel pronounced on its final letter is an error
      acoustic    : optional callback (pred_run_start, pred_run_end_exclusive) -> measured harakat (float) of a madd
                    run, from the audio; used for madd muttasil so the decoded repeat count cannot hide a short madd
      alts        : optional per-character runner-up letters {char: posterior} (rival confusion pairs)
    """
    best: Optional[Tuple[str, Alignment]] = None
    form_costs: Dict[str, float] = {}
    for f in dict.fromkeys(x for x in forms if x):
        al = align(f, seg, tail_free_ref=False, pred_lead_free=lead, tail_ins_chars=trail)
        form_costs[f] = al.cost
        if best is None or al.cost < best[1].cost - 1e-9:
            best = (f, al)
    if best is None:
        return [], None, "", form_costs
    exp, al = best
    uniq = [f for f in dict.fromkeys(x for x in forms if x)]
    if partner is None:
        partner = {uniq[1]: uniq[0]} if len(uniq) == 2 else {}
    if exp in partner:
        # a stop form won on cost, but if the reciter kept a final vowel he was NOT stopping:
        # judge against the continuation form so the vowel mistake is reported properly
        core_end = len(seg) - trail
        letter_idx = [i for i in range(lead, core_end) if not is_mark(seg[i]) and seg[i] not in ALWAYS_MADD]
        if letter_idx and any(seg[i] in CORE_VOWELS for i in range(letter_idx[-1] + 1, core_end)):
            exp = partner[exp]
            al = align(exp, seg, tail_free_ref=False, pred_lead_free=lead, tail_ins_chars=trail)
    M, L = len(exp), len(seg)
    r2p = {ri: pj for ri, pj in al.pairs if ri >= 0 and pj >= 0}
    aligned_p = set(r2p.values())
    aligned_p_to_r = {pj: ri for ri, pj in r2p.items()}

    def left_p(ri: int) -> Optional[int]:
        for k in range(ri - 1, -1, -1):
            if k in r2p:
                return r2p[k]
        return None

    def right_p(ri: int) -> Optional[int]:
        for k in range(ri + 1, M):
            if k in r2p:
                return r2p[k]
        return None

    def del_conf(ri: int) -> float:
        return _conf(probs, [x for x in (left_p(ri), right_p(ri)) if x is not None])

    issues: List[Issue] = []

    # ---------------------------------------------------------------- vowel marks
    for ri, c in enumerate(exp):
        if not is_mark(c) or c not in CORE_VOWELS:
            continue
        owner = prev_letter(exp, ri)
        pj = r2p.get(ri)
        if pj is not None:
            pc = seg[pj]
            if pc != c and pc in CORE_VOWELS:
                if owner and prev_letter(seg, pj) != owner:
                    continue              # the heard vowel sits on a different letter: alignment drift, not a swap
                issues.append(Issue("tashkeel", "vowel_swap", note_vowel_swap(owner, c, pc),
                                    _conf(probs, [pj]), ri, c, pc, owner))
            continue
        if all(is_mark(x) for x in exp[ri + 1:]):
            continue                      # final vowel: dropped at a stop (waqf) is legitimate
        if preset.flag_missing_vowel:
            issues.append(Issue("tashkeel", "vowel_missing", note_vowel_missing(owner, c),
                                del_conf(ri), ri, c, "", owner))

    # ---------------------------------------------------------------- letters and runs
    for c, s, e in split_runs(exp):
        if is_mark(c):
            continue
        E = e - s
        same = [r2p[ri] for ri in range(s, e) if ri in r2p and seg[r2p[ri]] == c]
        if same:
            a = b = same[0]
            while a - 1 >= 0 and seg[a - 1] == c:
                a -= 1
            while b + 1 < L and seg[b + 1] == c:
                b += 1
            p, conf = b - a + 1, _conf(probs, range(a, b + 1))
        else:
            p, conf = 0, 0.0

        if p == 0:
            subs = [r2p[ri] for ri in range(s, e) if ri in r2p and not is_mark(seg[r2p[ri]])]
            if subs:
                pj = subs[0]
                pc = seg[pj]
                conf = _conf(probs, [pj])
                if c in ALWAYS_MADD and pc in ALWAYS_MADD and c != pc:
                    # a different madd letter (fatha-madd / damma-madd / kasra-madd) = the vowel changed
                    issues.append(Issue("tashkeel", "madd_vowel_swap", note_madd_vowel_swap(c, pc),
                                        conf, s, c, pc, ""))
                elif (c in BASIC_LETTERS or c in ALWAYS_MADD) and (pc in BASIC_LETTERS or pc in ALWAYS_MADD):
                    issues.append(Issue("letters", "replace", note_replace(c, pc),
                                        _replace_conf(probs, alts, pj, c, pc), s, c, pc,
                                        c if c in BASIC_LETTERS else ""))
                else:
                    issues.append(Issue("tajweed", "rule", NOTE_RULE, conf, s, c, pc, ""))
                continue
            conf = del_conf(s)
            if _madd_is_run(exp, s, e) and E >= 2:
                lo, _hi = madd_bounds(exp, s, e)
                if 0 < lo - preset.short_slack:
                    issues.append(Issue("tajweed", "madd_dropped", note_madd_dropped(c, _need_text(lo, E)),
                                        conf, s, c * E, "", c))
            elif c in BASIC_LETTERS:
                issues.append(Issue("letters", "missing", note_missing_letter(c), conf, s, c, "", c))
            elif c not in MADD_LETTERS:
                issues.append(Issue("tajweed", "rule", NOTE_RULE, conf, s, c, "", ""))
            continue

        if p == E and not (acoustic is not None and E >= 3 and _madd_is_run(exp, s, e)
                           and madd_kind(exp, s, e) == "muttasil"):
            continue                      # (a muttasil run is still checked on the audio: the decoded count can lie)
        sub_in_run = [r2p[ri] for ri in range(s, e)
                      if ri in r2p and seg[r2p[ri]] != c and not is_mark(seg[r2p[ri]])]
        if sub_in_run and c in BASIC_LETTERS and seg[sub_in_run[0]] in BASIC_LETTERS:
            pc = seg[sub_in_run[0]]          # a different letter was spoken inside the run: it wins
            issues.append(Issue("letters", "replace", note_replace(c, pc),
                                _replace_conf(probs, alts, sub_in_run[0], c, pc), s, c, pc, c))
            continue
        if _madd_is_run(exp, s, e) and E == 1:
            if p > 1 + preset.long_slack:          # a liin letter (e.g. the ya of بَيْنَ) stretched like a madd
                issues.append(Issue("tajweed", "lin_extended", note_lin_extended(c), conf, s, c, c * p, c))
        elif _madd_is_run(exp, s, e):
            kind = madd_kind(exp, s, e)
            lo, hi, sslack, lslack = madd_policy(kind, *madd_bounds(exp, s, e), preset)
            p_eff = p
            if acoustic is not None and kind == "muttasil":
                h = acoustic(a, b + 1)                 # harakat measured on the AUDIO of this run
                if h is not None and math.isfinite(h) and abs(h - p) <= ACOUSTIC_MAX_DISAGREE:
                    # decoded count says "fine" but the audio is clearly outside 4..6 -> the audio wins;
                    # it never removes a flag the decoded count already raised (grossly different = alignment failure)
                    if h < lo - ACOUSTIC_MARGIN_H:
                        p_eff = min(p, max(0, int(math.floor(h + 0.5))), lo - 1)
                    elif h > hi + ACOUSTIC_MARGIN_H:
                        p_eff = max(p, int(math.floor(h + 0.5)), hi + 1)
            if p_eff < lo - sslack:
                issues.append(Issue("tajweed", "madd_short", note_madd_short(c, _need_text(lo, E)),
                                    conf, s, c * E, c * max(p_eff, 0), c))
            elif p_eff > hi + lslack:
                issues.append(Issue("tajweed", "madd_long", note_madd_long(c, min(E, hi)),
                                    conf, s, c * E, c * p_eff, c))
        elif c in GHUNNA_SYMBOLS:
            min_ok = E if preset.name == "strict" else ((2 if E == 2 else E - 1) if preset.name == "balanced" else 1)
            if E >= 2 and p < min_ok:
                issues.append(Issue("tajweed", "ghunna_short", note_ghunna_short(E), conf, s, c * E, c * p, ""))
            elif p > E and preset.flag_extra_shaddah:
                issues.append(Issue("tajweed", "ghunna_long", note_ghunna_long(), conf, s, c * E, c * p, ""))
        elif c in BASIC_LETTERS:
            if preset.name == "strict":
                min_ok = E
            elif preset.name == "balanced":
                min_ok = 2 if E == 2 else E - 1
            else:
                min_ok = 1
            if E >= 2 and p < min_ok:
                issues.append(Issue("tajweed", "shaddah_dropped", note_shaddah_dropped(c),
                                    conf, s, c * E, c * p, c))
            elif E >= 1 and p > E and preset.flag_extra_shaddah:
                issues.append(Issue("tajweed", "shaddah_extra", note_shaddah_extra(c),
                                    conf, s, c, c * p, c))

    # ---------------------------------------------------------------- extra predicted chars
    first_core, last_core = lead, L - trail
    for pj in range(first_core + 1, last_core - 1):
        if pj in aligned_p:
            continue
        # 'interior' means real LETTERS are aligned on both sides (a vowel mark alone does not count)
        before_ok = any(q in aligned_p and not is_mark(seg[q]) for q in range(first_core, pj))
        after_ok = any(q in aligned_p and not is_mark(seg[q]) for q in range(pj + 1, last_core))
        if not (before_ok and after_ok):
            continue
        pc = seg[pj]
        if pc in CORE_VOWELS and preset.flag_extra_vowel and (pj - 1) in aligned_p:
            ri_prev = aligned_p_to_r[pj - 1]
            nxt = exp[ri_prev + 1] if ri_prev + 1 < M else ""
            if not is_mark(seg[pj - 1]) and nxt not in CORE_VOWELS:
                owner = seg[pj - 1]
                issues.append(Issue("tashkeel", "vowel_extra", note_vowel_extra(owner, pc),
                                    _conf(probs, [pj]), ri_prev, "", pc, owner))
        elif (preset.flag_extra_letter and pc in BASIC_LETTERS and pc != seg[pj - 1]
              and pj + 1 < L and pc != seg[pj + 1]):
            issues.append(Issue("letters", "extra", note_extra_letter(pc), _conf(probs, [pj]), 0, "", pc, pc))

    # ---------------------------------------------------------------- vowel kept on a word that is stopped on
    if stop_target:
        last_ri = max((i for i, ch in enumerate(exp) if not is_mark(ch)), default=-1)
        if last_ri >= 0 and not any(ch in CORE_VOWELS for ch in exp[last_ri + 1:]):
            pj_last = r2p.get(last_ri)
            if (pj_last is not None and pj_last + 1 < L and (pj_last + 1) not in aligned_p
                    and seg[pj_last + 1] in CORE_VOWELS):
                issues.append(Issue("tashkeel", "waqf_vowel",
                                    note_waqf_vowel(exp[last_ri], seg[pj_last + 1]),
                                    _conf(probs, [pj_last + 1]), last_ri, "", seg[pj_last + 1], exp[last_ri]))

    # ---------------------------------------------------------------- a word whose consonants are mostly missing is never "ok"
    cons_runs = [(c, s0, e0) for c, s0, e0 in split_runs(exp)
                 if c in BASIC_LETTERS and c not in ALWAYS_MADD and c != "ا"]
    mostly_missing = False
    if len(cons_runs) >= 2:
        miss = sum(1 for c, s0, e0 in cons_runs
                   if not any(ri in r2p and seg[r2p[ri]] == c for ri in range(s0, e0)))
        mostly_missing = miss / len(cons_runs) > 0.5

    # ---------------------------------------------------------------- confidence gate
    kept: List[Issue] = []
    edge_kinds = ("missing", "replace", "extra", "rule", "vowel_missing")
    for it in issues:
        need = preset.conf_vowel if it.t == "tashkeel" else preset.conf_letter
        if it.k == "replace" and (it.exp[:1], it.pred[:1]) in RIVAL_PAIRS:
            need *= RIVAL_GATE_FACTOR              # rival pair: the posterior mass is split, do not lose the mistake
        if it.conf < need:
            continue
        if approx and it.k in edge_kinds and (it.pos < 1 or it.pos >= M - 1) and ctx_chars:
            # boundaries are only +-1 char accurate: an edge issue is ambiguous only when the letter
            # involved also sits at the neighbouring word's adjacent edge (it may belong to that word)
            if it.exp[:1] in ctx_chars or it.pred[:1] in ctx_chars:
                continue
        kept.append(it)
    if any(i.k == "vowel_swap" for i in kept):          # same mistake seen twice (vowel + its madd letter)
        kept = [i for i in kept if i.k != "madd_vowel_swap"]
    kept.sort(key=lambda i: (i.pos, -SEVERITY[i.t]))

    letter_issues = [i for i in kept if i.t == "letters"]
    if len(letter_issues) >= UNCLEAR_ISSUES:
        kept = [Issue("letters", "unclear", NOTE_UNCLEAR,
                      min(i.conf for i in letter_issues), 0, exp, seg, "")]
    elif mostly_missing and not any(i.t == "letters" for i in kept):
        kept = [Issue("letters", "unclear", NOTE_UNCLEAR, 1.0, 0, exp, seg, "")] + kept
    return kept[:3], al, exp, form_costs


# ==============================================================================
# Audio buffer + voice activity
# ==============================================================================
class AudioBuffer:
    """Thread-safe growing mono buffer (float32, 16 kHz) with 20 ms frame energies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf = np.zeros(SR * 30, dtype=np.float32)
        self.n = 0
        self._frames: List[float] = []

    def append(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        with self._lock:
            room = MAX_SESSION_S * SR - self.n
            if room <= 0:
                return
            x = x[:room]
            need = self.n + x.size
            if need > self._buf.size:
                nb = np.zeros(max(need, self._buf.size * 2), dtype=np.float32)
                nb[: self.n] = self._buf[: self.n]
                self._buf = nb
            self._buf[self.n:need] = x
            self.n = need
            nf = self.n // FRAME
            have = len(self._frames)
            if nf > have:
                seg = self._buf[have * FRAME: nf * FRAME].reshape(-1, FRAME)
                self._frames.extend(np.sqrt((seg * seg).mean(axis=1) + 1e-12).tolist())

    def snapshot(self, a: int, b: int) -> np.ndarray:
        with self._lock:
            return self._buf[max(0, a): min(self.n, b)].copy()

    def frames_array(self) -> np.ndarray:
        with self._lock:
            return np.asarray(self._frames, dtype=np.float64)


def voiced_mask(frames: np.ndarray, thr: float, fill: int = 4, min_run: int = 2) -> np.ndarray:
    m = frames >= thr
    n = len(m)
    out = m.copy()
    i = 0
    while i < n:                           # bridge short gaps (stop consonants, breaths)
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= fill:
                out[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < n:                           # drop tiny islands (clicks)
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_run:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


# ==============================================================================
# CTC frame track (timing of every decoded character) + zero-copy Int16 gate + acoustic madd meter
# ==============================================================================
@dataclass
class FrameTrack:
    """Where the model emitted each decoded character. Frames are `stride_s` long (about 40 ms: 99 feature frames of
    20 ms -> 50 CTC frames) and counted from the start of the wave handed to the model."""
    stride_s: float
    n_frames: int
    first: List[int]                                  # first CTC frame of every decoded character
    last: List[int]                                   # last CTC frame of every decoded character
    probs: Optional[List[float]] = None
    alts: Optional[List[Optional[Dict[str, float]]]] = None    # rival letters: runner-up {char: posterior}

    @property
    def n_chars(self) -> int:
        return len(self.first)

    def start_s(self, i: int) -> float:
        return self.first[i] * self.stride_s

    def end_s(self, i: int) -> float:
        return (self.last[i] + 1) * self.stride_s

    @staticmethod
    def from_logits(logits: Any, text: str, wave_s: float, id2char: Optional[Dict[int, str]] = None,
                    blank_ids: Sequence[int] = (0,)) -> Optional["FrameTrack"]:
        """Greedy CTC collapse of the phoneme logits [T, C]. Returns None unless the collapse reproduces `text`
        character for character (so a wrong timing can never be used)."""
        lg = np.asarray(logits, dtype=np.float64)
        if lg.ndim != 2 or lg.shape[0] < 2 or not text:
            return None
        T, C = lg.shape
        ex = np.exp(lg - lg.max(axis=1, keepdims=True))
        pr = ex / ex.sum(axis=1, keepdims=True)
        ids = pr.argmax(axis=1)
        for blank in tuple(blank_ids) + (C - 1,):
            segs: List[List[int]] = []
            prev = -1
            for f in range(T):
                i = int(ids[f])
                if i == blank:
                    prev = -1
                elif i != prev:
                    segs.append([f, f, i])
                    prev = i
                else:
                    segs[-1][1] = f
            if len(segs) != len(text):
                continue
            mapping: Dict[int, str] = {}
            consistent = True
            for sg, ch in zip(segs, text):
                if mapping.setdefault(sg[2], ch) != ch:
                    consistent = False
                    break
            if not consistent:
                continue
            if id2char is not None:
                id2char.update(mapping)
            sp = [float(pr[sg[0]:sg[1] + 1, sg[2]].max()) for sg in segs]
            alts: List[Optional[Dict[str, float]]] = []
            for sg, ch in zip(segs, text):
                if ch not in _RIVAL_LETTERS or id2char is None:
                    alts.append(None)
                    continue
                f_best = sg[0] + int(np.argmax(pr[sg[0]:sg[1] + 1, sg[2]]))
                top = np.argsort(-pr[f_best])[:5]
                alts.append({id2char[int(j)]: float(pr[f_best, j]) for j in top
                             if int(j) != sg[2] and int(j) in id2char and int(j) != blank} or None)
            return FrameTrack(wave_s / T, T, [sg[0] for sg in segs], [sg[1] for sg in segs], sp, alts)
        return None


_RIVAL_LETTERS = frozenset(ch for pair in RIVAL_PAIRS for ch in pair)


class PcmGate:
    """Zero-copy Int16 acoustic gate. Works on 10 ms frames straight from the PCM16 bytes (`memoryview.cast('h')`,
    no float conversion): RMS energy, zero-crossing rate and the mean first sample difference (a cheap spectral-tilt /
    formant proxy). `steady_ms()` says how long the vowel has been acoustically steady; a formant shift (imalah,
    tafkhim drift) or a glottal closure changes the ZCR / first difference / energy and ends the steady state at once."""
    FR = 160                                           # 10 ms
    VOICED_FLOOR = 0.012                               # RMS (full scale = 1.0) above which a frame is voiced
    TOL_RMS, TOL_ZCR, TOL_DLT = 0.25, 0.06, 0.25
    ZCR_MAX_VOWEL = 0.25                               # above this the frames are noise / frication, not a held vowel

    def __init__(self) -> None:
        self.rms: List[float] = []
        self.zcr: List[float] = []
        self.dlt: List[float] = []
        self._carry = np.zeros(0, dtype=np.int16)

    def feed_i16(self, arr: np.ndarray) -> None:
        x = np.concatenate((self._carry, arr)) if self._carry.size else arr
        n = x.size // self.FR
        if n:
            f = x[: n * self.FR].reshape(n, self.FR)                       # int16 view, no copy
            f64 = f.astype(np.int64)
            self.rms.extend((np.sqrt((f64 * f64).sum(axis=1) / self.FR) / 32768.0).tolist())
            self.zcr.extend((((f[:, 1:] ^ f[:, :-1]) < 0).sum(axis=1) / (self.FR - 1)).tolist())
            self.dlt.extend((np.abs(np.diff(f.astype(np.int32), axis=1)).mean(axis=1) / 32768.0).tolist())
        self._carry = x[n * self.FR:].copy()

    def feed_f32(self, x: np.ndarray) -> None:
        self.feed_i16(np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16))

    def _steady_at(self, end: int, win: int = 8) -> bool:
        if end < 2 * win:
            return False
        a, b = slice(end - win, end), slice(end - 2 * win, end - win)
        r1, r0 = float(np.mean(self.rms[a])), float(np.mean(self.rms[b]))
        if r1 < self.VOICED_FLOOR or r0 < self.VOICED_FLOOR:
            return False
        z1, z0 = float(np.mean(self.zcr[a])), float(np.mean(self.zcr[b]))
        d1, d0 = float(np.mean(self.dlt[a])), float(np.mean(self.dlt[b]))
        if z1 > self.ZCR_MAX_VOWEL or z0 > self.ZCR_MAX_VOWEL:
            return False
        return (abs(r1 - r0) <= self.TOL_RMS * r0 and abs(z1 - z0) <= self.TOL_ZCR
                and abs(d1 - d0) <= self.TOL_DLT * max(d0, 1e-6))

    def steady_ms(self, win: int = 8) -> int:
        end, ms = len(self.rms), 0
        while self._steady_at(end, win):
            ms += win * 10
            end -= win
        return ms


def env10(wave: np.ndarray) -> np.ndarray:
    """10 ms RMS envelope."""
    n = len(wave) // 160
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    x = wave[: n * 160].astype(np.float64).reshape(n, 160)
    return np.sqrt((x * x).mean(axis=1) + 1e-12)


def measure_madd_voiced(env: np.ndarray, t0_s: float, t_cap_s: float, dip_ratio: float = 0.35,
                        bridge_frames: int = 3, onset_frames: int = 6) -> Optional[Tuple[float, float]]:
    """Duration (ms) of the voiced vowel that starts at `t0_s` and cannot extend past `t_cap_s` (the CTC start of the
    next phoneme, e.g. the hamza of شَاءَ). Voiced-energy continuity bridges dips of <= `bridge_frames` x 10 ms; the
    silent glottal occlusion before a hamza (50-100 ms) breaks the run and is therefore trimmed. Resolution is 10 ms,
    so neither the 160 ms hop nor the 40 ms CTC frame quantises the result.
    Returns (voiced_ms, trimmed_ms) or None when the audio does not contain a usable vowel."""
    i0 = max(0, int(round(t0_s / ENV_HOP_S)))
    i1 = min(len(env), int(round(t_cap_s / ENV_HOP_S)))
    if i1 - i0 < 4:
        return None
    seg = env[i0:i1]
    peak = float(np.percentile(seg, 75))
    if peak < 1e-4:
        return None
    voiced = seg >= dip_ratio * peak
    start = next((i for i in range(min(onset_frames, len(seg))) if voiced[i]), None)
    if start is None:
        return None
    last, gap = start, 0
    for j in range(start, len(seg)):
        if voiced[j]:
            last, gap = j, 0
        else:
            gap += 1
            if gap > bridge_frames:
                break
    return (last - start + 1) * ENV_HOP_S * 1000.0, (len(seg) - (last + 1)) * ENV_HOP_S * 1000.0


def clamp_mph(ms: float) -> float:
    return float(min(MPH_MAX_MS, max(MPH_MIN_MS, ms)))


def harakat_from_audio(env: np.ndarray, t0_s: float, t_cap_s: float,
                       mph_ms: float = MPH_DEFAULT_MS) -> Optional[float]:
    """Harakat (float) of a madd measured on the audio. `mph_ms` = the reciter's milliseconds per harakah (clamped to
    [85, 185]; the deterministic default is used in isolated tests)."""
    res = measure_madd_voiced(env, t0_s, t_cap_s)
    if res is None:
        return None
    return res[0] / clamp_mph(mph_ms)


@dataclass
class Recognition:
    text: str
    probs: Optional[List[float]] = None
    sifat: Optional[List[Dict[str, Any]]] = None
    frames: Optional[FrameTrack] = None


# ==============================================================================
# Streaming session engine
# ==============================================================================
class SessionEngine:
    """
    One reciter, one ayah. Feed audio with `add_pcm16` / `add_audio`, call `step` whenever `ready_to_run()`.

      cursor    number of words already judged and HARD-LOCKED (never revisited)         [stage 2]
      tracked   index of the word the reciter is on right now (last word reached)        [stage 1, soft]
      win_start first sample of the next model window: LEFT_CTX_S before the first unlocked word

    Stage 1 (soft): `tracked` moves as soon as the alignment enters a word and is sent immediately.
    Stage 2 (hard): word k is judged and locked only when >= LOCK_STABLE_PHONEMES phonemes of word k+1 were heard in two
    consecutive passes, or when a pause arrived (pause frame / silence). Every word traversed is aligned and judged
    before the lock moves past it (a word the reciter really skipped is reported as skipped, never silently dropped).
    """

    def __init__(self, ref: RefAya, recognize: Callable[[np.ndarray], Recognition],
                 preset: Preset, meta: Optional[Dict[str, Any]] = None) -> None:
        self.ref = ref
        self.recognize = recognize
        self.preset = preset
        self.buf = AudioBuffer()
        self.gate = PcmGate()
        self.cursor = 0
        self.tracked = -1
        self.win_start = 0
        self.last_run_end = 0
        self.pause_checked_at = -1
        self.results: Dict[int, Tuple[str, List[Issue]]] = {}
        self.seq = 0
        self.rtf: Optional[float] = None
        self.complete = False
        self.last_window: Tuple[int, int] = (0, 0)
        self.skip_mode = False
        self._flush_iters = 0
        self.word_log: Dict[int, Dict[str, Any]] = {}
        self.pauses: List[Dict[str, int]] = []            # real acoustic pauses announced by the client
        self.phrase_start = 0
        self._cap_level = 0
        self._final_stable = 0
        self._tail_hist: "deque[Tuple[str, float]]" = deque(maxlen=3)
        self._skipped_in_row = 0
        self.skips = 0
        self.mph_samples: List[float] = []
        self.resumed_from = 0
        self._cur: Optional[Dict[str, Any]] = None
        self.debug: Dict[str, Any] = {
            "created": time.time(), "sura": ref.sura, "aya": ref.aya, "preset": preset.name,
            "approx_segmentation": ref.approx, "ref_info": ref.info, "meta": meta or {}, "steps": [],
        }

    # ------------------------------------------------------------------ audio in
    def add_audio(self, x: np.ndarray) -> None:
        """float32 samples (tests / local tools)."""
        if not self.complete and x.size:
            self.gate.feed_f32(x)
            self.buf.append(x)

    def add_pcm16(self, raw: Any) -> None:
        """Websocket frames: 16 kHz mono int16 little-endian. The bytes are inspected zero-copy
        (memoryview.cast('h') -> numpy view) BEFORE any float conversion."""
        if self.complete:
            return
        mv = memoryview(raw)
        if mv.format != "B":
            mv = mv.cast("B")
        k = (len(mv) // 2) * 2
        if not k:
            return
        arr = np.frombuffer(mv[:k].cast("h"), dtype=np.int16)           # zero-copy view
        if np.little_endian is False:                                    # pragma: no cover
            arr = arr.byteswap()
        self.gate.feed_i16(arr)
        self.buf.append(arr.astype(np.float32) * np.float32(1.0 / 32768.0))

    def on_pause(self, duration_ms: int, padding_ms: int = 500) -> None:
        """The client saw > 500 ms of continuous silence and stopped streaming. The last `padding_ms` of the buffer are
        hangover padding, NOT speech: the true voicing offset is kept separately so room noise in the padding can never
        stretch a terminal madd, and the pause counts as real acoustic evidence for waqf / ibtida'."""
        n = self.buf.n
        pad = int(max(0, min(int(padding_ms), int(duration_ms))) * SR / 1000)
        frames = self.buf.frames_array()
        voice_end = max(0, n - pad)
        if len(frames):
            thr = self._thr(frames)
            hi = min(len(frames), voice_end // FRAME + 1)
            idx = np.nonzero(frames[:hi] >= thr)[0]
            if len(idx):
                voice_end = min(voice_end, (int(idx[-1]) + 1) * FRAME)
        unsettled = False
        if len(frames):                                    # speech since the last lock that nobody has judged yet
            f0 = min(len(frames), self.win_start // FRAME)
            seg_end = max(f0, voice_end // FRAME)
            unsettled = bool(np.any(frames[f0:seg_end] >= self._thr(frames))) and self.tracked >= self.cursor
        self.pauses.append({"at": n, "ms": int(duration_ms), "voice_end": int(voice_end), "done": not unsettled})
        del self.pauses[:-32]

    def resume_from(self, word_idx: int) -> None:
        """Stateless checkpoint: the client keeps the verdicts of the words before `word_idx`; this (new) engine starts
        its lock there."""
        k = max(0, min(self.ref.n, int(word_idx)))
        self.cursor, self.tracked, self.resumed_from, self.phrase_start = k, k - 1, k, k
        self.win_start = 0
        self.pauses = []
        self.debug["resume_word_idx"] = k
        if k >= self.ref.n:
            self.complete = True

    @property
    def audio_ms(self) -> int:
        return int(self.buf.n * 1000 / SR)

    # ------------------------------------------------------------------ vad helpers
    @staticmethod
    def _thr(frames: np.ndarray) -> float:
        if len(frames) < 8:
            return 0.010
        floor = float(np.percentile(frames[-400:], 10))
        return float(min(max(3.0 * floor, 0.005), 0.015))

    def _mask_from(self, frames: np.ndarray, start: int, stop: Optional[int] = None) -> np.ndarray:
        """Voiced mask for frames[start:stop] (local indexing)."""
        seg = frames[start:stop]
        if len(seg) == 0:
            return np.zeros(0, dtype=bool)
        return voiced_mask(seg, self._thr(frames))

    def _mph_raw(self) -> Optional[float]:
        if len(self.mph_samples) < MPH_MIN_SAMPLES:
            return None
        recent = sorted(self.mph_samples[-12:])
        if len(recent) >= 6:                              # trimmed mean: averages the +-1 frame quantisation jitter
            recent = recent[1:-1]
        return float(np.mean(recent))

    def mph_trusted(self) -> Optional[float]:
        """local_mph(), or None when the measured tempo is so far outside the clamp range [85, 185] ms that the clamp
        would distort the scale (a very fast or very slow reciter): the audio channel then abstains."""
        raw = self._mph_raw()
        if raw is None or raw < MPH_MIN_MS * 0.92 or raw > MPH_MAX_MS * 1.08:
            return None
        return clamp_mph(raw)

    def local_mph(self) -> Optional[float]:
        """The reciter's own milliseconds per harakah (trimmed mean of measured madd tabii runs, clamped to [85, 185]);
        None until MPH_MIN_SAMPLES runs were measured."""
        raw = self._mph_raw()
        return None if raw is None else clamp_mph(raw)

    # ------------------------------------------------------------------ scheduling
    def _pending_pause(self, ws: int) -> Optional[Dict[str, int]]:
        """Earliest announced pause whose speech has not been settled yet (stale ones are retired)."""
        for p in self.pauses:
            if p["done"]:
                continue
            if p["at"] <= ws:
                p["done"] = True
                continue
            return p
        return None

    def _hop_s(self) -> float:
        return HOP_SLOW_S if (self.rtf is not None and self.rtf > 0.5) else HOP_S

    def ready_to_run(self) -> bool:
        if self.complete:
            return False
        frames = self.buf.frames_array()
        nfr = len(frames)
        if nfr < 5:
            return False
        n = self.buf.n
        new_s = (n - self.last_run_end) / SR
        f0 = min(nfr, self.last_run_end // FRAME)
        if new_s >= self._hop_s():
            voiced_new = float(self._mask_from(frames, f0).sum()) * FRAME / SR
            if voiced_new >= MIN_NEW_VOICED_S:
                return True
        if self._pending_pause(self.win_start) is not None and self.tracked >= self.cursor:
            return True                    # a pause frame arrived: settle the last word now
        if self.tracked >= self.cursor:
            tail0 = max(0, nfr - 200)
            vidx = np.nonzero(self._mask_from(frames, tail0))[0]
            if len(vidx):
                last_v = tail0 + int(vidx[-1])
                if (nfr - 1 - last_v) * FRAME / SR >= TAIL_SILENCE_S and last_v != self.pause_checked_at:
                    return True            # the reciter just paused: settle the last word
        return False

    def _sustaining(self) -> bool:
        """A long vowel / madd (or slow tahqiq) is being held right now."""
        if self._tail_hist and self._tail_hist[-1][0] in ALWAYS_MADD:
            return True
        return self.gate.steady_ms() >= 280

    def _tail_cap_s(self, n: int, ws: int) -> float:
        levels = (TAIL_BASE_S, TAIL_ELASTIC_S, MAX_WINDOW_S)
        cap = levels[min(self._cap_level, 2)]
        if cap < TAIL_ELASTIC_S and self._sustaining():
            cap = TAIL_ELASTIC_S
        backlog_s = (n - ws) / SR
        if backlog_s > cap:                              # the lock lags the live edge: the window must reach it, or
            cap = max(cap, min(BURST_WINDOW_S, backlog_s + 0.05))   # the tail can never catch up (catch-up: <= 6 s)
        return cap

    def _gate_may_skip(self) -> bool:
        """Interleaved gating: while a high-confidence vowel/madd is held (2 neural passes in a row > 0.95) and the
        incoming Int16 audio is acoustically steady, ONE neural hop may be skipped (never two in a row)."""
        if self._skipped_in_row >= GATE_MAX_SKIP_IN_ROW or len(self._tail_hist) < 2:
            return False
        for ch, p in list(self._tail_hist)[-2:]:
            if ch not in ALWAYS_MADD or p <= GATE_HIGH_CONF:
                return False
        return self.gate.steady_ms() >= 160

    # ------------------------------------------------------------------ one model window
    def step(self, flush: bool = False) -> Optional[Dict[str, Any]]:
        """Judges the next chunk of audio chronologically (never skips a backlog)."""
        if self.complete:
            return None
        t0 = time.time()
        n = self.buf.n
        frames = self.buf.frames_array()
        nfr = len(frames)
        if nfr < 5:
            return None

        ws = self.win_start
        forced = self.skip_mode
        if n - ws > int(MAX_BACKLOG_S * SR):                  # hopelessly behind: skip old audio
            ws = n - int(MAX_WINDOW_S * SR)
            forced = True
        pre_fr = int(math.ceil(PRE_PAD_S * SR / FRAME))
        post_fr = int(math.ceil(POST_PAD_S * SR / FRAME))
        nz = np.nonzero(self._mask_from(frames, ws // FRAME))[0]
        if len(nz):                                   # leading silence is not part of the tail: start at the first speech
            ws_v = (ws // FRAME + int(nz[0]) - pre_fr) * FRAME
            if ws_v > ws:
                ws = ws_v
                self.win_start = max(self.win_start, ws)
        pend = self._pending_pause(ws)
        settle_hint = flush or pend is not None
        if not settle_hint and not forced and self._gate_may_skip():
            self._skipped_in_row += 1
            self.skips += 1
            self.last_run_end = n
            return None
        self._skipped_in_row = 0
        cap_s = self._tail_cap_s(n, ws)
        we = min(n, ws + int(cap_s * SR))
        pause_reg = False
        if pend is not None and pend["at"] <= we + FRAME:
            we = min(we, pend["at"])          # judge up to the pause; later audio belongs to the next pass
            pause_reg = True
        self.last_run_end = we
        at_live_edge = we >= n - FRAME
        wf0, wf1 = ws // FRAME, min(nfr, we // FRAME)
        vidx = np.nonzero(self._mask_from(frames, wf0, wf1))[0]
        if len(vidx) * FRAME / SR < MIN_VOICED_S:
            if not at_live_edge:                               # silent stretch: move on
                self.win_start = max(self.win_start, we - int(0.5 * SR))
            return None
        ts = max(0, int(vidx[0]) - pre_fr)
        te = min(wf1 - wf0, int(vidx[-1]) + 1 + post_fr)
        wave = self.buf.snapshot((wf0 + ts) * FRAME, (wf0 + te) * FRAME)
        last_v_abs = wf0 + int(vidx[-1])
        tail_sil = (wf1 - 1 - last_v_abs) * FRAME / SR
        pause_end = tail_sil >= TAIL_SILENCE_S
        settle = pause_end or pause_reg or (flush and at_live_edge)
        vsel = vidx[(vidx >= ts) & (vidx < te)] + wf0          # voiced frames inside the window
        if wave.size < int(0.25 * SR):
            return None

        self.last_window = ((wf0 + ts) * FRAME, (wf0 + te) * FRAME)
        t_inf = time.time()
        rec = self.recognize(wave)
        inf_s = time.time() - t_inf
        r = inf_s / max(0.25, wave.size / SR)
        self.rtf = r if self.rtf is None else 0.7 * self.rtf + 0.3 * r

        pred = clean_phonemes(rec.text)
        if not pred and not at_live_edge and not flush:
            self.win_start = max(self.win_start, we - int(1.0 * SR))     # voiced noise, no phonemes: nothing to lose
        probs = rec.probs if (rec.probs is not None and pred == rec.text
                              and len(rec.probs) == len(pred)) else None
        ft = rec.frames if (rec.frames is not None and pred == rec.text and rec.frames.n_chars == len(pred)) else None
        self._cur = {"origin": self.last_window[0], "wave_s": wave.size / SR, "env": env10(wave), "frames": ft}
        self._tail_hist.append((pred[-1] if pred else "", float(probs[-1]) if probs else 0.0))
        pred_gmap = (group_index_map(rec.sifat, pred) or tolerant_group_map(rec.sifat, pred)) if rec.sifat else None
        step_dbg: Dict[str, Any] = {
            "seq": self.seq + 1, "win_s": [round(wf0 * FRAME / SR, 2), round(n / SR, 2)],
            "wave_s": round(wave.size / SR, 2), "inf_s": round(inf_s, 2), "forced": forced,
            "flush": flush, "pause_end": pause_end, "pause_frame": pause_reg, "pred": pred,
            "cursor_in": self.cursor, "cap_s": cap_s, "frames": ft is not None,
        }
        step_dbg["sifat"] = {"pred_groups": len(rec.sifat) if rec.sifat else 0,
                             "pred_mapped": pred_gmap is not None,
                             "ref_tiles": self.ref.group_of_char is not None,
                             "probs": probs is not None}
        self.seq += 1
        updates: List[Dict[str, Any]] = []
        tracked_before = self.tracked

        cursor_in = self.cursor
        if pred:
            updates = self._align_and_commit(
                pred, probs, rec, pred_gmap, vsel, forced, settle, flush, step_dbg, we)
        if pause_reg and pend is not None:
            pend["done"] = True
        if (pause_end and at_live_edge) or pause_reg:
            self.pause_checked_at = last_v_abs
            self.phrase_start = self.cursor
        if self.cursor > cursor_in:
            self.skip_mode = False
            self._cap_level = 0
        else:
            if not at_live_edge and not settle:
                self._cap_level = min(2, self._cap_level + 1)   # a cap-limited window made no progress: widen
            if (we - ws) >= int(MAX_WINDOW_S * SR) - 5 * FRAME and not self.complete:
                self.win_start = max(self.win_start, ws + int(4 * SR))   # nothing usable in 10 s: move on
                self.skip_mode = True
        step_dbg["total_s"] = round(time.time() - t0, 2)
        step_dbg["tracked"] = self.tracked
        step_dbg["cursor_out"] = self.cursor
        dbg = self.debug["steps"]
        dbg.append(step_dbg)
        if len(dbg) > DEBUG_KEEP_STEPS:
            del dbg[0]
        if not updates and self.tracked == tracked_before:
            return None
        return self._progress(updates, round((time.time() - t0) * 1000), we)

    # ------------------------------------------------------------------ alignment (3 stages) + commit
    def _align_stage(self, pred: str, c: int, a: int, b: int, forced: bool, restart: bool) -> Dict[str, Any]:
        ref, N = self.ref, self.ref.n
        base = ref.starts[a]
        ref_str = ref.full_ph[base: ref.starts[b]]
        lead_free = ref.starts[c] - base
        lead_pen = 0.0
        if forced:
            lead_free = ref.starts[min(N, c + 8)] - base
            lead_pen = 0.05
        L = len(pred)
        al = align(
            ref_str, pred,
            lead_free=lead_free, lead_pen=lead_pen,
            pred_lead_free=(L if restart else (60 if c == 0 else 0)),
            tail_free_ref=True,
            tail_ins_chars=60 if b == N else 0,
        )
        pj_of_word: Dict[int, List[int]] = {}
        run_id: List[int] = [-1] * len(ref_str)
        rid = -1
        last_letter, last_w = "", -1
        for i, ch in enumerate(ref_str):
            w = ref.word_of_char[base + i]
            if w != last_w:
                last_letter, last_w = "", w
            if is_mark(ch):
                continue
            if ch != last_letter:
                rid += 1
            last_letter = ch
            run_id[i] = rid
        matched_runs: Dict[int, set] = {}
        matched_abs: Dict[int, int] = {}                 # absolute reference char -> predicted char
        matched_letters = 0
        first_pair_pj = None
        for ri, pj in al.pairs:
            if ri < 0 or pj < 0:
                continue
            if first_pair_pj is None:
                first_pair_pj = pj
            w = ref.word_of_char[base + ri]
            pj_of_word.setdefault(w, []).append(pj)
            if ref_str[ri] == pred[pj] and not is_mark(pred[pj]):
                matched_runs.setdefault(w, set()).add(run_id[ri])
                matched_abs[base + ri] = pj
                if w >= c:
                    matched_letters += 1

        def cov(w: int) -> float:
            total = len(ref.core[w])
            return 1.0 if total == 0 else len(matched_runs.get(w, ())) / total

        last_pair_pj = max((pj for ri, pj in al.pairs if ri >= 0 and pj >= 0), default=-1)
        tail_extra = L - 1 - last_pair_pj                  # predicted characters left over behind the alignment
        junk = (first_pair_pj or 0) if restart else 0
        consumed = max(1, al.i_end - lead_free)
        cost_pc = max(0.0, al.cost - 0.08 * junk) / consumed
        reached = -1
        for w in range(c, b):
            if cov(w) >= WORD_REACH_COV and w in pj_of_word:
                reached = w
        quality_ok = cost_pc <= QUALITY_MAX_COST and matched_letters >= MIN_MATCHED_LETTERS
        return {"a": a, "b": b, "base": base, "ref_str": ref_str, "lead_free": lead_free, "al": al,
                "pj_of_word": pj_of_word, "matched_abs": matched_abs, "matched_letters": matched_letters,
                "cov": cov, "reached": reached, "cost_pc": cost_pc, "quality_ok": quality_ok,
                "tail_extra": tail_extra}

    def _stable_enter(self, w: int, st: Dict[str, Any], probs: Optional[List[float]]) -> bool:
        """The reciter really is inside word w: >= LOCK_STABLE_PHONEMES of its phonemes are matched with a solid
        posterior (a lone phoneme at the window edge, or a low-probability guess, never locks the previous word)."""
        s0, s1 = self.ref.starts[w], self.ref.starts[w + 1]
        letters = [i for i in range(s0, s1) if not is_mark(self.ref.full_ph[i])]
        need = min(LOCK_STABLE_PHONEMES, len(letters))
        have = 0
        for i in letters:
            pj = st["matched_abs"].get(i)
            if pj is not None and (probs is None or probs[pj] >= LOCK_STABLE_PROB):
                have += 1
        return have >= need

    def _align_and_commit(self, pred: str, probs: Optional[List[float]], rec: Recognition,
                          pred_gmap: Optional[List[int]], vsel: np.ndarray, forced: bool,
                          settle: bool, flush: bool, dbg: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        ref, N, c = self.ref, self.ref.n, self.cursor
        a0 = max(0, c - CTX_WORDS)
        b_wide = min(N, c + REGION_AHEAD_WORDS)
        b_loc = min(b_wide, max(c, self.tracked) + LOCAL_AHEAD_WORDS)
        # ...but never shorter than what the window can contain: enough reference words to cover the predicted phonemes
        need_chars = ref.starts[c] + int(len(pred) * 1.2) + 6
        b_loc = min(b_wide, max(b_loc, bisect.bisect_right(ref.starts, need_chars)))
        a_back = max(0, min(a0, max(self.phrase_start, c - PHRASE_BACK_MAX)))
        stages: List[Tuple[str, int, int, bool]] = [("local", a0, b_loc, False)]
        if b_loc < b_wide:
            stages.append(("wide", a0, b_wide, False))
        stages.append(("restart", a_back, b_wide, True))     # phrase-start escape hatch: reciter repeated / backtracked
        chosen: Optional[Dict[str, Any]] = None
        fallback: Optional[Dict[str, Any]] = None
        tried: List[Dict[str, Any]] = []
        for name, a, b, restart in stages:
            st = self._align_stage(pred, c, a, b, forced, restart)
            st["stage"] = name
            tried.append({"stage": name, "region": [a, b], "cost": round(st["cost_pc"], 3),
                          "ok": st["quality_ok"], "reached": st["reached"]})
            if not st["quality_ok"] or st["reached"] < c:
                continue
            if name == "local" and b < b_wide and (st["al"].i_end >= len(st["ref_str"]) or st["tail_extra"] >= 3):
                fallback = fallback or st                   # the reciter may be further ahead: look wider first
                continue
            chosen = st
            break
        chosen = chosen or fallback
        dbg["stages"] = tried
        if chosen is None:
            dbg.update({"quality_ok": False, "reached": -1})
            return []
        st = chosen
        al, base, ref_str = st["al"], st["base"], st["ref_str"]
        pj_of_word, cov, reached = st["pj_of_word"], st["cov"], st["reached"]
        L = len(pred)
        dbg.update({
            "stage": st["stage"], "region": [st["a"], st["b"]], "cost_per_char": round(st["cost_pc"], 3),
            "matched_letters": st["matched_letters"], "reached": reached, "i_end": al.i_end,
            "lead_free": st["lead_free"], "quality_ok": True,
        })

        self.tracked = max(self.tracked, reached)                      # stage 1 (soft): immediate
        reached_stable = max((w for w in range(c, st["b"]) if cov(w) >= WORD_REACH_COV and w in pj_of_word
                              and self._stable_enter(w, st, probs)), default=c)
        last_done = (reached == N - 1 and cov(N - 1) >= 0.999 and al.i_end >= len(ref_str))
        self._final_stable = self._final_stable + 1 if last_done else 0
        settle_last = settle or self._final_stable >= 2                # the last word is judged the moment it is complete
        if settle_last and reached == N - 1 and cov(reached) >= LAST_WORD_SETTLE_COV:
            end_elig = N
        elif settle and cov(reached) >= LAST_WORD_SETTLE_COV:
            end_elig = reached + 1
        else:
            end_elig = reached_stable
        dbg["end_elig"] = end_elig
        if end_elig <= c:
            return []

        words_with_pj = sorted(pj_of_word)
        first_matched_word = next((w for w in words_with_pj if w >= c and cov(w) >= WORD_SKIP_COV), c)
        updates: List[Dict[str, Any]] = []
        judged_dbg: List[Dict[str, Any]] = []
        for w in range(c, end_elig):
            jinfo: Dict[str, Any] = {}
            if forced and w < first_matched_word:
                self.results[w] = ("unchecked", [])
            elif w not in pj_of_word or cov(w) < WORD_SKIP_COV:
                self.results[w] = ("letters", [Issue("letters", "skipped", NOTE_SKIPPED, 1.0)])
            else:
                issues, jinfo = self._judge(w, st, pred, probs, rec, pred_gmap, vsel)
                self.results[w] = (status_of(issues), issues)
            st_w, iss = self.results[w]
            updates.append(self._update_item(w))
            entry = {"w": w, "s": st_w, "issues": [i.to_debug() for i in iss]}
            entry.update(jinfo)
            self.word_log[w] = entry
            judged_dbg.append(entry)
        dbg["judged"] = judged_dbg
        self.cursor = end_elig
        if self.cursor >= N:
            self.complete = True
        else:
            self._advance_window(pj_of_word, words_with_pj, L, vsel, n, dbg)
        return updates

    # ------------------------------------------------------------------ one word
    def _judge(self, w: int, st: Dict[str, Any], pred: str, probs: Optional[List[float]], rec: Recognition,
               pred_gmap: Optional[List[int]], vsel: np.ndarray) -> Tuple[List[Issue], Dict[str, Any]]:
        ref, L = self.ref, len(pred)
        pj_of_word, al, base, ref_str = st["pj_of_word"], st["al"], st["base"], st["ref_str"]
        words_with_pj = sorted(pj_of_word)
        pjs = pj_of_word[w]
        first, last = min(pjs), max(pjs) + 1
        k = words_with_pj.index(w)
        prev_end = (max(pj_of_word[words_with_pj[k - 1]]) + 1) if k > 0 else 0
        next_start = min(pj_of_word[words_with_pj[k + 1]]) if k + 1 < len(words_with_pj) else L
        lo = max(prev_end, first - 3)
        hi = min(next_start, last + 3)
        hi = max(hi, last)
        seg = pred[lo:hi]
        seg_probs = probs[lo:hi] if probs is not None else None
        ft = self._cur["frames"] if self._cur else None
        alts_seg = ft.alts[lo:hi] if (ft is not None and ft.alts) else None
        approx_w = ref.word_approx[w]
        ctx = ""
        if approx_w:
            if w > 0:
                ctx += ref.wasl[w - 1][-2:]
            if w + 1 < ref.n:
                ctx += ref.wasl[w + 1][:2]
        forms, partner = ref.forms_ext(w)
        ac_trace: List[Any] = []
        issues, wal, form, costs = judge_word(
            forms, seg, seg_probs, self.preset,
            lead=first - lo, trail=hi - last, approx=approx_w, ctx_chars=ctx, partner=partner,
            stop_target=ref.stop_target[w], acoustic=self._acoustic_fn(lo, ac_trace), alts=alts_seg)
        extra = self._waqf_without_pause(w, form, costs, pj_of_word, L, vsel)
        if extra is not None:
            issues = issues + [extra]
        extra2 = self._start_without_pause(w, form, costs, pj_of_word, L, vsel)
        if extra2 is not None:
            issues = issues + [extra2]
        trace: List[Any] = []
        if self.preset.sifat_levels and ref.group_of_char is not None and pred_gmap is not None and rec.sifat:
            clean = costs.get(form, 99.0) <= 0.25
            s_issues, trace = self._sifat_issues(w, al, base, ref_str, pred, pred_gmap, rec.sifat, clean)
            issues = issues + s_issues
        if wal is not None:
            self._calibrate_mph(form, wal, seg, lo, issues)
        label = (("waqf" if ref.stop_target[w] and form == ref.wasl[w] else "wasl") if form == ref.wasl[w]
                 else "waqf" if form == ref.waqf[w]
                 else "start" if form == ref.start_forms[w] else "start+waqf")
        # the displayed segment never carries a neighbour's letters: leading unassigned characters (previous word's
        # silah / madd) and trailing unassigned LETTERS (next word's first phoneme) are cut; trailing marks / madd stay.
        d0 = first - lo
        d1 = len(seg) - (hi - last)
        tail = seg[d1:]
        keep = 0
        while keep < len(tail) and (is_mark(tail[keep]) or tail[keep] in ALWAYS_MADD):
            keep += 1
        shown = seg[d0:d1 + keep]
        info = {"seg": shown, "exp": form, "form": label,
                "sifat_cmp": trace[:14], "exact": not approx_w}
        if ac_trace:
            info["madd_audio"] = ac_trace[:4]
        return issues[:3], info

    # ------------------------------------------------------------------ acoustic madd (muttasil) + tempo calibration
    def _cap_by_pause(self, t_cap_s: float, t0_s: float, origin: int) -> float:
        for p in self.pauses:
            ve = (p["voice_end"] - origin) / SR
            if t0_s < ve < t_cap_s:
                t_cap_s = ve                            # trailing hangover / room noise is never counted as vowel
        return t_cap_s

    def _acoustic_fn(self, lo_g: int, trace: List[Any]) -> Optional[Callable[[int, int], Optional[float]]]:
        cur = self._cur
        mph = self.mph_trusted()
        if not cur or cur["frames"] is None or mph is None:
            return None
        ft, env, origin = cur["frames"], cur["env"], cur["origin"]

        def fn(a: int, b_excl: int) -> Optional[float]:
            pa, pn = lo_g + a, lo_g + b_excl
            if pa < 0 or pa >= ft.n_chars or pn >= ft.n_chars:      # the next phoneme (hamza) must have been heard
                return None
            t0 = ft.start_s(pa)
            t_cap = self._cap_by_pause(ft.start_s(pn), t0, origin)
            res = measure_madd_voiced(env, t0, t_cap)
            if res is None:
                return None
            h = res[0] / mph
            trace.append({"run": [pa, pn - 1], "voiced_ms": round(res[0]), "trimmed_ms": round(res[1]),
                          "mph": round(mph, 1), "harakat": round(h, 2)})
            return h
        return fn

    def _calibrate_mph(self, exp: str, al: Alignment, seg: str, lo_g: int, issues: List[Issue]) -> None:
        """Learns the reciter's tempo from clean 2-harakat madd tabii runs (same measurement as the judged madd)."""
        cur = self._cur
        if not cur or cur["frames"] is None or any(i.k.startswith("madd") for i in issues):
            return
        ft, env = cur["frames"], cur["env"]
        r2p = {ri: pj for ri, pj in al.pairs if ri >= 0 and pj >= 0}
        for c, s0, e0 in split_runs(exp):
            if c not in ALWAYS_MADD or e0 - s0 != 2 or madd_kind(exp, s0, e0) != "tabii":
                continue
            pj = [r2p[ri] for ri in range(s0, e0) if ri in r2p and seg[r2p[ri]] == c]
            if len(pj) != 2 or pj[1] != pj[0] + 1:
                continue
            pa, pn = lo_g + pj[0], lo_g + pj[1] + 1
            if pa >= ft.n_chars or pn >= ft.n_chars:
                continue
            t0 = ft.start_s(pa)
            res = measure_madd_voiced(env, t0, self._cap_by_pause(ft.start_s(pn), t0, cur["origin"]))
            if res is not None and 40.0 <= res[0] / 2.0 <= 300.0:
                self.mph_samples.append(res[0] / 2.0)
                del self.mph_samples[:-24]

    # ------------------------------------------------------------------ pauses (audio evidence)
    def _pause_near(self, t_a: int, t_b: int) -> bool:
        lo, hi = min(t_a, t_b) - SR // 2, max(t_a, t_b) + SR // 2
        return any(lo <= p["voice_end"] <= hi for p in self.pauses)

    def _silence_between(self, t_a: int, t_b: int, vsel: np.ndarray) -> Optional[bool]:
        """True if >= 100 ms of silence (or a pause frame) lies within +-0.5 s of the boundary between two times;
        None = unknown."""
        if self._pause_near(t_a, t_b):
            return True
        if len(vsel) < 6:
            return None
        lo_f = max(min(t_a, t_b) // FRAME - 25, int(vsel[0]))
        hi_f = min(max(t_a, t_b) // FRAME + 25, int(vsel[-1]))
        if hi_f - lo_f < 10:
            return None
        vset = set(int(x) for x in vsel)
        run = best = 0
        for f in range(lo_f, hi_f + 1):
            if f in vset:
                run = 0
            else:
                run += 1
                best = max(best, run)
        return best >= 5

    def _start_without_pause(self, w: int, form: str, costs: Dict[str, float],
                             pj_of_word: Dict[int, List[int]], L: int, vsel: np.ndarray) -> Optional[Issue]:
        """The word was said as if starting fresh (hamzat al-wasl pronounced / no idgham with the previous word)
        although the reciter did not stop before it: a tajweed mistake in continuous reading."""
        ref = self.ref
        starts = (ref.start_forms[w], ref.start_waqf[w])
        if (not self.preset.flag_waqf_no_pause or w == 0 or form not in [s for s in starts if s]
                or (w - 1) not in pj_of_word or w not in pj_of_word):
            return None
        cont = [f for f in (ref.wasl[w], ref.waqf[w]) if f]
        if min(costs.get(f, 99.0) for f in cont) - costs.get(form, 99.0) < 0.12:
            return None                    # the continuation form fits about as well: no evidence
        t_prev = self._pred_to_sample(max(pj_of_word[w - 1]), L, vsel)
        t_this = self._pred_to_sample(min(pj_of_word[w]), L, vsel)
        if t_prev is None or t_this is None:
            return None
        if self._silence_between(t_prev, t_this, vsel) is not False:
            return None
        return Issue("tajweed", "start_no_pause", note_start_no_pause(), 0.9, 0, form, ref.wasl[w], "")

    def _waqf_without_pause(self, w: int, form: str, costs: Dict[str, float],
                            pj_of_word: Dict[int, List[int]], L: int,
                            vsel: np.ndarray) -> Optional[Issue]:
        """Word-final vowel dropped (stop form) although the reciter kept going: tashkeel mistake.
        A real stop shows up as a pause frame or as a silence in the audio next to the word boundary."""
        ref = self.ref
        waqf = ref.waqf[w]
        if (not self.preset.flag_waqf_no_pause or not waqf or waqf == ref.wasl[w] or form != waqf
                or w + 1 >= ref.n or (w + 1) not in pj_of_word):
            return None
        if costs.get(ref.wasl[w], 99.0) - costs.get(waqf, 99.0) < 0.3:
            return None                    # the stop form is not clearly better than the continuation
        t_end = self._pred_to_sample(max(pj_of_word[w]), L, vsel)
        t_next = self._pred_to_sample(min(pj_of_word[w + 1]), L, vsel)
        if t_end is None or t_next is None:
            return None
        if self._pause_near(t_end, t_next):
            return None                    # the client announced a real pause: the stop form is fine
        if len(vsel) < 6:
            return None
        lo_f = max(min(t_end, t_next) // FRAME - 25, int(vsel[0]))      # +-0.5 s around the boundary
        hi_f = min(max(t_end, t_next) // FRAME + 25, int(vsel[-1]))
        if hi_f - lo_f < 10:
            return None
        vset = set(int(x) for x in vsel)
        run = best = 0
        for f in range(lo_f, hi_f + 1):
            if f in vset:
                run = 0
            else:
                run += 1
                best = max(best, run)
        if best >= 5:                      # >= 100 ms of silence: a real pause, the stop form is fine
            return None
        vowel = next((ch for ch in reversed(ref.wasl[w]) if ch in CORE_VOWELS), "")
        return Issue("tashkeel", "waqf_no_pause", note_waqf_no_pause(vowel), 0.9, 0, waqf, ref.wasl[w], "")

    # ------------------------------------------------------------------ sifat
    @staticmethod
    def _is_rare(lv: str, lab: Optional[str]) -> bool:
        key = SIFA_RARE.get(lv)
        if lab is None or key is None or "PAD" in lab:
            return False
        if lv == "tafkheem_or_taqeeq" and "low" in lab:
            return False
        return key in lab and "not_" not in lab and "no_" not in lab

    def _best_label_shift(self, pairs: List[Tuple[int, int, str]], pred_sifat: List[Dict[str, Any]],
                          levels: Sequence[str]) -> Tuple[Dict[str, int], Dict[str, Dict[int, int]]]:
        """Each of the ten sifat heads is its own CTC head, so its label sequence can slide against the phoneme groups
        independently of the others (long madd tiles, geminates, edits): the live reports showed every property of a word
        one group to the left (w43) and only `tafashie` two groups to the right (w39). A shift s is therefore decided PER
        LEVEL, only on evidence, and only if it fixes that level:
          * levels with a rare marked property (tikraar, istitala, tafashie, safeer, qalqla, ghonna, itbaq, tafkheem, hams):
            the rare property must line up better after the shift, AND the unshifted reading must show an impossible
            property (tikraar on ء) or >= 2 aligned rare matches;
          * the other levels (shidda_or_rakhawa): the full labels must agree at least 2 more times after the shift.
        Returns ({level: shift}, {level: {shift: score}})."""
        groups = self.ref.exp_groups
        shifts: Dict[str, int] = {}
        all_scores: Dict[str, Dict[int, int]] = {}
        for lv in levels:
            rare_lv = lv in SIFA_RARE
            scores: Dict[int, int] = {}
            halluc0 = 0
            for sft in (-2, -1, 0, 1, 2):
                sc = 0
                for g, h, el in pairs:
                    h2 = h + sft
                    if not 0 <= h2 < len(pred_sifat):
                        continue
                    e = groups[g]["L"].get(lv, (None, None))[0]
                    p = pred_sifat[h2]["L"].get(lv, (None, None))[0]
                    if e is None or p is None or "PAD" in e or "PAD" in p:
                        continue
                    if rare_lv:
                        er, pr = self._is_rare(lv, e), self._is_rare(lv, p)
                        if er and pr:
                            sc += 1
                        elif er != pr:
                            sc -= 1
                        if sft == 0 and pr and lv in SIFA_DOMAIN and lv in ("tafashie", "tikraar", "istitala", "safeer") \
                                and el not in SIFA_DOMAIN[lv]:
                            halluc0 += 1
                    elif e == p:
                        sc += 1
                scores[sft] = sc
            all_scores[lv] = scores
            best = max((-2, -1, 1, 2), key=lambda t: (scores[t], -abs(t)))
            if rare_lv:
                accept = scores[best] >= 1 and scores[best] - scores[0] >= 2 and (scores[best] >= 2 or halluc0 >= 1)
            else:
                accept = scores[best] - scores[0] >= 2 and scores[best] >= max(2, int(0.8 * len(pairs)))
            if accept:
                shifts[lv] = best
        return shifts, all_scores

    def _sifat_issues(self, w: int, al: Alignment, base: int, ref_str: str, pred: str,
                      pred_gmap: List[int], pred_sifat: List[Dict[str, Any]],
                      clean: bool = False) -> Tuple[List[Issue], List[Any]]:
        """Compares the model's per-letter properties with the expected ones. Returns (issues, trace) where trace lists
        [letter, level, expected, predicted, prob, flagged] for the report.
          * pred_sifat is matched to the reference phonemes by base-consonant identity (and shifted as a whole only on
            evidence, see _best_label_shift) - never by a naive running index;
          * a contrast is evaluated only on letters where it is phonetically valid (SIFA_DOMAIN);
          * voicing / manner contrasts on a clean, ultra-short function word (فِي) are not trusted (vowel bleed);
          * every confidence is a probability in [0, 1]."""
        ref = self.ref
        out: List[Issue] = []
        trace: List[Any] = []
        pos = lambda s, key: key in s and "not_" not in s and "no_" not in s
        pairs: List[Tuple[int, int, str]] = []
        seen = set()
        for ri, pj in al.pairs:
            if ri < 0 or pj < 0 or ref.word_of_char[base + ri] != w:
                continue
            if ref_str[ri] != pred[pj] or is_mark(pred[pj]):
                continue
            g, h = ref.group_of_char[base + ri], pred_gmap[pj]
            if (g, h) in seen or g >= len(ref.exp_groups) or h >= len(pred_sifat):
                continue
            seen.add((g, h))
            el = next(iter(letters_only(ref.exp_groups[g]["g"])), "")
            if el:
                pairs.append((g, h, el))
        if not pairs:
            return out, trace
        levels = self.preset.sifat_levels
        shifts, all_scores = self._best_label_shift(pairs, pred_sifat, levels)
        for lv_, sh_ in sorted(shifts.items()):
            trace.append(["~", "label_shift:" + lv_, str(sh_), "%d->%d" % (all_scores[lv_][0], all_scores[lv_][sh_]), 0.0, False])
        ultra_short = clean and len(ref.core[w]) <= BLEED_MAX_CORE
        for g, h, el in pairs:
            eg = ref.exp_groups[g]
            for lv in levels:
                shift = shifts.get(lv, 0)
                h2 = h + shift
                if not 0 <= h2 < len(pred_sifat):
                    continue
                pg = pred_sifat[h2]
                if shift == 0:
                    pl = next(iter(letters_only(pg["g"])), "")
                    if el != pl:
                        continue                # the label group belongs to another letter: nothing to compare
                dom = SIFA_DOMAIN.get(lv)
                if dom is not None and el not in dom:
                    continue                    # phonetic domain gating (tafashie only on ش, tikraar only on ر, ...)
                elab, _ = eg["L"].get(lv, (None, None))
                plab, pprob = pg["L"].get(lv, (None, None))
                if elab is None or plab is None:
                    continue
                if "PAD" in elab or "PAD" in plab:
                    continue                     # the model abstains for letters where this property is undefined
                if lv == "ghonna" and el not in GHUNNA_LETTERS:
                    continue                     # ghunna on ل/ت/... is impossible: it is a label misalignment
                if lv == "tafkheem_or_taqeeq" and el not in TAFKHEEM_LETTERS:
                    continue
                conf = min(1.0, max(0.0, pprob if pprob is not None else 0.92))
                flagged = False
                if elab != plab and shift == 0 and self._label_shift(lv, g, h, elab, plab, pred_sifat):
                    trace.append([el, lv + "/shift-ignored", elab, plab, round(conf, 3), False])
                    continue
                if elab != plab and ultra_short and lv in SIFA_BLEED_LEVELS:
                    trace.append([el, lv + "/bleed-ignored", elab, plab, round(conf, 3), False])
                    continue
                if elab != plab and conf >= self.preset.conf_sifat:
                    if lv == "qalqla" and pos(elab, "moqalqal") and not pos(plab, "moqalqal"):
                        out.append(Issue("tajweed", "qalqla", note_qalqla(el), conf, 0, elab, plab, el))
                        flagged = True
                    elif lv == "ghonna":
                        e_yes, p_yes = pos(elab, "maghnoon"), pos(plab, "maghnoon")
                        if e_yes != p_yes:
                            out.append(Issue("tajweed", "ghonna", note_ghonna(el, e_yes), conf, 0, elab, plab, el))
                            flagged = True
                    elif lv == "tafkheem_or_taqeeq":
                        e_h = "mofakham" in elab and "low" not in elab
                        p_h = "mofakham" in plab and "low" not in plab
                        e_l, p_l = "moraqaq" in elab, "moraqaq" in plab
                        if (e_h and p_l) or (e_l and p_h):
                            out.append(Issue("tajweed", "tafkheem", note_tafkheem(el, e_h), conf, 0, elab, plab, el))
                            flagged = True
                if (lv in SIFA_AR and elab != plab and not flagged and el in BASIC_LETTERS
                        and conf >= max(self.preset.conf_sifat, 0.8)):
                    out.append(Issue("tajweed", "sifa:" + lv, note_sifa(lv, el), conf, 0, elab, plab, el))
                    flagged = True
                    trace.append([el, lv, elab, plab, round(conf, 3), True])
                interesting = (lv == "tafkheem_or_taqeeq" and elab != plab) or \
                              (lv != "tafkheem_or_taqeeq" and (pos(elab, "moqalqal") or pos(plab, "moqalqal")
                                                                 or pos(elab, "maghnoon") or pos(plab, "maghnoon")))
                if interesting:
                    trace.append([el, lv, elab, plab, round(conf, 3), flagged])
        return out, trace

    def _label_shift(self, lv: str, g: int, h: int, elab: str, plab: str,
                     pred_sifat: List[Dict[str, Any]]) -> bool:
        """The live report showed neighbouring letters with SWAPPED labels (م loses its ghunna while the
        next ل gains one): that is the model's label sequence sliding by one group, not a mistake."""
        groups = self.ref.exp_groups
        for dg in (-1, 1):
            g2, h2 = g + dg, h + dg
            if 0 <= g2 < len(groups) and 0 <= h2 < len(pred_sifat):
                e2 = groups[g2]["L"].get(lv, (None, None))[0]
                p2 = pred_sifat[h2]["L"].get(lv, (None, None))[0]
                if e2 is not None and p2 is not None and e2 == plab and p2 == elab and e2 != elab:
                    return True
        return False

    # ------------------------------------------------------------------ window management
    def _pred_to_sample(self, pj: int, L: int, vsel: np.ndarray) -> Optional[int]:
        cur = self._cur
        ft = cur["frames"] if cur else None
        if ft is not None and 0 <= pj < ft.n_chars:      # exact CTC frame of the character
            return int(cur["origin"] + (ft.first[pj] + 0.5) * ft.stride_s * SR)
        if len(vsel) == 0 or L == 0:
            return None
        k = min(len(vsel) - 1, int((pj + 0.5) / L * len(vsel)))
        return int(vsel[k]) * FRAME

    def _snap(self, sample: int) -> int:
        """Quietest frame within 120 ms BEFORE `sample`: the window may start earlier, never later, so the left-context
        overlap is never shorter than LEFT_CTX_S."""
        frames = self.buf.frames_array()
        if len(frames) == 0:
            return max(0, sample)
        f = sample // FRAME
        lo, hi = max(0, f - 6), min(len(frames), f + 1)
        if hi <= lo:
            return max(0, sample)
        return (lo + int(np.argmin(frames[lo:hi]))) * FRAME

    def _advance_window(self, pj_of_word: Dict[int, List[int]], words_with_pj: List[int],
                        L: int, vsel: np.ndarray, n: int, dbg: Dict[str, Any]) -> None:
        """The next window starts LEFT_CTX_S before the first UNLOCKED word - or, when the reciter has not started it yet
        (a pause), LEFT_CTX_S before the end of the last locked word. Either way the overlap is never shorter."""
        pj = None
        for w in words_with_pj:
            if w >= self.cursor:
                pj = min(pj_of_word[w])
                break
        if pj is None and (self.cursor - 1) in pj_of_word:
            pj = max(pj_of_word[self.cursor - 1])
        if pj is None:
            return
        t = self._pred_to_sample(pj, L, vsel)
        if t is None:
            return
        new_ws = self._snap(t - int(LEFT_CTX_S * SR))
        new_ws = min(new_ws, max(self.win_start, n - int(0.8 * SR)))
        if new_ws > self.win_start:
            self.win_start = new_ws
        dbg["next_win_start_s"] = round(self.win_start / SR, 2)

    # ------------------------------------------------------------------ messages
    def _update_item(self, w: int) -> Dict[str, Any]:
        st, issues = self.results[w]
        d: Dict[str, Any] = {"i": w, "s": st}
        if issues:
            d["issues"] = [i.to_client() for i in issues]
        return d

    def _progress(self, updates: List[Dict[str, Any]], busy_ms: int, n: int) -> Dict[str, Any]:
        return {
            "type": "progress", "seq": self.seq, "audio_ms": int(n * 1000 / SR),
            "tracked": self.tracked, "committed": self.cursor, "lock_ms": int(self.win_start * 1000 / SR),
            "updates": updates, "busy_ms": busy_ms, "rtf": round(self.rtf, 3) if self.rtf is not None else None,
        }

    def final_message(self) -> Dict[str, Any]:
        words = [self._update_item(w) for w in sorted(self.results)]
        counts = {"ok": 0, "tashkeel": 0, "tajweed": 0, "letters": 0, "unchecked": 0}
        for w in self.results:
            counts[self.results[w][0]] = counts.get(self.results[w][0], 0) + 1
        summary = dict(counts)
        summary["total"] = self.ref.n
        summary["judged"] = len(self.results)
        summary["complete"] = bool(self.complete or self.cursor >= self.ref.n)
        return {"type": "final", "audio_ms": self.audio_ms, "tracked": self.tracked,
                "committed": self.cursor, "words": words, "summary": summary}

    def flush_step(self) -> Tuple[Optional[Dict[str, Any]], bool]:
        """One more window while finishing (the reciter pressed stop). Returns (progress message, done)."""
        before = (self.cursor, self.tracked, self.win_start, self.last_run_end)
        msg = self.step(flush=True)
        after = (self.cursor, self.tracked, self.win_start, self.last_run_end)
        self._flush_iters += 1
        done = (self.complete or self._flush_iters >= 80
                or (after == before and self.last_run_end >= self.buf.n - FRAME))
        return msg, done

    def flush(self) -> Dict[str, Any]:
        """Judge everything that was heard, in order, then build the final report."""
        while True:
            _msg, done = self.flush_step()
            if done:
                break
        return self.final_message()


def compact_debug(eng: "SessionEngine", steps: int = 60) -> Dict[str, Any]:
    """Small, complete technical report: one entry per judged word + a thin step list (fits in a chat)."""
    d = {k: v for k, v in eng.debug.items() if k != "steps"}
    d["words"] = [eng.word_log[w] for w in sorted(eng.word_log)]
    d["engine_stats"] = {"skips": eng.skips, "mph_ms": eng.local_mph(), "pauses": len(eng.pauses),
                         "resumed_from": eng.resumed_from}
    d["steps"] = [{"seq": s.get("seq"), "win_s": s.get("win_s"), "pred": s.get("pred"), "stage": s.get("stage"),
                   "tracked": s.get("tracked"), "cursor": s.get("cursor_out"), "ok": s.get("quality_ok")}
                  for s in eng.debug.get("steps", [])][-steps:]
    return d


# ==============================================================================
# Audio helpers
# ==============================================================================
def prepare_wave(x: np.ndarray) -> np.ndarray:
    """Remove DC offset and clip. (The model's feature extractor normalises loudness itself.)"""
    x = x.astype(np.float32)
    if x.size:
        x = x - float(np.mean(x))
        np.clip(x, -1.0, 1.0, out=x)
    return x


# ==============================================================================
# Model runtime (loaded once per container)
# ==============================================================================
class QuranLib:
    """Thin wrapper around quran-transcript (the only place that touches that library)."""

    def __init__(self) -> None:
        from quran_transcript import MoshafAttributes
        self.moshaf = MoshafAttributes(
            rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4,
            madd_mottasel_waqf=4, madd_aared_len=4,
        )

    def get_uthmani(self, sura: int, aya: int) -> str:
        from quran_transcript import Aya
        return Aya(sura, aya).get().uthmani

    def phonetize(self, text: str, remove_spaces: bool) -> Any:
        from quran_transcript import quran_phonetizer
        return quran_phonetizer(text, self.moshaf, remove_spaces=remove_spaces)


SESSIONS: "deque[Dict[str, Any]]" = deque(maxlen=3)      # debug traces of the latest sessions

_HEAD_TOKENS = ("lm_head", "head", "classifier", "proj_out", "final_proj", "level", "output_proj")


def _extract_phoneme_logits(output: Any) -> Optional[np.ndarray]:
    """Phoneme-level CTC logits [T, C] from whatever the model's forward returns (tuple / ModelOutput / dict)."""
    o = output
    if isinstance(o, (tuple, list)) and o:
        o = o[0]
    if hasattr(o, "logits") and not hasattr(o, "detach"):
        o = o.logits
    if isinstance(o, dict):
        o = o.get("phonemes")
    if o is None or not hasattr(o, "detach"):
        return None
    arr = o.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return arr if arr.ndim == 2 else None


class ModelRuntime:
    def __init__(self) -> None:
        self.muaalem: Any = None
        self.lib: Optional[QuranLib] = None
        self.device = "cpu"
        self.int8 = False
        self.quant_info: Dict[str, Any] = {}
        self.bench_rtf: Optional[float] = None
        self.bench_rtf_int8: Optional[float] = None
        self.load_seconds: Optional[float] = None
        self.load_error: Optional[str] = None
        self.diag: Dict[str, Any] = {}
        self.id2char: Dict[int, str] = {}
        self._capture: Dict[str, Any] = {"logits": None}
        self._refs: Dict[Tuple[int, int], RefAya] = {}
        self._load_lock = threading.Lock()
        self._ref_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # ---------------------------------------------------------------- loading
    def ensure_loaded(self) -> None:
        if self.muaalem is not None:
            return
        with self._load_lock:
            if self.muaalem is not None:
                return
            t0 = time.time()
            import torch
            from quran_muaalem import Muaalem

            torch.set_num_threads(NUM_THREADS)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.lib = QuranLib()
            m = Muaalem(model_name_or_path=MODEL_ID, device=self.device, dtype=dtype)
            try:
                ref: Optional[RefAya] = self.get_ref(DEFAULT_SURA, DEFAULT_AYA)   # also pre-computes waqf forms
            except Exception:
                ref = None                                          # retried at the first session
            if self.device == "cpu" and ref is not None:
                self._tune_precision(m, ref)
            self._install_capture(m)
            self.diag["internals"] = self._probe_internals(m)
            self.muaalem = m
            self.load_seconds = round(time.time() - t0, 1)

    def warmup(self) -> None:
        """Dummy inference so every lazy initialisation (feature extractor, kernels, the logit hook) is part of the
        memory snapshot."""
        try:
            ref = self.get_ref(DEFAULT_SURA, DEFAULT_AYA)
            rng = np.random.default_rng(3)
            self.recognize((rng.standard_normal(int(SR * TAIL_BASE_S)) * 0.03).astype(np.float32), ref)
        except Exception as exc:                                    # a failed warm-up must never break start-up
            self.diag["warmup_error"] = repr(exc)[:160]

    def snapshot_prepare(self) -> None:
        """Right before Modal takes the memory snapshot: give freed pages back to the kernel and freeze the surviving
        objects so the garbage collector never touches (= copy-on-write dirties) them in restored containers."""
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        if hasattr(gc, "freeze"):
            gc.freeze()

    def _install_capture(self, m: Any) -> None:
        """Forward hook on the (final) model: the logits of the pass that is running anyway are kept, so CTC frame
        timings and rival-letter posteriors cost no extra inference."""
        model = getattr(m, "model", None)
        if model is None or not hasattr(model, "register_forward_hook"):
            return
        holder = self._capture

        def hook(_module: Any, _args: Any, output: Any) -> None:
            try:
                holder["logits"] = _extract_phoneme_logits(output)
            except Exception:
                holder["logits"] = None

        model.register_forward_hook(hook)

    def _time_inference(self, m: Any, ref: RefAya, seconds: float = 3.0) -> Optional[float]:
        """Real-time factor of one forward pass (content independent). None if it cannot be measured."""
        import torch
        rng = np.random.default_rng(0)
        wave_arr = (rng.standard_normal(int(SR * seconds)) * 0.03).astype(np.float32)
        try:
            with torch.inference_mode():
                m([wave_arr], [ref.full_out], sampling_rate=SR)           # warm-up
                t = time.time()
                m([wave_arr], [ref.full_out], sampling_rate=SR)
                return (time.time() - t) / seconds
        except Exception:
            pass
        try:
            proc, model = getattr(m, "processor", None), getattr(m, "model", None)
            if proc is None or model is None:
                return None
            feats = proc([wave_arr], sampling_rate=SR, return_tensors="pt")
            with torch.inference_mode():
                model(**feats)
                t = time.time()
                model(**feats)
                return (time.time() - t) / seconds
        except Exception:
            return None

    def _probe_internals(self, m: Any) -> Dict[str, Any]:
        """Records the shape of the library internals. Any failure is only recorded."""
        info: Dict[str, Any] = {}
        try:
            import torch
            model, proc = getattr(m, "model", None), getattr(m, "processor", None)
            info.update({"has_model": model is not None, "has_processor": proc is not None,
                         "model_class": type(model).__name__})
            if model is not None and proc is not None:
                rng = np.random.default_rng(1)
                feats = proc([(rng.standard_normal(SR * 2) * 0.03).astype(np.float32)],
                             sampling_rate=SR, return_tensors="pt")
                info["feature_shapes"] = {k: list(v.shape) for k, v in feats.items()}
                with torch.inference_mode():
                    out = model(**feats, return_dict=False)
                o0 = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(o0, dict):
                    info["logit_levels"] = {str(k): list(v.shape) for k, v in o0.items()}
                else:
                    info["output_type"] = type(o0).__name__
        except Exception as exc:
            info["error"] = repr(exc)[:200]
        return info

    # ---------------------------------------------------------------- hybrid INT8
    def _quantize_encoder_only(self, model: Any) -> Any:
        """Dynamic INT8 on the Transformer ENCODER Linear layers only. The front-end projection, the adapter and all 11
        multilevel output heads (phonemes + 10 sifat) stay in full precision, so low-energy hams / qalqalah / sifat
        probabilities are never crushed by quantisation noise."""
        import torch
        qfn = getattr(getattr(torch, "ao", None), "quantization", None) or torch.quantization
        names = {n for n, mod in model.named_modules()
                 if isinstance(mod, torch.nn.Linear) and (".encoder." in "." + n + "." or n.startswith("encoder."))
                 and not any(tok in n for tok in _HEAD_TOKENS)}
        self.quant_info = {"scope": "encoder-linear" if names else "all-linear", "layers": len(names)}
        if names:
            return qfn.quantize_dynamic(model, names, dtype=torch.qint8)
        return qfn.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    def _quant_close(self, m: Any, ref_model: Any, q_model: Any) -> Optional[float]:
        """Smallest cosine similarity between fp32 and int8 logits over all levels (synthetic voiced signals)."""
        import torch
        proc = getattr(m, "processor", None)
        if proc is None:
            return None
        worst = 1.0
        t = np.arange(int(SR * 2.0)) / SR
        sigs = [(0.2 * np.sin(2 * np.pi * 140 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
                 + 0.05 * np.random.default_rng(5).standard_normal(t.size)).astype(np.float32),
                (0.15 * (np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 880 * t))).astype(np.float32)]
        with torch.inference_mode():
            for sig in sigs:
                feats = proc([sig], sampling_rate=SR, return_tensors="pt")
                a = ref_model(**feats, return_dict=False)[0]
                b = q_model(**feats, return_dict=False)[0]
                if not isinstance(a, dict) or not isinstance(b, dict):
                    return None
                for k in a:
                    x, y = a[k].float().reshape(-1), b[k].float().reshape(-1)
                    cos = float(torch.nn.functional.cosine_similarity(x, y, dim=0))
                    worst = min(worst, cos)
        return worst

    def _tune_precision(self, m: Any, ref: RefAya) -> None:
        """Keeps fp32 when the CPU is fast enough; otherwise tries hybrid INT8 (and keeps it only if it really is
        faster, still runs, and its logits stay close to fp32). Any failure leaves the original fp32 model."""
        rtf = self._time_inference(m, ref)
        self.bench_rtf = None if rtf is None else round(rtf, 3)
        want_int8 = PRECISION == "int8" or (PRECISION == "auto" and rtf is not None and rtf > AUTO_INT8_ABOVE_RTF)
        if not want_int8 or getattr(m, "model", None) is None:
            return
        old = m.model
        try:
            q = self._quantize_encoder_only(old)
            close = self._quant_close(m, old, q)
            self.quant_info["min_logit_cosine"] = None if close is None else round(close, 4)
            if close is not None and close < 0.985:
                self.quant_info["rejected"] = "logits drifted"
                del q
                return
            m.model = q
            rq = self._time_inference(m, ref)
            if rq is not None and (rtf is None or rq < rtf * 0.9):
                self.int8 = True
                self.bench_rtf_int8 = round(rq, 3)
                del old
            else:
                m.model = old
                del q
        except Exception as exc:
            m.model = old
            self.quant_info["error"] = repr(exc)[:160]
        finally:
            gc.collect()

    # ---------------------------------------------------------------- references
    def get_ref(self, sura: int, aya: int) -> RefAya:
        key = (int(sura), int(aya))
        ref = self._refs.get(key)
        if ref is not None:
            return ref
        with self._ref_lock:
            ref = self._refs.get(key)
            if ref is None:
                if self.lib is None:
                    self.lib = QuranLib()
                ref = build_ref_aya(key[0], key[1], self.lib)
                self._refs[key] = ref
        return ref

    # ---------------------------------------------------------------- inference
    def recognize(self, wave_arr: np.ndarray, ref: RefAya) -> Recognition:
        self.ensure_loaded()
        import torch
        x = prepare_wave(wave_arr)
        with self._infer_lock:
            self._capture["logits"] = None
            with torch.inference_mode():
                outs = self.muaalem([x], [ref.full_out], sampling_rate=SR)
            logits = self._capture["logits"]
        out = outs[0]
        units = getattr(out, "phonemes", None)
        text = str(getattr(units, "text", "") or "")
        probs: Optional[List[float]] = None
        raw_probs = getattr(units, "probs", None)
        if raw_probs is not None:
            try:
                if hasattr(raw_probs, "detach"):
                    raw_probs = raw_probs.detach().float().cpu().numpy()
                arr = np.asarray(raw_probs, dtype=np.float64).reshape(-1)
                probs = arr.tolist() if arr.size == len(text) else None
            except Exception:
                probs = None
        frames: Optional[FrameTrack] = None
        if logits is not None and text:
            try:
                frames = FrameTrack.from_logits(logits, clean_phonemes(text), x.size / SR, self.id2char)
            except Exception:
                frames = None
        raw_sifat = getattr(out, "sifat", None)
        sifat = convert_sifat(raw_sifat)
        if "pred_sifat_n" not in self.diag or (raw_sifat and "pred_sifat_sample" not in self.diag):
            try:                                   # shape of the library objects, for the debug report
                self.diag["pred_sifat_n"] = len(raw_sifat) if raw_sifat else 0
                if raw_sifat:
                    self.diag["pred_sifat_sample"] = repr(raw_sifat[0])[:300]
                self.diag["pred_sifat_converted"] = bool(sifat)
                self.diag["phoneme_probs"] = probs is not None
                self.diag["ctc_frames"] = frames is not None
                self.diag["out_fields"] = [a for a in dir(out) if not a.startswith("_")][:20]
            except Exception:
                pass
        return Recognition(text, probs, sifat, frames)

    def status(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.muaalem is not None else ("error" if self.load_error else "loading"),
            "engine": ENGINE_NAME, "model": MODEL_ID, "device": self.device, "int8": self.int8,
            "quant": self.quant_info,
            "bench_rtf_fp32": self.bench_rtf, "bench_rtf_int8": self.bench_rtf_int8,
            "load_seconds": self.load_seconds, "load_error": self.load_error, "diag": self.diag,
            "refs": [{"sura": k[0], "aya": k[1], "words": v.n, "approx": v.approx,
                      "segmentation": v.info.get("segmentation"), "waqf_forms": v.info.get("waqf_forms"),
                      "sifat": v.group_of_char is not None} for k, v in self._refs.items()],
        }


RUNTIME = ModelRuntime()


def debug_snapshot(eng: "SessionEngine", runtime: ModelRuntime, steps: int = 60) -> Dict[str, Any]:
    """Compact trace of a session, sent with the final message (the server may scale to zero right
    after a session, so the app keeps its own copy for the 'technical report' button)."""
    d = compact_debug(eng, steps)
    srv = runtime.status()
    d["server"] = {k: srv.get(k) for k in ("status", "engine", "device", "int8", "quant", "bench_rtf_fp32",
                                           "load_seconds", "diag") if k in srv}
    return d


# ==============================================================================
# FastAPI application (live websocket only)
# ==============================================================================
def _dumps(obj: Any) -> str:
    if _orjson is not None:
        return _orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def uvicorn_kwargs(port: int = 8000) -> Dict[str, Any]:
    """Server settings: no access log, NO websocket permessage-deflate (compressing 5 KB PCM frames burns CPU for ~0
    bytes saved), uvloop / httptools when installed."""
    import importlib.util
    kw: Dict[str, Any] = dict(host="0.0.0.0", port=port, log_level="warning", access_log=False,
                              ws_per_message_deflate=False, timeout_keep_alive=5)
    if importlib.util.find_spec("uvloop"):
        kw["loop"] = "uvloop"
    if importlib.util.find_spec("httptools"):
        kw["http"] = "httptools"
    return kw


def build_fastapi_app(runtime: ModelRuntime = RUNTIME):
    from fastapi import FastAPI, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    api = FastAPI(title="Rafiq Quran recitation corrector", version=VERSION + ".0")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @api.get("/health")
    async def health() -> Dict[str, Any]:
        return runtime.status()

    @api.get("/debug/last")
    async def debug_last(steps: int = 30) -> JSONResponse:
        if not SESSIONS:
            return JSONResponse({"info": "no session yet", "server": runtime.status()})
        d = dict(SESSIONS[-1])
        keep = max(1, min(int(steps), DEBUG_KEEP_STEPS))
        d["steps"] = list(d.get("steps", []))[-keep:]
        d["server"] = runtime.status()
        return JSONResponse(d)

    @api.get("/debug/ref")
    async def debug_ref(sura_idx: int = DEFAULT_SURA, aya_idx: int = DEFAULT_AYA) -> JSONResponse:
        ref = await asyncio.to_thread(runtime.get_ref, sura_idx, aya_idx)
        return JSONResponse(ref.describe())

    @api.websocket("/ws/live-recitation")
    async def ws_live(websocket: WebSocket) -> None:
        await websocket.accept()
        qp = websocket.query_params
        cfg = {
            "sura": int(qp.get("sura_idx", DEFAULT_SURA)),
            "aya": int(qp.get("aya_idx", DEFAULT_AYA)),
            "strictness": qp.get("strictness", "balanced"),
        }
        state: Dict[str, Any] = {"engine": None, "last_rx": time.monotonic(), "end": "final"}
        wake, stop_evt = asyncio.Event(), asyncio.Event()
        send_lock = asyncio.Lock()
        last_sig: Dict[str, Any] = {"v": None}

        async def send(obj: Dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_text(_dumps(obj))

        async def send_progress(msg: Dict[str, Any]) -> None:
            """State-diff emission: a frame goes out only if the tracked word, the lock or a verdict changed."""
            sig = (msg.get("tracked"), msg.get("committed"),
                   tuple((u["i"], u["s"], len(u.get("issues", ()))) for u in msg.get("updates", ())))
            if sig == last_sig["v"]:
                return
            last_sig["v"] = sig
            await send(msg)

        async def new_engine() -> SessionEngine:
            await asyncio.to_thread(runtime.ensure_loaded)
            ref = await asyncio.to_thread(runtime.get_ref, cfg["sura"], cfg["aya"])
            eng = SessionEngine(ref, lambda w: runtime.recognize(w, ref), get_preset(cfg["strictness"]),
                                meta={"int8": runtime.int8, "device": runtime.device})
            SESSIONS.append(eng.debug)
            last_sig["v"] = None
            return eng

        def ready_msg(eng: SessionEngine) -> Dict[str, Any]:
            return {"type": "ready", "n_words": eng.ref.n, "words": eng.ref.words, "approx": eng.ref.approx,
                    "int8": runtime.int8, "strictness": eng.preset.name, "engine": ENGINE_NAME}

        async def worker() -> None:
            try:
                while not stop_evt.is_set():
                    await wake.wait()
                    wake.clear()
                    eng: SessionEngine = state["engine"]
                    while not stop_evt.is_set() and not eng.complete and eng.ready_to_run():
                        try:
                            msg = await asyncio.to_thread(eng.step, False)
                        except Exception as exc:            # keep the session alive on a bad window
                            await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                            break
                        if msg:
                            await send_progress(msg)
                    if state["engine"].complete:
                        break
                eng = state["engine"]
                while True:                    # settle whatever was heard, streaming progress to the app
                    msg, done = await asyncio.to_thread(eng.flush_step)
                    if msg:
                        await send_progress(msg)
                    if done:
                        break
                if state["end"] == "idle":     # nobody is talking: hand the checkpoint back and let the container go
                    await send({"type": "paused_idle", "resume_word_idx": eng.cursor, "audio_ms": eng.audio_ms,
                                "lock_ms": int(eng.win_start * 1000 / SR)})
                else:
                    final = eng.final_message()
                    final["debug"] = debug_snapshot(eng, runtime)
                    await send(final)
            except Exception as exc:
                try:
                    await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                except Exception:
                    pass
            finally:
                try:
                    await websocket.close(code=1000)
                except Exception:
                    pass

        task: Optional["asyncio.Task[None]"] = None
        recv: Optional["asyncio.Future[Any]"] = None
        try:
            state["engine"] = await new_engine()
            await send(ready_msg(state["engine"]))
            state["last_rx"] = time.monotonic()
            task = asyncio.create_task(worker())
            poll = min(1.0, max(0.05, IDLE_KILL_S / 4.0))
            while True:
                if recv is None:
                    recv = asyncio.ensure_future(websocket.receive())
                done, _pending = await asyncio.wait({recv}, timeout=poll)
                if not done:                   # idle watchdog: no audio and no control frame at all
                    if (not stop_evt.is_set() and time.monotonic() - state["last_rx"] >= IDLE_KILL_S):
                        state["end"] = "idle"
                        stop_evt.set()
                        wake.set()
                    continue
                msg = recv.result()
                recv = None
                if msg.get("type") == "websocket.disconnect":
                    break
                state["last_rx"] = time.monotonic()
                data = msg.get("bytes")
                if data:
                    state["engine"].add_pcm16(data)
                    wake.set()
                txt = msg.get("text")
                if txt:
                    try:
                        payload = json.loads(txt)
                    except Exception:
                        continue
                    kind = payload.get("type", "")
                    eng = state["engine"]
                    if kind == "stop":
                        state["end"] = "final"
                        stop_evt.set()
                        wake.set()
                    elif kind == "pause":
                        eng.on_pause(int(payload.get("duration_ms", PAUSE_MIN_MS)),
                                     int(payload.get("speech_tail_padding_ms", int(POST_PAD_S * 1000))))
                        wake.set()
                    elif kind == "resume":
                        eng.resume_from(int(payload.get("word_idx", 0)))
                        wake.set()
                    elif kind == "reset":
                        state["engine"] = await new_engine()
                        await send(ready_msg(state["engine"]))
        except Exception as exc:
            try:
                await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            stop_evt.set()
            if recv is not None and not recv.done():
                recv.cancel()
            if task is not None and not task.done():
                task.cancel()

    return api


# ==============================================================================
# Modal deployment (CPU, memory snapshot, scale-to-zero)
# ==============================================================================
MODAL_CPU = (0.125, 2.0)                                   # (reserved, burst limit) physical cores
MODAL_MEMORY = (int(os.environ.get("RAFIQ_MEM_MIB", "1536")), int(os.environ.get("RAFIQ_MEM_LIMIT_MIB", "4096")))
MODAL_SCALEDOWN_S = 2
MODAL_MAX_INPUTS = 16
MODAL_TARGET_INPUTS = int(os.environ.get("RAFIQ_TARGET_INPUTS", "1"))   # inference is CPU-bound: 1 live session per core budget
SERVER_MODE = os.environ.get("RAFIQ_SERVER", "uvicorn")    # "uvicorn" (compression off, no access log) | "asgi"

if modal is not None:
    app = modal.App(APP_NAME)

    def download_muaalem_weights() -> None:
        """Runs once at image build time so cold starts do not download the 2 GB model."""
        import torch
        from quran_muaalem import Muaalem
        from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        Muaalem(model_name_or_path=MODEL_ID, device=device, dtype=dtype)
        moshaf = MoshafAttributes(
            rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4,
            madd_mottasel_waqf=4, madd_aared_len=4,
        )
        quran_phonetizer(Aya(2, 255).get().uthmani, moshaf, remove_spaces=True)

    muaalem_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1")
        .env({"OMP_NUM_THREADS": str(NUM_THREADS), "MKL_NUM_THREADS": str(NUM_THREADS),
              "OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0", "MALLOC_TRIM_THRESHOLD_": "65536",
              "TOKENIZERS_PARALLELISM": "false"})
        .pip_install(
            "quran-muaalem>=0.2.2",
            "transformers>=4.45.0,<5.0.0",
            "accelerate",
            "quran-transcript",
            "librosa",
            "torch",
            "fastapi[standard]",
            "uvicorn[standard]",
            "websockets",
            "orjson",
            "numba>=0.61.2",
            "soundfile",
            "numpy",
        )
        .run_function(download_muaalem_weights)
    )

    # lowest bill: fractional CPU reservation with burst, scale to zero after 2 s, default (pre-emptible) capacity,
    # no region pinning (both would multiply the price).
    _cls_kwargs: Dict[str, Any] = dict(
        image=muaalem_image, cpu=MODAL_CPU, memory=MODAL_MEMORY, scaledown_window=MODAL_SCALEDOWN_S, timeout=900,
    )
    if USE_MEMORY_SNAPSHOT:
        _cls_kwargs["enable_memory_snapshot"] = True
    if USE_GPU:
        _cls_kwargs["gpu"] = "T4"
    _concurrent = getattr(modal, "concurrent", None)
    if _concurrent is None:                                  # very old Modal versions
        _cls_kwargs["allow_concurrent_inputs"] = MODAL_MAX_INPUTS

    def _serve_decorator() -> Callable[[Callable[..., Any]], Any]:
        if SERVER_MODE == "asgi" or not hasattr(modal, "web_server"):
            return modal.asgi_app()
        return modal.web_server(8000, startup_timeout=60)

    def _wrap_class(cls: Any) -> Any:
        if _concurrent is not None:
            try:                                   # many sockets may share a container; scale out at TARGET inputs
                cls = _concurrent(max_inputs=MODAL_MAX_INPUTS, target_inputs=MODAL_TARGET_INPUTS)(cls)
            except TypeError:
                cls = _concurrent(max_inputs=MODAL_MAX_INPUTS)(cls)
        return app.cls(**_cls_kwargs)(cls)

    # Class `Fastapi` + method `asgi_app` keeps the public hostname Modal derives from the names
    # (...-rafiq-quran-muaalem-backend-fastapi-asgi-app) identical to the function-based v14.4 deployment.
    def _enter_hook() -> Callable[[Callable[..., Any]], Any]:
        # snap=True is only legal together with enable_memory_snapshot=True
        return modal.enter(snap=True) if USE_MEMORY_SNAPSHOT else modal.enter()

    class Fastapi:
        @_enter_hook()
        def load(self) -> None:
            """Executed once while the snapshot is created: load (and hybrid-quantise) the model, warm it up,
            return freed memory to the kernel and freeze the heap. Restored containers start with everything ready."""
            try:
                RUNTIME.ensure_loaded()
                RUNTIME.warmup()
            except Exception:                      # stay up and explain at /health instead of crash-looping
                RUNTIME.load_error = traceback.format_exc()[-1500:]
            RUNTIME.snapshot_prepare()

        @_serve_decorator()
        def asgi_app(self) -> Any:
            api = build_fastapi_app(RUNTIME)
            if SERVER_MODE == "asgi" or not hasattr(modal, "web_server"):
                return api
            import uvicorn
            server = uvicorn.Server(uvicorn.Config(api, **uvicorn_kwargs(8000)))
            threading.Thread(target=server.run, name="rafiq-uvicorn", daemon=True).start()
            return None

    Fastapi = _wrap_class(Fastapi)
else:  # pragma: no cover
    app = None
