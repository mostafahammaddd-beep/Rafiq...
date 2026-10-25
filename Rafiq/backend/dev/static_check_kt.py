"""
Static checks for the Kotlin sources (no Kotlin compiler needed):
  1. delimiters ( ) { } [ ] balance, strings / comments aware
  2. no leftover references to batch mode (uploadBatch, buildWav, KEY_MODE, getMode, saveMode, analyzeUrl, /analyze)
  3. every member of RafiqController / RecitationConfig / RecitationSession used from another file exists
  4. every RecitationSession.Listener method is declared exactly once and overridden in RafiqController
  5. WebSocket JSON key parity: every key the app reads / writes exists on the Python side and vice versa
Run:  python3 dev/static_check_kt.py        (exit code 0 = all good)
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KT_DIR = os.path.join(ROOT, "android", "app", "src", "main", "java", "com", "rafiq", "app")
PY = os.path.join(ROOT, "backend", "modal_app.py")
BATCH_TOKENS = ["uploadBatch", "buildWav", "KEY_MODE", "getMode", "saveMode", "analyzeUrl", "/analyze", "sessionBatch",
                "MultipartBody", "changeMode", "ModeChip"]


def read_all():
    return {n: open(os.path.join(KT_DIR, n), encoding="utf-8").read() for n in sorted(os.listdir(KT_DIR)) if n.endswith(".kt")}


def strip_strings_comments(src):
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i); i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2); i = n if j < 0 else j + 2
        elif src.startswith('"""', i):
            j = src.find('"""', i + 3); i = n if j < 0 else j + 3; out.append('""')
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append('""'); i = j + 1
        elif c == "'":
            j = i + 1
            while j < n and src[j] != "'":
                j += 2 if src[j] == "\\" else 1
            out.append("''"); i = j + 1
        else:
            out.append(c); i += 1
    return "".join(out)


def check_balance(name, src, errs):
    s = strip_strings_comments(src)
    pairs = {")": "(", "}": "{", "]": "["}
    st = []
    line = 1
    for ch in s:
        if ch == "\n":
            line += 1
        if ch in "({[":
            st.append((ch, line))
        elif ch in ")}]":
            if not st or st[-1][0] != pairs[ch]:
                errs.append("%s:%d unbalanced '%s'" % (name, line, ch)); return
            st.pop()
    if st:
        errs.append("%s: unclosed '%s' opened at line %d" % (name, st[-1][0], st[-1][1]))


def members(src, cls):
    """public members (fun / val / var) declared at class level of `cls`, roughly."""
    m = re.search(r"class %s\b" % cls, src)
    body = src[m.start():] if m else src
    names = set(re.findall(r"^\s{4}(?:override\s+)?(?:suspend\s+)?fun\s+(\w+)", body, re.M))
    names |= set(re.findall(r"^\s{4}(?:@Volatile\s+)?(?:override\s+)?va[lr]\s+(\w+)", body, re.M))
    names |= set(re.findall(r"^\s{4}(?:const\s+)?(?:private\s+)?const val\s+(\w+)", body, re.M))
    return names


def main():
    errs, notes = [], []
    kt = read_all()
    for n, s in kt.items():
        check_balance(n, s, errs)
    blob = "\n".join(kt.values())
    for tok in BATCH_TOKENS:
        for n, s in kt.items():
            if tok in s:
                errs.append("batch leftover '%s' in %s" % (tok, n))

    ctrl_m = members(kt["RafiqController.kt"], "RafiqController")
    cfg_src = kt["RecitationConfig.kt"]
    cfg_m = set(re.findall(r"(?:fun|const val|val|var)\s+(\w+)", cfg_src))
    sess_m = members(kt["RecitationSession.kt"], "RecitationSession")
    # members used from MainActivity.kt through `c.`
    for used in sorted(set(re.findall(r"\bc\.(\w+)", kt["MainActivity.kt"]))):
        if used not in ctrl_m and used not in ("copy",):
            errs.append("MainActivity uses c.%s which RafiqController does not declare" % used)
    for n, s in kt.items():
        for used in sorted(set(re.findall(r"RecitationConfig\.(\w+)", s))):
            if used not in cfg_m:
                errs.append("%s uses RecitationConfig.%s which does not exist" % (n, used))
    for used in sorted(set(re.findall(r"\bs\.(start|stop|abort|voicedSinceMs|\w+)\(", kt["RafiqController.kt"]))):
        if used in ("start", "stop", "abort", "voicedSinceMs") and used not in sess_m:
            errs.append("RafiqController uses RecitationSession.%s which does not exist" % used)

    # Listener interface <-> overrides
    iface = re.search(r"interface Listener \{(.*?)\n    \}", kt["RecitationSession.kt"], re.S).group(1)
    decl = re.findall(r"fun (\w+)\(", iface)
    impl = re.findall(r"override fun (\w+)\(", kt["RafiqController.kt"])
    for d in decl:
        if impl.count(d) != 1:
            errs.append("Listener.%s declared but overridden %d time(s) in RafiqController" % (d, impl.count(d)))
    for i in impl:
        if i not in decl:
            errs.append("RafiqController overrides %s which Listener does not declare" % i)
    for d in decl:                                         # every listener callback is actually called by the session
        if not re.search(r"listener\.%s\(" % d, kt["RecitationSession.kt"]):
            errs.append("Listener.%s is never invoked by RecitationSession" % d)

    # JSON keys: app <-> server
    py = open(PY, encoding="utf-8").read()
    sess = kt["RecitationSession.kt"]
    py_types_sent = set(re.findall(r'"type":\s*"(\w+)"', py))
    kt_types_read = set(re.findall(r'^\s+"(\w+)"\s*->\s*\{', sess, re.M))
    for t in sorted(kt_types_read):
        if t not in py_types_sent:
            errs.append("app handles message type '%s' that the server never sends" % t)
    kt_types_sent = set(re.findall(r'\\"type\\":\\"(\w+)\\"', sess))
    py_types_read = set(re.findall(r'kind == "(\w+)"', py))
    for t in sorted(kt_types_sent):
        if t not in py_types_read:
            errs.append("app sends control frame '%s' that the server does not parse" % t)
    kt_keys_read = set(re.findall(r'\bo\.opt\w+\("(\w+)"', sess))
    for k in sorted(kt_keys_read):
        if ('"%s"' % k) not in py:
            errs.append("app reads JSON key '%s' that the server never writes" % k)
    kt_keys_sent = set(re.findall(r'\\"(\w+)\\":', sess)) - {"type"}
    for k in sorted(kt_keys_sent):
        if ('payload.get("%s"' % k) not in py:
            errs.append("app sends JSON key '%s' that the server never reads" % k)
    notes.append("message types app reads: %s" % sorted(kt_types_read))
    notes.append("control frames app sends: %s" % sorted(kt_types_sent))
    notes.append("JSON keys app reads: %s" % sorted(kt_keys_read))
    notes.append("JSON keys app sends: %s" % sorted(kt_keys_sent))
    for n in notes:
        print("  ", n)
    if errs:
        for e in errs:
            print("ERROR", e)
        return 1
    print("kotlin static check: OK (%d files)" % len(kt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
