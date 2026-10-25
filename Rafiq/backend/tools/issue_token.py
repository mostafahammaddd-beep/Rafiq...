"""
Issue (or help create the secret for) Rafiq access tokens.

The signing secret is read from the ENVIRONMENT, never from the command line (shell history, process lists):

    # once: create a secret and store it where the server reads it (Modal example)
    python tools/issue_token.py --new-secret
    modal secret create rafiq-auth RAFIQ_AUTH_SECRET=<that value> RAFIQ_ADMIN_TOKEN=<another --new-secret value>

    # per learner / teacher / device (the value of RAFIQ_AUTH_SECRET must equal the one stored in Modal)
    export RAFIQ_AUTH_SECRET=<that value>            # Windows cmd:  set RAFIQ_AUTH_SECRET=<that value>
    python tools/issue_token.py --sub student-017 --days 30

Give the printed token to that person; they paste it in the app under  ⚙ -> رمز الدخول.
A token is a bearer credential: anyone holding it can use the server until it expires. Use short lifetimes for
children's devices, one subject per person (the server limits each subject to 2 concurrent sessions), and rotate
RAFIQ_AUTH_SECRET to revoke every token at once. There is no per-token revocation list (see docs/SECURITY.md).
"""
import argparse
import os
import secrets
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_app as m  # noqa: E402  (works without Modal installed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-secret", action="store_true", help="print a fresh random secret (for RAFIQ_AUTH_SECRET / RAFIQ_ADMIN_TOKEN) and exit")
    ap.add_argument("--sub", help="subject: who the token is for, [A-Za-z0-9_.:@-]{1,64} (no real names needed)")
    ap.add_argument("--days", type=float, default=30.0, help="lifetime in days (default 30, max 366)")
    a = ap.parse_args()
    if a.new_secret:
        print(secrets.token_urlsafe(48))
        return 0
    secret = os.environ.get(m.ENV_AUTH_SECRET, "").strip()
    if not secret:
        print("error: set %s in the environment first (same value as the server's)." % m.ENV_AUTH_SECRET, file=sys.stderr)
        return 2
    if not a.sub:
        print("error: --sub is required", file=sys.stderr)
        return 2
    try:
        print(m.issue_token(secret, a.sub, int(a.days * 86400)))
    except ValueError as exc:
        print("error:", exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
