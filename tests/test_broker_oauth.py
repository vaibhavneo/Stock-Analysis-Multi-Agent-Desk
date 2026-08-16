"""
Verification for broker/oauth.py — PKCE + authorize-URL construction.

Run: python3 tests/test_broker_oauth.py

Offline/deterministic. No network calls — register_client/exchange_code_for_token/
refresh_access_token are HTTP calls to Robinhood and are exercised live by
broker/register_client.py and broker/discover_tools.py instead (a live OAuth
flow can't be faked meaningfully offline; see M2 in the plan for how those
were actually verified against the real server).
"""
import base64
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from broker.oauth import build_authorize_url, generate_pkce_pair, generate_state

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:58s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def test_pkce_rfc7636_vector():
    """RFC 7636 Appendix B's own worked example: an independent recomputation
    of the S256 algorithm (base64url(sha256(verifier)), no padding) must
    match the RFC's published challenge for its published verifier — this
    checks the ALGORITHM this codebase implements is the one the spec
    defines, not just that the code agrees with itself."""
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    expected_challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    check("S256(RFC 7636 test vector) matches published challenge",
          computed == expected_challenge, f"got {computed}")


def test_generate_pkce_pair():
    verifier, challenge = generate_pkce_pair()
    check("verifier length within RFC 7636 bounds [43,128]",
          43 <= len(verifier) <= 128, f"len={len(verifier)}")
    check("verifier/challenge contain no base64 padding",
          "=" not in verifier and "=" not in challenge)
    recomputed = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    check("generate_pkce_pair()'s challenge matches independent recomputation",
          challenge == recomputed)

    v2, c2 = generate_pkce_pair()
    check("two calls produce different verifiers (not deterministic/reused)",
          verifier != v2 and challenge != c2)


def test_generate_state():
    s1, s2 = generate_state(), generate_state()
    check("state values differ across calls", s1 != s2)
    check("state has no obviously-guessable length", len(s1) >= 16, f"len={len(s1)}")


def test_build_authorize_url():
    url = build_authorize_url("client-abc", "https://example.com/callback",
                               "state-xyz", "challenge-123")
    check("uses Robinhood's real authorization endpoint",
          url.startswith("https://robinhood.com/oauth?"))
    check("carries response_type=code", "response_type=code" in url)
    check("carries the client_id", "client_id=client-abc" in url)
    check("carries code_challenge_method=S256", "code_challenge_method=S256" in url)
    check("carries the code_challenge", "code_challenge=challenge-123" in url)
    check("carries the state", "state=state-xyz" in url)
    check("redirect_uri is present (URL-encoded)", "redirect_uri=" in url)


if __name__ == "__main__":
    test_pkce_rfc7636_vector()
    test_generate_pkce_pair()
    test_generate_state()
    test_build_authorize_url()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — broker.oauth: PKCE + authorize URL")
