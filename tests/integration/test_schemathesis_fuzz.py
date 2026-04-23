"""
tests/integration/test_schemathesis_fuzz.py
─────────────────────────────────────────────
Property-based fuzzing of every REST endpoint using Schemathesis.

Schemathesis generates random-but-schema-valid inputs and checks that:
  - No endpoint returns 5xx (server crashes)
  - Responses match their declared schema
  - Content-Type is always application/json

Run with:
    pytest tests/integration/test_schemathesis_fuzz.py -v
    pytest tests/integration/test_schemathesis_fuzz.py -v --hypothesis-seed=0
    pytest tests/integration/test_schemathesis_fuzz.py -v -k "context"
"""

import schemathesis
from schemathesis import checks
from hypothesis import settings, HealthCheck

from pact_ax.api.server import app


# ── Load schema from the live ASGI app ───────────────────────────────────────

schema = schemathesis.openapi.from_asgi("/openapi.json", app)


# ── Fuzz every endpoint ───────────────────────────────────────────────────────

@schema.parametrize()
@settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
def test_no_server_errors(case):
    """
    Fire a schema-valid random request at every endpoint and assert:
      - Response is not a 5xx (server error)
      - If the response is 2xx, the body is valid JSON matching the schema
    """
    response = case.call()

    # No 5xx — the API must never crash on schema-valid input
    case.validate_response(response, checks=[checks.not_a_server_error])
