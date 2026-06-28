from graph_layout_rag.harvest.providers import (
    OpenAlexPolicy,
    OutcomeKind,
    ProviderPolicy,
    RequestOutcome,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_openalex_free_budget_reservation_stops_metered_but_not_singletons():
    policy = OpenAlexPolicy(concurrency=1, rps=100, clock=lambda: 0.0, sleep=lambda _: None)
    policy._daily_budget = 0.001
    policy.metrics.budget_remaining_usd = 0.001
    assert policy.reserve("search")
    assert not policy.reserve("list")
    assert policy.reserve("singleton")


def _key_recording_policy(outcomes_by_key):
    """OpenAlexPolicy whose .request() records the api_key and returns a
    canned outcome per key, without doing real HTTP."""
    policy = OpenAlexPolicy(concurrency=1, rps=100, clock=lambda: 0.0, sleep=lambda _: None)
    tried = []

    def fake_request(method, url, *, params=None, before_attempt=None, **kwargs):
        key = (params or {}).get("api_key")
        tried.append(key)
        # honour the budget reservation so the client-side cap path still works
        if before_attempt is not None and not before_attempt():
            return RequestOutcome(OutcomeKind.BUDGET_EXHAUSTED)
        return outcomes_by_key[key]

    policy.request = fake_request  # type: ignore[method-assign]
    return policy, tried


def test_openalex_rotates_to_next_key_on_budget_429(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEYS", "KEY_A,KEY_B")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    policy, tried = _key_recording_policy(
        {
            # KEY_A: OpenAlex "Insufficient budget" → 429 with multi-hour Retry-After
            "KEY_A": RequestOutcome(OutcomeKind.RATE_LIMITED, status_code=429, retry_after=7200),
            # KEY_B: healthy
            "KEY_B": RequestOutcome(OutcomeKind.SUCCESS, data={"ok": True}, status_code=200),
        }
    )
    out = policy.request_openalex("GET", "https://api.openalex.org/works", operation="search")
    assert out.kind is OutcomeKind.SUCCESS
    assert tried == ["KEY_A", "KEY_B"]
    # KEY_A retired; a subsequent call goes straight to KEY_B
    tried.clear()
    policy.request_openalex("GET", "https://api.openalex.org/works", operation="search")
    assert tried == ["KEY_B"]


def test_openalex_returns_last_outcome_when_all_keys_depleted(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEYS", "KEY_A,KEY_B")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    depleted = RequestOutcome(OutcomeKind.RATE_LIMITED, status_code=429, retry_after=7200)
    policy, tried = _key_recording_policy({"KEY_A": depleted, "KEY_B": depleted})
    out = policy.request_openalex("GET", "https://api.openalex.org/works", operation="search")
    assert out.kind is OutcomeKind.RATE_LIMITED
    assert tried == ["KEY_A", "KEY_B"]  # both tried once, then give up (no infinite loop)


def test_openalex_no_keys_passes_through(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEYS", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    policy, tried = _key_recording_policy({None: RequestOutcome(OutcomeKind.SUCCESS, status_code=200)})
    out = policy.request_openalex("GET", "https://api.openalex.org/works", operation="search")
    assert out.kind is OutcomeKind.SUCCESS
    assert tried == [None]


def test_provider_circuit_waits_then_resets():
    clock = FakeClock()
    policy = ProviderPolicy(
        "test", concurrency=1, rps=0, cooldown_seconds=5, clock=clock.clock, sleep=clock.sleep
    )
    policy._open_circuit(5)
    policy._wait_for_slot()
    assert clock.now == 5
    assert policy.metrics.cooldown_seconds == 5


def test_sub_one_rps_policy_paces_requests():
    clock = FakeClock()
    policy = ProviderPolicy(
        "slow", concurrency=1, rps=10 / 60, clock=clock.clock, sleep=clock.sleep
    )
    policy._wait_for_slot()
    policy._wait_for_slot()
    assert clock.now == 6
