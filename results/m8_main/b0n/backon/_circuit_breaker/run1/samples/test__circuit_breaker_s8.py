import asyncio
import threading
import time

import pytest

from backon._circuit_breaker import (
    BreakerRetrying,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitOpenError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClock:
    """Controllable replacement for time.monotonic."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr("backon._circuit_breaker.time.monotonic", c)
    return c


def boom(*args, **kwargs):
    raise ValueError("boom")


def ok(value="ok"):
    return value


async def async_ok(value="ok"):
    return value


async def async_boom(*args, **kwargs):
    raise ValueError("async boom")


# ---------------------------------------------------------------------------
# Enum / exception
# ---------------------------------------------------------------------------

def test_state_enum_members():
    assert CircuitBreakerState.CLOSED.value == "CLOSED"
    assert CircuitBreakerState.OPEN.value == "OPEN"
    assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"
    assert {s.name for s in CircuitBreakerState} == {"CLOSED", "OPEN", "HALF_OPEN"}


def test_circuit_open_error_is_exception():
    assert issubclass(CircuitOpenError, Exception)
    err = CircuitOpenError("my-breaker")
    assert str(err) == "my-breaker"


# ---------------------------------------------------------------------------
# CircuitBreaker: constructor / properties
# ---------------------------------------------------------------------------

def test_defaults():
    cb = CircuitBreaker()
    assert cb.name == ""
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_custom_name_and_params():
    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=10.0,
        half_open_max_calls=2,
        name="svc",
    )
    assert cb.name == "svc"
    assert cb._failure_threshold == 3
    assert cb._recovery_timeout == 10.0
    assert cb._half_open_max_calls == 2


# ---------------------------------------------------------------------------
# record_success / record_failure
# ---------------------------------------------------------------------------

def test_record_success_resets_failures_increments_successes():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    assert cb.success_count == 0

    cb.record_success()
    assert cb.failure_count == 0
    assert cb.success_count == 1

    cb.record_success()
    assert cb.success_count == 2


def test_record_failure_opens_at_threshold(clock):
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.failure_count == 3
    assert cb.state == CircuitBreakerState.OPEN


def test_record_failure_resets_success_count(clock):
    cb = CircuitBreaker()
    cb.record_success()
    cb.record_success()
    assert cb.success_count == 2
    cb.record_failure()
    assert cb.success_count == 0


# ---------------------------------------------------------------------------
# State transitions via time
# ---------------------------------------------------------------------------

def test_open_transitions_to_half_open_after_timeout(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(4.999)
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(0.001)
    assert cb.state == CircuitBreakerState.HALF_OPEN


def test_half_open_transition_resets_half_open_calls(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=5)
    cb.record_failure()
    cb._half_open_calls = 42
    clock.advance(2.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0


def test_record_success_in_half_open_closes(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED


def test_record_failure_in_half_open_reopens(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# call()
# ---------------------------------------------------------------------------

def test_call_success_returns_result_and_counts():
    cb = CircuitBreaker()
    assert cb.call(lambda a, b: a + b, 2, 3) == 5
    assert cb.success_count == 1
    assert cb.failure_count == 0
    assert cb.state == CircuitBreakerState.CLOSED


def test_call_passes_kwargs():
    cb = CircuitBreaker()
    assert cb.call(ok, value="hi") == "hi"


def test_call_failure_records_and_reraises(clock):
    cb = CircuitBreaker(failure_threshold=2)
    with pytest.raises(ValueError, match="boom"):
        cb.call(boom)
    assert cb.failure_count == 1
    assert cb.success_count == 0
    assert cb.state == CircuitBreakerState.CLOSED


def test_call_opens_after_threshold(clock):
    cb = CircuitBreaker(failure_threshold=2)
    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(boom)
    assert cb.state == CircuitBreakerState.OPEN


def test_call_raises_circuit_open_when_open(clock):
    cb = CircuitBreaker(failure_threshold=1, name="mycb")
    with pytest.raises(ValueError):
        cb.call(boom)
    assert cb.state == CircuitBreakerState.OPEN

    called = []
    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(lambda: called.append(1))
    assert exc_info.value.args[0] == "mycb"
    assert called == []


def test_call_half_open_limits_calls(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1)
    with pytest.raises(ValueError):
        cb.call(boom)
    clock.advance(2.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # First call in half-open allowed -> success closes it.
    assert cb.call(ok) == "ok"
    assert cb.state == CircuitBreakerState.CLOSED


def test_call_half_open_exhaustion_raises(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1)
    with pytest.raises(ValueError):
        cb.call(boom)
    clock.advance(2.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Manually consume the half-open slot without changing state.
    cb._half_open_calls = 1
    with pytest.raises(CircuitOpenError):
        cb.call(ok)


def test_call_half_open_failure_reopens(clock):
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0, half_open_max_calls=5)
    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(boom)
    clock.advance(2.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    with pytest.raises(ValueError):
        cb.call(boom)
    assert cb.state == CircuitBreakerState.OPEN


def test_dunder_call_delegates():
    cb = CircuitBreaker()
    assert cb(lambda x: x * 2, 4) == 8
    assert cb.success_count == 1


def test_context_manager_sync():
    cb = CircuitBreaker()
    with cb as entered:
        assert entered is cb


# ---------------------------------------------------------------------------
# async_call()
# ---------------------------------------------------------------------------

def test_async_call_success():
    cb = CircuitBreaker()
    result = asyncio.run(cb.async_call(async_ok, value="v"))
    assert result == "v"
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_async_call_failure_records(clock):
    cb = CircuitBreaker(failure_threshold=1)
    with pytest.raises(ValueError, match="async boom"):
        asyncio.run(cb.async_call(async_boom))
    assert cb.failure_count == 1
    assert cb.state == CircuitBreakerState.OPEN


def test_async_call_open_raises(clock):
    cb = CircuitBreaker(failure_threshold=1, name="asynccb")
    asyncio.run(cb.async_call(async_boom)) if False else None
    with pytest.raises(ValueError):
        asyncio.run(cb.async_call(async_boom))
    assert cb.state == CircuitBreakerState.OPEN

    async def should_not_run():
        raise AssertionError("should not be called")

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(cb.async_call(should_not_run))
    assert exc_info.value.args[0] == "asynccb"


def test_async_call_half_open_limiting(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1)
    with pytest.raises(ValueError):
        asyncio.run(cb.async_call(async_boom))
    clock.advance(2.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    cb._half_open_calls = 1
    with pytest.raises(CircuitOpenError):
        asyncio.run(cb.async_call(async_ok))


def test_async_context_manager():
    async def run():
        cb = CircuitBreaker()
        async with cb as entered:
            assert entered is cb

    asyncio.run(run())


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------

def test_breaker_retrying_default_breaker():
    br = BreakerRetrying(max_tries=1)
    assert isinstance(br.breaker, CircuitBreaker)


def test_breaker_retrying_uses_provided_breaker():
    cb = CircuitBreaker(name="x")
    br = BreakerRetrying(breaker=cb, max_tries=1)
    assert br.breaker is cb


def test_breaker_retrying_call_success():
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, max_tries=1)
    assert br.call(lambda a, b: a * b, 3, 4) == 12
    assert cb.success_count == 1


def test_breaker_retrying_call_retries_then_succeeds(clock):
    cb = CircuitBreaker(failure_threshold=10)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("transient")
        return "done"

    br = BreakerRetrying(
        breaker=cb,
        max_tries=3,
        sleep=lambda _: None,
        jitter=None,
        exception=ValueError,
    )
    assert br.call(flaky) == "done"
    assert calls["n"] == 2
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_breaker_retrying_call_failure_records(clock):
    cb = CircuitBreaker(failure_threshold=1, name="failcb")
    br = BreakerRetrying(
        breaker=cb,
        max_tries=1,
        exception=ValueError,
        raise_on_giveup=True,
    )
    with pytest.raises(ValueError):
        br.call(boom)
    assert cb.failure_count == 1
    assert cb.state == CircuitBreakerState.OPEN


def test_breaker_retrying_open_raises(clock):
    cb = CircuitBreaker(failure_threshold=1, name="openbr")
    br = BreakerRetrying(breaker=cb, max_tries=1, exception=ValueError)
    with pytest.raises(ValueError):
        br.call(boom)
    assert cb.state == CircuitBreakerState.OPEN

    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(lambda: 1)
    assert exc_info.value.args[0] == "openbr"


def test_breaker_retrying_half_open_exhaustion(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1)
    br = BreakerRetrying(breaker=cb, max_tries=1, exception=ValueError)
    with pytest.raises(ValueError):
        br.call(boom)
    clock.advance(2.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1
    with pytest.raises(CircuitOpenError):
        br.call(lambda: 1)


def test_breaker_retrying_async_call_success():
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, max_tries=1)
    result = asyncio.run(br.async_call(async_ok, value="y"))
    assert result == "y"
    assert cb.success_count == 1


def test_breaker_retrying_async_call_failure(clock):
    cb = CircuitBreaker(failure_threshold=1, name="abr")
    br = BreakerRetrying(breaker=cb, max_tries=1, exception=ValueError)
    with pytest.raises(ValueError):
        asyncio.run(br.async_call(async_boom))
    assert cb.failure_count == 1
    assert cb.state == CircuitBreakerState.OPEN


def test_breaker_retrying_async_open_raises(clock):
    cb = CircuitBreaker(failure_threshold=1, name="abr2")
    br = BreakerRetrying(breaker=cb, max_tries=1, exception=ValueError)
    with pytest.raises(ValueError):
        asyncio.run(br.async_call(async_boom))
    assert cb.state == CircuitBreakerState.OPEN

    async def should_not_run():
        raise AssertionError("nope")

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(br.async_call(should_not_run))
    assert exc_info.value.args[0] == "abr2"


def test_breaker_retrying_async_half_open_exhaustion(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1)
    br = BreakerRetrying(breaker=cb, max_tries=1, exception=ValueError)
    with pytest.raises(ValueError):
        asyncio.run(br.async_call(async_boom))
    clock.advance(2.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1
    with pytest.raises(CircuitOpenError):
        asyncio.run(br.async_call(async_ok))
