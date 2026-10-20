import asyncio
import threading
from unittest import mock

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
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr("backon._circuit_breaker.time.monotonic", fake.monotonic)
    yield fake


def fail(*args, **kwargs):
    raise ValueError("boom")


def succeed(*args, **kwargs):
    return "ok"


# ---------------------------------------------------------------------------
# CircuitBreakerState / CircuitOpenError
# ---------------------------------------------------------------------------

def test_states_are_distinct_enum_members():
    assert CircuitBreakerState.CLOSED.value == "CLOSED"
    assert CircuitBreakerState.OPEN.value == "OPEN"
    assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"
    assert len(CircuitBreakerState) == 3


def test_circuit_open_error_is_exception():
    assert issubclass(CircuitOpenError, Exception)
    err = CircuitOpenError("myname")
    assert err.args == ("myname",)


# ---------------------------------------------------------------------------
# CircuitBreaker construction / simple properties
# ---------------------------------------------------------------------------

def test_defaults():
    cb = CircuitBreaker()
    assert cb.name == ""
    assert cb.failure_count == 0
    assert cb.success_count == 0
    assert cb.state is CircuitBreakerState.CLOSED


def test_custom_name_and_configuration_stored():
    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=12.5,
        half_open_max_calls=4,
        name="svc",
    )
    assert cb.name == "svc"
    assert cb._failure_threshold == 3
    assert cb._recovery_timeout == 12.5
    assert cb._half_open_max_calls == 4


# ---------------------------------------------------------------------------
# record_success / record_failure
# ---------------------------------------------------------------------------

def test_record_failure_increments_and_resets_successes():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_success()
    cb.record_success()
    assert cb.success_count == 2
    cb.record_failure()
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_record_success_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.success_count == 1


def test_failure_threshold_opens_circuit(clock):
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(2):
        cb.record_failure()
    assert cb.state is CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    assert cb.failure_count == 3


def test_record_failure_sets_last_failure_time(clock):
    cb = CircuitBreaker(failure_threshold=10)
    clock.now = 1234.5
    cb.record_failure()
    assert cb._last_failure_time == 1234.5


# ---------------------------------------------------------------------------
# state transitions and recovery timeout
# ---------------------------------------------------------------------------

def test_open_transitions_to_half_open_after_timeout(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN

    clock.advance(9.999)
    assert cb.state is CircuitBreakerState.OPEN

    clock.now += 0.0
    clock.advance(0.001)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0


def test_half_open_resets_half_open_calls(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0,
                        half_open_max_calls=5)
    cb.record_failure()
    cb._half_open_calls = 5  # simulate leaked counter
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0


def test_success_in_half_open_closes_circuit(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 1


def test_failure_in_half_open_reopens(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb.record_failure()
    # failure_count >= threshold triggers OPEN again
    assert cb.state is CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# call()
# ---------------------------------------------------------------------------

def test_call_returns_result_and_records_success():
    cb = CircuitBreaker()
    assert cb.call(lambda x, y: x + y, 2, 3) == 5
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_call_passes_kwargs():
    cb = CircuitBreaker()
    assert cb.call(lambda **kw: kw["a"], a=42) == 42


def test_call_propagates_exception_and_records_failure():
    cb = CircuitBreaker(failure_threshold=5)
    with pytest.raises(ValueError, match="boom"):
        cb.call(fail)
    assert cb.failure_count == 1


def test_call_opens_after_threshold_then_raises_circuit_open(clock):
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=100.0, name="n")
    with pytest.raises(ValueError):
        cb.call(fail)
    with pytest.raises(ValueError):
        cb.call(fail)
    assert cb.state is CircuitBreakerState.OPEN

    with pytest.raises(CircuitOpenError) as excinfo:
        cb.call(succeed)
    assert excinfo.value.args == ("n",)


def test_call_in_half_open_within_max_calls():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0,
                        half_open_max_calls=2)
    cb.record_failure()  # opens; recovery_timeout 0 -> immediate half-open
    assert cb.state is CircuitBreakerState.HALF_OPEN

    assert cb.call(succeed) == "ok"
    # after one success in half-open, state goes CLOSED
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_in_half_open_exceeds_max_calls_raises(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0,
                        half_open_max_calls=1, name="half")
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    # Consume the single half-open slot with a failing call so state
    # reopens... but to test the max-calls branch directly, manipulate
    # the counter and keep state HALF_OPEN.
    with cb._lock:
        cb._half_open_calls = 1

    with pytest.raises(CircuitOpenError) as excinfo:
        cb.call(succeed)
    assert excinfo.value.args == ("half",)


def test_call_failure_recorded_in_half_open_reopens(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0,
                        half_open_max_calls=5)
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    with pytest.raises(ValueError):
        cb.call(fail)
    assert cb.state is CircuitBreakerState.OPEN


def test_dunder_call_delegates():
    cb = CircuitBreaker()
    assert cb(lambda x: x * 2, 4) == 8
    assert cb.success_count == 1


# ---------------------------------------------------------------------------
# async_call()
# ---------------------------------------------------------------------------

def test_async_call_returns_and_records_success():
    cb = CircuitBreaker()

    async def coro(x):
        return x + 1

    assert asyncio.run(cb.async_call(coro, 4)) == 5
    assert cb.success_count == 1


def test_async_call_records_failure_and_propagates():
    cb = CircuitBreaker()

    async def coro():
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        asyncio.run(cb.async_call(coro))
    assert cb.failure_count == 1


def test_async_call_open_raises(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, name="as")
    cb.record_failure()

    async def coro():
        return 1

    with pytest.raises(CircuitOpenError) as excinfo:
        asyncio.run(cb.async_call(coro))
    assert excinfo.value.args == ("as",)


def test_async_call_half_open_max_calls(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0,
                        half_open_max_calls=1, name="ho")
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    with cb._lock:
        cb._half_open_calls = 1

    async def coro():
        return 1

    with pytest.raises(CircuitOpenError) as excinfo:
        asyncio.run(cb.async_call(coro))
    assert excinfo.value.args == ("ho",)


# ---------------------------------------------------------------------------
# context managers
# ---------------------------------------------------------------------------

def test_sync_context_manager_returns_self():
    cb = CircuitBreaker()
    with cb as cm:
        assert cm is cb


def test_async_context_manager_returns_self():
    cb = CircuitBreaker()

    async def run():
        async with cb as cm:
            return cm

    assert asyncio.run(run()) is cb


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------

def test_breaker_retrying_creates_default_breaker():
    br = BreakerRetrying(giveup=lambda e: True)
    assert isinstance(br.breaker, CircuitBreaker)
    assert br.breaker.name == ""


def test_breaker_retrying_uses_provided_breaker():
    cb = CircuitBreaker(name="provided")
    br = BreakerRetrying(breaker=cb, giveup=lambda e: True)
    assert br.breaker is cb


def test_breaker_retrying_retries_then_succeeds():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("nope")
        return "done"

    br = BreakerRetrying(
        giveup=lambda e: False,
        exception=(ValueError,),
        max_tries=5,
        sleep=lambda s: None,
        jitter=None,
        giveup_log_level=50,
    )
    assert br.call(target) == "done"
    assert calls["n"] == 3
    # A success resets and counts.
    assert br.breaker.failure_count == 0
    assert br.breaker.success_count == 1


def test_breaker_retrying_records_failure_on_giveup():
    br = BreakerRetrying(
        giveup=lambda e: True,
        exception=(ValueError,),
        max_tries=3,
        sleep=lambda s: None,
        jitter=None,
        raise_on_giveup=True,
    )

    def target():
        raise ValueError("always")

    with pytest.raises(ValueError, match="always"):
        br.call(target)
    assert br.breaker.failure_count == 1


def test_breaker_retrying_open_circuit_raises():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100.0, name="br")
    cb.record_failure()
    br = BreakerRetrying(breaker=cb, giveup=lambda e: True)
    with pytest.raises(CircuitOpenError) as excinfo:
        br.call(lambda: 1)
    assert excinfo.value.args == ("br",)


def test_breaker_retrying_half_open_max_calls(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0,
                        half_open_max_calls=1, name="bh")
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    with cb._lock:
        cb._half_open_calls = 1

    br = BreakerRetrying(breaker=cb, giveup=lambda e: True)
    with pytest.raises(CircuitOpenError) as excinfo:
        br.call(lambda: 1)
    assert excinfo.value.args == ("bh",)


def test_breaker_retrying_success_in_half_open_closes(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0,
                        half_open_max_calls=5)
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    br = BreakerRetrying(breaker=cb, giveup=lambda e: True)
    assert br.call(lambda: "yes") == "yes"
    assert cb.state is CircuitBreakerState.CLOSED


def test_breaker_retrying_async_call_returns_value():
    br = BreakerRetrying(giveup=lambda e: True, sleep=lambda s: None)

    async def target(x):
        return x * 2

    assert asyncio.run(br.async_call(target, 5)) == 10
    assert br.breaker.success_count == 1


def test_breaker_retrying_async_call_failure_records():
    br = BreakerRetrying(
        giveup=lambda e: True,
        exception=(RuntimeError,),
        max_tries=1,
        sleep=lambda s: None,
    )

    async def target():
        raise RuntimeError("bad")

    with pytest.raises(RuntimeError, match="bad"):
        asyncio.run(br.async_call(target))
    assert br.breaker.failure_count == 1


def test_breaker_retrying_async_open_circuit():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100.0, name="ao")
    cb.record_failure()
    br = BreakerRetrying(breaker=cb, giveup=lambda e: True)

    async def target():
        return 1

    with pytest.raises(CircuitOpenError) as excinfo:
        asyncio.run(br.async_call(target))
    assert excinfo.value.args == ("ao",)


def test_breaker_retrying_async_half_open_max_calls(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0,
                        half_open_max_calls=1, name="aho")
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    with cb._lock:
        cb._half_open_calls = 1

    br = BreakerRetrying(breaker=cb, giveup=lambda e: True)

    async def target():
        return 1

    with pytest.raises(CircuitOpenError) as excinfo:
        asyncio.run(br.async_call(target))
    assert excinfo.value.args == ("aho",)
