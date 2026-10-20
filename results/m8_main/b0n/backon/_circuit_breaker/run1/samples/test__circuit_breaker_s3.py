import asyncio
import threading

import pytest

from backon._circuit_breaker import (
    BreakerRetrying,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitOpenError,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeClock:
    """Monotonic clock we can advance manually."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr("backon._circuit_breaker.time.monotonic", fake)
    return fake


def make_breaker(clock=None, **kwargs) -> CircuitBreaker:
    return CircuitBreaker(**kwargs)


def fail():
    raise RuntimeError("boom")


def succeed(value=42):
    return value


# ---------------------------------------------------------------------------
# CircuitBreakerState / CircuitOpenError
# ---------------------------------------------------------------------------

def test_state_enum_members():
    assert CircuitBreakerState.CLOSED.value == "CLOSED"
    assert CircuitBreakerState.OPEN.value == "OPEN"
    assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"
    assert {s.name for s in CircuitBreakerState} == {"CLOSED", "OPEN", "HALF_OPEN"}


def test_circuit_open_error_is_exception():
    err = CircuitOpenError("name")
    assert isinstance(err, Exception)
    assert err.args == ("name",)


# ---------------------------------------------------------------------------
# CircuitBreaker construction / basic props
# ---------------------------------------------------------------------------

def test_initial_state():
    cb = CircuitBreaker()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 0
    assert cb.name == ""


def test_name_property():
    cb = CircuitBreaker(name="my-breaker")
    assert cb.name == "my-breaker"


# ---------------------------------------------------------------------------
# record_success / record_failure
# ---------------------------------------------------------------------------

def test_record_success_resets_failure_and_counts():
    cb = CircuitBreaker()
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.success_count == 1
    cb.record_success()
    assert cb.success_count == 2


def test_record_failure_increments_and_resets_success(clock):
    cb = CircuitBreaker(failure_threshold=10)
    cb.record_success()
    cb.record_success()
    assert cb.success_count == 2
    cb.record_failure()
    assert cb.failure_count == 1
    assert cb.success_count == 0
    assert cb.state is CircuitBreakerState.CLOSED


def test_record_failure_opens_at_threshold(clock):
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    assert cb.failure_count == 3


# ---------------------------------------------------------------------------
# OPEN -> HALF_OPEN transition via recovery timeout
# ---------------------------------------------------------------------------

def test_open_does_not_recover_before_timeout(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    clock.advance(9.99)
    assert cb.state is CircuitBreakerState.OPEN


def test_open_recovers_to_half_open_after_timeout(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    clock.advance(10.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    # half_open_calls reset
    assert cb._half_open_calls == 0


# ---------------------------------------------------------------------------
# record_success from HALF_OPEN -> CLOSED
# ---------------------------------------------------------------------------

def test_record_success_closes_from_half_open(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    clock.advance(5.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# CircuitBreaker.call
# ---------------------------------------------------------------------------

def test_call_success_returns_result_and_records():
    cb = CircuitBreaker()
    result = cb.call(succeed, 7)
    assert result == 7
    assert cb.success_count == 1
    assert cb.failure_count == 0
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_passes_args_kwargs():
    cb = CircuitBreaker()

    def fn(a, b, *, c):
        return a + b + c

    assert cb.call(fn, 1, 2, c=3) == 6


def test_call_failure_reraises_and_counts():
    cb = CircuitBreaker(failure_threshold=5)
    with pytest.raises(RuntimeError, match="boom"):
        cb.call(fail)
    assert cb.failure_count == 1
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_opens_circuit_after_threshold():
    cb = CircuitBreaker(failure_threshold=2)
    with pytest.raises(RuntimeError):
        cb.call(fail)
    with pytest.raises(RuntimeError):
        cb.call(fail)
    assert cb.state is CircuitBreakerState.OPEN
    with pytest.raises(CircuitOpenError) as excinfo:
        cb.call(succeed)
    assert excinfo.value.args == ("",)


def test_call_open_raises_circuit_open_with_name(clock):
    cb = CircuitBreaker(failure_threshold=1, name="svc")
    with pytest.raises(RuntimeError):
        cb.call(fail)
    assert cb.state is CircuitBreakerState.OPEN
    with pytest.raises(CircuitOpenError) as excinfo:
        cb.call(succeed)
    assert excinfo.value.args == ("svc",)


def test_call_half_open_allows_up_to_max(clock):
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
    )
    with pytest.raises(RuntimeError):
        cb.call(fail)
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb.call(succeed, 5) == 5
    # success closed it
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_half_open_rejects_above_max(clock):
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
    )
    with pytest.raises(RuntimeError):
        cb.call(fail)
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    # first call in half-open: bump counter, then fail to stay open but
    # keep half_open_calls consumed.  Actually a failure re-opens circuit,
    # so instead consume via a slow-ish check: simulate by manually
    # bumping counter to its max and verifying subsequent raise.
    cb._half_open_calls = cb._half_open_max_calls
    with pytest.raises(CircuitOpenError):
        cb.call(succeed)


# ---------------------------------------------------------------------------
# CircuitBreaker.async_call
# ---------------------------------------------------------------------------

def test_async_call_success():
    cb = CircuitBreaker()

    async def afn(x):
        return x * 2

    result = asyncio.run(cb.async_call(afn, 21))
    assert result == 42
    assert cb.success_count == 1


def test_async_call_failure_reraises():
    cb = CircuitBreaker(failure_threshold=5)

    async def afn():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        asyncio.run(cb.async_call(afn))
    assert cb.failure_count == 1


def test_async_call_open_raises(clock):
    cb = CircuitBreaker(failure_threshold=1, name="async-name")

    async def afn():
        raise ValueError("x")

    with pytest.raises(ValueError):
        asyncio.run(cb.async_call(afn))
    assert cb.state is CircuitBreakerState.OPEN

    async def ok():
        return 1

    with pytest.raises(CircuitOpenError) as excinfo:
        asyncio.run(cb.async_call(ok))
    assert excinfo.value.args == ("async-name",)


def test_async_call_half_open_limit(clock):
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
    )

    async def fail_afn():
        raise ValueError("x")

    with pytest.raises(ValueError):
        asyncio.run(cb.async_call(fail_afn))
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = cb._half_open_max_calls

    async def ok():
        return 1

    with pytest.raises(CircuitOpenError):
        asyncio.run(cb.async_call(ok))


def test_async_call_half_open_success_closes(clock):
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
    )

    async def fail_afn():
        raise ValueError("x")

    with pytest.raises(ValueError):
        asyncio.run(cb.async_call(fail_afn))
    clock.advance(1.0)

    async def ok():
        return "yes"

    assert asyncio.run(cb.async_call(ok)) == "yes"
    assert cb.state is CircuitBreakerState.CLOSED


# ---------------------------------------------------------------------------
# __call__ and context managers
# ---------------------------------------------------------------------------

def test_dunder_call_delegates_to_call():
    cb = CircuitBreaker()
    assert cb(succeed, 3) == 3
    assert cb.success_count == 1


def test_sync_context_manager():
    cb = CircuitBreaker()
    with cb as entered:
        assert entered is cb


def test_async_context_manager():
    cb = CircuitBreaker()

    async def use():
        async with cb as entered:
            return entered

    assert asyncio.run(use()) is cb


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------

def test_breaker_retrying_defaults():
    br = BreakerRetrying()
    assert isinstance(br.breaker, CircuitBreaker)
    assert br.breaker.name == ""


def test_breaker_retrying_custom_breaker():
    cb = CircuitBreaker(name="custom")
    br = BreakerRetrying(breaker=cb)
    assert br.breaker is cb


def test_breaker_retrying_call_success():
    br = BreakerRetrying(max_tries=1)
    assert br.call(succeed, 11) == 11
    assert br.breaker.success_count == 1
    assert br.breaker.state is CircuitBreakerState.CLOSED


def test_breaker_retrying_call_failure_records_and_raises():
    br = BreakerRetrying(max_tries=1)
    with pytest.raises(RuntimeError, match="boom"):
        br.call(fail)
    assert br.breaker.failure_count == 1


def test_breaker_retrying_opens_circuit(clock):
    cb = CircuitBreaker(failure_threshold=1, name="br")
    br = BreakerRetrying(breaker=cb, max_tries=1)
    with pytest.raises(RuntimeError):
        br.call(fail)
    assert cb.state is CircuitBreakerState.OPEN
    with pytest.raises(CircuitOpenError) as excinfo:
        br.call(succeed)
    assert excinfo.value.args == ("br",)


def test_breaker_retrying_half_open_limit(clock):
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
    )
    br = BreakerRetrying(breaker=cb, max_tries=1)
    with pytest.raises(RuntimeError):
        br.call(fail)
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = cb._half_open_max_calls
    with pytest.raises(CircuitOpenError):
        br.call(succeed)


def test_breaker_retrying_async_call_success():
    br = BreakerRetrying(max_tries=1)

    async def afn(x):
        return x + 1

    assert asyncio.run(br.async_call(afn, 1)) == 2
    assert br.breaker.success_count == 1


def test_breaker_retrying_async_call_failure():
    br = BreakerRetrying(max_tries=1)

    async def afn():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        asyncio.run(br.async_call(afn))
    assert br.breaker.failure_count == 1


def test_breaker_retrying_async_open(clock):
    cb = CircuitBreaker(failure_threshold=1, name="async-br")
    br = BreakerRetrying(breaker=cb, max_tries=1)

    async def bad():
        raise ValueError("x")

    with pytest.raises(ValueError):
        asyncio.run(br.async_call(bad))
    assert cb.state is CircuitBreakerState.OPEN

    async def ok():
        return 1

    with pytest.raises(CircuitOpenError) as excinfo:
        asyncio.run(br.async_call(ok))
    assert excinfo.value.args == ("async-br",)


# ---------------------------------------------------------------------------
# Thread-safety-ish sanity check
# ---------------------------------------------------------------------------

def test_concurrent_record_failures_opens():
    cb = CircuitBreaker(failure_threshold=100)

    def worker():
        for _ in range(10):
            cb.record_failure()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cb.failure_count == 100
    assert cb.state is CircuitBreakerState.OPEN
