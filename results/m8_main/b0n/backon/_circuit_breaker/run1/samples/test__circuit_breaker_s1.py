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
    """Patchable monotonic clock used to control the breaker's timing logic."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


def _fail():
    raise ValueError("boom")


def _ok():
    return "ok"


# ---------------------------------------------------------------------------
# CircuitBreakerState / CircuitOpenError
# ---------------------------------------------------------------------------

def test_state_enum_values():
    assert CircuitBreakerState.CLOSED.value == "CLOSED"
    assert CircuitBreakerState.OPEN.value == "OPEN"
    assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"
    assert len(CircuitBreakerState) == 3


def test_circuit_open_error_is_exception():
    err = CircuitOpenError("mybreaker")
    assert isinstance(err, Exception)
    assert str(err) == "mybreaker"


# ---------------------------------------------------------------------------
# CircuitBreaker construction & properties
# ---------------------------------------------------------------------------

def test_defaults():
    cb = CircuitBreaker()
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.name == ""
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_custom_construction():
    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5.0,
        half_open_max_calls=2,
        name="cb",
    )
    assert cb.name == "cb"
    assert cb._failure_threshold == 3
    assert cb._recovery_timeout == 5.0
    assert cb._half_open_max_calls == 2


def test_record_success_increments_consecutive():
    cb = CircuitBreaker()
    cb.record_success()
    cb.record_success()
    assert cb.success_count == 2
    assert cb.failure_count == 0


def test_record_failure_resets_and_counts():
    cb = CircuitBreaker(failure_threshold=10)
    cb.record_success()
    assert cb.success_count == 1
    cb.record_failure()
    assert cb.failure_count == 1
    assert cb.success_count == 0
    # success resets failure count
    cb.record_success()
    assert cb.failure_count == 0


def test_record_failure_opens_at_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.failure_count == 3


# ---------------------------------------------------------------------------
# CircuitBreaker.call
# ---------------------------------------------------------------------------

def test_call_success():
    cb = CircuitBreaker()
    result = cb.call(lambda a, b: a + b, 1, 2)
    assert result == 3
    assert cb.success_count == 1
    assert cb.failure_count == 0
    assert cb.state == CircuitBreakerState.CLOSED


def test_call_dunder_call_delegates():
    cb = CircuitBreaker()
    assert cb(lambda x: x * 2, 5) == 10


def test_call_failure_records_and_reraises():
    cb = CircuitBreaker(failure_threshold=5)
    with pytest.raises(ValueError, match="boom"):
        cb.call(_fail)
    assert cb.failure_count == 1
    assert cb.state == CircuitBreakerState.CLOSED


def test_call_opens_after_threshold_then_rejects(clock):
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0, name="n")
    with pytest.raises(ValueError):
        cb.call(_fail)
    with pytest.raises(ValueError):
        cb.call(_fail)
    assert cb.state == CircuitBreakerState.OPEN

    # Further calls rejected without invoking function
    called = []

    def f():
        called.append(1)
        return 1

    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(f)
    assert exc_info.value.args[0] == "n"
    assert called == []


def test_custom_exception_caught_base_exception():
    cb = CircuitBreaker()

    def raises_base():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        cb.call(raises_base)
    assert cb.failure_count == 1


# ---------------------------------------------------------------------------
# Recovery / half-open transition via state property
# ---------------------------------------------------------------------------

def test_recovery_transitions_to_half_open(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    with pytest.raises(ValueError):
        cb.call(_fail)
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(9.99)
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(0.01)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0


def test_half_open_success_closes(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    with pytest.raises(ValueError):
        cb.call(_fail)
    clock.advance(5.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    assert cb.call(_ok) == "ok"
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 1


def test_half_open_failure_reopens(clock):
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5.0)
    with pytest.raises(ValueError):
        cb.call(_fail)
    with pytest.raises(ValueError):
        cb.call(_fail)
    clock.advance(5.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    with pytest.raises(ValueError):
        cb.call(_fail)
    assert cb.state == CircuitBreakerState.OPEN


def test_half_open_limits_concurrent_calls(clock):
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=5.0,
        half_open_max_calls=1,
    )
    with pytest.raises(ValueError):
        cb.call(_fail)
    clock.advance(5.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # First half-open call body is invoked
    assert cb.call(lambda: "first") == "first"

    # reset to half-open state to test the max-calls gate directly
    cb._state = CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1
    with pytest.raises(CircuitOpenError):
        cb.call(_ok)


def test_half_open_max_calls_multiple(clock):
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=5.0,
        half_open_max_calls=2,
    )
    with pytest.raises(ValueError):
        cb.call(_fail)
    clock.advance(5.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    cb._state = CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 2
    with pytest.raises(CircuitOpenError):
        cb.call(_ok)


# ---------------------------------------------------------------------------
# async_call
# ---------------------------------------------------------------------------

def test_async_call_success():
    cb = CircuitBreaker()

    async def coro(a, b):
        return a * b

    result = asyncio.run(cb.async_call(coro, 3, 4))
    assert result == 12
    assert cb.success_count == 1


def test_async_call_failure():
    cb = CircuitBreaker(failure_threshold=5)

    async def coro():
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        asyncio.run(cb.async_call(coro))
    assert cb.failure_count == 1


def test_async_call_open_rejects():
    cb = CircuitBreaker(failure_threshold=1, name="async-cb")

    async def coro():
        raise RuntimeError()

    with pytest.raises(RuntimeError):
        asyncio.run(cb.async_call(coro))
    assert cb.state == CircuitBreakerState.OPEN

    called = []

    async def other():
        called.append(1)
        return 1

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(cb.async_call(other))
    assert exc_info.value.args[0] == "async-cb"
    assert called == []


def test_async_half_open_limit(clock):
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=5.0,
        half_open_max_calls=1,
        name="h",
    )

    async def fail_coro():
        raise RuntimeError()

    with pytest.raises(RuntimeError):
        asyncio.run(cb.async_call(fail_coro))
    clock.advance(5.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1

    async def ok_coro():
        return "ok"

    with pytest.raises(CircuitOpenError):
        asyncio.run(cb.async_call(ok_coro))


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

def test_sync_context_manager():
    cb = CircuitBreaker()
    with cb as c:
        assert c is cb
    # __exit__ returns None (does not suppress)
    assert cb.__exit__(None, None, None) is None


@pytest.mark.asyncio
async def test_async_context_manager():
    cb = CircuitBreaker()
    async with cb as c:
        assert c is cb
    assert await cb.__aexit__(None, None, None) is None


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------

def test_breaker_retrying_creates_default_breaker():
    br = BreakerRetrying()
    assert isinstance(br.breaker, CircuitBreaker)
    assert br.breaker.state == CircuitBreakerState.CLOSED


def test_breaker_retrying_uses_provided_breaker():
    cb = CircuitBreaker(name="provided")
    br = BreakerRetrying(breaker=cb)
    assert br.breaker is cb


def test_breaker_retrying_call_success():
    cb = CircuitBreaker()
    br = BreakerRetrying(
        breaker=cb,
        predicate=lambda r: False,
        exception=(ValueError,),
        max_tries=1,
        sleep=lambda _: None,
    )
    result = br.call(lambda a: a + 1, 41)
    assert result == 42
    assert cb.success_count == 1


def test_breaker_retrying_call_retries_then_succeeds():
    cb = CircuitBreaker()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("flaky")
        return "done"

    br = BreakerRetrying(
        breaker=cb,
        exception=(ValueError,),
        max_tries=5,
        sleep=lambda _: None,
    )
    assert br.call(flaky) == "done"
    assert calls["n"] == 3
    # outer record_success resets failure_count
    assert cb.failure_count == 0
    assert cb.state == CircuitBreakerState.CLOSED


def test_breaker_retrying_records_failure_and_reraises():
    cb = CircuitBreaker(failure_threshold=2, name="br")
    br = BreakerRetrying(
        breaker=cb,
        exception=(ValueError,),
        max_tries=1,
        sleep=lambda _: None,
    )
    with pytest.raises(ValueError):
        br.call(_fail)
    assert cb.failure_count == 1


def test_breaker_retrying_open_raises_circuit_open():
    cb = CircuitBreaker(failure_threshold=1, name="br")
    br = BreakerRetrying(
        breaker=cb,
        exception=(ValueError,),
        max_tries=1,
        sleep=lambda _: None,
    )
    with pytest.raises(ValueError):
        br.call(_fail)
    assert cb.state == CircuitBreakerState.OPEN

    called = []

    def target():
        called.append(1)
        return 1

    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(target)
    assert exc_info.value.args[0] == "br"
    assert called == []


def test_breaker_retrying_half_open_limit(clock):
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=5.0,
        half_open_max_calls=1,
        name="br",
    )
    br = BreakerRetrying(
        breaker=cb,
        exception=(ValueError,),
        max_tries=1,
        sleep=lambda _: None,
    )
    with pytest.raises(ValueError):
        br.call(_fail)
    clock.advance(5.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1

    with pytest.raises(CircuitOpenError):
        br.call(_ok)


def test_breaker_retrying_async_call_success():
    cb = CircuitBreaker()
    br = BreakerRetrying(
        breaker=cb,
        exception=(ValueError,),
        max_tries=1,
        sleep=lambda _: None,
    )

    async def coro(x):
        return x + 1

    result = asyncio.run(br.async_call(coro, 10))
    assert result == 11
    assert cb.success_count == 1


def test_breaker_retrying_async_call_failure():
    cb = CircuitBreaker(failure_threshold=5)
    br = BreakerRetrying(
        breaker=cb,
        exception=(ValueError,),
        max_tries=1,
        sleep=lambda _: None,
    )

    async def coro():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        asyncio.run(br.async_call(coro))
    assert cb.failure_count == 1


def test_breaker_retrying_async_call_open():
    cb = CircuitBreaker(failure_threshold=1, name="abc")
    br = BreakerRetrying(
        breaker=cb,
        exception=(RuntimeError,),
        max_tries=1,
        sleep=lambda _: None,
    )

    async def fail_coro():
        raise RuntimeError()

    with pytest.raises(RuntimeError):
        asyncio.run(br.async_call(fail_coro))
    assert cb.state == CircuitBreakerState.OPEN

    async def other():
        return 1

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(br.async_call(other))
    assert exc_info.value.args[0] == "abc"


def test_breaker_retrying_async_half_open_limit(clock):
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=5.0,
        half_open_max_calls=1,
        name="abc",
    )
    br = BreakerRetrying(
        breaker=cb,
        exception=(RuntimeError,),
        max_tries=1,
        sleep=lambda _: None,
    )

    async def fail_coro():
        raise RuntimeError()

    with pytest.raises(RuntimeError):
        asyncio.run(br.async_call(fail_coro))
    clock.advance(5.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1

    async def ok_coro():
        return 1

    with pytest.raises(CircuitOpenError):
        asyncio.run(br.async_call(ok_coro))


def test_breaker_retrying_thread_safety_of_record():
    cb = CircuitBreaker(failure_threshold=1000)

    def worker():
        for _ in range(50):
            cb.record_failure()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cb.failure_count == 200
