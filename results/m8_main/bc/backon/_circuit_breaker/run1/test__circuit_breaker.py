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
# CircuitBreakerState / CircuitOpenError
# ---------------------------------------------------------------------------


def test_state_enum_values():
    assert CircuitBreakerState.CLOSED.value == "CLOSED"
    assert CircuitBreakerState.OPEN.value == "OPEN"
    assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"


def test_circuit_open_error_is_exception():
    assert issubclass(CircuitOpenError, Exception)
    with pytest.raises(CircuitOpenError):
        raise CircuitOpenError("boom")


# ---------------------------------------------------------------------------
# CircuitBreaker construction / properties
# ---------------------------------------------------------------------------


def test_initial_state_and_counts():
    cb = CircuitBreaker()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 0
    assert cb.name == ""


def test_name_property():
    cb = CircuitBreaker(name="my-breaker")
    assert cb.name == "my-breaker"


def test_default_attributes():
    cb = CircuitBreaker()
    assert cb._failure_threshold == 5
    assert cb._recovery_timeout == 60.0
    assert cb._half_open_max_calls == 1


def test_custom_attributes():
    cb = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=1.5,
        half_open_max_calls=3,
        name="custom",
    )
    assert cb._failure_threshold == 2
    assert cb._recovery_timeout == 1.5
    assert cb._half_open_max_calls == 3
    assert cb.name == "custom"


# ---------------------------------------------------------------------------
# record_success / record_failure
# ---------------------------------------------------------------------------


def test_record_success_increments_success_count_and_resets_failures():
    cb = CircuitBreaker()
    cb.record_failure()
    assert cb.failure_count == 1
    assert cb.success_count == 0
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.success_count == 1
    cb.record_success()
    assert cb.success_count == 2


def test_record_failure_increments_and_resets_successes():
    cb = CircuitBreaker()
    cb.record_success()
    cb.record_success()
    assert cb.success_count == 2
    cb.record_failure()
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_record_failure_opens_at_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN


def test_record_success_from_half_open_closes():
    cb = CircuitBreaker()
    cb._state = CircuitBreakerState.HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.success_count == 1


def test_record_success_in_open_state_does_not_close():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    cb.record_success()
    # failure count reset, but state is not CLOSED (only HALF_OPEN transitions to CLOSED)
    assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# state property / recovery timeout
# ---------------------------------------------------------------------------


def test_open_state_does_not_recover_before_timeout(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    now[0] = 1059.0
    assert cb.state is CircuitBreakerState.OPEN
    now[0] = 1060.0
    assert cb.state is CircuitBreakerState.HALF_OPEN


def test_recovery_transition_resets_half_open_calls(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    cb.record_failure()
    cb._half_open_calls = 5
    now[0] = 11.0
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0


# ---------------------------------------------------------------------------
# CircuitBreaker.call
# ---------------------------------------------------------------------------


def test_call_success():
    cb = CircuitBreaker()
    result = cb.call(lambda a, b: a + b, 2, 3)
    assert result == 5
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_call_success_with_kwargs():
    cb = CircuitBreaker()

    def fn(a, b=0):
        return a * b

    assert cb.call(fn, 4, b=5) == 20


def test_call_failure_records_and_reraises():
    cb = CircuitBreaker(failure_threshold=10)

    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        cb.call(boom)
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_call_reaches_threshold_and_opens():
    cb = CircuitBreaker(failure_threshold=2)

    def boom():
        raise RuntimeError("bad")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.state is CircuitBreakerState.OPEN
    # Next call is refused without executing fn
    called = []

    with pytest.raises(CircuitOpenError):
        cb.call(lambda: called.append(1))
    assert called == []


def test_call_open_raises_with_name():
    cb = CircuitBreaker(failure_threshold=1, name="foo")
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(lambda: None)
    assert exc_info.value.args == ("foo",)


def test_double_underscore_call_delegates():
    cb = CircuitBreaker()
    assert cb(lambda x: x * 2, 21) == 42
    assert cb.success_count == 1


def test_call_half_open_allows_up_to_max(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=2
    )
    cb.record_failure()
    now[0] = 11.0
    assert cb.state is CircuitBreakerState.HALF_OPEN
    # Two allowed calls
    assert cb.call(lambda: 1) == 1
    # After successful call, state closed already
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_half_open_limit_exceeded(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=1
    )
    cb.record_failure()
    now[0] = 11.0
    assert cb.state is CircuitBreakerState.HALF_OPEN
    # Fail the first half-open call so we stay in HALF_OPEN
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))

    cb._state = CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: 1)


def test_call_half_open_limit_exceeded_with_name(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=0, name="x"
    )
    cb.record_failure()
    now[0] = 11.0
    assert cb.state is CircuitBreakerState.HALF_OPEN
    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(lambda: 1)
    assert exc_info.value.args == ("x",)


def test_call_half_open_increments_half_open_calls(monkeypatch):
    """Exercise the branch where a half-open call is admitted (line 103)."""
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=2
    )
    cb.record_failure()
    now[0] = 11.0
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0

    # Fail a call so we stay HALF_OPEN but observe the counter increment.
    def boom():
        # While executing, _half_open_calls should already be incremented.
        assert cb._half_open_calls == 1
        raise RuntimeError("half-open-fail")

    with pytest.raises(RuntimeError, match="half-open-fail"):
        cb.call(boom)
    assert cb._half_open_calls == 1
    # Force back to HALF_OPEN (failure at threshold re-opened the circuit).
    cb._state = CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1

    def boom2():
        assert cb._half_open_calls == 2
        raise RuntimeError("half-open-fail2")

    with pytest.raises(RuntimeError, match="half-open-fail2"):
        cb.call(boom2)
    assert cb._half_open_calls == 2

    # Third call is refused; counter must not advance.
    cb._state = CircuitBreakerState.HALF_OPEN
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: 1)
    assert cb._half_open_calls == 2


# ---------------------------------------------------------------------------
# CircuitBreaker async call
# ---------------------------------------------------------------------------


def test_async_call_success():
    async def main():
        cb = CircuitBreaker()

        async def fn(a, b):
            return a + b

        result = await cb.async_call(fn, 2, 3)
        assert result == 5
        assert cb.success_count == 1
        assert cb.failure_count == 0

    asyncio.run(main())


def test_async_call_failure_records_and_reraises():
    async def main():
        cb = CircuitBreaker(failure_threshold=10)

        async def boom():
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            await cb.async_call(boom)
        assert cb.failure_count == 1

    asyncio.run(main())


def test_async_call_open_raises():
    async def main():
        cb = CircuitBreaker(failure_threshold=1, name="asyncname")

        async def boom():
            raise RuntimeError()

        with pytest.raises(RuntimeError):
            await cb.async_call(boom)

        executed = []

        async def ok():
            executed.append(True)

        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.async_call(ok)
        assert exc_info.value.args == ("asyncname",)
        assert executed == []

    asyncio.run(main())


def test_async_call_half_open_limit(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    async def main():
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=1
        )
        cb.record_failure()
        now[0] = 11.0
        assert cb.state is CircuitBreakerState.HALF_OPEN

        async def ok():
            return 1

        assert await cb.async_call(ok) == 1
        assert cb.state is CircuitBreakerState.CLOSED

    asyncio.run(main())


def test_async_call_half_open_limit_exceeded(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    async def main():
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=1
        )
        cb.record_failure()
        now[0] = 11.0
        cb._state = CircuitBreakerState.HALF_OPEN
        cb._half_open_calls = 1

        ran = []

        async def ok():
            ran.append(True)

        with pytest.raises(CircuitOpenError):
            await cb.async_call(ok)
        assert ran == []

    asyncio.run(main())


def test_async_call_half_open_increments_half_open_calls(monkeypatch):
    """Exercise the branch where an async half-open call is admitted (line 122)."""
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    async def main():
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=2
        )
        cb.record_failure()
        now[0] = 11.0
        assert cb.state is CircuitBreakerState.HALF_OPEN
        assert cb._half_open_calls == 0

        async def boom():
            assert cb._half_open_calls == 1
            raise RuntimeError("async-half-open-fail")

        with pytest.raises(RuntimeError, match="async-half-open-fail"):
            await cb.async_call(boom)
        assert cb._half_open_calls == 1

        cb._state = CircuitBreakerState.HALF_OPEN
        cb._half_open_calls = 1

        async def boom2():
            assert cb._half_open_calls == 2
            raise RuntimeError("async-half-open-fail2")

        with pytest.raises(RuntimeError, match="async-half-open-fail2"):
            await cb.async_call(boom2)
        assert cb._half_open_calls == 2

        cb._state = CircuitBreakerState.HALF_OPEN
        called = []

        async def refused():
            called.append(True)

        with pytest.raises(CircuitOpenError):
            await cb.async_call(refused)
        assert called == []
        assert cb._half_open_calls == 2

    asyncio.run(main())


# ---------------------------------------------------------------------------
# context managers
# ---------------------------------------------------------------------------


def test_sync_context_manager():
    cb = CircuitBreaker()
    with cb as entered:
        assert entered is cb


def test_async_context_manager():
    async def main():
        cb = CircuitBreaker()
        async with cb as entered:
            assert entered is cb

    asyncio.run(main())


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------


def test_breaker_retrying_default_breaker():
    br = BreakerRetrying()
    assert isinstance(br.breaker, CircuitBreaker)
    assert br.breaker.state is CircuitBreakerState.CLOSED


def test_breaker_retrying_custom_breaker():
    cb = CircuitBreaker(name="custom")
    br = BreakerRetrying(breaker=cb)
    assert br.breaker is cb
    assert br.breaker.name == "custom"


def test_breaker_retrying_call_success():
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, max_tries=1, predicate=lambda r: False)

    result = br.call(lambda a, b: a + b, 1, 2)
    assert result == 3
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_breaker_retrying_call_retries_then_succeeds():
    cb = CircuitBreaker()
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("try again")
        return "ok"

    br = BreakerRetrying(
        breaker=cb,
        max_tries=5,
        exception=ValueError,
        sleep=lambda _: None,
        predicate=lambda r: False,
    )
    assert br.call(flaky) == "ok"
    assert attempts["n"] == 3
    assert cb.success_count == 1


def test_breaker_retrying_call_failure_records():
    cb = CircuitBreaker(failure_threshold=1)
    br = BreakerRetrying(
        breaker=cb,
        max_tries=1,
        exception=ValueError,
        sleep=lambda _: None,
    )

    with pytest.raises(ValueError, match="kaboom"):
        br.call(lambda: (_ for _ in ()).throw(ValueError("kaboom")))

    assert cb.failure_count == 1
    assert cb.state is CircuitBreakerState.OPEN


def test_breaker_retrying_call_open_raises_without_running():
    cb = CircuitBreaker(failure_threshold=1, name="rb")
    cb.record_failure()
    br = BreakerRetrying(breaker=cb)

    calls = []

    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(lambda: calls.append(1))
    assert exc_info.value.args == ("rb",)
    assert calls == []


def test_breaker_retrying_call_half_open_limit(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=0, name="x"
    )
    cb._state = CircuitBreakerState.OPEN
    cb._last_failure_time = 0.0
    now[0] = 11.0
    assert cb.state is CircuitBreakerState.HALF_OPEN

    br = BreakerRetrying(breaker=cb)
    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(lambda: 1)
    assert exc_info.value.args == ("x",)


def test_breaker_retrying_call_half_open_increments_counter(monkeypatch):
    """Exercise BreakerRetrying.call's half-open admit branch (lines 213/215)."""
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=2
    )
    cb.record_failure()
    now[0] = 11.0
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0

    br = BreakerRetrying(breaker=cb, max_tries=1, exception=ValueError,
                         sleep=lambda _: None, predicate=lambda r: False)

    def boom():
        assert cb._half_open_calls == 1
        raise ValueError("br-half-open-fail")

    with pytest.raises(ValueError, match="br-half-open-fail"):
        br.call(boom)
    assert cb._half_open_calls == 1

    # Circuit re-opened by the failure; force HALF_OPEN with one slot used.
    cb._state = CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1

    def boom2():
        assert cb._half_open_calls == 2
        raise ValueError("br-half-open-fail2")

    with pytest.raises(ValueError, match="br-half-open-fail2"):
        br.call(boom2)
    assert cb._half_open_calls == 2

    # Limit reached: no execution and no counter bump.
    cb._state = CircuitBreakerState.HALF_OPEN
    called = []
    with pytest.raises(CircuitOpenError):
        br.call(lambda: called.append(True))
    assert called == []
    assert cb._half_open_calls == 2


def test_breaker_retrying_async_call_success():
    async def main():
        cb = CircuitBreaker()
        br = BreakerRetrying(breaker=cb, max_tries=1, predicate=lambda r: False)

        async def fn(a, b):
            return a * b

        result = await br.async_call(fn, 3, 4)
        assert result == 12
        assert cb.success_count == 1

    asyncio.run(main())


def test_breaker_retrying_async_call_failure_records():
    async def main():
        cb = CircuitBreaker(failure_threshold=1)
        br = BreakerRetrying(breaker=cb, max_tries=1, exception=ValueError,
                             sleep=lambda _: None)

        async def boom():
            raise ValueError("async-boom")

        with pytest.raises(ValueError, match="async-boom"):
            await br.async_call(boom)

        assert cb.failure_count == 1
        assert cb.state is CircuitBreakerState.OPEN

    asyncio.run(main())


def test_breaker_retrying_async_call_open_raises():
    async def main():
        cb = CircuitBreaker(failure_threshold=1, name="asyncrety")
        cb.record_failure()
        br = BreakerRetrying(breaker=cb)

        ran = []

        async def ok():
            ran.append(True)

        with pytest.raises(CircuitOpenError) as exc_info:
            await br.async_call(ok)
        assert exc_info.value.args == ("asyncrety",)
        assert ran == []

    asyncio.run(main())


def test_breaker_retrying_async_call_half_open_limit(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    async def main():
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=0,
            name="x",
        )
        cb._state = CircuitBreakerState.OPEN
        cb._last_failure_time = 0.0
        now[0] = 11.0
        assert cb.state is CircuitBreakerState.HALF_OPEN

        br = BreakerRetrying(breaker=cb)

        async def ok():
            return 1

        with pytest.raises(CircuitOpenError):
            await br.async_call(ok)

    asyncio.run(main())


def test_breaker_retrying_async_call_half_open_increments_counter(monkeypatch):
    """Exercise BreakerRetrying.async_call's half-open admit branch (lines 232/234)."""
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    async def main():
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=10.0, half_open_max_calls=2
        )
        cb.record_failure()
        now[0] = 11.0
        assert cb.state is CircuitBreakerState.HALF_OPEN
        assert cb._half_open_calls == 0

        br = BreakerRetrying(
            breaker=cb, max_tries=1, exception=ValueError,
            sleep=lambda _: None, predicate=lambda r: False,
        )

        async def boom():
            assert cb._half_open_calls == 1
            raise ValueError("br-async-half-open-fail")

        with pytest.raises(ValueError, match="br-async-half-open-fail"):
            await br.async_call(boom)
        assert cb._half_open_calls == 1

        cb._state = CircuitBreakerState.HALF_OPEN
        cb._half_open_calls = 1

        async def boom2():
            assert cb._half_open_calls == 2
            raise ValueError("br-async-half-open-fail2")

        with pytest.raises(ValueError, match="br-async-half-open-fail2"):
            await br.async_call(boom2)
        assert cb._half_open_calls == 2

        cb._state = CircuitBreakerState.HALF_OPEN
        called = []

        async def refused():
            called.append(True)

        with pytest.raises(CircuitOpenError):
            await br.async_call(refused)
        assert called == []
        assert cb._half_open_calls == 2

    asyncio.run(main())


def test_breaker_retrying_async_retries_then_succeeds():
    async def main():
        cb = CircuitBreaker()
        attempts = {"n": 0}

        async def flaky():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ValueError("again")
            return "ok"

        async def async_sleep(_):
            return None

        br = BreakerRetrying(
            breaker=cb,
            max_tries=5,
            exception=ValueError,
            sleep=async_sleep,
            predicate=lambda r: False,
        )
        assert await br.async_call(flaky) == "ok"
        assert attempts["n"] == 2
        assert cb.success_count == 1

    asyncio.run(main())
