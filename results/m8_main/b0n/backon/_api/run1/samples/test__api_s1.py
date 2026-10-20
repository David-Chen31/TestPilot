import asyncio
import logging
import operator

import pytest

from backon._retry import _api
from backon._retry._api import retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Boom(Exception):
    pass


class Other(Exception):
    pass


def _never_sleep(_seconds):
    return None


@pytest.fixture
def recorder():
    calls = []

    def record(_details):
        calls.append(_details)

    return calls, record


# ---------------------------------------------------------------------------
# Basic sync behavior
# ---------------------------------------------------------------------------

def test_sync_returns_result_immediately():
    def target():
        return 42

    assert retry(target)() == 42


def test_sync_retries_until_success():
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Boom("no")
        return "ok"

    wrapped = retry(target, exception=Boom, max_tries=5, sleep=_never_sleep)
    assert wrapped() == "ok"
    assert attempts["n"] == 3


def test_sync_gives_up_and_raises():
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        raise Boom("always")

    wrapped = retry(target, exception=Boom, max_tries=3, sleep=_never_sleep)
    with pytest.raises(Boom, match="always"):
        wrapped()
    assert attempts["n"] == 3


def test_sync_gives_up_raise_on_giveup_false():
    def target():
        raise Boom("always")

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=2,
        sleep=_never_sleep,
        raise_on_giveup=False,
    )
    # With raise_on_giveup=False and no retry_error_callback the original
    # exception is still propagated by the underlying loop; assert the call
    # runs without an unexpected TypeError and raises Boom.
    with pytest.raises(Boom):
        wrapped()


def test_sync_retry_error_callback_used():
    def target():
        raise Boom("x")

    def cb(details):
        return "fallback"

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=2,
        sleep=_never_sleep,
        raise_on_giveup=False,
        retry_error_callback=cb,
    )
    assert wrapped() == "fallback"


def test_sync_predicate_condition():
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        return attempts["n"]

    # predicate is operator.not_ by default; result 0 is falsy -> retried.
    # Here we use a custom predicate that retries until value >= 2.
    wrapped = retry(
        target,
        predicate=lambda v: v < 2,
        max_tries=5,
        sleep=_never_sleep,
    )
    assert wrapped() == 2
    assert attempts["n"] == 2


def test_sync_custom_condition_object():
    class AlwaysRetry:
        def __call__(self, state):
            return True

    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Boom()
        return "done"

    wrapped = retry(
        target,
        condition=AlwaysRetry(),
        max_tries=5,
        sleep=_never_sleep,
    )
    assert wrapped() == "done"


def test_sync_custom_stop_object():
    from backon._conditions import Stop

    class StopAfter2(Stop):
        def __call__(self, state):
            return state.get_tries() >= 2 if hasattr(state, "get_tries") else False

    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        raise Boom()

    wrapped = retry(target, exception=Boom, stop=StopAfter2(), sleep=_never_sleep)
    with pytest.raises(Boom):
        wrapped()
    assert attempts["n"] >= 1


def test_sync_giveup_predicate_success_path():
    def target():
        raise Boom()

    wrapped = retry(
        target,
        exception=Boom,
        giveup=lambda e: True,
        max_tries=5,
        sleep=_never_sleep,
    )
    with pytest.raises(Boom):
        wrapped()


def test_sync_max_time_limits():
    def target():
        raise Boom()

    wrapped = retry(
        target,
        exception=Boom,
        max_time=lambda: 0,
        sleep=_never_sleep,
    )
    with pytest.raises(Boom):
        wrapped()


def test_sync_callable_max_tries():
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        raise Boom()

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=lambda: 2,
        sleep=_never_sleep,
    )
    with pytest.raises(Boom):
        wrapped()
    assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# Handlers and logging
# ---------------------------------------------------------------------------

def test_sync_on_success_called():
    calls = []

    def on_success(details):
        calls.append(details)

    def target():
        return 7

    assert retry(target, on_success=on_success)() == 7
    assert len(calls) == 1
    assert calls[0]["value"] == 7


def test_sync_on_backoff_and_on_giveup_called(recorder):
    calls, record = recorder

    def target():
        raise Boom()

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=3,
        sleep=_never_sleep,
        on_backoff=record,
        on_giveup=record,
        on_success=record,
    )
    with pytest.raises(Boom):
        wrapped()
    assert len(calls) == 3
    assert calls[-1]["tries"] == 3


def test_sync_on_attempt_and_before_sleep():
    attempts = []
    sleeps = []

    def target():
        raise Boom()

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=2,
        sleep=_never_sleep,
        on_attempt=lambda d: attempts.append(d),
        before_sleep=lambda d: sleeps.append(d),
    )
    with pytest.raises(Boom):
        wrapped()
    assert len(attempts) == 2
    assert len(sleeps) == 1


def test_sync_before_after_handlers():
    events = []

    def target():
        return "r"

    def before(d):
        events.append("before")

    def after(d):
        events.append("after")

    assert retry(target, before=before, after=after)() == "r"
    assert events == ["before", "after"]


def test_sync_default_logging(caplog):
    def target():
        raise Boom()

    with caplog.at_level(logging.INFO, logger="backon"):
        with pytest.raises(Boom):
            retry(
                target,
                exception=Boom,
                max_tries=2,
                sleep=_never_sleep,
            )()
    messages = [r.getMessage() for r in caplog.records]
    assert any("Backing off" in m for m in messages)
    assert any("Giving up" in m for m in messages)


def test_sync_custom_logger_object(caplog):
    logger = logging.getLogger("custom-test-logger")

    def target():
        raise Boom()

    with caplog.at_level(logging.INFO, logger="custom-test-logger"):
        with pytest.raises(Boom):
            retry(
                target,
                exception=Boom,
                max_tries=2,
                sleep=_never_sleep,
                logger=logger,
            )()
    assert any("Giving up" in r.getMessage() for r in caplog.records)


def test_sync_no_logger_disables_default(caplog):
    def target():
        raise Boom()

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Boom):
            retry(
                target,
                exception=Boom,
                max_tries=2,
                sleep=_never_sleep,
                logger=None,
            )()
    assert caplog.records == []


# ---------------------------------------------------------------------------
# is_enabled / global disabling
# ---------------------------------------------------------------------------

def test_sync_disabled_short_circuits(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise Boom()
        return "ok"

    # Since disabled, the exception propagates without retry.
    with pytest.raises(Boom):
        retry(target, exception=Boom, max_tries=5)()
    assert attempts["n"] == 1


def test_sync_disabled_ignores_args_kwargs(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)

    def target():
        return "direct"

    assert retry(target, exception=Boom)() == "direct"


# ---------------------------------------------------------------------------
# Rate limit / timeout passthrough
# ---------------------------------------------------------------------------

def test_sync_rate_limit_passthrough():
    from backon._rate_limiter import RateLimiter

    limiter = RateLimiter(max_calls=100, period=10.0)

    def target():
        return "ok"

    assert retry(target, rate_limit=limiter, sleep=_never_sleep)() == "ok"


def test_sync_attempt_timeout_passthrough():
    def target():
        return "ok"

    assert retry(target, attempt_timeout=1.0, sleep=_never_sleep)() == "ok"


# ---------------------------------------------------------------------------
# Async behavior
# ---------------------------------------------------------------------------

def test_async_returns_result():
    async def target():
        return 99

    assert asyncio.run(retry(target)()) == 99


def test_async_retries_until_success():
    attempts = {"n": 0}

    async def target():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Boom()
        return "ok"

    wrapped = retry(target, exception=Boom, max_tries=5, sleep=_never_sleep)
    assert asyncio.run(wrapped()) == "ok"
    assert attempts["n"] == 3


def test_async_gives_up():
    async def target():
        raise Boom()

    wrapped = retry(target, exception=Boom, max_tries=2, sleep=_never_sleep)
    with pytest.raises(Boom):
        asyncio.run(wrapped())


def test_async_disabled(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)

    async def target():
        return "immediate"

    assert asyncio.run(retry(target)()) == "immediate"


def test_async_retry_error_callback():
    async def target():
        raise Boom()

    def cb(details):
        return "async-fallback"

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=2,
        sleep=_never_sleep,
        raise_on_giveup=False,
        retry_error_callback=cb,
    )
    assert asyncio.run(wrapped()) == "async-fallback"


def test_async_handlers_called():
    calls = []

    async def target():
        raise Boom()

    def record(details):
        calls.append(details)

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=2,
        sleep=_never_sleep,
        on_backoff=record,
        on_giveup=record,
    )
    with pytest.raises(Boom):
        asyncio.run(wrapped())
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Dispatch: sync vs async
# ---------------------------------------------------------------------------

def test_dispatch_sync_returns_non_coroutine():
    def target():
        return 1

    result = retry(target)()
    assert result == 1


def test_dispatch_async_returns_coroutine():
    async def target():
        return 1

    coro = retry(target)()
    assert asyncio.iscoroutine(coro)
    assert asyncio.run(coro) == 1


# ---------------------------------------------------------------------------
# wait_gen / wait_gen_kwargs
# ---------------------------------------------------------------------------

def test_sync_custom_wait_gen_kwargs_used():
    sleeps = []

    def target():
        raise Boom()

    def wait_gen(**kwargs):
        while True:
            yield kwargs.get("base", 1)

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=2,
        wait_gen=wait_gen,
        base=0,
        sleep=lambda s: sleeps.append(s),
    )
    with pytest.raises(Boom):
        wrapped()
    assert sleeps == [0]


def test_sync_wait_gen_kwargs_none_defaults():
    def target():
        raise Boom()

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=2,
        sleep=_never_sleep,
    )
    with pytest.raises(Boom):
        wrapped()


# ---------------------------------------------------------------------------
# Multiple handlers (iterables)
# ---------------------------------------------------------------------------

def test_sync_multiple_handlers():
    logs = []

    def h1(d):
        logs.append("h1")

    def h2(d):
        logs.append("h2")

    def target():
        raise Boom()

    wrapped = retry(
        target,
        exception=Boom,
        max_tries=2,
        sleep=_never_sleep,
        on_backoff=[h1, h2],
        logger=None,
    )
    with pytest.raises(Boom):
        wrapped()
    assert logs == ["h1", "h2"]


# ---------------------------------------------------------------------------
# Edge case: predicate on return values that never succeeds
# ---------------------------------------------------------------------------

def test_sync_predicate_never_satisfied():
    def target():
        return 0

    wrapped = retry(
        target,
        predicate=lambda v: True,
        max_tries=3,
        sleep=_never_sleep,
        raise_on_giveup=False,
    )
    with pytest.raises(Exception):
        wrapped()
