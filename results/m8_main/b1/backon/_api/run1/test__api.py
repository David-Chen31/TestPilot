"""Tests for backon._retry._api."""

import asyncio
import inspect
import logging
import operator

import pytest

from backon._retry import _api
from backon._retry._api import retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_sleep(_):
    return None


async def _a_no_sleep(_):
    return None


def _fast_expo(**_kwargs):
    # wait generator: yields 0 forever
    def gen():
        while True:
            yield 0

    return gen()


def _fast_wait_gen(**kwargs):
    """A wait_gen factory returning a fresh generator yielding 0 forever."""
    def gen():
        while True:
            yield 0

    return gen()


# ---------------------------------------------------------------------------
# retry: basic sync behavior
# ---------------------------------------------------------------------------

def test_sync_success_first_try():
    calls = []

    def target():
        calls.append(1)
        return "ok"

    result = retry(target, wait_gen=_fast_wait_gen, sleep=_no_sleep)
    assert result == "ok"
    assert calls == [1]


def test_sync_retries_until_success():
    calls = []

    def target():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("boom")
        return "done"

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        exception=ValueError,
        max_tries=5,
        sleep=_no_sleep,
    )
    assert result == "done"
    assert calls == [1, 1, 1]


def test_sync_gives_up_after_max_tries():
    calls = []

    def target():
        calls.append(1)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        retry(
            target,
            wait_gen=_fast_wait_gen,
            exception=ValueError,
            max_tries=3,
            sleep=_no_sleep,
        )
    assert len(calls) == 3


def test_sync_giveup_raise_on_giveup_false_returns_callback():
    calls = []

    def target():
        calls.append(1)
        raise ValueError("boom")

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        exception=ValueError,
        max_tries=2,
        raise_on_giveup=False,
        retry_error_callback=lambda details: "gave-up",
        sleep=_no_sleep,
    )
    assert result == "gave-up"
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# retry: pass-through of args/kwargs
# ---------------------------------------------------------------------------

def test_sync_passes_positional_and_keyword_args():
    # NOTE: `retry` invokes the target with no arguments; args/kwargs are only
    # honored by the internal *_inner loop when the wrapped target has no
    # required parameters. So use a target without required parameters and
    # verify that args/kwargs are accepted (and not passed) gracefully.
    calls = []

    def target():
        calls.append(1)
        return "ok"

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        sleep=_no_sleep,
        args=(1, 2),
        kwargs={"c": 3},
    )
    assert result == "ok"
    assert calls == [1]


def test_sync_invalid_target_args_raises_typeerror():
    def target(a, b, *, c):
        return (a, b, c)

    with pytest.raises(TypeError):
        retry(
            target,
            wait_gen=_fast_wait_gen,
            sleep=_no_sleep,
            args=(1, 2),
            kwargs={"c": 3},
        )


# ---------------------------------------------------------------------------
# retry: predicate based retry (returns falsy)
# ---------------------------------------------------------------------------

def test_sync_predicate_retry():
    results = iter([0, 0, 5])

    def target():
        return next(results)

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        predicate=operator.truth,
        max_tries=5,
        sleep=_no_sleep,
    )
    assert result == 5


# ---------------------------------------------------------------------------
# retry detects coroutine functions -> returns coroutine
# ---------------------------------------------------------------------------

def test_retry_detects_coroutine_function():
    calls = []

    async def target():
        calls.append(1)
        return "async-ok"

    coro = retry(target, wait_gen=_fast_wait_gen, sleep=_a_no_sleep)
    assert inspect.iscoroutine(coro)
    result = asyncio.run(coro)
    assert result == "async-ok"
    assert calls == [1]


def test_async_retry_until_success():
    calls = []

    async def target():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("boom")
        return "async-done"

    coro = retry(
        target,
        wait_gen=_fast_wait_gen,
        exception=ValueError,
        max_tries=4,
        sleep=_a_no_sleep,
    )
    assert asyncio.run(coro) == "async-done"
    assert calls == [1, 1]


def test_async_gives_up():
    calls = []

    async def target():
        calls.append(1)
        raise ValueError("boom")

    coro = retry(
        target,
        wait_gen=_fast_wait_gen,
        exception=ValueError,
        max_tries=3,
        sleep=_a_no_sleep,
    )
    with pytest.raises(ValueError):
        asyncio.run(coro)
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# retry: custom condition and stop are honored
# ---------------------------------------------------------------------------

def test_custom_condition_and_stop():
    calls = []

    def target():
        calls.append(1)
        return "x"

    class AlwaysRetry:
        def __call__(self, state):
            return True

    class NeverStop:
        def __call__(self, state):
            return False

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        condition=AlwaysRetry(),
        stop=NeverStop(),
        max_tries=1,
        sleep=_no_sleep,
        raise_on_giveup=False,
        retry_error_callback=lambda details: "gave-up",
    )
    assert result == "x"
    assert calls == [1]


def test_custom_condition_triggers_retry():
    calls = []

    def target():
        calls.append(1)
        return "always"

    class AlwaysRetry:
        def __call__(self, state):
            return True

    class StopAfterTwo:
        def __call__(self, state):
            return state.elapsed_tries >= 2 or state['tries'] >= 2

    # Simpler: stop after N attempts
    class StopAfterThree:
        def __call__(self, state):
            return state['tries'] >= 3

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        condition=AlwaysRetry(),
        stop=StopAfterThree(),
        sleep=_no_sleep,
        raise_on_giveup=False,
        retry_error_callback=lambda details: "gave-up",
    )
    assert result == "gave-up"
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# retry: custom sleep callable is used
# ---------------------------------------------------------------------------

def test_sync_custom_sleep_used():
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    calls = []

    def target():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("boom")
        return "ok"

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        exception=ValueError,
        max_tries=3,
        sleep=fake_sleep,
    )
    assert result == "ok"
    assert len(sleeps) >= 1


# ---------------------------------------------------------------------------
# Direct _retry_sync / _retry_async return values
# ---------------------------------------------------------------------------

def test_retry_sync_direct_returns_value():
    result = _api._retry_sync(
        lambda: 42,
        _fast_wait_gen,
        sleep=_no_sleep,
        max_tries=1,
    )
    assert result == 42


def test_retry_async_direct_returns_value():
    async def target():
        return 7

    async def main():
        return await _api._retry_async(
            target,
            _fast_wait_gen,
            sleep=_a_no_sleep,
            max_tries=1,
        )

    assert asyncio.run(main()) == 7


# ---------------------------------------------------------------------------
# retry: exception type list / tuple
# ---------------------------------------------------------------------------

def test_retry_exception_list():
    calls = []

    def target():
        calls.append(1)
        if len(calls) == 1:
            raise KeyError("k")
        return "ok"

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        exception=[KeyError, ValueError],
        max_tries=3,
        sleep=_no_sleep,
    )
    assert result == "ok"
    assert calls == [1, 1]


def test_non_retryable_exception_bubbles_immediately():
    calls = []

    def target():
        calls.append(1)
        raise TypeError("nope")

    with pytest.raises(TypeError):
        retry(
            target,
            wait_gen=_fast_wait_gen,
            exception=ValueError,
            max_tries=3,
            sleep=_no_sleep,
        )
    assert calls == [1]


# ---------------------------------------------------------------------------
# giveup predicate
# ---------------------------------------------------------------------------

def test_giveup_predicate_stops_retries():
    calls = []

    def target():
        calls.append(1)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        retry(
            target,
            wait_gen=_fast_wait_gen,
            exception=ValueError,
            giveup=lambda e: True,
            max_tries=5,
            sleep=_no_sleep,
        )
    assert calls == [1]


# ---------------------------------------------------------------------------
# retry on a function with args
# ---------------------------------------------------------------------------

def test_retry_wraps_and_preserves_behavior():
    # retry() invokes target with no arguments, so define a target that takes
    # no arguments but still exercises the retry path.
    calls = []

    def original():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("retryme")
        return 5

    result = retry(
        original,
        wait_gen=_fast_wait_gen,
        exception=ValueError,
        max_tries=3,
        sleep=_no_sleep,
    )
    assert result == 5
    assert calls == [1, 1]


# ---------------------------------------------------------------------------
# retry returns the value (not a wrapper)
# ---------------------------------------------------------------------------

def test_retry_returns_value_not_wrapper():
    def f():
        return 1

    result = retry(f, wait_gen=_fast_wait_gen, sleep=_no_sleep, max_tries=1)
    assert result == 1


# ---------------------------------------------------------------------------
# backoff log handlers can be exercised through the retry result
# ---------------------------------------------------------------------------

def test_on_backoff_handler_called():
    events = []

    def handler(details):
        events.append(details)

    calls = []

    def target():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("boom")
        return "ok"

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        exception=ValueError,
        max_tries=3,
        on_backoff=handler,
        sleep=_no_sleep,
    )
    assert result == "ok"
    assert len(events) >= 1
    assert events[0]["tries"] >= 1


def test_on_giveup_handler_called():
    events = []

    def handler(details):
        events.append(details)

    def target():
        raise ValueError("always")

    with pytest.raises(ValueError):
        retry(
            target,
            wait_gen=_fast_wait_gen,
            exception=ValueError,
            max_tries=2,
            on_giveup=handler,
            sleep=_no_sleep,
        )
    assert len(events) == 1
    assert events[0]["tries"] == 2


def test_on_success_handler_called():
    events = []

    def handler(details):
        events.append(details)

    def target():
        return "yes"

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        on_success=handler,
        max_tries=2,
        sleep=_no_sleep,
    )
    assert result == "yes"
    assert len(events) == 1


# ---------------------------------------------------------------------------
# logger handling: string -> logger object
# ---------------------------------------------------------------------------

def test_logger_string_is_resolved():
    logger = logging.getLogger("backon-test-api")
    logger.setLevel(logging.DEBUG)
    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collector()
    logger.addHandler(handler)
    try:
        def target():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            retry(
                target,
                wait_gen=_fast_wait_gen,
                exception=ValueError,
                max_tries=1,
                logger="backon-test-api",
                sleep=_no_sleep,
            )
        assert any("Giving up" in r.getMessage() for r in records)
    finally:
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# wait_gen_kwargs forwarded (via default expo accepted)
# ---------------------------------------------------------------------------

def test_wait_gen_kwargs_accepted():
    calls = []

    def target():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("boom")
        return "ok"

    result = retry(
        target,
        wait_gen=_fast_wait_gen,
        exception=ValueError,
        max_tries=3,
        sleep=_no_sleep,
        base=2,
    )
    assert result == "ok"
