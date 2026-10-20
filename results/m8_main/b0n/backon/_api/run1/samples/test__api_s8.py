import asyncio
import logging
import operator

import pytest

from backon._retry import _api
from backon._retry._api import retry, _retry_sync, _retry_async
from backon._wait_gen import expo


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def no_jitter():
    return lambda v: v


@pytest.fixture
def fast_expo():
    # tiny, deterministic wait generator
    def gen():
        yield 0.0
        yield 0.0
        yield 0.0
        yield 0.0
    return gen


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, details):
        self.calls.append(details)


# ---------------------------------------------------------------------------
# retry() dispatch
# ---------------------------------------------------------------------------

def test_retry_dispatches_to_sync_for_sync_function(no_jitter):
    attempts = {"n": 0}

    @retry(max_tries=3, jitter=no_jitter, sleep=lambda s: None)
    def f():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("boom")
        return "ok"

    assert f() == "ok"
    assert attempts["n"] == 2


def test_retry_dispatches_to_async_for_coroutine_function(no_jitter):
    attempts = {"n": 0}

    @retry(max_tries=3, jitter=no_jitter, sleep=_async_noop)
    async def f():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("boom")
        return "ok"

    result = asyncio.run(f())
    assert result == "ok"
    assert attempts["n"] == 2


async def _async_noop(_):
    return None


def test_retry_returns_wrapper_with_metadata(no_jitter):
    @retry(max_tries=2, jitter=no_jitter, sleep=lambda s: None)
    def f():
        return 1

    # functools.wraps preserves name
    assert f.__name__ == "f"
    assert f() == 1


# ---------------------------------------------------------------------------
# _retry_sync dispatch / behavior
# ---------------------------------------------------------------------------

def test_retry_sync_success_on_first_try(no_jitter):
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return 42

    result = _retry_sync(
        target,
        expo,
        max_tries=3,
        jitter=no_jitter,
        sleep=lambda s: None,
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    assert result == 42
    assert calls["n"] == 1


def test_retry_sync_gives_up_after_max_tries(no_jitter):
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("always fails")

    with pytest.raises(ValueError):
        _retry_sync(
            target,
            expo,
            max_tries=3,
            jitter=no_jitter,
            sleep=lambda s: None,
            wait_gen_kwargs={"base": 1, "factor": 1},
        )
    # max_tries attempts total
    assert calls["n"] == 3


def test_retry_sync_raise_on_giveup_false(no_jitter):
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("nope")

    result = _retry_sync(
        target,
        expo,
        max_tries=2,
        jitter=no_jitter,
        raise_on_giveup=False,
        sleep=lambda s: None,
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    # last exception is re-raised? No: when raise_on_giveup False, we expect None
    # (the loop returns retry_error_callback result or None)
    assert calls["n"] == 2
    # If the implementation re-raises anyway, this assertion is intentional.
    # Per current implementation it should not raise; result is None.
    assert result is None


def test_retry_sync_retry_error_callback(no_jitter):
    def target():
        raise ValueError("nope")

    sentinel = object()

    def on_error(details):
        return sentinel

    result = _retry_sync(
        target,
        expo,
        max_tries=2,
        jitter=no_jitter,
        retry_error_callback=on_error,
        sleep=lambda s: None,
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    assert result is sentinel


def test_retry_sync_custom_condition(no_jitter):
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        return attempts["n"]

    # custom condition that retries once then stops
    calls = {"n": 0}

    def condition(state):
        calls["n"] += 1
        return calls["n"] < 2

    result = _retry_sync(
        target,
        expo,
        condition=condition,
        max_tries=5,
        jitter=no_jitter,
        sleep=lambda s: None,
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    assert result == 2
    assert attempts["n"] == 2


def test_retry_sync_custom_stop(no_jitter):
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        raise ValueError("x")

    def stop(state):
        return state.tries >= 2

    with pytest.raises(ValueError):
        _retry_sync(
            target,
            expo,
            stop=stop,
            jitter=no_jitter,
            sleep=lambda s: None,
            wait_gen_kwargs={"base": 1, "factor": 1},
        )
    assert attempts["n"] == 2


def test_retry_sync_handlers_called(no_jitter):
    on_success = Recorder()
    on_backoff = Recorder()
    on_giveup = Recorder()
    on_attempt = Recorder()

    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("boom")
        return "done"

    result = _retry_sync(
        target,
        expo,
        max_tries=3,
        jitter=no_jitter,
        sleep=lambda s: None,
        on_success=on_success,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
        on_attempt=on_attempt,
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    assert result == "done"
    assert len(on_success.calls) == 1
    assert len(on_backoff.calls) == 1
    assert len(on_attempt.calls) >= 1
    # The backoff details record the exception
    assert isinstance(on_backoff.calls[0]["exception"], ValueError)


def test_retry_sync_giveup_handler(no_jitter):
    on_giveup = Recorder()

    def target():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        _retry_sync(
            target,
            expo,
            max_tries=2,
            jitter=no_jitter,
            sleep=lambda s: None,
            on_giveup=on_giveup,
            wait_gen_kwargs={"base": 1, "factor": 1},
        )
    assert len(on_giveup.calls) == 1


def test_retry_sync_before_sleep_called(no_jitter):
    before_sleep = Recorder()

    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("boom")
        return "ok"

    result = _retry_sync(
        target,
        expo,
        max_tries=3,
        jitter=no_jitter,
        sleep=lambda s: None,
        before_sleep=before_sleep,
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    assert result == "ok"
    assert len(before_sleep.calls) == 1


def test_retry_sync_args_kwargs_passthrough(no_jitter):
    def target(a, b, *, c):
        return (a, b, c)

    result = _retry_sync(
        target,
        expo,
        max_tries=1,
        jitter=no_jitter,
        sleep=lambda s: None,
        args=(1, 2),
        kwargs={"c": 3},
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    assert result == (1, 2, 3)


def test_retry_sync_predicate(no_jitter):
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        return False

    # predicate True means retry
    result = _retry_sync(
        target,
        expo,
        max_tries=3,
        predicate=lambda v: v is False,
        jitter=no_jitter,
        sleep=lambda s: None,
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    assert attempts["n"] == 3


def test_retry_sync_exception_type_filter(no_jitter):
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("retryable")
        raise KeyError("not retryable")

    with pytest.raises(KeyError):
        _retry_sync(
            target,
            expo,
            max_tries=5,
            exception=ValueError,
            jitter=no_jitter,
            sleep=lambda s: None,
            wait_gen_kwargs={"base": 1, "factor": 1},
        )
    # KeyError is not in the retry exceptions so should propagate after 2 attempts
    assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# is_enabled() short-circuit
# ---------------------------------------------------------------------------

def test_retry_sync_disabled_calls_target_without_args(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)

    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return "direct"

    # When disabled, target is called with no args/kwargs
    result = _retry_sync(target, expo, max_tries=3)
    assert result == "direct"
    assert calls["n"] == 1


def test_retry_async_disabled_calls_target(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)

    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        return "direct"

    result = asyncio.run(_retry_async(target, expo, max_tries=3))
    assert result == "direct"
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# _retry_async behavior
# ---------------------------------------------------------------------------

def test_retry_async_success(no_jitter):
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("boom")
        return "ok"

    result = asyncio.run(
        _retry_async(
            target,
            expo,
            max_tries=3,
            jitter=no_jitter,
            sleep=_async_noop,
            wait_gen_kwargs={"base": 1, "factor": 1},
        )
    )
    assert result == "ok"
    assert calls["n"] == 2


def test_retry_async_gives_up(no_jitter):
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        raise ValueError("always")

    with pytest.raises(ValueError):
        asyncio.run(
            _retry_async(
                target,
                expo,
                max_tries=2,
                jitter=no_jitter,
                sleep=_async_noop,
                wait_gen_kwargs={"base": 1, "factor": 1},
            )
        )
    assert calls["n"] == 2


def test_retry_async_handlers(no_jitter):
    on_success = Recorder()
    on_backoff = Recorder()

    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("boom")
        return "ok"

    result = asyncio.run(
        _retry_async(
            target,
            expo,
            max_tries=3,
            jitter=no_jitter,
            sleep=_async_noop,
            on_success=on_success,
            on_backoff=on_backoff,
            wait_gen_kwargs={"base": 1, "factor": 1},
        )
    )
    assert result == "ok"
    assert len(on_success.calls) == 1
    assert len(on_backoff.calls) == 1


def test_retry_async_retry_error_callback(no_jitter):
    sentinel = object()

    async def target():
        raise ValueError("nope")

    result = asyncio.run(
        _retry_async(
            target,
            expo,
            max_tries=2,
            jitter=no_jitter,
            sleep=_async_noop,
            retry_error_callback=lambda details: sentinel,
            wait_gen_kwargs={"base": 1, "factor": 1},
        )
    )
    assert result is sentinel


def test_retry_async_args_kwargs(no_jitter):
    async def target(a, b, *, c):
        return (a, b, c)

    result = asyncio.run(
        _retry_async(
            target,
            expo,
            max_tries=1,
            jitter=no_jitter,
            sleep=_async_noop,
            args=(1, 2),
            kwargs={"c": 3},
            wait_gen_kwargs={"base": 1, "factor": 1},
        )
    )
    assert result == (1, 2, 3)


def test_retry_async_exception_filter(no_jitter):
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("retryable")
        raise KeyError("not retryable")

    with pytest.raises(KeyError):
        asyncio.run(
            _retry_async(
                target,
                expo,
                max_tries=5,
                exception=ValueError,
                jitter=no_jitter,
                sleep=_async_noop,
                wait_gen_kwargs={"base": 1, "factor": 1},
            )
        )
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------

def test_retry_sync_default_logger_logs_backoff(no_jitter, caplog):
    caplog.set_level(logging.INFO, logger="backon")

    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("boom")
        return "ok"

    result = _retry_sync(
        target,
        expo,
        max_tries=3,
        jitter=no_jitter,
        sleep=lambda s: None,
        backoff_log_level=logging.WARNING,
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    assert result == "ok"
    # Some backoff log message should have been emitted
    messages = [r.message for r in caplog.records]
    assert any("Backing off" in m for m in messages)


def test_retry_sync_custom_logger(no_jitter, caplog):
    logger = logging.getLogger("custom-test-logger")
    caplog.set_level(logging.DEBUG, logger="custom-test-logger")

    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("boom")
        return "ok"

    result = _retry_sync(
        target,
        expo,
        max_tries=3,
        jitter=no_jitter,
        sleep=lambda s: None,
        logger=logger,
        wait_gen_kwargs={"base": 1, "factor": 1},
    )
    assert result == "ok"
    assert any("Backing off" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Decorator with default wait_gen
# ---------------------------------------------------------------------------

def test_retry_decorator_default_wait_gen(no_jitter, monkeypatch):
    monkeypatch.setattr(_api, "full_jitter", no_jitter)
    attempts = {"n": 0}

    @retry(max_tries=2, sleep=lambda s: None)
    def f():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("x")
        return 7

    # default expo generator accepts base/factor kwargs; call without sleep issues
    assert f() == 7
    assert attempts["n"] == 2


def test_retry_decorator_raises_on_giveup_false(no_jitter):
    calls = {"n": 0}

    @retry(max_tries=2, jitter=no_jitter, sleep=lambda s: None,
           raise_on_giveup=False)
    def f():
        calls["n"] += 1
        raise ValueError("nope")

    assert f() is None
    assert calls["n"] == 2


def test_retry_decorator_on_attempt(no_jitter):
    attempts = Recorder()

    @retry(max_tries=2, jitter=no_jitter, sleep=lambda s: None,
           on_attempt=attempts)
    def f():
        return "done"

    assert f() == "done"
    assert len(attempts.calls) == 1


# ---------------------------------------------------------------------------
# giveup predicate with exception
# ---------------------------------------------------------------------------

def test_retry_sync_giveup_predicate(no_jitter):
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        raise ValueError("x")

    with pytest.raises(ValueError):
        _retry_sync(
            target,
            expo,
            max_tries=5,
            exception=ValueError,
            giveup=lambda e: True,
            jitter=no_jitter,
            sleep=lambda s: None,
            wait_gen_kwargs={"base": 1, "factor": 1},
        )
    # Giveup immediately on the first attempt -> only one call
    assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# Sleep override is used
# ---------------------------------------------------------------------------

def test_retry_sync_uses_sleep_override(no_jitter):
    sleep_calls = []

    def custom_sleep(s):
        sleep_calls.append(s)

    def target():
        if len(sleep_calls) < 2:
            raise ValueError("x")
        return "ok"

    result = _retry_sync(
        target,
        expo,
        max_tries=3,
        jitter=no_jitter,
        sleep=custom_sleep,
        wait_gen_kwargs={"base": 1, "factor": 2},
    )
    assert result == "ok"
    assert len(sleep_calls) == 2


def test_retry_async_uses_sleep_override(no_jitter):
    sleep_calls = []

    async def custom_sleep(s):
        sleep_calls.append(s)

    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("x")
        return "ok"

    result = asyncio.run(
        _retry_async(
            target,
            expo,
            max_tries=3,
            jitter=no_jitter,
            sleep=custom_sleep,
            wait_gen_kwargs={"base": 1, "factor": 2},
        )
    )
    assert result == "ok"
    assert len(sleep_calls) == 1
