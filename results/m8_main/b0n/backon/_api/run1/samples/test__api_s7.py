import asyncio
import inspect
import logging
import operator

import pytest

import backon._common as _common
from backon._retry import _api
from backon._retry._api import retry
from backon._wait_gen import expo


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_test_config():
    """Snapshot & restore the global test config and enabled flag."""
    saved_config = dict(_common._TEST_CONFIG)
    enabled_attr = "_GLOBAL_ENABLED"
    saved_enabled = getattr(_common, enabled_attr, None)
    yield
    _common._TEST_CONFIG.clear()
    _common._TEST_CONFIG.update(saved_config)
    if saved_enabled is not None:
        setattr(_common, enabled_attr, saved_enabled)


def _no_sleep(_seconds):
    return None


class _Counter:
    def __init__(self, fail_until=0, exc_factory=None):
        self.calls = 0
        self.fail_until = fail_until
        self.exc_factory = exc_factory or (lambda: ValueError("boom"))

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_until:
            raise self.exc_factory()
        return ("ok", args, kwargs)


# ---------------------------------------------------------------------------
# retry: dispatch (sync vs async)
# ---------------------------------------------------------------------------

def test_retry_dispatches_sync_target_returns_value_directly():
    # A sync function that succeeds immediately should return the value
    # directly (not a wrapped callable), since retry() is called on a target.
    result = retry(lambda: 42)
    assert result == 42


def test_retry_dispatches_async_target_based_on_iscoroutinefunction():
    async def coro():
        return 7

    assert inspect.iscoroutinefunction(coro)

    coro_obj = retry(coro)
    # For a coroutine function target, retry() should return a coroutine
    # that resolves to the value.
    assert inspect.iscoroutine(coro_obj)
    assert asyncio.get_event_loop().run_until_complete(coro_obj) == 7


def test_retry_sync_dispatch_uses_retry_sync(monkeypatch):
    calls = {}

    def fake_sync(target, wait_gen, **kwargs):
        calls["sync"] = True
        calls["target"] = target
        calls["wait_gen"] = wait_gen
        calls["kwargs"] = kwargs
        return "sync-result"

    def fake_async(target, wait_gen, **kwargs):  # pragma: no cover
        calls["async"] = True
        return "async-result"

    monkeypatch.setattr(_api, "_retry_sync", fake_sync)
    monkeypatch.setattr(_api, "_retry_async", fake_async)

    def target():
        return 1

    result = retry(target)
    assert result == "sync-result"
    assert calls.get("sync") is True
    assert "async" not in calls
    assert calls["target"] is target
    assert calls["wait_gen"] is expo


def test_retry_async_dispatch_uses_retry_async(monkeypatch):
    calls = {}

    def fake_sync(target, wait_gen, **kwargs):  # pragma: no cover
        calls["sync"] = True
        return "sync-result"

    async def fake_async(target, wait_gen, **kwargs):
        calls["async"] = True
        return "async-result"

    monkeypatch.setattr(_api, "_retry_sync", fake_sync)
    monkeypatch.setattr(_api, "_retry_async", fake_async)

    async def target():
        return 1

    result = retry(target)
    assert inspect.iscoroutine(result)
    assert asyncio.get_event_loop().run_until_complete(result) == "async-result"
    assert calls.get("async") is True
    assert "sync" not in calls


# ---------------------------------------------------------------------------
# retry: forwarding of arguments / wait_gen_kwargs
# ---------------------------------------------------------------------------

def test_retry_sync_forwards_kwargs_and_wait_gen_kwargs(monkeypatch):
    captured = {}

    def fake_sync(target, wait_gen, **kwargs):
        captured["target"] = target
        captured["wait_gen"] = wait_gen
        captured["kwargs"] = kwargs
        return "done"

    monkeypatch.setattr(_api, "_retry_sync", fake_sync)
    monkeypatch.setattr(_api, "_retry_async", lambda *a, **k: None)

    def target():
        return 1

    result = retry(
        target,
        max_tries=5,
        max_time=10.0,
        raise_on_giveup=False,
        name="ignored",
        jitter=None,
        logger="mylogger",
        backoff_log_level=logging.DEBUG,
        giveup_log_level=logging.WARNING,
        base=3,
        factor=2,
    )
    assert result == "done"
    kwargs = captured["kwargs"]
    assert kwargs["max_tries"] == 5
    assert kwargs["max_time"] == 10.0
    assert kwargs["raise_on_giveup"] is False
    assert kwargs["jitter"] is None
    assert kwargs["logger"] == "mylogger"
    assert kwargs["backoff_log_level"] == logging.DEBUG
    assert kwargs["giveup_log_level"] == logging.WARNING
    # extra kwargs go to wait_gen_kwargs
    assert kwargs["wait_gen_kwargs"] == {"base": 3, "factor": 2}
    # name is not forwarded
    assert "name" not in kwargs


def test_retry_async_forwards_kwargs(monkeypatch):
    captured = {}

    async def fake_async(target, wait_gen, **kwargs):
        captured["target"] = target
        captured["wait_gen"] = wait_gen
        captured["kwargs"] = kwargs
        return "done"

    monkeypatch.setattr(_api, "_retry_async", fake_async)
    monkeypatch.setattr(_api, "_retry_sync", lambda *a, **k: None)

    async def target():
        return 1

    result = retry(target, max_tries=2, base=5)
    assert asyncio.get_event_loop().run_until_complete(result) == "done"
    assert captured["kwargs"]["max_tries"] == 2
    assert captured["kwargs"]["wait_gen_kwargs"] == {"base": 5}


# ---------------------------------------------------------------------------
# retry: real behavior - sync success and retry
# ---------------------------------------------------------------------------

def test_retry_sync_eventually_succeeds():
    counter = _Counter(fail_until=2)
    result = retry(counter, max_tries=5, sleep=_no_sleep)
    assert result == ("ok", (), {})
    assert counter.calls == 3


def test_retry_sync_exhausts_and_raises():
    counter = _Counter(fail_until=100)
    with pytest.raises(ValueError):
        retry(counter, max_tries=3, sleep=_no_sleep)
    assert counter.calls == 3


def test_retry_sync_raise_on_giveup_false_returns_none():
    counter = _Counter(fail_until=100)
    result = retry(counter, max_tries=2, sleep=_no_sleep, raise_on_giveup=False)
    assert result is None
    assert counter.calls == 2


def test_retry_sync_retry_error_callback_used():
    counter = _Counter(fail_until=100)

    def cb(details):
        return ("failed", details["tries"])

    result = retry(
        counter, max_tries=2, sleep=_no_sleep, retry_error_callback=cb
    )
    assert result == ("failed", 2)


# ---------------------------------------------------------------------------
# retry: real behavior - async
# ---------------------------------------------------------------------------

def test_retry_async_eventually_succeeds():
    state = {"calls": 0}

    async def target():
        state["calls"] += 1
        if state["calls"] < 3:
            raise RuntimeError("nope")
        return "done"

    async def fake_sleep(_):
        return None

    coro = retry(target, max_tries=5, sleep=fake_sleep)
    assert asyncio.get_event_loop().run_until_complete(coro) == "done"
    assert state["calls"] == 3


def test_retry_async_exhausts_and_raises():
    async def target():
        raise RuntimeError("always")

    async def fake_sleep(_):
        return None

    coro = retry(target, max_tries=2, sleep=fake_sleep)
    with pytest.raises(RuntimeError):
        asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# retry: disabled via global flag (returns target directly)
# ---------------------------------------------------------------------------

def test_retry_sync_disabled_calls_target_without_retry(monkeypatch):
    monkeypatch.setattr(_common, "_GLOBAL_ENABLED", False)
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("no retry")

    with pytest.raises(ValueError):
        retry(target, max_tries=5, sleep=_no_sleep)
    # Disabled -> only one attempt, no retry.
    assert calls["n"] == 1


def test_retry_async_disabled_calls_target_directly(monkeypatch):
    monkeypatch.setattr(_common, "_GLOBAL_ENABLED", False)
    state = {"n": 0}

    async def target():
        state["n"] += 1
        raise ValueError("no retry")

    coro = retry(target, max_tries=5)
    with pytest.raises(ValueError):
        asyncio.get_event_loop().run_until_complete(coro)
    assert state["n"] == 1


# ---------------------------------------------------------------------------
# retry: custom condition / stop objects are forwarded
# ---------------------------------------------------------------------------

def test_retry_passes_custom_condition_and_stop(monkeypatch):
    captured = {}

    def fake_sync(target, wait_gen, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(_api, "_retry_sync", fake_sync)
    monkeypatch.setattr(_api, "_retry_async", lambda *a, **k: None)

    sentinel_condition = object()
    sentinel_stop = object()

    result = retry(
        lambda: 1,
        condition=sentinel_condition,
        stop=sentinel_stop,
    )
    assert result == "ok"
    assert captured["condition"] is sentinel_condition
    assert captured["stop"] is sentinel_stop


# ---------------------------------------------------------------------------
# _retry_sync internals
# ---------------------------------------------------------------------------

def test_retry_sync_passes_defaults_when_none(monkeypatch):
    # Exercise _retry_sync to make sure default condition/stop and sleep are
    # set up. Capture what gets forwarded to _retry_fast_sync_inner.
    captured = {}

    def fake_inner(target, wait_gen, **kwargs):
        captured["target"] = target
        captured["wait_gen"] = wait_gen
        captured["kwargs"] = kwargs
        return "inner-result"

    monkeypatch.setattr(_api, "_retry_fast_sync_inner", fake_inner)

    def target():
        return 1

    result = _api._retry_sync(target, expo)
    assert result == "inner-result"
    assert captured["target"] is target
    assert captured["wait_gen"] is expo
    kwargs = captured["kwargs"]
    # wait_gen_kwargs defaults to {}.
    assert kwargs["wait_gen_kwargs"] == {}
    # default condition and stop are constructed.
    assert callable(kwargs["condition"])
    assert callable(kwargs["stop"])
    # default sleep is time.sleep (not None).
    assert kwargs["sleep"] is not None
    # default handlers of on_backoff / on_giveup configured.
    assert len(kwargs["on_backoff"]) == 1
    assert len(kwargs["on_giveup"]) == 1


def test_retry_sync_custom_sleep_used(monkeypatch):
    captured = {}

    def fake_inner(target, wait_gen, **kwargs):
        captured["sleep"] = kwargs["sleep"]
        return "ok"

    monkeypatch.setattr(_api, "_retry_fast_sync_inner", fake_inner)

    def my_sleep(secs):
        return None

    _api._retry_sync(lambda: 1, expo, sleep=my_sleep)
    assert captured["sleep"] is my_sleep


def test_retry_sync_disabled_returns_target_result(monkeypatch):
    monkeypatch.setattr(_common, "_GLOBAL_ENABLED", False)

    def target():
        return "direct"

    result = _api._retry_sync(target, expo)
    assert result == "direct"


# ---------------------------------------------------------------------------
# _retry_async internals
# ---------------------------------------------------------------------------

def test_retry_async_passes_defaults(monkeypatch):
    captured = {}

    async def fake_inner(target, wait_gen, **kwargs):
        captured["target"] = target
        captured["wait_gen"] = wait_gen
        captured["kwargs"] = kwargs
        return "inner-result"

    monkeypatch.setattr(_api, "_retry_fast_async_inner", fake_inner)

    async def target():
        return 1

    result = asyncio.get_event_loop().run_until_complete(
        _api._retry_async(target, expo)
    )
    assert result == "inner-result"
    kwargs = captured["kwargs"]
    assert kwargs["wait_gen_kwargs"] == {}
    assert callable(kwargs["condition"])
    assert callable(kwargs["stop"])
    # default for async is asyncio.sleep
    assert kwargs["sleep"] is asyncio.sleep


def test_retry_async_custom_sleep_used(monkeypatch):
    captured = {}

    async def fake_inner(target, wait_gen, **kwargs):
        captured["sleep"] = kwargs["sleep"]
        return "ok"

    monkeypatch.setattr(_api, "_retry_fast_async_inner", fake_inner)

    async def my_sleep(secs):
        return None

    async def target():
        return 1

    asyncio.get_event_loop().run_until_complete(
        _api._retry_async(target, expo, sleep=my_sleep)
    )
    assert captured["sleep"] is my_sleep


def test_retry_async_disabled_returns_target_result(monkeypatch):
    monkeypatch.setattr(_common, "_GLOBAL_ENABLED", False)

    async def target():
        return "direct"

    result = asyncio.get_event_loop().run_until_complete(
        _api._retry_async(target, expo)
    )
    assert result == "direct"


# ---------------------------------------------------------------------------
# _retry_sync / _retry_async: handler configuration
# ---------------------------------------------------------------------------

def test_retry_sync_custom_handlers_are_configured(monkeypatch):
    captured = {}

    def fake_inner(target, wait_gen, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(_api, "_retry_fast_sync_inner", fake_inner)

    on_success = lambda d: None
    on_backoff = lambda d: None
    on_giveup = lambda d: None
    on_attempt = lambda d: None
    before_sleep = lambda d: None
    before = lambda d: None
    after = lambda d: None

    _api._retry_sync(
        lambda: 1,
        expo,
        on_success=on_success,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
        on_attempt=on_attempt,
        before_sleep=before_sleep,
        before=before,
        after=after,
        logger=None,
    )
    # Without a logger, _config_handlers omits the default handler and just
    # wraps the user-provided handlers.
    assert captured["on_success"] == [on_success]
    assert on_backoff in captured["on_backoff"]
    assert on_giveup in captured["on_giveup"]
    assert captured["on_attempt"] == [on_attempt]
    assert captured["before_sleep"] == [before_sleep]
    assert captured["before"] == [before]
    assert captured["after"] == [after]
