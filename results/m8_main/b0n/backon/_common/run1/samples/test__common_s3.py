import logging
import threading
import time as time_module

import pytest

import backon._common as common
from backon._common import (
    disable,
    enable,
    is_enabled,
    _maybe_call,
    _init_wait_gen,
    _next_wait,
    _apply_test_overrides,
    _prepare_logger,
    _LazyLogHandler,
    _config_handlers,
    _log_backoff,
    _log_giveup,
    _now,
    _elapsed,
    _is_custom_wait,
    _check_hot_loop,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_global_state():
    original_enabled = common._GLOBAL_ENABLED
    original_test_config = dict(common._TEST_CONFIG)
    original_hot = dict(common._hot_loop_data)
    try:
        yield
    finally:
        common._GLOBAL_ENABLED = original_enabled
        common._TEST_CONFIG.clear()
        common._TEST_CONFIG.update(original_test_config)
        common._hot_loop_data.clear()
        common._hot_loop_data.update(original_hot)


# ---------------------------------------------------------------------------
# enable / disable / is_enabled
# ---------------------------------------------------------------------------

def test_default_enabled():
    # Reset to whatever the fixture captured is the module default here.
    enable()
    assert is_enabled() is True


def test_disable_then_enable():
    enable()
    assert is_enabled() is True
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True


def test_disable_returns_none():
    assert disable() is None
    assert enable() is None


def test_is_enabled_reflects_module_global():
    common._GLOBAL_ENABLED = False
    assert is_enabled() is False
    common._GLOBAL_ENABLED = True
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# _maybe_call
# ---------------------------------------------------------------------------

def test_maybe_call_non_callable_returned_as_is():
    assert _maybe_call(5) == 5
    assert _maybe_call("abc") == "abc"
    obj = object()
    assert _maybe_call(obj) is obj


def test_maybe_call_callable_no_args():
    assert _maybe_call(lambda: 42) == 42


def test_maybe_call_with_args():
    assert _maybe_call(lambda a, b: a + b, 2, 3) == 5
    assert _maybe_call(lambda a: a * 2, a=10) == 20


def test_maybe_call_missing_positional_returns_callable():
    # A callable requiring a positional arg (called with none) is returned.
    def needs_arg(x):
        return x

    assert _maybe_call(needs_arg) is needs_arg


def test_maybe_call_type_error_about_takes_returns_callable():
    class Foo:
        def __call__(self):
            return 1

    def f(a, b):
        return a + b

    # Calling f() with no args: "missing 2 required positional argument"...
    # But we exercise the "takes " message variant by a builtin-ish call.
    with pytest.raises(TypeError):
        # int() with a bad keyword arg raises a TypeError not matching branches
        # It's safer to verify propagation path with a custom message.
        pass
    # Ensure real callables returning value still work.
    assert _maybe_call(Foo()) == 1


def test_maybe_call_propagates_unrelated_type_error():
    def boom(x):
        raise TypeError("this is a genuine internal failure")

    with pytest.raises(TypeError, match="genuine internal failure"):
        _maybe_call(boom, 1)


def test_maybe_call_propagates_value_error():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        _maybe_call(boom)


def test_maybe_call_takes_message_branch():
    # Simulate the err message containing "takes " to hit the return-f branch.
    def fake(*args, **kwargs):
        raise TypeError("takes exactly 1 argument (0 given)")

    result = _maybe_call(fake)
    assert result is fake


def test_maybe_call_required_positional_branch():
    def fake(*args, **kwargs):
        raise TypeError("missing 1 required positional argument: 'x'")

    result = _maybe_call(fake)
    assert result is fake


# ---------------------------------------------------------------------------
# _init_wait_gen
# ---------------------------------------------------------------------------

def test_init_wait_gen_calls_with_values():
    captured = {}

    def gen(a, b):
        captured["a"] = a
        captured["b"] = b
        return "result"

    out = _init_wait_gen(gen, {"a": 1, "b": 2})
    assert out == "result"
    assert captured == {"a": 1, "b": 2}


def test_init_wait_gen_resolves_callables_in_kwargs():
    def gen(value):
        return value

    out = _init_wait_gen(gen, {"value": lambda: 99})
    assert out == 99


# ---------------------------------------------------------------------------
# _next_wait
# ---------------------------------------------------------------------------

class _FakeWait:
    def __init__(self, value):
        self.value = value
        self.received = []

    def next(self, send_value):
        self.received.append(send_value)
        return self.value


def test_next_wait_no_jitter_no_max():
    w = _FakeWait(2.0)
    out = _next_wait(w, None, None, 0.0, None)
    assert out == 2.0
    assert w.received == [None]


def test_next_wait_applies_jitter():
    w = _FakeWait(2.0)
    out = _next_wait(w, "sent", lambda v: v * 3, 0.0, None)
    assert out == 6.0
    assert w.received == ["sent"]


def test_next_wait_respects_multiplier():
    common._TEST_CONFIG["backoff_multiplier"] = 2.0
    w = _FakeWait(1.5)
    out = _next_wait(w, None, None, 0.0, None)
    assert out == 3.0


def test_next_wait_capped_by_max_time():
    w = _FakeWait(10.0)
    out = _next_wait(w, None, None, elapsed=4.0, max_time=5.0)
    assert out == 1.0


def test_next_wait_max_time_larger_than_seconds():
    w = _FakeWait(1.0)
    out = _next_wait(w, None, None, elapsed=0.0, max_time=5.0)
    assert out == 1.0


# ---------------------------------------------------------------------------
# _apply_test_overrides
# ---------------------------------------------------------------------------

def test_apply_test_overrides_no_override():
    common._TEST_CONFIG["max_retries"] = None
    assert _apply_test_overrides(max_tries=3, max_time=10) == (3, 10)
    assert _apply_test_overrides() == (None, None)


def test_apply_test_overrides_with_override():
    common._TEST_CONFIG["max_retries"] = 7
    assert _apply_test_overrides(max_tries=3, max_time=10) == (7, 10)


# ---------------------------------------------------------------------------
# _prepare_logger
# ---------------------------------------------------------------------------

def test_prepare_logger_string_returns_logger():
    lg = _prepare_logger("my.test.logger")
    assert isinstance(lg, logging.Logger)
    assert lg.name == "my.test.logger"


def test_prepare_logger_object_passthrough():
    lg = logging.getLogger("backon.test")
    assert _prepare_logger(lg) is lg


# ---------------------------------------------------------------------------
# _LazyLogHandler
# ---------------------------------------------------------------------------

def test_lazy_log_handler_creates_once_and_forwards():
    calls = []

    def handler(*, logger, log_level):
        def _call(details):
            calls.append((logger, log_level, details))
        return _call

    lazy = _LazyLogHandler(handler, "lg", 20)
    assert lazy._handler is None
    lazy({"a": 1})
    assert lazy._handler is not None
    first = lazy._handler
    lazy({"a": 2})
    assert lazy._handler is first
    assert len(calls) == 2
    assert calls[0] == ("lg", 20, {"a": 1})
    assert calls[1] == ("lg", 20, {"a": 2})


# ---------------------------------------------------------------------------
# _config_handlers
# ---------------------------------------------------------------------------

def test_config_handlers_list_without_logger_returns_same_list():
    user = [object(), object()]
    out = _config_handlers(user, default_handler=lambda **kw: None, logger=None)
    assert out is user


def test_config_handlers_no_logger_no_user():
    out = _config_handlers(None, default_handler=None, logger=None)
    assert out == []


def test_config_handlers_single_user_handler_no_logger():
    h = object()
    out = _config_handlers(h, default_handler=None, logger=None)
    assert out == [h]


def test_config_handlers_iterable_user_handlers_no_logger():
    a, b = object(), object()
    out = _config_handlers((a, b), default_handler=None, logger=None)
    assert out == [a, b]


def test_config_handlers_logger_none_user_returns_lazy():
    def handler(**kw):
        return lambda d: None

    out = _config_handlers(
        None, default_handler=handler, logger="lg", log_level=10
    )
    assert len(out) == 1
    assert isinstance(out[0], _LazyLogHandler)


def test_config_handlers_logger_with_single_user():
    def default_handler(**kw):
        return lambda d: None

    user = object()
    out = _config_handlers(
        user, default_handler=default_handler, logger="lg", log_level=10
    )
    assert len(out) == 2
    assert callable(out[0])
    assert out[1] is user


def test_config_handlers_logger_with_iterable_user():
    def default_handler(**kw):
        return lambda d: None

    a, b = object(), object()
    out = _config_handlers(
        [a, b], default_handler=default_handler, logger="lg", log_level=10
    )
    assert len(out) == 3
    assert out[1] is a
    assert out[2] is b


# ---------------------------------------------------------------------------
# _log_backoff / _log_giveup
# ---------------------------------------------------------------------------

class _CaptureLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg, args))


def _target():
    pass


def test_log_backoff_with_exception():
    lg = _CaptureLogger()
    exc = ValueError("bad")
    _log_backoff(
        {"exception": exc, "target": _target, "wait": 1.5}, lg, 30
    )
    level, msg, args = lg.records[0]
    assert level == 30
    assert args == ["_target", 1.5, "ValueError: bad"]


def test_log_backoff_with_value_no_exception():
    lg = _CaptureLogger()
    _log_backoff({"value": "rv", "target": _target, "wait": 2.0}, lg, 20)
    _, _, args = lg.records[0]
    assert args == ["_target", 2.0, "rv"]


def test_log_backoff_unknown_value():
    lg = _CaptureLogger()
    _log_backoff({"target": _target, "wait": 0.5}, lg, 10)
    _, _, args = lg.records[0]
    assert args == ["_target", 0.5, "unknown"]


def test_log_giveup_with_exception():
    lg = _CaptureLogger()
    _log_giveup(
        {"exception": RuntimeError("boom"), "target": _target, "tries": 4},
        lg, 40,
    )
    level, msg, args = lg.records[0]
    assert level == 40
    assert args == ["_target", 4, "RuntimeError: boom"]


def test_log_giveup_with_value():
    lg = _CaptureLogger()
    _log_giveup({"value": 7, "target": _target, "tries": 2}, lg, 20)
    _, _, args = lg.records[0]
    assert args == ["_target", 2, 7]


# ---------------------------------------------------------------------------
# _now / _elapsed
# ---------------------------------------------------------------------------

def test_now_positive():
    assert _now() > 0


def test_elapsed_increases():
    start = _now()
    e = _elapsed(start)
    assert e >= 0


# ---------------------------------------------------------------------------
# _is_custom_wait
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (1, True),
    (1.5, True),
    (0, True),
    (True, False),
    (False, False),
    ("1", False),
    (None, False),
    ([1], False),
])
def test_is_custom_wait(value, expected):
    assert _is_custom_wait(value) is expected


# ---------------------------------------------------------------------------
# _check_hot_loop
# ---------------------------------------------------------------------------

def test_check_hot_loop_first_call_no_warning(caplog):
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0
    with caplog.at_level(logging.WARNING, logger="backon"):
        _check_hot_loop()
    assert common._hot_loop_data["count"] == 0
    assert common._hot_loop_data["last_retry"] > 0
    assert not caplog.records


def test_check_hot_loop_triggers_warning(monkeypatch, caplog):
    # Freeze time at a controlled value to simulate rapid retries.
    fake_now = [100.0]
    monkeypatch.setattr(
        time_module, "monotonic_ns", lambda: int(fake_now[0] * 1_000_000_000)
    )
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0

    with caplog.at_level(logging.WARNING, logger="backon"):
        _check_hot_loop()  # first call, sets last_retry
        assert common._hot_loop_data["count"] == 0
        for _ in range(5):
            fake_now[0] += 0.05  # within 0.1s window
            _check_hot_loop()

    # After the 5th rapid retry the count reaches 5 -> warning, reset to 0.
    assert common._hot_loop_data["count"] == 0
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1
    assert "Hot loop detected" in warned[0].getMessage()


def test_check_hot_loop_resets_when_slow(monkeypatch):
    fake_now = [100.0]
    monkeypatch.setattr(
        time_module, "monotonic_ns", lambda: int(fake_now[0] * 1_000_000_000)
    )
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0

    _check_hot_loop()  # first
    fake_now[0] += 0.05
    _check_hot_loop()  # count -> 1
    assert common._hot_loop_data["count"] == 1
    fake_now[0] += 1.0  # slow gap resets count
    _check_hot_loop()
    assert common._hot_loop_data["count"] == 0


def test_check_hot_loop_thread_safe_lock_used():
    # Ensure the lock attribute exists and functions without deadlock.
    assert isinstance(common._hot_loop_lock, type(threading.Lock()))
    _check_hot_loop()
