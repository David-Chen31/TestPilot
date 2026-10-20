import functools
import logging
import time as time_module

import pytest

from backon import _common
from backon._common import (
    disable,
    enable,
    is_enabled,
    _log_backoff,
    _log_giveup,
    _maybe_call,
    _prepare_logger,
    _is_custom_wait,
    _config_handlers,
    _LazyLogHandler,
    _check_hot_loop,
    _now,
    _elapsed,
)


@pytest.fixture(autouse=True)
def _restore_global_state():
    original_enabled = _common._GLOBAL_ENABLED
    original_test_config = dict(_common._TEST_CONFIG)
    original_hot_loop = dict(_common._hot_loop_data)
    try:
        yield
    finally:
        _common._GLOBAL_ENABLED = original_enabled
        _common._TEST_CONFIG.clear()
        _common._TEST_CONFIG.update(original_test_config)
        _common._hot_loop_data.clear()
        _common._hot_loop_data.update(original_hot_loop)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_is_enabled_true_by_default():
    _common.enable()
    assert is_enabled() is True


def test_disable_sets_flag_false():
    enable()
    assert is_enabled() is True

    disable()
    assert is_enabled() is False
    assert _common._GLOBAL_ENABLED is False


def test_enable_sets_flag_true():
    disable()
    assert is_enabled() is False

    enable()
    assert is_enabled() is True
    assert _common._GLOBAL_ENABLED is True


def test_disable_is_idempotent():
    disable()
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True


def test_enable_is_idempotent():
    enable()
    enable()
    assert is_enabled() is True


def test_is_enabled_returns_bool():
    enable()
    assert is_enabled() is True
    disable()
    assert is_enabled() is False


def test_enable_disable_round_trip():
    for _ in range(3):
        disable()
        assert is_enabled() is False
        enable()
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

def test_logger_name_and_null_handler():
    assert _common._logger.name == "backon"
    assert any(
        isinstance(h, logging.NullHandler) for h in _common._logger.handlers
    )


# ---------------------------------------------------------------------------
# _maybe_call
# ---------------------------------------------------------------------------

def test_maybe_call_non_callable_returns_value():
    assert _maybe_call(5) == 5
    assert _maybe_call("abc") == "abc"
    assert _maybe_call(None) is None


def test_maybe_call_callable_without_args_invokes():
    assert _maybe_call(lambda: 42) == 42


def test_maybe_call_callable_with_args():
    def add(a, b):
        return a + b

    assert _maybe_call(add, 1, 2) == 3
    assert _maybe_call(add, a=1, b=2) == 3


def test_maybe_call_requires_args_returns_callable():
    def needs_arg(x):
        return x

    assert _maybe_call(needs_arg) is needs_arg


def test_maybe_call_propagates_legitimate_type_error():
    def raiser():
        raise TypeError("boom inside the function")

    with pytest.raises(TypeError, match="boom inside the function"):
        _maybe_call(raiser)


def test_maybe_call_propagates_other_exceptions():
    def raiser():
        raise ValueError("not a type error")

    with pytest.raises(ValueError, match="not a type error"):
        _maybe_call(raiser)


def test_maybe_call_inner_type_error_with_required_pos_arg_string():
    def needs_arg(x):
        return x

    result = _maybe_call(needs_arg)
    assert result is needs_arg


# ---------------------------------------------------------------------------
# _prepare_logger
# ---------------------------------------------------------------------------

def test_prepare_logger_string_returns_logger():
    logger = _prepare_logger("backon.test.logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "backon.test.logger"


def test_prepare_logger_none_returns_none():
    assert _prepare_logger(None) is None


def test_prepare_logger_passes_through_logger_instance():
    existing = logging.getLogger("backon.some.logger")
    assert _prepare_logger(existing) is existing


# ---------------------------------------------------------------------------
# _is_custom_wait
# ---------------------------------------------------------------------------

def test_is_custom_wait_accepts_numbers():
    assert _is_custom_wait(1) is True
    assert _is_custom_wait(1.5) is True
    assert _is_custom_wait(0) is True


def test_is_custom_wait_rejects_bool():
    assert _is_custom_wait(True) is False
    assert _is_custom_wait(False) is False


def test_is_custom_wait_rejects_non_numbers():
    assert _is_custom_wait("1") is False
    assert _is_custom_wait(None) is False
    assert _is_custom_wait([1]) is False


# ---------------------------------------------------------------------------
# _log_backoff / _log_giveup
# ---------------------------------------------------------------------------

class _RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg, args))


def _make_target():
    def target():
        pass

    return target


def test_log_backoff_with_exception():
    logger = _RecordingLogger()
    target = _make_target()
    exc = ValueError("bad value")
    details = {"target": target, "wait": 1.5, "exception": exc}

    _log_backoff(details, logger, logging.WARNING)

    assert len(logger.records) == 1
    level, msg, args = logger.records[0]
    assert level == logging.WARNING
    assert msg == "Backing off %s(...) for %.1fs (%s)"
    assert args == (target.__name__, 1.5, "ValueError: bad value")


def test_log_backoff_without_exception_uses_value():
    logger = _RecordingLogger()
    target = _make_target()
    details = {"target": target, "wait": 2.0, "value": "some value"}

    _log_backoff(details, logger, logging.INFO)

    level, msg, args = logger.records[0]
    assert level == logging.INFO
    assert args == (target.__name__, 2.0, "some value")


def test_log_backoff_without_exception_or_value_uses_unknown():
    logger = _RecordingLogger()
    target = _make_target()
    details = {"target": target, "wait": 0.5}

    _log_backoff(details, logger, logging.DEBUG)

    _, _, args = logger.records[0]
    assert args == (target.__name__, 0.5, "unknown")


def test_log_giveup_with_exception():
    logger = _RecordingLogger()
    target = _make_target()
    exc = KeyError("missing")
    details = {"target": target, "tries": 3, "exception": exc}

    _log_giveup(details, logger, logging.ERROR)

    level, msg, args = logger.records[0]
    assert level == logging.ERROR
    assert msg == "Giving up %s(...) after %d tries (%s)"
    assert args == (target.__name__, 3, "KeyError: 'missing'")


def test_log_giveup_without_exception_uses_value():
    logger = _RecordingLogger()
    target = _make_target()
    details = {"target": target, "tries": 2, "value": "retval"}

    _log_giveup(details, logger, logging.INFO)

    _, _, args = logger.records[0]
    assert args == (target.__name__, 2, "retval")


def test_log_giveup_without_exception_or_value_uses_unknown():
    logger = _RecordingLogger()
    target = _make_target()
    details = {"target": target, "tries": 1}

    _log_giveup(details, logger, logging.DEBUG)

    _, _, args = logger.records[0]
    assert args == (target.__name__, 1, "unknown")


# ---------------------------------------------------------------------------
# _TEST_CONFIG-aware helpers
# ---------------------------------------------------------------------------

def test_apply_test_overrides_no_override():
    _common._TEST_CONFIG["max_retries"] = None
    assert _common._apply_test_overrides(max_tries=5, max_time=10) == (5, 10)


def test_apply_test_overrides_with_max_retries():
    _common._TEST_CONFIG["max_retries"] = 3
    assert _common._apply_test_overrides(max_tries=5, max_time=10) == (3, 10)


def test_apply_test_overrides_zero_max_retries():
    # 0 is not None, so it must override.
    _common._TEST_CONFIG["max_retries"] = 0
    assert _common._apply_test_overrides(max_tries=5, max_time=10) == (0, 10)


def test_apply_test_overrides_defaults():
    _common._TEST_CONFIG["max_retries"] = None
    assert _common._apply_test_overrides() == (None, None)


def test_init_wait_gen_resolves_kwargs():
    calls = {}

    def gen(**kwargs):
        calls.update(kwargs)
        return "generator"

    result = _common._init_wait_gen(gen, {"a": lambda: 1, "b": 2})
    assert result == "generator"
    assert calls == {"a": 1, "b": 2}


def test_init_wait_gen_empty_kwargs():
    def gen(**kwargs):
        return kwargs

    assert _common._init_wait_gen(gen, {}) == {}


class _FakeWait:
    def __init__(self, value):
        self.value = value
        self.sent = []

    def next(self, send_value):
        self.sent.append(send_value)
        return self.value


def test_next_wait_without_jitter_or_max_time():
    _common._TEST_CONFIG["backoff_multiplier"] = 1.0
    wait = _FakeWait(2.5)
    seconds = _common._next_wait(wait, None, None, elapsed=0.0, max_time=None)
    assert seconds == 2.5
    assert wait.sent == [None]


def test_next_wait_forwards_send_value():
    _common._TEST_CONFIG["backoff_multiplier"] = 1.0
    wait = _FakeWait(1.0)
    _common._next_wait(wait, "token", None, elapsed=0.0, max_time=None)
    assert wait.sent == ["token"]


def test_next_wait_applies_jitter():
    _common._TEST_CONFIG["backoff_multiplier"] = 1.0
    wait = _FakeWait(3.0)
    seconds = _common._next_wait(
        wait, None, jitter=lambda v: v * 2, elapsed=0.0, max_time=None
    )
    assert seconds == 6.0


def test_next_wait_applies_multiplier():
    _common._TEST_CONFIG["backoff_multiplier"] = 0.5
    wait = _FakeWait(4.0)
    seconds = _common._next_wait(wait, None, None, elapsed=0.0, max_time=None)
    assert seconds == 2.0


def test_next_wait_clamps_to_remaining_max_time():
    _common._TEST_CONFIG["backoff_multiplier"] = 1.0
    wait = _FakeWait(10.0)
    seconds = _common._next_wait(wait, None, None, elapsed=8.0, max_time=10.0)
    assert seconds == 2.0


def test_next_wait_within_max_time_not_clamped():
    _common._TEST_CONFIG["backoff_multiplier"] = 1.0
    wait = _FakeWait(1.0)
    seconds = _common._next_wait(wait, None, None, elapsed=1.0, max_time=10.0)
    assert seconds == 1.0


def test_next_wait_jitter_then_multiplier_order():
    # jitter applied first, then multiplier.
    _common._TEST_CONFIG["backoff_multiplier"] = 3.0
    wait = _FakeWait(2.0)
    seconds = _common._next_wait(
        wait, None, jitter=lambda v: v + 1, elapsed=0.0, max_time=None
    )
    assert seconds == 9.0


# ---------------------------------------------------------------------------
# _now / _elapsed
# ---------------------------------------------------------------------------

def test_now_returns_seconds_scale(monkeypatch):
    monkeypatch.setattr(time_module, "monotonic_ns", lambda: 2_500_000_000)
    assert _now() == 2.5


def test_elapsed_subtracts_start(monkeypatch):
    monkeypatch.setattr(time_module, "monotonic_ns", lambda: 3_000_000_000)
    assert _elapsed(1.0) == 2.0


# ---------------------------------------------------------------------------
# _config_handlers
# ---------------------------------------------------------------------------

def _default_handler(*, logger, log_level):
    def handler(details):
        return ("default", logger, log_level, details)

    return handler


def test_config_handlers_list_and_no_logger_returns_same_list():
    h1 = object()
    user = [h1]
    result = _config_handlers(user)
    assert result is user
    assert result == [h1]


def test_config_handlers_empty_list_no_logger():
    user = []
    assert _config_handlers(user) is user


def test_config_handlers_lazy_default_only():
    lazy = _config_handlers(
        None,
        default_handler=_default_handler,
        logger="some.logger",
        log_level=logging.INFO,
    )
    assert len(lazy) == 1
    assert isinstance(lazy[0], _LazyLogHandler)


def test_config_handlers_lazy_requires_log_level():
    with pytest.raises(AssertionError):
        _config_handlers(None, default_handler=_default_handler, logger="x")


def test_config_handlers_with_logger_and_user_handlers_iterable():
    user = [object(), object()]
    handlers = _config_handlers(
        user,
        default_handler=_default_handler,
        logger="L",
        log_level=logging.WARNING,
    )
    assert len(handlers) == 3
    # first is a partial of the default handler with logger/log_level bound
    first = handlers[0]
    assert isinstance(first, functools.partial)
    assert first.func is _default_handler
    assert first.keywords == {"logger": "L", "log_level": logging.WARNING}
    assert handlers[1:] == user


def test_config_handlers_with_logger_and_single_user_handler():
    single = object()
    handlers = _config_handlers(
        single,
        default_handler=_default_handler,
        logger="L",
        log_level=logging.ERROR,
    )
    assert len(handlers) == 2
    assert handlers[1] is single


def test_config_handlers_with_logger_and_no_user_handlers_branch():
    # user_handlers is None and logger is provided -> lazy branch.
    handlers = _config_handlers(
        None,
        default_handler=_default_handler,
        logger="L",
        log_level=logging.DEBUG,
    )
    assert len(handlers) == 1
    assert isinstance(handlers[0], _LazyLogHandler)


def test_config_handlers_no_logger_iterable_user_handlers():
    user = [object(), object()]
    handlers = _config_handlers(user)
    # user_handlers is a list without logger -> returned as-is
    assert handlers is user


def test_config_handlers_no_logger_single_user_handler():
    single = object()
    handlers = _config_handlers(single)
    assert handlers == [single]


def test_config_handlers_no_logger_none_user_handlers():
    handlers = _config_handlers(None)
    assert handlers == []


# ---------------------------------------------------------------------------
# _LazyLogHandler
#
# NOTE: The factory stored by `_LazyLogHandler` wraps the given `handler` in
# `functools.partial(handler, logger=..., log_level=...)`. It does NOT invoke
# `handler` to obtain the final callable. Therefore the `handler` passed in
# must itself be a callable accepting `(details, *, logger, log_level)`.
# ---------------------------------------------------------------------------

class _RecordingHandler:
    def __init__(self):
        self.calls = []

    def __call__(self, details, *, logger, log_level):
        self.calls.append((logger, log_level, details))

    def __call_record__(self):  # pragma: no cover - helper only
        return self.calls


def test_lazy_log_handler_init_state():
    lazy = _LazyLogHandler(_RecordingHandler(), "L", logging.INFO)
    assert lazy._handler is None
    assert callable(lazy._factory)


def test_lazy_log_handler_builds_and_calls_on_first_use():
    inner = _RecordingHandler()

    lazy = _LazyLogHandler(inner, "L", logging.INFO)
    details = {"x": 1}
    lazy(details)

    # `handler` is invoked by the partial with logger/log_level bound and
    # `details` supplied positionally.
    assert inner.calls == [("L", logging.INFO, details)]

    first_handler = lazy._handler
    assert first_handler is not None
    lazy(details)
    assert lazy._handler is first_handler
    assert inner.calls == [
        ("L", logging.INFO, details),
        ("L", logging.INFO, details),
    ]


def test_lazy_log_handler_factory_passes_params():
    seen = []

    def handler(details, *, logger, log_level):
        seen.append((logger, log_level, details))

    lazy = _LazyLogHandler(handler, "mylogger", logging.WARNING)
    lazy({"a": 1})
    assert seen == [("mylogger", logging.WARNING, {"a": 1})]


def test_lazy_log_handler_factory_is_partial():
    inner = _RecordingHandler()
    lazy = _LazyLogHandler(inner, "L", logging.INFO)
    built = lazy._factory()
    assert isinstance(built, functools.partial)
    assert built.func is inner
    assert built.keywords == {"logger": "L", "log_level": logging.INFO}


# ---------------------------------------------------------------------------
# _check_hot_loop
# ---------------------------------------------------------------------------

def test_check_hot_loop_single_retry_no_warning(monkeypatch, caplog):
    _common._hot_loop_data["last_retry"] = 0.0
    _common._hot_loop_data["count"] = 0
    monkeypatch.setattr(time_module, "monotonic_ns", lambda: 1_000_000_000)
    with caplog.at_level(logging.WARNING, logger="backon"):
        _check_hot_loop()
    assert _common._hot_loop_data["count"] == 0
    assert _common._hot_loop_data["last_retry"] == 1.0
    assert not [r for r in caplog.records if "Hot loop" in r.message]


def test_check_hot_loop_triggers_warning_after_six_rapid_retries(
    monkeypatch, caplog
):
    _common._hot_loop_data["last_retry"] = 0.0
    _common._hot_loop_data["count"] = 0
    # Fixed small time step < 0.1s so gaps qualify as rapid.
    times = iter([1.0 + i * 0.01 for i in range(10)])
    monkeypatch.setattr(time_module, "monotonic_ns",
                        lambda: int(next(times) * 1_000_000_000))
    with caplog.at_level(logging.WARNING, logger="backon"):
        for _ in range(6):
            _check_hot_loop()
    warnings = [r for r in caplog.records if "Hot loop" in r.message]
    assert len(warnings) >= 1
    # Count resets after warning
    assert _common._hot_loop_data["count"] == 0


def test_check_hot_loop_resets_when_gap_large(monkeypatch):
    _common._hot_loop_data["last_retry"] = 1.0
    _common._hot_loop_data["count"] = 3
    monkeypatch.setattr(time_module, "monotonic_ns", lambda: 2_000_000_000)
    _check_hot_loop()
    assert _common._hot_loop_data["count"] == 0
    assert _common._hot_loop_data["last_retry"] == 2.0
