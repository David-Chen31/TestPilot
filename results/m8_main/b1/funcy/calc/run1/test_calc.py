from datetime import timedelta
import functools
import inspect

import pytest

from funcy.calc import (
    SkipMemory,
    memoize,
    cache,
    CacheMemory,
    has_arg_types,
    make_lookuper,
    silent_lookuper,
)


# ---------------------------------------------------------------------------
# SkipMemory
# ---------------------------------------------------------------------------

def test_skip_memory_is_exception():
    assert issubclass(SkipMemory, Exception)
    e = SkipMemory("some message")
    assert e.args == ("some message",)


def test_memoize_skip_alias():
    assert memoize.skip is SkipMemory


def test_cache_skip_alias():
    assert cache.skip is SkipMemory


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_plain_decorator():
    calls = []

    @memoize
    def double(x):
        calls.append(x)
        return x * 2

    assert double(3) == 6
    assert double(3) == 6
    assert calls == [3]


def test_memoize_called_with_args_no_kwargs():
    calls = []

    @memoize()
    def add(a, b):
        calls.append((a, b))
        return a + b

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert add(1, 3) == 4
    assert calls == [(1, 2), (1, 3)]


def test_memoize_with_kwargs():
    calls = []

    @memoize
    def fn(a, b=0):
        calls.append((a, b))
        return a + b

    assert fn(1, b=2) == 3
    assert fn(1, b=2) == 3
    assert calls == [(1, 2)]


def test_memoize_kwargs_order_independent():
    calls = []

    @memoize
    def fn(**kwargs):
        calls.append(kwargs)
        return sorted(kwargs.items())

    assert fn(a=1, b=2) == [("a", 1), ("b", 2)]
    assert fn(b=2, a=1) == [("a", 1), ("b", 2)]
    assert len(calls) == 1


def test_memoize_kwargs_vs_positional_distinct():
    calls = []

    @memoize
    def fn(a, b=0):
        calls.append((a, b))
        return a + b

    assert fn(1, 2) == 3
    assert fn(1, b=2) == 3
    assert len(calls) == 2


def test_memoize_no_kwargs_key_is_args_tuple():
    @memoize
    def f(a, b):
        return a + b

    f(1, 2)
    assert (1, 2) in f.memory
    assert f.memory[(1, 2)] == 3


def test_memoize_with_kwargs_key_format():
    from funcy.calc import _KWARGS_MARK

    @memoize
    def f(a, b=0):
        return a + b

    f(1, b=2)
    expected_key = (_KWARGS_MARK, (1,), frozenset({("b", 2)}))
    assert f.memory[expected_key] == 3


def test_memoize_key_func():
    calls = []

    @memoize(key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(3) == 1  # same key, cached value from first call
    assert calls == [1]
    assert f(2) == 2


def test_memoize_memory_attribute_is_dict():
    @memoize
    def f(x):
        return x

    assert isinstance(f.memory, dict)
    f(5)
    assert f.memory == {(5,): 5}


def test_memoize_invalidate():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 10

    assert f(2) == 20
    assert f(2) == 20
    assert calls == [2]
    f.invalidate(2)
    assert f(2) == 20
    assert calls == [2, 2]


def test_memoize_invalidate_missing_key_is_noop():
    @memoize
    def f(x):
        return x

    f.invalidate(123)  # shouldn't raise
    assert f.memory == {}


def test_memoize_invalidate_with_kwargs():
    calls = []

    @memoize
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    f(1, b=2)
    f.invalidate(1, b=2)
    assert f.memory == {}
    f(1, b=2)
    assert calls == [(1, 2), (1, 2)]


def test_memoize_invalidate_all():
    @memoize
    def f(x):
        return x

    f(1)
    f(2)
    assert len(f.memory) == 2
    f.invalidate_all()
    assert f.memory == {}


def test_memoize_skip_returns_arg_or_none():
    @memoize
    def f(x):
        raise SkipMemory(x * 2)

    assert f(3) == 6


def test_memoize_skip_no_args_returns_none():
    @memoize
    def f(x):
        raise SkipMemory

    assert f(3) is None


def test_memoize_preserves_metadata():
    @memoize
    def my_func(x):
        """docstring here"""
        return x

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "docstring here"


def test_memoize_does_not_cache_skipped_values():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        raise SkipMemory(x)

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1, 1]


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

class FakeClock(object):
    def __init__(self, value=1000.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, delta):
        self.value += delta


def test_cache_basic(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    calls = []

    @cache(10)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]


def test_cache_expires(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    calls = []

    @cache(10)
    def f(x):
        calls.append(x)
        return x

    f(1)
    clock.advance(5)
    f(1)
    assert calls == [1]
    clock.advance(6)
    f(1)
    assert calls == [1, 1]


def test_cache_with_timedelta(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    calls = []

    @cache(timedelta(seconds=3))
    def f(x):
        calls.append(x)
        return x

    f(1)
    clock.advance(1)
    f(1)
    assert calls == [1]
    clock.advance(3)
    f(1)
    assert calls == [1, 1]


def test_cache_memory_is_cache_memory():
    @cache(1)
    def f(x):
        return x

    assert isinstance(f.memory, CacheMemory)
    f(1)
    assert f.memory[(1,)] == 1


def test_cache_key_func():
    calls = []

    @cache(1000, key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(3) == 1
    assert calls == [1]


def test_cache_invalidate():
    calls = []

    @cache(100)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f.invalidate(1)
    f(1)
    assert calls == [1, 1]


def test_cache_skip(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    @cache(10)
    def f(x):
        raise SkipMemory(x + 100)

    assert f(1) == 101


def test_cache_skip_no_args(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    @cache(10)
    def f(x):
        raise SkipMemory

    assert f(1) is None


# ---------------------------------------------------------------------------
# CacheMemory
# ---------------------------------------------------------------------------

def test_cache_memory_setitem_getitem(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    m = CacheMemory(10)
    m["a"] = 1
    assert m["a"] == 1
    # Stored as (value, expires_at)
    assert dict.__getitem__(m, "a") == (1, 1010.0)


def test_cache_memory_expiry_raises_keyerror(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    m = CacheMemory(10)
    m["a"] = 1
    clock.advance(11)
    with pytest.raises(KeyError):
        m["a"]


def test_cache_memory_missing_key_raises(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    m = CacheMemory(10)
    with pytest.raises(KeyError):
        m["missing"]


def test_cache_memory_clear(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    m = CacheMemory(10)
    m["a"] = 1
    m["b"] = 2
    m.clear()
    assert dict.__len__(m) == 0
    assert list(m._keys) == []
    assert list(m._expires) == []


def test_cache_memory_timeout_attribute(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    m = CacheMemory(42)
    assert m.timeout == 42


def test_cache_memory_expire_purges_old_entries(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    m = CacheMemory(10)
    m["a"] = 1
    clock.advance(5)
    m["b"] = 2
    clock.advance(6)  # "a" expired, "b" not
    m.expire()
    assert "a" not in dict.keys(m)
    assert "b" in dict.keys(m)


def test_cache_memory_expire_nothing_when_unexpired(monkeypatch):
    from funcy import calc

    clock = FakeClock()
    monkeypatch.setattr(calc.time, "monotonic", clock)

    m = CacheMemory(10)
    m["a"] = 1
    m.expire()
    assert "a" in dict.keys(m)


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_positional():
    def f(a, b):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_keyword_only():
    def f(*, a):
        pass

    assert has_arg_types(f) == (False, True)


def test_has_arg_types_varargs_varkw():
    def f(*args, **kwargs):
        pass

    assert has_arg_types(f) == (True, True)


def test_has_arg_types_no_args():
    def f():
        pass

    assert has_arg_types(f) == (False, False)


def test_has_arg_types_positional_only():
    def f(a, /):
        pass

    assert has_arg_types(f) == (True, False)


# ---------------------------------------------------------------------------
# make_lookuper / silent_lookuper
# ---------------------------------------------------------------------------

def test_make_lookuper_basic():
    @make_lookuper
    def lookup():
        return {"a": 1, "b": 2}

    assert lookup("a") == 1
    assert lookup("b") == 2
    with pytest.raises(LookupError):
        lookup("c")


def test_silent_lookuper_basic():
    @silent_lookuper
    def lookup():
        return {"a": 1, "b": 2}

    assert lookup("a") == 1
    assert lookup("missing") is None


def test_silent_lookuper_name():
    assert silent_lookuper.__name__ == "silent_lookuper"


def test_lookuper_calls_builder_once():
    calls = []

    @make_lookuper
    def lookup():
        calls.append(1)
        return {"x": 10}

    lookup("x")
    lookup("x")
    assert len(calls) == 1


def test_lookuper_builder_failure_propagates_and_resets():
    state = {"count": 0}

    @make_lookuper
    def lookup():
        state["count"] += 1
        if state["count"] == 1:
            raise ValueError("boom")
        return {"a": 1}

    with pytest.raises(ValueError):
        lookup("a")
    # Retries since memory was cleared
    assert lookup("a") == 1


def test_lookuper_with_args():
    # When the builder function takes arguments, the decorated function itself
    # becomes a memoized factory: calling it returns a per-args lookuper.
    @make_lookuper
    def lookup(table):
        return {t: len(t) for t in table}

    sub = lookup(("a",))
    assert callable(sub)
    assert sub("a") == 1

    sub_ab = lookup(("ab",))
    assert callable(sub_ab)
    assert sub_ab("ab") == 2


def test_lookuper_with_args_lookup_error():
    @make_lookuper
    def lookup(table):
        return {t: len(t) for t in table}

    sub = lookup(("missing",))
    with pytest.raises(LookupError):
        sub("other")


def test_silent_lookuper_with_args():
    # With args, silent_lookuper behaves identically to make_lookuper in
    # structure: it returns a per-args lookuper.
    @silent_lookuper
    def lookup(table):
        return {t: len(t) for t in table}

    sub = lookup(("abc",))
    assert callable(sub)
    assert sub("abc") == 3


def test_lookuper_with_args_memoizes_factory():
    calls = []

    @make_lookuper
    def lookup(table):
        calls.append(table)
        return {t: len(t) for t in table}

    a1 = lookup(("a",))
    a2 = lookup(("a",))
    # Factory is memoized: same returned object; builder not called yet (lazy).
    assert a1 is a2
    assert calls == []
    # Builder is invoked lazily on first actual lookup, and only once.
    assert a1("a") == 1
    assert a1("a") == 1
    assert calls == [("a",)]


def test_lookuper_rejects_keyword_args():
    with pytest.raises(AssertionError):
        @make_lookuper
        def lookup(**kwargs):
            return {}


def test_lookuper_rejects_keyword_only_args():
    with pytest.raises(AssertionError):
        @make_lookuper
        def lookup(*, kw):
            return {}


def test_lookuper_preserves_metadata():
    @make_lookuper
    def lookup():
        """my doc"""
        return {}

    assert lookup.__name__ == "lookup"
    assert lookup.__doc__ == "my doc"
