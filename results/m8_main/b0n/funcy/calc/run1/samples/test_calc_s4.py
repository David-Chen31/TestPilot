import time
from datetime import timedelta

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


def test_skip_memory_args():
    e = SkipMemory("value")
    assert e.args == ("value",)


def test_skip_memory_no_args():
    e = SkipMemory()
    assert e.args == ()


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]
    assert f(4) == 8
    assert calls == [3, 4]


def test_memoize_exposes_memory():
    @memoize
    def f(x):
        return x + 1

    f(1)
    assert f.memory == {(1,): 2}


def test_memoize_with_kwargs():
    calls = []

    @memoize
    def f(a, b=1):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    assert f(1, b=3) == 4
    # positional args form a separate key than kwargs
    assert f(1, 2) == 3
    assert f(1, 2) == 3
    assert calls == [(1, 2), (1, 3), (1, 2)]


def test_memoize_positional_vs_keyword_distinct():
    @memoize
    def f(a, b):
        return a + b

    assert f(1, 2) == 3
    assert f(1, b=2) == 3
    # two distinct keys: positional args tuple, and marked kwargs key
    assert len(f.memory) == 2
    assert (1, 2) in f.memory


def test_memoize_key_func():
    calls = []

    @memoize(key_func=lambda a, b=0: a)
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1, b=5) == 6
    assert f(1, b=100) == 6  # same key, cached
    assert calls == [(1, 5)]


def test_memoize_key_func_modulo():
    @memoize(key_func=lambda x: x % 2)
    def f(x):
        return x

    assert f(2) == 2
    assert f(4) == 2  # cached under key 0


def test_memoize_invalidate():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1]
    f.invalidate(1)
    f(1)
    assert calls == [1, 1]


def test_memoize_invalidate_with_kwargs():
    calls = []

    @memoize
    def f(a, b=1):
        calls.append((a, b))
        return a + b

    f(1, b=2)
    f.invalidate(1, b=2)
    f(1, b=2)
    assert calls == [(1, 2), (1, 2)]


def test_memoize_invalidate_nonexistent():
    @memoize
    def f(x):
        return x

    f(1)
    f.invalidate(99)  # should not raise
    assert f.memory == {(1,): 1}


def test_memoize_invalidate_all():
    @memoize
    def f(x):
        return x

    f(1)
    f(2)
    assert len(f.memory) == 2
    f.invalidate_all()
    assert f.memory == {}


def test_memoize_skip():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        if x < 0:
            raise memoize.skip("negative")
        return x

    assert f(-1) == "negative"
    assert f(-1) == "negative"  # still skipped, not cached
    assert calls == [-1, -1]
    assert f(5) == 5
    assert f(5) == 5
    assert calls == [-1, -1, 5]
    assert -1 not in f.memory  # nothing cached under -1 key
    assert (5,) in f.memory


def test_memoize_skip_no_args():
    @memoize
    def f():
        raise memoize.skip

    assert f() is None


def test_memoize_skip_attribute_is_skipexception():
    assert memoize.skip is SkipMemory


def test_memoize_preserves_metadata():
    @memoize
    def foo(x):
        """docstring"""
        return x

    assert foo.__name__ == "foo"
    assert foo.__doc__ == "docstring"


def test_memoize_called_with_key_func():
    @memoize(key_func=lambda x: x)
    def f(x):
        return x * 10

    assert f(2) == 20
    assert f.memory == {2: 20}


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def test_cache_basic():
    calls = []

    @cache(10)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]
    assert isinstance(f.memory, CacheMemory)


def test_cache_with_timedelta():
    @cache(timedelta(seconds=10))
    def f(x):
        return x

    assert f(1) == 1
    assert isinstance(f.memory, CacheMemory)
    assert f.memory.timeout == 10.0


def test_cache_skip():
    @cache(10)
    def f(x):
        raise cache.skip("skipped")

    assert f(1) == "skipped"


def test_cache_skip_no_args():
    @cache(10)
    def f(x):
        raise cache.skip

    assert f(1) is None


def test_cache_skip_attribute_is_skipexception():
    assert cache.skip is SkipMemory


def test_cache_invalidate():
    calls = []

    @cache(10)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f.invalidate(1)
    f(1)
    assert calls == [1, 1]


def test_cache_invalidate_all():
    @cache(10)
    def f(x):
        return x

    f(1)
    f(2)
    assert len(f.memory) == 2
    f.invalidate_all()
    assert len(f.memory) == 0


def test_cache_key_func():
    calls = []

    @cache(10, key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(2) == 2
    assert f(4) == 2
    assert calls == [2]


# ---------------------------------------------------------------------------
# CacheMemory
# ---------------------------------------------------------------------------

def test_cache_memory_set_get():
    m = CacheMemory(1000)
    m["a"] = 1
    assert m["a"] == 1
    assert "a" in m


def test_cache_memory_expired():
    m = CacheMemory(0)
    m["a"] = 1
    time.sleep(0.001)
    with pytest.raises(KeyError):
        m["a"]


def test_cache_memory_clear():
    m = CacheMemory(1000)
    m["a"] = 1
    m["b"] = 2
    assert len(m) == 2
    m.clear()
    assert len(m) == 0
    assert len(m._keys) == 0
    assert len(m._expires) == 0


def test_cache_memory_missing_key():
    m = CacheMemory(1000)
    with pytest.raises(KeyError):
        m["nope"]


def test_cache_memory_init_clears():
    m = CacheMemory(5)
    assert m.timeout == 5
    assert len(m) == 0


def test_cache_memory_expire():
    m = CacheMemory(0)
    m["x"] = 10
    m["y"] = 20
    time.sleep(0.001)
    m.expire()
    assert len(m) == 0


def test_cache_memory_expire_leaves_fresh():
    m = CacheMemory(1000)
    m["x"] = 1
    m.expire()
    assert m["x"] == 1


def test_cache_memory_getitem_expires():
    m = CacheMemory(0)
    m["k"] = 1
    time.sleep(0.001)
    with pytest.raises(KeyError):
        m["k"]
    assert "k" not in m


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_no_args():
    def f():
        pass

    assert has_arg_types(f) == (False, False)


def test_has_arg_types_positional():
    def f(a, b):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_keyword_only():
    def f(*, a):
        pass

    assert has_arg_types(f) == (False, True)


def test_has_arg_types_var_positional():
    def f(*args):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_var_keyword():
    def f(**kwargs):
        pass

    assert has_arg_types(f) == (False, True)


def test_has_arg_types_mixed():
    def f(a, *, b):
        pass

    assert has_arg_types(f) == (True, True)


def test_has_arg_types_defaults():
    def f(a=1, *, b=2):
        pass

    assert has_arg_types(f) == (True, True)


def test_has_arg_types_positional_only_lambda():
    f = lambda a: a
    assert has_arg_types(f) == (True, False)


# ---------------------------------------------------------------------------
# make_lookuper / silent_lookuper
# ---------------------------------------------------------------------------

def test_make_lookuper_basic():
    calls = []

    @make_lookuper
    def lookup():
        calls.append(1)
        return {1: "one", 2: "two"}

    assert lookup(1) == "one"
    assert lookup(2) == "two"
    assert calls == [1]  # built once


def test_make_lookuper_miss_raises():
    @make_lookuper
    def lookup():
        return {1: "one"}

    assert lookup(1) == "one"
    with pytest.raises(LookupError):
        lookup(99)


def test_silent_lookuper_miss_returns_none():
    @silent_lookuper
    def lookup():
        return {1: "one"}

    assert lookup(1) == "one"
    assert lookup(99) is None


def test_make_lookuper_empty_result_builds_once():
    # After a successful (but empty) build, a sentinel prevents refill
    calls = []

    @make_lookuper
    def lookup():
        calls.append(1)
        return {}

    with pytest.raises(LookupError):
        lookup(1)
    with pytest.raises(LookupError):
        lookup(1)
    assert len(calls) == 1


def test_make_lookuper_with_args():
    @make_lookuper
    def lookup(x):
        return {x: x * 10}

    result = lookup(1)
    assert callable(result)
    assert result(1) == 10
    # memoized: same result object returned for same args
    assert lookup(1) is result


def test_make_lookuper_with_args_multiple():
    @make_lookuper
    def lookup(a, b):
        return {a + b: a * b}

    r1 = lookup(1, 2)  # func(1, 2) == {3: 2}
    r2 = lookup(2, 3)  # func(2, 3) == {5: 6}
    assert r1(3) == 2
    assert r2(5) == 6


def test_make_lookuper_with_kwargs_assertion():
    with pytest.raises(AssertionError):
        @make_lookuper
        def lookup(*, a):
            return {a: a}


def test_silent_lookuper_with_kwargs_assertion():
    with pytest.raises(AssertionError):
        @silent_lookuper
        def lookup(**kwargs):
            return kwargs


def test_silent_lookuper_result_build_error_clears():
    state = {"fail": True}

    @silent_lookuper
    def lookup():
        if state["fail"]:
            raise ValueError("boom")
        return {1: "one"}

    with pytest.raises(ValueError):
        lookup(1)

    state["fail"] = False
    assert lookup(1) == "one"


def test_make_lookuper_preserves_metadata():
    @make_lookuper
    def lookup():
        """doc"""
        return {}

    assert lookup.__name__ == "lookup"
    assert lookup.__doc__ == "doc"


def test_silent_lookuper_name():
    assert silent_lookuper.__name__ == "silent_lookuper"
