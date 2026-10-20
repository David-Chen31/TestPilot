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
    e = SkipMemory("x")
    assert e.args == ("x",)


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic_caching():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]
    assert f.memory == {2: 4}


def test_memoize_different_args():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x + 1

    assert f(1) == 2
    assert f(2) == 3
    assert f(1) == 2
    assert calls == [1, 2]
    assert f.memory == {1: 2, 2: 3}


def test_memoize_kwargs_distinct_from_positional():
    calls = []

    @memoize
    def f(*args, **kwargs):
        calls.append((args, kwargs))
        return len(args) + len(kwargs)

    assert f(1) == 1
    assert f(x=1) == 1
    assert calls == [((1,), {}), ((), {"x": 1})]
    assert len(f.memory) == 2


def test_memoize_kwargs_key_stable_regardless_of_order():
    calls = []

    @memoize
    def f(**kwargs):
        calls.append(dict(kwargs))
        return sum(kwargs.values())

    assert f(a=1, b=2) == 3
    assert f(b=2, a=1) == 3
    assert calls == [{"a": 1, "b": 2}]


def test_memoize_with_key_func():
    calls = []

    @memoize(key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x * 10

    assert f(1) == 10
    assert f(3) == 10  # same key (1 % 2 == 3 % 2 == 1)
    assert calls == [1]
    assert f.memory == {1: 10}


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
    assert f.memory == {}
    f(1)
    assert calls == [1, 1]


def test_memoize_invalidate_with_kwargs():
    @memoize
    def f(*args, **kwargs):
        return (args, kwargs)

    f(1, a=2)
    assert len(f.memory) == 1
    f.invalidate(1, a=2)
    assert f.memory == {}
    # invalidate with different key is a no-op
    f.invalidate(1)
    assert f.memory == {}


def test_memoize_invalidate_all():
    @memoize
    def f(x):
        return x

    f(1)
    f(2)
    assert len(f.memory) == 2
    f.invalidate_all()
    assert f.memory == {}


def test_memoize_skip_memory_returns_value():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        if x < 0:
            raise memoize.skip("skipped")
        return x

    assert f(-1) == "skipped"
    # not cached, called again
    assert f(-1) == "skipped"
    assert calls == [-1, -1]
    assert f.memory == {}


def test_memoize_skip_memory_no_args_returns_none():
    @memoize
    def f(x):
        raise memoize.skip()

    assert f(5) is None
    assert f.memory == {}


def test_memoize_skip_alias():
    assert memoize.skip is SkipMemory


def test_memoize_preserves_name():
    @memoize
    def my_func(x):
        return x

    assert my_func.__name__ == "my_func"


def test_memoize_used_without_parens_on_callable():
    # memoize(func) form
    def raw(x):
        return x + 1

    f = memoize(raw)
    assert f(1) == 2
    assert f.memory == {1: 2}


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def test_cache_basic():
    calls = []

    @cache(1000)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]
    assert isinstance(f.memory, CacheMemory)


def test_cache_skip_alias():
    assert cache.skip is SkipMemory


def test_cache_timedelta_timeout():
    @cache(timedelta(seconds=1000))
    def f(x):
        return x

    assert f(1) == 1
    assert f.memory.timeout == 1000.0


def test_cache_invalidate():
    calls = []

    @cache(1000)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1]
    f.invalidate(1)
    f(1)
    assert calls == [1, 1]


def test_cache_invalidate_all():
    @cache(1000)
    def f(x):
        return x

    f(1)
    f(2)
    f.invalidate_all()
    assert dict(f.memory) == {}


def test_cache_uses_key_func():
    @cache(1000, key_func=lambda x: x % 2)
    def f(x):
        return x

    assert f(1) == 1
    assert f(3) == 1  # cache hit on key 1


# ---------------------------------------------------------------------------
# CacheMemory
# ---------------------------------------------------------------------------

def test_cachememory_set_get():
    m = CacheMemory(1000)
    m["a"] = 1
    assert m["a"] == 1
    assert dict.__getitem__(m, "a")[0] == 1


def test_cachememory_missing_key_raises():
    m = CacheMemory(1000)
    with pytest.raises(KeyError):
        m["missing"]


def test_cachememory_clear_resets_structures():
    m = CacheMemory(1000)
    m["a"] = 1
    m.clear()
    assert dict(m) == {}
    assert list(m._keys) == []
    assert list(m._expires) == []


def test_cachememory_expire_immediately():
    # zero timeout => entry expires immediately
    m = CacheMemory(0)
    m["a"] = 1
    with pytest.raises(KeyError):
        m["a"]
    assert "a" not in m


def test_cachememory_expire_removes_entry():
    m = CacheMemory(1000)
    m["a"] = 1
    # Manually force expiry by manipulating stored expiration
    value, _ = dict.__getitem__(m, "a")
    dict.__setitem__(m, "a", (value, 0))
    m._expires[-1] = 0
    with pytest.raises(KeyError):
        m["a"]
    assert "a" not in m


def test_cachememory_expire_does_not_remove_fresher_duplicate():
    m = CacheMemory(1000)
    # Insert same key twice via __setitem__
    m["a"] = 1
    m["a"] = 2
    # Mark first expiration as expired, second as fresh
    m._expires[0] = 0
    m._expires[1] = time.monotonic() + 1000
    m.expire()
    # The newer entry should remain
    assert m["a"] == 2


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_plain_arg():
    def f(x):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_no_args():
    def f():
        pass

    assert has_arg_types(f) == (False, False)


def test_has_arg_types_kwargs():
    def f(**kwargs):
        pass

    assert has_arg_types(f) == (False, True)


def test_has_arg_types_keyword_only():
    def f(*, x):
        pass

    assert has_arg_types(f) == (False, True)


def test_has_arg_types_var_positional():
    def f(*args):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_mixed():
    def f(a, *args, **kwargs):
        pass

    assert has_arg_types(f) == (True, True)


# ---------------------------------------------------------------------------
# make_lookuper / silent_lookuper
# ---------------------------------------------------------------------------

def test_make_lookuper_basic():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {1: "one", 2: "two"}

    assert table(1) == "one"
    assert table(2) == "two"
    assert calls == [1]


def test_make_lookuper_raises_lookup_error():
    @make_lookuper
    def table():
        return {1: "one"}

    with pytest.raises(LookupError):
        table(3)


def test_make_lookuper_caches_empty_result_avoiding_refill():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {}

    with pytest.raises(LookupError):
        table(1)
    with pytest.raises(LookupError):
        table(2)
    # Should only be called once due to sentinel
    assert calls == [1]


def test_silent_lookuper_returns_none():
    @silent_lookuper
    def table():
        return {1: "one"}

    assert table(1) == "one"
    assert table(3) is None


def test_silent_lookuper_name():
    assert silent_lookuper.__name__ == "silent_lookuper"


def test_make_lookuper_with_args():
    @make_lookuper
    def table(prefix):
        return {1: prefix + "one", 2: prefix + "two"}

    assert table("a", 1) == "aone"
    assert table("a", 2) == "atwo"
    assert table("b", 1) == "bone"


def test_make_lookuper_with_args_raises():
    @make_lookuper
    def table(prefix):
        return {}

    with pytest.raises(LookupError):
        table("a", 1)


def test_make_lookuper_rejects_kwargs():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(**kwargs):
            return {}


def test_make_lookuper_rejects_keyword_only():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(*, x):
            return {}


def test_lookuper_propagates_exception_and_retries():
    calls = []

    @silent_lookuper
    def table():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("boom")
        return {1: "one"}

    with pytest.raises(ValueError):
        table(1)
    # After failure, memory is cleared, so it retries
    assert table(1) == "one"
    assert calls == [1, 1]


def test_make_lookuper_preserves_name():
    @make_lookuper
    def my_table():
        return {}

    assert my_table.__name__ == "my_table"
