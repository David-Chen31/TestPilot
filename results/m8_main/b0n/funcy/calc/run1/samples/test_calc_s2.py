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
    with pytest.raises(SkipMemory):
        raise SkipMemory("x")


def test_memoize_skip_is_skip_memory():
    assert memoize.skip is SkipMemory


def test_cache_skip_is_skip_memory():
    assert cache.skip is SkipMemory


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


def test_memoize_kwargs_key():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert calls == [(1, 2)]
    # positional vs kwargs yield distinct keys
    assert f(1, 2) == 3
    assert len(calls) == 2


def test_memoize_kwargs_order_independent():
    calls = []

    @memoize
    def f(**kw):
        calls.append(dict(kw))
        return sum(kw.values())

    assert f(a=1, b=2) == 3
    assert f(b=2, a=1) == 3
    assert len(calls) == 1


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


def test_memoize_invalidate_missing():
    @memoize
    def f(x):
        return x

    # Should not raise
    f.invalidate(42)


def test_memoize_invalidate_all():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f.invalidate_all()
    assert f.memory == {}
    f(1)
    f(2)
    assert calls == [1, 2, 1, 2]


def test_memoize_skip():
    @memoize
    def f(x):
        if x == 0:
            raise memoize.skip("zero")
        return x

    assert f(0) == "zero"
    assert f(0) == "zero"
    assert f.memory == {}


def test_memoize_skip_no_args():
    @memoize
    def f(x):
        raise memoize.skip

    assert f(1) is None


def test_memoize_key_func():
    calls = []

    @memoize(key_func=lambda x, y: x)
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 2) == 3
    assert f(1, 100) == 101  # different y but same key
    assert calls == [(1, 2)]
    assert f.memory == {1: 3}


def test_memoize_key_func_invalidate():
    calls = []

    @memoize(key_func=lambda x: x)
    def f(x):
        calls.append(x)
        return x * 10

    f(5)
    f.invalidate(5)
    f(5)
    assert calls == [5, 5]


def test_memoize_preserves_metadata():
    @memoize
    def f(x):
        """docstring"""
        return x

    assert f.__name__ == "f"
    assert f.__doc__ == "docstring"


def test_memoize_no_parens_and_parens():
    @memoize
    def f(x):
        return x

    @memoize()
    def g(x):
        return x

    assert f(1) == 1
    assert g(1) == 1


# ---------------------------------------------------------------------------
# cache & CacheMemory
# ---------------------------------------------------------------------------

def test_cache_basic():
    calls = []

    @cache(60)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_cache_exposes_memory():
    @cache(60)
    def f(x):
        return x

    f(1)
    assert isinstance(f.memory, CacheMemory)
    assert f.memory.timeout == 60


def test_cache_timedelta_timeout():
    @cache(timedelta(minutes=1))
    def f(x):
        return x

    assert f.memory.timeout == 60.0


def test_cache_invalidate_and_invalidate_all():
    calls = []

    @cache(60)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f.invalidate(1)
    f(1)
    assert calls == [1, 1]

    f(2)
    f.invalidate_all()
    assert f.memory == {}
    f(1)
    f(2)
    assert calls == [1, 1, 2, 1, 2]


def test_cache_skip():
    @cache(60)
    def f(x):
        if x < 0:
            raise cache.skip("neg")
        return x

    assert f(-1) == "neg"
    assert f.memory == {}


def test_cache_with_key_func():
    calls = []

    @cache(60, key_func=lambda x, y: x)
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 2) == 3
    assert f(1, 10) == 11
    assert calls == [(1, 2)]


def test_cache_memory_expiry():
    mem = CacheMemory(0)
    mem["k"] = "v"
    # timeout is 0, so expiry time is <= now on next access
    time.sleep(0.001)
    with pytest.raises(KeyError):
        mem["k"]
    assert "k" not in mem


def test_cache_memory_getitem_valid():
    mem = CacheMemory(1000)
    mem["k"] = "v"
    assert mem["k"] == "v"


def test_cache_memory_clear():
    mem = CacheMemory(1000)
    mem["a"] = 1
    mem["b"] = 2
    mem.clear()
    assert len(mem) == 0
    assert len(mem._keys) == 0
    assert len(mem._expires) == 0


def test_cache_memory_init_clears():
    mem = CacheMemory(1000)
    mem["x"] = 1
    # there is no re-init, but clear should reset
    mem.clear()
    assert dict(mem) == {}


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_plain():
    def f(a, b):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_no_args():
    def f():
        pass

    assert has_arg_types(f) == (False, False)


def test_has_arg_types_kwargs():
    def f(a, **kw):
        pass

    assert has_arg_types(f) == (True, True)


def test_has_arg_types_kwonly():
    def f(*, a):
        pass

    assert has_arg_types(f) == (False, True)


def test_has_arg_types_varargs():
    def f(*args):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_positional_only():
    def f(a, /):
        pass

    assert has_arg_types(f) == (True, False)


# ---------------------------------------------------------------------------
# make_lookuper / silent_lookuper (public-ish helpers used by has_arg_types)
# ---------------------------------------------------------------------------

def test_make_lookuper_no_args():
    @make_lookuper
    def table():
        return {"a": 1, "b": 2}

    assert table("a") == 1
    assert table("b") == 2
    with pytest.raises(LookupError):
        table("c")


def test_silent_lookuper_no_args():
    @silent_lookuper
    def table():
        return {"a": 1}

    assert table("a") == 1
    assert table("missing") is None


def test_lookuper_called_once():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {"x": 10}

    table("x")
    table("x")
    assert calls == [1]


def test_lookuper_with_args():
    @make_lookuper
    def table(prefix):
        return {prefix + "a": 1}

    assert table("p_")("p_a") == 1


def test_lookuper_keyword_args_rejected():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(**kw):
            return kw


def test_silent_lookuper_name():
    assert silent_lookuper.__name__ == "silent_lookuper"
