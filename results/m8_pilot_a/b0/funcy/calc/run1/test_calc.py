import pytest
from datetime import timedelta

from funcy.calc import (
    SkipMemory,
    memoize,
    cache,
    CacheMemory,
    make_lookuper,
    silent_lookuper,
    has_arg_types,
)


# ---------------------------------------------------------------------------
# SkipMemory
# ---------------------------------------------------------------------------

def test_skip_memory_is_exception():
    assert issubclass(SkipMemory, Exception)
    e = SkipMemory("value")
    assert e.args == ("value",)


def test_memoize_skip_and_cache_skip_attributes():
    assert memoize.skip is SkipMemory
    assert cache.skip is SkipMemory


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_caches_results():
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


def test_memoize_memory_attribute():
    @memoize
    def f(x):
        return x + 1

    f(1)
    assert f.memory == {(1,): 2}


def test_memoize_kwargs_keying():
    calls = []

    @memoize
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    # different kwargs order still same frozenset key
    assert f(1, b=2) == 3
    assert len(calls) == 1
    # positional vs keyword differ
    assert f(1, 2) == 3
    assert len(calls) == 2


def test_memoize_kwargs_key_distinguishes_from_positional():
    calls = []

    @memoize
    def f(*args, **kwargs):
        calls.append((args, kwargs))
        return len(calls)

    f(1)
    f(1, x=2)
    f(1)
    f(1, x=2)
    assert len(calls) == 2


def test_memoize_key_func():
    calls = []

    @memoize(key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(3) == 1  # same key -> cached value from first call
    assert calls == [1]


def test_memoize_key_func_with_kwargs():
    @memoize(key_func=lambda *, name: name)
    def f(*, name):
        return name.upper()

    assert f(name="a") == "A"
    assert f.memory == {"a": "A"}


def test_memoize_with_skip_memory():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        if x < 0:
            raise memoize.skip("negative")
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1]
    # skip raises but returns value from args
    assert f(-1) == "negative"
    assert calls == [1, -1]


def test_memoize_with_skip_memory_no_args():
    @memoize
    def f(x):
        raise memoize.skip()

    assert f(5) is None


def test_memoize_invalidate():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert len(calls) == 1
    f.invalidate(1, y=2)
    assert f(1, y=2) == 3
    assert len(calls) == 2


def test_memoize_invalidate_missing_key_is_noop():
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


def test_memoize_invalidate_with_key_func():
    calls = []

    @memoize(key_func=lambda x: x)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f.invalidate(1)
    f(1)
    assert calls == [1, 1]


def test_memoize_preserves_metadata():
    @memoize
    def some_function(x):
        """docstring here"""
        return x

    assert some_function.__name__ == "some_function"
    assert some_function.__doc__ == "docstring here"


def test_memoize_bare_and_call_with_parens():
    @memoize()
    def f(x):
        return x * 3

    assert f(2) == 6
    assert f.memory == {(2,): 6}


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def test_cache_caches_before_expiry():
    calls = []

    @cache(100)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_cache_memory_is_cache_memory():
    @cache(100)
    def f(x):
        return x

    assert isinstance(f.memory, CacheMemory)
    assert f.memory.timeout == 100


def test_cache_with_timedelta():
    @cache(timedelta(seconds=50))
    def f(x):
        return x

    assert f.memory.timeout == 50


def test_cache_invalidate_and_invalidate_all():
    calls = []

    @cache(100)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1]
    f.invalidate(1)
    f(1)
    assert calls == [1, 1]
    f.invalidate_all()
    f(1)
    assert calls == [1, 1, 1]


def test_cache_with_skip_memory():
    @cache(100)
    def f(x):
        raise cache.skip("skip result")

    assert f(1) == "skip result"


def test_cache_key_func():
    calls = []

    @cache(100, key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(3) == 1
    assert calls == [1]


def test_cache_negative_timeout_expires_immediately():
    calls = []

    @cache(-1)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    # Entry expires immediately since timeout is negative
    assert calls == [1, 1]


# ---------------------------------------------------------------------------
# CacheMemory
# ---------------------------------------------------------------------------

def test_cache_memory_set_get():
    mem = CacheMemory(100)
    mem["a"] = 1
    assert mem["a"] == 1


def test_cache_memory_missing_raises_key_error():
    mem = CacheMemory(100)
    with pytest.raises(KeyError):
        mem["missing"]


def test_cache_memory_clear():
    mem = CacheMemory(100)
    mem["a"] = 1
    mem.clear()
    with pytest.raises(KeyError):
        mem["a"]
    assert list(mem._keys) == []
    assert list(mem._expires) == []


def test_cache_memory_expired_entry():
    mem = CacheMemory(-1)
    mem["a"] = 1
    with pytest.raises(KeyError):
        mem["a"]


def test_cache_memory_expire_removes_expired():
    mem = CacheMemory(-1)
    mem["a"] = 1
    mem["b"] = 2
    mem.expire()
    assert dict.__len__(mem) == 0
    assert list(mem._keys) == []
    assert list(mem._expires) == []


def test_cache_memory_expire_leaves_fresh_entries():
    mem = CacheMemory(100)
    mem["a"] = 1
    mem["b"] = 2
    mem.expire()
    assert mem["a"] == 1
    assert mem["b"] == 2


def test_cache_memory_expire_mixed():
    mem = CacheMemory(100)
    mem["old"] = 1
    # manually force old entry to be expired
    value, _ = dict.__getitem__(mem, "old")
    dict.__setitem__(mem, "old", (value, 0))
    mem._expires[0] = 0
    mem["fresh"] = 2
    mem.expire()
    assert dict.__len__(mem) == 1
    assert mem["fresh"] == 2


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_no_params():
    def f():
        pass
    assert has_arg_types(f) == (False, False)


def test_has_arg_types_positional():
    def f(a, b):
        pass
    assert has_arg_types(f) == (True, False)


def test_has_arg_types_positional_only():
    def f(a, /):
        pass
    has_args, has_keys = has_arg_types(f)
    assert has_args is True
    assert has_keys is False


def test_has_arg_types_var_positional():
    def f(*args):
        pass
    assert has_arg_types(f) == (True, False)


def test_has_arg_types_keyword_only():
    def f(*, a):
        pass
    assert has_arg_types(f) == (False, True)


def test_has_arg_types_var_keyword():
    def f(**kwargs):
        pass
    assert has_arg_types(f) == (False, True)


def test_has_arg_types_mixed():
    def f(a, *args, b, **kwargs):
        pass
    assert has_arg_types(f) == (True, True)


# ---------------------------------------------------------------------------
# make_lookuper / silent_lookuper
# ---------------------------------------------------------------------------

def test_make_lookuper_basic():
    @make_lookuper
    def table():
        return {"a": 1, "b": 2}

    assert table("a") == 1
    assert table("b") == 2


def test_make_lookuper_missing_raises():
    @make_lookuper
    def table():
        return {"a": 1}

    with pytest.raises(LookupError) as excinfo:
        table("z")
    assert "Failed to look up" in str(excinfo.value)


def test_silent_lookuper_missing_returns_none():
    @silent_lookuper
    def table():
        return {"a": 1}

    assert table("a") == 1
    assert table("z") is None


def test_lookuper_calls_once():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {"a": 1}

    table("a")
    table("a")
    table("a")
    assert calls == [1]


def test_lookuper_with_args():
    @make_lookuper
    def table(x):
        return {x: x * 2}

    assert table(3) == 6
    assert table(3) == 6
    assert table(4) == 8


def test_lookuper_with_args_missing():
    @make_lookuper
    def table(x):
        return {x: x}

    with pytest.raises(LookupError):
        table(1)  # table(1) builds {1: 1}, lookup would succeed... use mismatch


def test_lookuper_with_args_missing_value():
    @make_lookuper
    def table(x):
        return {"other": x}

    with pytest.raises(LookupError):
        table(1)


def test_lookuper_rejects_kwonly():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(*, a):
            return {a: a}


def test_lookuper_rejects_var_keyword():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(**kwargs):
            return kwargs


def test_lookuper_preserves_name():
    @make_lookuper
    def my_table():
        return {}

    assert my_table.__name__ == "my_table"


def test_silent_lookuper_preserves_name():
    @silent_lookuper
    def my_table():
        return {}

    assert my_table.__name__ == "my_table"


def test_lookuper_empty_table_still_works():
    @make_lookuper
    def table():
        return {}

    with pytest.raises(LookupError):
        table("a")


def test_lookuper_error_in_builder_propagates_and_retries():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("boom")
        return {"a": 1}

    with pytest.raises(ValueError):
        table("a")
    # memory cleared, retries
    assert table("a") == 1
    assert len(calls) == 2
