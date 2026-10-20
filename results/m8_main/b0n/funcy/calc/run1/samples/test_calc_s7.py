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


# --------------------------- SkipMemory ---------------------------

def test_skip_memory_is_exception():
    assert issubclass(SkipMemory, Exception)


def test_memoize_skip_attribute():
    assert memoize.skip is SkipMemory


def test_cache_skip_attribute():
    assert cache.skip is SkipMemory


# --------------------------- memoize ---------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert f(3) == 6
    assert calls == [2, 3]
    assert f.memory == {2: 4, 3: 6}


def test_memoize_no_args():
    calls = []

    @memoize
    def f():
        calls.append(1)
        return 42

    assert f() == 42
    assert f() == 42
    assert calls == [1]


def test_memoize_with_kwargs():
    calls = []

    @memoize
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    assert f(1, b=3) == 4
    assert calls == [(1, 2), (1, 3)]


def test_memoize_kwargs_order_insensitive():
    calls = []

    @memoize
    def f(**kw):
        calls.append(dict(kw))
        return 1

    f(a=1, b=2)
    f(b=2, a=1)
    assert len(calls) == 1


def test_memoize_positional_vs_kwarg_distinct():
    calls = []

    @memoize
    def f(a):
        calls.append(a)
        return a

    f(1)
    f(a=1)
    # positional vs keyword produce different keys -> both called
    assert len(calls) == 2


def test_memoize_with_key_func():
    calls = []

    @memoize(key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(3) == 1  # same key -> cached value from f(1)
    assert f(2) == 2
    assert calls == [1, 2]


def test_memoize_called_with_key_func_only():
    calls = []

    @memoize(key_func=lambda x: x)
    def f(x):
        calls.append(x)
        return x + 1

    assert f(5) == 6
    assert f(5) == 6
    assert calls == [5]


def test_memoize_skip_memory():
    @memoize
    def f(x):
        raise memoize.skip(100)

    # SkipMemory value not cached; returns the args[0]
    assert f(1) == 100
    assert f(1) == 100
    assert 1 not in f.memory


def test_memoize_skip_memory_empty():
    @memoize
    def f(x):
        raise memoize.skip()

    assert f(1) is None


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
    # invalidate nonexistent key does not raise
    f.invalidate(99)


def test_memoize_invalidate_kwargs():
    calls = []

    @memoize
    def f(a=0):
        calls.append(a)
        return a

    f(a=1)
    f(a=1)
    assert calls == [1]
    f.invalidate(a=1)
    f(a=1)
    assert calls == [1, 1]


def test_memoize_invalidate_with_key_func():
    calls = []

    @memoize(key_func=lambda x: x)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    f.invalidate(1)
    f(1)
    assert calls == [1, 1]


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


def test_memoize_preserves_metadata():
    @memoize
    def my_function(x):
        "doc string"
        return x

    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'doc string'


# --------------------------- cache ---------------------------

def test_cache_basic():
    calls = []

    @cache(100)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]


def test_cache_timeout_parameter():
    @cache(timedelta(seconds=100))
    def f(x):
        return x

    assert f.memory.timeout == 100


def test_cache_timeout_float():
    @cache(50)
    def f(x):
        return x

    assert f.memory.timeout == 50


def test_cache_expires(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])

    calls = []

    @cache(10)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]

    now[0] = 1005.0
    # Not expired yet
    assert f(1) == 2
    assert calls == [1]

    now[0] = 1011.0
    # expired
    assert f(1) == 2
    assert calls == [1, 1]


def test_cache_skip():
    @cache(10)
    def f(x):
        raise cache.skip(7)

    assert f(1) == 7


def test_cache_invalidate():
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


# --------------------------- CacheMemory ---------------------------

def test_cachememory_init():
    m = CacheMemory(10)
    assert m.timeout == 10
    assert m == {}
    assert list(m._keys) == []
    assert list(m._expires) == []


def test_cachememory_set_get(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])

    m = CacheMemory(10)
    m['a'] = 1
    assert m['a'] == 1
    assert 'a' in m

    now[0] = 5.0
    assert m['a'] == 1

    now[0] = 10.0
    with pytest.raises(KeyError):
        m['a']


def test_cachememory_clear(monkeypatch):
    monkeypatch.setattr(time, 'monotonic', lambda: 0.0)
    m = CacheMemory(10)
    m['a'] = 1
    m.clear()
    assert m == {}
    assert list(m._keys) == []
    assert list(m._expires) == []


def test_cachememory_expire(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])

    m = CacheMemory(10)
    m['a'] = 1
    now[0] = 5.0
    m['b'] = 2
    now[0] = 12.0

    # Accessing 'a' triggers expire
    with pytest.raises(KeyError):
        m['a']
    # 'a' (expires at 10) is removed, 'b' (expires at 15) remains
    assert 'a' not in m
    assert m['b'] == 2


# --------------------------- has_arg_types ---------------------------

def test_has_arg_types_positional():
    def f(a, b):
        pass
    assert has_arg_types(f) == (True, False)


def test_has_arg_types_none():
    def f():
        pass
    assert has_arg_types(f) == (False, False)


def test_has_arg_types_kwargs():
    def f(*args, **kwargs):
        pass
    assert has_arg_types(f) == (True, True)


def test_has_arg_types_keyword_only():
    def f(*, a):
        pass
    assert has_arg_types(f) == (False, True)


def test_has_arg_types_positional_only():
    def f(a, /):
        pass
    assert has_arg_types(f) == (True, False)


# --------------------------- make_lookuper ---------------------------

def test_make_lookuper_basic():
    @make_lookuper
    def table():
        return {1: 'one', 2: 'two'}

    assert table(1) == 'one'
    assert table(2) == 'two'
    with pytest.raises(LookupError):
        table(3)


def test_make_lookuper_called_once():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {1: 'one'}

    table(1)
    table(1)
    assert calls == [1]


def test_make_lookuper_no_arg_missing_raises():
    @make_lookuper
    def table():
        return {}

    with pytest.raises(LookupError):
        table(1)


def test_make_lookuper_error_propagation():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        raise ValueError('boom')

    with pytest.raises(ValueError):
        table(1)
    # retried on next call since memory was cleared
    with pytest.raises(ValueError):
        table(1)
    assert calls == [1, 1]


def test_silent_lookuper_basic():
    @silent_lookuper
    def table():
        return {1: 'one'}

    assert table(1) == 'one'
    assert table(2) is None


def test_silent_lookuper_name():
    assert silent_lookuper.__name__ == 'silent_lookuper'


def test_make_lookuper_with_args():
    @make_lookuper
    def table(prefix):
        return {1: prefix + 'one', 2: prefix + 'two'}

    assert table('a')(1) == 'aone'
    assert table('a')(2) == 'atwo'
    with pytest.raises(LookupError):
        table('a')(3)


def test_make_lookuper_with_args_cached():
    state = []

    @make_lookuper
    def table(prefix):
        state.append(prefix)
        return {1: prefix + 'one'}

    t1 = table('a')
    assert t1(1) == 'aone'
    t2 = table('a')
    assert t1 is t2
    assert state == ['a']


def test_make_lookuper_with_args_missing_raises():
    @make_lookuper
    def table(prefix):
        return {1: prefix}

    table('a')
    with pytest.raises(LookupError):
        table('a')(2)


def test_make_lookuper_kwargs_assert():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(**kwargs):
            return {}


def test_make_lookuper_keyword_only_assert():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(*, a):
            return {}


def test_make_lookuper_preserves_metadata():
    @make_lookuper
    def my_table():
        "docstring"
        return {}

    assert my_table.__name__ == 'my_table'
    assert my_table.__doc__ == 'docstring'
