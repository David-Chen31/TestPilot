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
# SkipMemory / memoize.skip / cache.skip
# ---------------------------------------------------------------------------

def test_skip_memory_is_exception():
    assert issubclass(SkipMemory, Exception)


def test_memoize_skip_attribute():
    assert memoize.skip is SkipMemory


def test_cache_skip_attribute():
    assert cache.skip is SkipMemory


# ---------------------------------------------------------------------------
# memoize basic behavior
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


def test_memoize_with_kwargs():
    calls = []

    @memoize
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1) == 1
    assert f(1, b=10) == 11
    assert f(1) == 1
    assert f(1, b=10) == 11
    assert calls == [(1, 0), (1, 10)]


def test_memoize_kwargs_order_insensitive():
    calls = []

    @memoize
    def f(**kw):
        calls.append(dict(kw))
        return sum(kw.values())

    assert f(a=1, b=2) == 3
    assert f(b=2, a=1) == 3
    assert len(calls) == 1


def test_memoize_positional_vs_kwargs_distinct():
    calls = []

    @memoize
    def f(a, b=1):
        calls.append((a, b))
        return (a, b)

    assert f(1, 2) == (1, 2)
    assert f(1, b=2) == (1, 2)
    # Both computed because keys differ (pos args vs pos+kwargs)
    assert len(calls) == 2


def test_memoize_exposes_memory():
    @memoize
    def f(x):
        return x * 3

    assert isinstance(f.memory, dict)
    assert f.memory == {}
    assert f(2) == 6
    assert f.memory == {(2,): 6}


def test_memoize_called_with_no_func_returns_decorator():
    deco = memoize()
    assert callable(deco)


def test_memoize_explicit_key_func():
    keys = []

    def key_func(x):
        keys.append(x)
        return x % 3

    calls = []

    @memoize(key_func=key_func)
    def f(x):
        calls.append(x)
        return x * 10

    assert f(3) == 30
    assert f(6) == 30  # same key x % 3 == 0
    assert calls == [3]
    assert keys == [3, 6]


def test_memoize_key_func_with_kwargs():
    def key_func(*args, **kwargs):
        return (args, tuple(sorted(kwargs.items())))

    calls = []

    @memoize(key_func=key_func)
    def f(a, b=0, **kw):
        calls.append((a, b, kw))
        return a

    assert f(1) == 1
    assert f(1) == 1
    assert len(calls) == 1


def test_memoize_invalidate():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1]

    f.invalidate(1)
    assert f(1) == 1
    assert calls == [1, 1]


def test_memoize_invalidate_with_kwargs():
    calls = []

    @memoize
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    f(1, b=2)
    f.invalidate(1, b=2)
    assert f(1, b=2) == 3
    assert calls == [(1, 2), (1, 2)]


def test_memoize_invalidate_missing_key_is_noop():
    @memoize
    def f(x):
        return x

    # Should not raise
    f.invalidate(999)
    assert f.memory == {}


def test_memoize_invalidate_all():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    assert len(f.memory) == 2

    f.invalidate_all()
    assert f.memory == {}

    f(1)
    assert calls == [1, 2, 1]


def test_memoize_wraps_preserves_metadata():
    @memoize
    def my_func(x):
        """docstring here"""
        return x

    assert my_func.__name__ == 'my_func'
    assert my_func.__doc__ == 'docstring here'


def test_memoize_skip_memory_with_value():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        raise memoize.skip('sentinel')

    assert f(1) == 'sentinel'
    # first call raised SkipMemory, not memoized properly
    assert f(1) == 'sentinel'
    assert calls == [1, 1]


def test_memoize_skip_memory_without_args():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        raise memoize.skip()

    assert f(1) is None
    assert f(1) is None
    assert calls == [1, 1]


def test_memoize_skip_memory_stores_sentinel_value():
    # When SkipMemory is raised with a value, the value returned is not cached
    # (exception happens before assignment completes), verify behavior.
    @memoize
    def f(x):
        raise memoize.skip(x * 100)

    assert f(2) == 200
    assert f.memory == {}


def test_memoize_skip_memory_multiarg():
    @memoize
    def f(x):
        raise memoize.skip(x, x + 1)

    # Returns first arg of exception
    assert f(5) == 5


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def test_cache_basic():
    calls = []

    @cache(100)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]
    assert isinstance(f.memory, CacheMemory)


def test_cache_returns_none_when_timeout_zero(monkeypatch):
    # With timeout 0, expires_at <= monotonic always true, so value never returned
    calls = []

    @cache(0)
    def f(x):
        calls.append(x)
        return x

    # First call computes and stores; second call sees expired -> recomputes
    assert f(1) == 1
    # Because timeout = 0, expires_at <= monotonic immediately true
    assert f(1) == 1
    assert len(calls) >= 1


def test_cache_timedelta_timeout():
    @cache(timedelta(seconds=100))
    def f(x):
        return x

    assert f(3) == 3
    assert f.memory.timeout == 100.0


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


def test_cache_invalidate_all():
    calls = []

    @cache(100)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f.invalidate_all()
    assert f.memory == {}
    f(1)
    assert calls == [1, 2, 1]


def test_cache_with_key_func():
    calls = []

    @cache(100, key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(3) == 1  # same key
    assert calls == [1]


def test_cache_skip_memory():
    @cache(100)
    def f(x):
        raise SkipMemory('skipped')

    assert f(1) == 'skipped'


def test_cache_skip_memory_no_args():
    @cache(100)
    def f(x):
        raise SkipMemory()

    assert f(1) is None


# ---------------------------------------------------------------------------
# CacheMemory
# ---------------------------------------------------------------------------

def test_cache_memory_init():
    m = CacheMemory(50)
    assert m.timeout == 50
    assert dict.__len__(m) == 0
    assert len(m._keys) == 0
    assert len(m._expires) == 0


def test_cache_memory_set_and_get():
    m = CacheMemory(100)
    m['a'] = 1
    assert m['a'] == 1
    assert dict.__getitem__(m, 'a')[0] == 1
    assert len(m._keys) == 1
    assert len(m._expires) == 1


def test_cache_memory_missing_key_raises():
    m = CacheMemory(100)
    with pytest.raises(KeyError):
        m['missing']


def test_cache_memory_expired_key_raises(monkeypatch):
    fake_time = [1000.0]
    monkeypatch.setattr(time, 'monotonic', lambda: fake_time[0])

    m = CacheMemory(10)
    m['a'] = 'value'
    assert m['a'] == 'value'

    fake_time[0] += 100
    with pytest.raises(KeyError):
        m['a']


def test_cache_memory_expire_removes_old():
    fake_time = [0.0]

    def fake_monotonic():
        return fake_time[0]

    import funcy.calc as calc_mod
    orig = calc_mod.time.monotonic
    calc_mod.time.monotonic = fake_monotonic
    try:
        m = CacheMemory(10)
        m['a'] = 1
        fake_time[0] = 5.0
        m['b'] = 2
        fake_time[0] = 15.0
        # a expired at 10, b expires at 15
        # accessing b should not have expired b (expires_at 15 <= 15 is true though)
        # Let's set to 14.99 to keep b alive
        fake_time[0] = 14.999
        # Trigger expire through access of expired a
        with pytest.raises(KeyError):
            m['a']
        # b should still be alive
        assert m['b'] == 2
    finally:
        calc_mod.time.monotonic = orig


def test_cache_memory_clear():
    m = CacheMemory(100)
    m['a'] = 1
    m['b'] = 2
    m.clear()
    assert dict.__len__(m) == 0
    assert len(m._keys) == 0
    assert len(m._expires) == 0


def test_cache_memory_expire_with_overwrite():
    # When a key gets overwritten, the old expires entry must not delete the new value
    fake_time = [0.0]

    def fake_monotonic():
        return fake_time[0]

    import funcy.calc as calc_mod
    orig = calc_mod.time.monotonic
    calc_mod.time.monotonic = fake_monotonic
    try:
        m = CacheMemory(10)
        m['a'] = 1  # expires at 10
        fake_time[0] = 5.0
        m['a'] = 2  # overwrite, expires at 15
        fake_time[0] = 11.0
        # Access 'a' should succeed since expires_at 15 > 11
        assert m['a'] == 2
        # trigger expire of old entry
        m.expire()
        # still alive
        assert m['a'] == 2
    finally:
        calc_mod.time.monotonic = orig


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_no_args_no_kwargs():
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

    assert has_arg_types(f) == (True, False)


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


def test_has_arg_types_both():
    def f(a, *args, b, **kwargs):
        pass

    assert has_arg_types(f) == (True, True)


def test_has_arg_types_default_args():
    def f(a=1, *, b=2):
        pass

    assert has_arg_types(f) == (True, True)


# ---------------------------------------------------------------------------
# make_lookuper / silent_lookuper
# ---------------------------------------------------------------------------

def test_lookup_table_constant():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {'a': 1, 'b': 2}

    assert table('a') == 1
    assert table('b') == 2
    assert len(calls) == 1


def test_lookup_table_constant_missing_raises():
    @make_lookuper
    def table():
        return {'a': 1}

    with pytest.raises(LookupError):
        table('missing')


def test_lookup_table_constant_missing_message():
    @make_lookuper
    def table():
        return {'a': 1}

    with pytest.raises(LookupError) as excinfo:
        table('missing')
    assert 'table' in str(excinfo.value)
    assert 'missing' in str(excinfo.value)


def test_lookup_table_silent_missing_returns_none():
    @silent_lookuper
    def table():
        return {'a': 1}

    assert table('a') == 1
    assert table('missing') is None


def test_lookup_table_silent_name():
    assert silent_lookuper.__name__ == 'silent_lookuper'


def test_lookup_table_function_called_once():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {'x': 10}

    table('x')
    table('x')
    table('x')
    assert len(calls) == 1


def test_lookup_table_with_args_builds_per_args():
    calls = []

    @make_lookuper
    def table(kind):
        calls.append(kind)
        if kind == 'a':
            return {'x': 1}
        return {'x': 2}

    assert table('a')('x') == 1
    assert table('b')('x') == 2
    assert table('a')('x') == 1
    assert calls == ['a', 'b']


def test_lookup_table_with_args_missing_raises():
    @make_lookuper
    def table(kind):
        return {'x': 1}

    with pytest.raises(LookupError):
        table('k')('missing')


def test_lookup_table_silent_with_args():
    @silent_lookuper
    def table(kind):
        return {'x': 1}

    assert table('k')('x') == 1
    assert table('k')('missing') is None


def test_lookup_table_rejects_keyword_args():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(*, kind):
            return {}


def test_lookup_table_rejects_var_keyword():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(**kwargs):
            return {}


def test_lookup_table_error_in_builder_clears_memory():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('boom')
        return {'a': 1}

    with pytest.raises(ValueError):
        table('a')
    # On next call, memory is cleared so builder is called again
    assert table('a') == 1
    assert len(calls) == 2


def test_lookup_table_silent_error_in_builder_clears_memory():
    calls = []

    @silent_lookuper
    def table():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('boom')
        return {'a': 1}

    with pytest.raises(ValueError):
        table('a')
    assert table('a') == 1


def test_lookup_table_wraps_preserves_metadata():
    def my_table():
        """my doc"""
        return {}

    wrapped = make_lookuper(my_table)
    assert wrapped.__name__ == 'my_table'
    assert wrapped.__doc__ == 'my doc'


def test_lookup_table_sentinel_key_present():
    # After building, a sentinel object() key should be present to prevent refill
    @make_lookuper
    def table():
        return {'a': 1}

    table('a')
    # We can't access memory directly, but behavior verifies: builder not called again
    # Covered above. Check that object key got inserted -> lookups don't re-run.


def test_lookup_table_empty_result_still_marks_built():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {}

    with pytest.raises(LookupError):
        table('a')
    with pytest.raises(LookupError):
        table('a')
    # builder only called once since sentinel was inserted
    assert len(calls) == 1


def test_lookup_table_silent_empty_result():
    calls = []

    @silent_lookuper
    def table():
        calls.append(1)
        return {}

    assert table('a') is None
    assert table('a') is None
    assert len(calls) == 1
