import time
import pytest
from datetime import timedelta

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

def test_skipmemory_instantiable_and_raisable():
    with pytest.raises(SkipMemory) as exc:
        raise SkipMemory(42)
    assert exc.value.args == (42,)


def test_memoize_skip_is_skipmemory():
    assert memoize.skip is SkipMemory


def test_cache_skip_is_skipmemory():
    assert cache.skip is SkipMemory


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic_and_call_count():
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

    assert f.memory == {}
    f(1)
    assert f.memory == {(1,): 2}


def test_memoize_with_kwargs():
    calls = []

    @memoize
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    assert calls == [(1, 2)]
    # different kwarg values -> different key
    assert f(1, b=3) == 4
    assert calls == [(1, 2), (1, 3)]
    # positional value vs kwarg value produce different keys
    assert f(1, 3) == 4
    assert calls == [(1, 2), (1, 3), (1, 3)]


def test_memoize_kwargs_key_marker():
    @memoize
    def f(*args, **kwargs):
        return (args, tuple(sorted(kwargs.items())))

    f()
    f(1)
    f(1, a=2)
    # keys distinguish no-kwargs path (args tuple) vs kwargs path
    keys = set(f.memory.keys())
    assert () in keys
    assert (1,) in keys
    # the kwargs entry contains the marker, args, and frozenset
    kw_entry = [k for k in keys if isinstance(k, tuple) and len(k) == 3]
    assert len(kw_entry) == 1
    marker, args, frozen = kw_entry[0]
    assert args == (1,)
    assert frozen == frozenset({('a', 2)})


def test_memoize_with_key_func():
    calls = []

    @memoize(key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(3) == 1  # same key (3 % 2 == 1), cached result from f(1) = 1
    assert calls == [1]
    assert f(2) == 2
    assert calls == [1, 2]


def test_memoize_key_func_called_with_kwargs():
    seen = []

    def key_func(*args, **kwargs):
        seen.append((args, kwargs))
        return 'k'

    @memoize(key_func=key_func)
    def f(*args, **kwargs):
        return 99

    assert f(1, a=2) == 99
    assert seen == [((1,), {'a': 2})]


def test_memoize_without_args_via_parentheses():
    @memoize()
    def f(x):
        return x

    assert f(5) == 5


def test_memoize_invalidate():
    @memoize
    def f(x):
        return x

    f(1)
    assert (1,) in f.memory
    f.invalidate(1)
    assert (1,) not in f.memory
    # invalidating missing key is a no-op
    f.invalidate(42)


def test_memoize_invalidate_with_kwargs():
    @memoize
    def f(a, b=0):
        return a + b

    f(1, b=2)
    f.invalidate(1, b=2)
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


def test_memoize_skip_returns_value():
    @memoize
    def f(x):
        if x < 0:
            raise SkipMemory(-1)
        return x

    assert f(1) == 1
    assert f(-1) == -1
    # skipped results are still cached
    assert (-1,) in f.memory


def test_memoize_skip_no_args_returns_none():
    @memoize
    def f(x):
        raise SkipMemory

    assert f(1) is None
    assert (1,) in f.memory


def test_memoize_wraps_preserves_name():
    @memoize
    def my_func(x):
        """docstring"""
        return x

    assert my_func.__name__ == 'my_func'
    assert my_func.__doc__ == 'docstring'


def test_memoize_separate_instances_have_separate_memory():
    @memoize
    def f(x):
        return x

    @memoize
    def g(x):
        return x

    f(1)
    assert g.memory == {}


# ---------------------------------------------------------------------------
# cache / CacheMemory
# ---------------------------------------------------------------------------

def test_cache_basic():
    calls = []

    @cache(1000)
    def f(x):
        calls.append(x)
        return x * 10

    assert f(1) == 10
    assert f(1) == 10
    assert calls == [1]
    assert f(2) == 20
    assert calls == [1, 2]


def test_cache_timeout_attribute():
    @cache(5)
    def f(x):
        return x

    assert isinstance(f.memory, CacheMemory)
    assert f.memory.timeout == 5


def test_cache_timedelta_timeout():
    @cache(timedelta(seconds=10))
    def f(x):
        return x

    assert f.memory.timeout == 10.0


def test_cache_memory_is_cache_memory_instance():
    @cache(100)
    def f():
        return 1

    assert isinstance(f.memory, CacheMemory)


def test_cache_memory_set_get():
    m = CacheMemory(1000)
    m['a'] = 5
    assert m['a'] == 5
    # raw stored value is a (value, expires_at) tuple
    raw = dict.__getitem__(m, 'a')
    assert raw[0] == 5
    assert raw[1] > time.monotonic()


def test_cache_memory_expiry(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr(time, 'monotonic', lambda: fake_now[0])

    m = CacheMemory(10)
    m['a'] = 'v'
    assert m['a'] == 'v'

    fake_now[0] = 1011.0  # past expiry
    with pytest.raises(KeyError):
        m['a']
    # entry expired and removed
    assert 'a' not in m


def test_cache_memory_expire_removes_expired_keys(monkeypatch):
    fake_now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: fake_now[0])

    m = CacheMemory(10)
    m['a'] = 1  # expires at 10
    fake_now[0] = 5.0
    m['b'] = 2  # expires at 15
    fake_now[0] = 12.0
    m.expire()
    # 'a' expired, 'b' still valid
    assert 'a' not in m
    assert dict.__getitem__(m, 'b')[0] == 2


def test_cache_memory_clear():
    m = CacheMemory(100)
    m['a'] = 1
    m['b'] = 2
    m.clear()
    assert dict.__len__(m) == 0
    assert len(m._keys) == 0
    assert len(m._expires) == 0


def test_cache_memory_reset_on_reinit():
    m = CacheMemory(100)
    m['a'] = 1
    m.__init__(50)
    assert m.timeout == 50
    assert dict.__len__(m) == 0


def test_cache_invalidate():
    @cache(1000)
    def f(x):
        return x

    f(1)
    assert (1,) in f.memory or 1 in f.memory or (1,) in dict.keys(f.memory)
    f.invalidate(1)
    # after invalidation, cache misses
    calls = []

    @cache(1000)
    def g(x):
        calls.append(x)
        return x

    g(1)
    g.invalidate(1)
    g(1)
    assert calls == [1, 1]


def test_cache_invalidate_all():
    @cache(1000)
    def f(x):
        return x

    f(1)
    f(2)
    f.invalidate_all()
    assert len(f.memory._keys) == 0


def test_cache_skip():
    @cache(1000)
    def f(x):
        if x < 0:
            raise SkipMemory('skipped')
        return x

    assert f(-1) == 'skipped'
    assert f(1) == 1


def test_cache_skip_no_args():
    @cache(1000)
    def f(x):
        raise SkipMemory

    assert f(1) is None


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_positional():
    def f(a, b=1):
        pass

    has_args, has_keys = has_arg_types(f)
    assert has_args is True
    assert has_keys is False


def test_has_arg_types_no_params():
    def f():
        pass

    has_args, has_keys = has_arg_types(f)
    assert has_args is False
    assert has_keys is False


def test_has_arg_types_keyword_only():
    def f(*, a):
        pass

    has_args, has_keys = has_arg_types(f)
    assert has_args is False
    assert has_keys is True


def test_has_arg_types_varargs_varkw():
    def f(*args, **kwargs):
        pass

    has_args, has_keys = has_arg_types(f)
    assert has_args is True
    assert has_keys is True


def test_has_arg_types_positional_only():
    ns = {}
    exec("def f(a, /, b): pass", ns)
    has_args, has_keys = has_arg_types(ns['f'])
    assert has_args is True
    assert has_keys is False


# ---------------------------------------------------------------------------
# make_lookuper / silent_lookuper
# ---------------------------------------------------------------------------

def test_make_lookuper_no_args():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {'a': 1, 'b': 2}

    assert table('a') == 1
    assert table('b') == 2
    # building function called only once
    assert calls == [1]


def test_make_lookuper_raises_lookuperror():
    @make_lookuper
    def table():
        return {'a': 1}

    assert table('a') == 1
    with pytest.raises(LookupError):
        table('missing')


def test_silent_lookuper_returns_none():
    @silent_lookuper
    def table():
        return {'a': 1}

    assert table('a') == 1
    assert table('missing') is None


def test_make_lookuper_builder_error_propagates_and_clears():
    attempts = []

    @make_lookuper
    def table():
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("boom")
        return {'a': 1}

    with pytest.raises(ValueError):
        table('a')
    # memory cleared after error, retried
    assert table('a') == 1
    assert len(attempts) == 2


def test_make_lookuper_with_args():
    calls = []

    @make_lookuper
    def table(x):
        calls.append(x)
        return {'a': x}

    assert table('a', 5) == 5
    assert table('a', 5) == 5
    assert calls == [5]


def test_silent_lookuper_name_preserved():
    assert silent_lookuper.__name__ == 'silent_lookuper'


def test_make_lookuper_requires_no_kwargs():
    with pytest.raises(AssertionError):

        @make_lookuper
        def table(*, a):
            return {}


def test_make_lookuper_wraps_name():
    @make_lookuper
    def my_table():
        return {'x': 10}

    assert my_table.__name__ == 'my_table'
