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
    _KWARGS_MARK,
)


# ---------------------------------------------------------------------------
# SkipMemory
# ---------------------------------------------------------------------------

def test_skip_memory_is_exception():
    assert issubclass(SkipMemory, Exception)
    # memoize.skip and cache.skip are the same class
    assert memoize.skip is SkipMemory
    assert cache.skip is SkipMemory


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_returns_cached_values():
    calls = []

    @memoize
    def add(a, b):
        calls.append((a, b))
        return a + b

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_distinguishes_args():
    calls = []

    @memoize
    def ident(x):
        calls.append(x)
        return x

    assert ident(1) == 1
    assert ident(2) == 2
    assert ident(1) == 1
    assert calls == [1, 2]


def test_memoize_kwargs_keys():
    calls = []

    @memoize
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    assert calls == [(1, 2)]
    # positional vs keyword are distinct calls
    assert f(1, 2) == 3
    assert calls == [(1, 2), (1, 2)]


def test_memoize_keyword_order_independent():
    calls = []

    @memoize
    def f(**kwargs):
        calls.append(kwargs)
        return sum(kwargs.values())

    assert f(a=1, b=2) == 3
    assert f(b=2, a=1) == 3
    assert len(calls) == 1


def test_memoize_exposes_memory():
    @memoize
    def f(x):
        return x * 10

    assert f.memory == {}
    assert f(3) == 30
    # The key is the args tuple used by the wrapper (no kwargs -> plain tuple)
    assert f.memory[(3,)] == 30


def test_memoize_with_key_func():
    calls = []

    @memoize(key_func=lambda a, b=0: a)
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    # different b, same key -> cached
    assert f(1, b=100) == 3
    assert calls == [(1, 2)]


def test_memoize_called_with_func_directly():
    calls = []

    def raw(x):
        calls.append(x)
        return x * 2

    wrapped = memoize(raw)
    assert wrapped(4) == 8
    assert wrapped(4) == 8
    assert calls == [4]


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


def test_memoize_invalidate_missing_key_no_error():
    @memoize
    def f(x):
        return x

    # invalidating a never-called key should not raise
    f.invalidate(99)
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


def test_memoize_skip_memory():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        raise SkipMemory(x * 100)

    result = f(2)
    assert result == 200
    # not cached, calls again
    assert f(2) == 200
    assert calls == [2, 2]
    assert f.memory == {}


def test_memoize_skip_memory_without_args():
    @memoize
    def f(x):
        raise SkipMemory()

    assert f(1) is None


def test_memoize_wraps_metadata():
    @memoize
    def named_func(x):
        """docstring here"""
        return x

    assert named_func.__name__ == 'named_func'
    assert named_func.__doc__ == 'docstring here'


# ---------------------------------------------------------------------------
# cache / CacheMemory
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


def test_cache_timedelta_timeout():
    # CacheMemory does not convert timedelta; only the cache() decorator does.
    c = CacheMemory(timedelta(seconds=60))
    assert c.timeout == timedelta(seconds=60)


def test_cache_decorator_timedelta():
    @cache(timedelta(minutes=1))
    def f(x):
        return x

    assert f(1) == 1
    assert f.memory.timeout == 60


def test_cache_memory_set_and_get():
    m = CacheMemory(1000)
    m['a'] = 42
    assert m['a'] == 42
    assert len(m) == 1


def test_cache_memory_expire_removes_entries():
    m = CacheMemory(1000)
    m['a'] = 1
    # Force expiration by manipulating stored expiry directly
    value, _ = dict.__getitem__(m, 'a')
    dict.__setitem__(m, 'a', (value, 0))
    # also fix the _expires deque entry
    m._expires[0] = 0
    m.expire()
    assert 'a' not in m


def test_cache_memory_expired_get_raises_keyerror():
    m = CacheMemory(1000)
    m['a'] = 1
    value, _ = dict.__getitem__(m, 'a')
    dict.__setitem__(m, 'a', (value, 0))
    m._expires[0] = 0
    with pytest.raises(KeyError):
        m['a']


def test_cache_memory_clear():
    m = CacheMemory(1000)
    m['a'] = 1
    m.clear()
    assert len(m) == 0
    assert len(m._keys) == 0
    assert len(m._expires) == 0


def test_cache_decorator_invalidate_and_all():
    calls = []

    @cache(1000)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1]
    f.invalidate(1)
    assert f(1) == 1
    assert calls == [1, 1]
    f.invalidate_all()
    assert f.memory == {}


def test_cache_skip_memory():
    @cache(1000)
    def f(x):
        raise SkipMemory('skipped')

    assert f(1) == 'skipped'
    assert len(f.memory) == 0


def test_cache_memory_expire_with_replaced_key():
    """Exercise the branch where a key was replaced in dict but old entry
    is still in the deques: entry is not None and entry[1] <= expires_at."""
    m = CacheMemory(1000)
    m['a'] = 1
    # Replace 'a' with a new value+expiry: dict now has a later expiry, but
    # the old deque entry for 'a' remains at the front.
    value, _ = dict.__getitem__(m, 'a')
    # Set the dict entry to a *later* expiry than the deque entry.
    dict.__setitem__(m, 'a', (value, 10**18))
    # Force the first deque entry to be expired, keeping dict entry valid.
    m._expires[0] = 0
    m.expire()
    # Branch taken: entry exists but entry[1] (large) > expires_at (0),
    # so the key must remain.
    assert 'a' in m
    assert m['a'] == 1


def test_cache_memory_expire_removes_stale_entry():
    """Exercise branch where entry exists and its stored expiry <= expires_at:
    the key gets popped."""
    m = CacheMemory(1000)
    m['a'] = 1
    value, _ = dict.__getitem__(m, 'a')
    # Align dict entry expiry with deque entry so both are expired.
    dict.__setitem__(m, 'a', (value, 0))
    m._expires[0] = 0
    m.expire()
    assert 'a' not in m
    assert len(m._keys) == 0
    assert len(m._expires) == 0


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_none():
    def f():
        pass

    assert has_arg_types(f) == (False, False)


def test_has_arg_types_positional():
    def f(a, b):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_keyword():
    def f(*, a):
        pass

    assert has_arg_types(f) == (False, True)


def test_has_arg_types_both():
    def f(a, *, b):
        pass

    assert has_arg_types(f) == (True, True)


def test_has_arg_types_varargs():
    def f(*args, **kwargs):
        pass

    assert has_arg_types(f) == (True, True)


def test_has_arg_types_positional_only():
    def f(a, /):
        pass

    assert has_arg_types(f) == (True, False)


# ---------------------------------------------------------------------------
# make_lookuper / silent_lookuper (supporting public behavior)
# ---------------------------------------------------------------------------

def test_make_lookuper_no_args():
    @make_lookuper
    def table():
        return {'a': 1, 'b': 2}

    assert table('a') == 1
    assert table('b') == 2
    with pytest.raises(LookupError):
        table('c')


def test_silent_lookuper_no_args():
    @silent_lookuper
    def table():
        return {'a': 1}

    assert table('a') == 1
    assert table('missing') is None


def test_make_lookuper_with_args():
    # With args, the decorated function takes the *builder* arguments and
    # returns a per-args lookuper function which must then be called with
    # a single lookup argument.
    calls = []

    @make_lookuper
    def table(prefix):
        calls.append(prefix)
        return {prefix + '1': 1, prefix + '2': 2}

    lookup = table('x')
    assert callable(lookup)
    assert lookup('x1') == 1
    assert lookup('x2') == 2
    with pytest.raises(LookupError):
        lookup('zzz')

    # Re-calling with the same builder args returns the memoized lookuper
    lookup2 = table('x')
    assert lookup2 is lookup
    assert calls == ['x']


def test_make_lookuper_rejects_kwargs_builder():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(**kwargs):
            return {}


def test_lookuper_builder_error_propagates():
    calls = []

    def builder():
        calls.append(1)
        raise ValueError('boom')

    looked = make_lookuper(builder)
    with pytest.raises(ValueError):
        looked('a')
    # memory cleared on error, retries builder
    with pytest.raises(ValueError):
        looked('a')
    assert calls == [1, 1]


def test_silent_lookuper_missing_none():
    @silent_lookuper
    def table():
        return {}

    assert table('anything') is None
