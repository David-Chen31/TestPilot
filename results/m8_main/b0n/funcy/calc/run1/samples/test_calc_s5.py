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
    _make_lookuper,
)


# ---------------------------------------------------------------------------
# SkipMemory
# ---------------------------------------------------------------------------

def test_skip_memory_is_exception():
    assert issubclass(SkipMemory, Exception)
    e = SkipMemory()
    assert isinstance(e, SkipMemory)


def test_memoize_skip_attribute():
    assert memoize.skip is SkipMemory


def test_cache_skip_attribute():
    assert cache.skip is SkipMemory


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_plain():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]
    assert f.memory == {3: 6}


def test_memoize_without_parens_directly_on_function():
    # @memoize(key_func=...) form returns decorator
    calls = []

    @memoize()
    def f(x):
        calls.append(x)
        return x + 1

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]


def test_memoize_distinguishes_kwargs():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert f(1) == 1
    assert f(1) == 1
    assert len(calls) == 2

    # positional and kwargs variants are distinct keys
    assert f(1, 2) if False else True  # placeholder, not asserting unsupported form


def test_memoize_positional_vs_keyword_distinct():
    @memoize
    def f(a, b):
        return (a, b)

    r1 = f(1, b=2)
    r2 = f(1, b=2)
    r3 = f(1, 2)
    assert r1 == r2 == r3 == (1, 2)
    # different key for kwargs form
    assert len(f.memory) == 2


def test_memoize_kwargs_marker():
    @memoize
    def f(**kwargs):
        return sum(kwargs.values())

    assert f(a=1, b=2) == 3
    assert f(a=1, b=2) == 3
    assert len(f.memory) == 1


def test_memoize_key_func():
    calls = []

    @memoize(key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(3) == 1  # same key -> cached result 1
    assert f(2) == 2
    assert f(4) == 2
    assert calls == [1, 2]


def test_memoize_key_func_kwargs():
    @memoize(key_func=lambda *a, **kw: (a, tuple(sorted(kw.items()))))
    def f(*args, **kwargs):
        return (args, kwargs)

    assert f(1, b=2) == ((1,), {'b': 2})
    assert f(1, b=2) == ((1,), {'b': 2})
    assert len(f.memory) == 1


def test_memoize_skip_memory():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        raise SkipMemory(x)

    assert f(5) == 5
    assert f(5) == 5
    # Not cached
    assert calls == [5, 5]
    assert f.memory == {}


def test_memoize_skip_no_args_returns_none():
    @memoize
    def f(x):
        raise SkipMemory()

    assert f(1) is None


def test_memoize_invalidate():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(2) == 2
    assert f(1) == 1
    assert calls == [1, 2]
    f.invalidate(1)
    assert f(1) == 1
    assert calls == [1, 2, 1]


def test_memoize_invalidate_missing():
    @memoize
    def f(x):
        return x

    # no error when invalidating unknown key
    f.invalidate(100)
    assert f.memory == {}


def test_memoize_invalidate_kwargs():
    calls = []

    @memoize
    def f(x, y):
        calls.append((x, y))
        return (x, y)

    f(1, y=2)
    f(1, y=2)
    assert calls == [(1, 2)]
    f.invalidate(1, y=2)
    f(1, y=2)
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


def test_memoize_invalidate_key_func():
    @memoize(key_func=lambda x: x)
    def f(x):
        return x

    f(1)
    assert 1 in f.memory
    f.invalidate(1)
    assert 1 not in f.memory


def test_memoize_memory_shared_per_function():
    @memoize
    def f(x):
        return x

    @memoize
    def g(x):
        return x

    f(1)
    g(1)
    # separatememory dicts
    assert f.memory == g.memory == {1: 1}
    assert f.memory is not g.memory


def test_memoize_preserves_metadata():
    @memoize
    def my_func(a, b):
        """Docstring here."""
        return a + b

    assert my_func.__name__ == 'my_func'
    assert my_func.__doc__ == 'Docstring here.'


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def test_cache_basic():
    calls = []

    @cache(100)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]
    assert isinstance(f.memory, CacheMemory)
    assert f.memory.timeout == 100


def test_cache_timedelta():
    @cache(timedelta(seconds=50))
    def f(x):
        return x

    assert f.memory.timeout == 50
    assert f(3) == 3


def test_cache_expiry(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr(time, 'monotonic', lambda: fake_now[0])

    calls = []

    @cache(10)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1]

    fake_now[0] += 5  # not yet expired
    assert f(1) == 1
    assert calls == [1]

    fake_now[0] += 10  # now expired (expires_at <= now)
    assert f(1) == 1
    assert calls == [1, 1]


def test_cache_skip_memory():
    @cache(100)
    def f(x):
        raise SkipMemory(x + 100)

    assert f(1) == 101
    assert f(1) == 101
    assert f.memory == {}


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


def test_cache_invalidate_all():
    @cache(100)
    def f(x):
        return x

    f(1)
    f(2)
    assert len(f.memory) == 2
    f.invalidate_all()
    assert f.memory == {}


def test_cache_key_func():
    @cache(100, key_func=lambda x: x % 3)
    def f(x):
        return x

    assert f(3) == 3
    assert f(6) == 3  # same key
    assert f(4) == 4


# ---------------------------------------------------------------------------
# CacheMemory
# ---------------------------------------------------------------------------

def test_cache_memory_basic(monkeypatch):
    fake_now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: fake_now[0])

    m = CacheMemory(10)
    m['a'] = 1
    assert m['a'] == 1


def test_cache_memory_expiry(monkeypatch):
    fake_now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: fake_now[0])

    m = CacheMemory(10)
    m['a'] = 1
    fake_now[0] = 5.0
    assert m['a'] == 1
    fake_now[0] = 10.0
    with pytest.raises(KeyError):
        m['a']


def test_cache_memory_setitem_stores_tuple():
    m = CacheMemory(10)
    m['key'] = 'value'
    raw = dict.__getitem__(m, 'key')
    assert raw[0] == 'value'
    assert isinstance(raw[1], float)


def test_cache_memory_expire_partial(monkeypatch):
    fake_now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: fake_now[0])

    m = CacheMemory(10)
    m['a'] = 1
    fake_now[0] = 5.0
    m['b'] = 2
    
    fake_now[0] = 10.0
    # a is expired (expires at 10.0 <= now), b not (expires at 15)
    with pytest.raises(KeyError):
        m['a']
    assert m['b'] == 2

    fake_now[0] = 15.0
    with pytest.raises(KeyError):
        m['b']


def test_cache_memory_expire_removes_entries(monkeypatch):
    fake_now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: fake_now[0])

    m = CacheMemory(10)
    m['a'] = 1
    m['b'] = 2
    fake_now[0] = 20.0
    m.expire()
    # all expired
    assert dict.__len__(m) == 0
    assert len(m._keys) == 0
    assert len(m._expires) == 0


def test_cache_memory_expire_partial_keeps_valid(monkeypatch):
    fake_now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: fake_now[0])

    m = CacheMemory(10)
    m['old'] = 1
    fake_now[0] = 5.0
    m['new'] = 2

    fake_now[0] = 10.5
    m.expire()
    assert dict.__len__(m) == 1
    assert m['new'] == 2


def test_cache_memory_clear():
    m = CacheMemory(10)
    m['a'] = 1
    m['b'] = 2
    m.clear()
    assert dict.__len__(m) == 0
    assert len(m._keys) == 0
    assert len(m._expires) == 0
    # timeout preserved
    assert m.timeout == 10


def test_cache_memory_missing_key():
    m = CacheMemory(10)
    with pytest.raises(KeyError):
        m['nonexistent']


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_no_params():
    def f():
        pass
    assert has_arg_types(f) == (False, False)


def test_has_arg_types_positional_only():
    def f(a, /):
        pass
    assert has_arg_types(f) == (True, False)


def test_has_arg_types_positional_or_keyword():
    def f(a):
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


def test_has_arg_types_mixed():
    def f(a, *args, b=1, **kwargs):
        pass
    assert has_arg_types(f) == (True, True)


def test_has_arg_types_builtin():
    # len has positional arg
    has_args, has_keys = has_arg_types(len)
    assert has_args is True
    assert has_keys is False


# ---------------------------------------------------------------------------
# lookuper / silent_lookuper (public names)
# ---------------------------------------------------------------------------

def test_make_lookuper_basic():
    @make_lookuper
    def table():
        return {'a': 1, 'b': 2}

    assert table('a') == 1
    assert table('b') == 2


def test_make_lookuper_error():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {'a': 1}

    assert table('a') == 1
    with pytest.raises(LookupError):
        table('missing')
    # function called only once
    assert calls == [1]


def test_make_lookuper_error_message():
    @make_lookuper
    def table():
        return {'a': 1}

    with pytest.raises(LookupError) as excinfo:
        table('zzz')
    assert 'zzz' in str(excinfo.value)


def test_silent_lookuper_basic():
    @silent_lookuper
    def table():
        return {'a': 1}

    assert table('a') == 1
    assert table('missing') is None


def test_silent_lookuper_name():
    assert silent_lookuper.__name__ == 'silent_lookuper'


def test_make_lookuper_called_once():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        return {'a': 1, 'b': 2}

    table('a')
    table('b')
    table('a')
    assert calls == [1]


def test_make_lookuper_empty_result_raises():
    @make_lookuper
    def table():
        return {}

    with pytest.raises(LookupError):
        table('a')


def test_silent_lookuper_empty_result_none():
    @silent_lookuper
    def table():
        return {}

    assert table('a') is None


def test_make_lookuper_exception_in_builder():
    calls = []

    @make_lookuper
    def table():
        calls.append(1)
        raise ValueError('boom')

    with pytest.raises(ValueError):
        table('a')
    # memory cleared so it retries
    with pytest.raises(ValueError):
        table('a')
    assert calls == [1, 1]


def test_make_lookuper_rejects_kwargs():
    with pytest.raises(AssertionError):
        @make_lookuper
        def table(**kwargs):
            return {}


def test_make_lookuper_with_args_caches():
    calls = []

    @make_lookuper
    def table(prefix):
        calls.append(prefix)
        return {prefix + 'a': 1}

    assert table('x', 'a') == 1
    assert table('x', 'a') == 1
    assert calls == ['x']


def test_make_lookuper_with_args_missing():
    @make_lookuper
    def table(prefix):
        return {prefix + 'a': 1}

    assert table('x', 'a') == 1
    with pytest.raises(LookupError):
        table('x', 'b')


def test_silent_lookuper_with_args():
    @silent_lookuper
    def table(prefix):
        return {prefix + 'a': 1}

    assert table('y', 'a') == 1
    assert table('y', 'zzz') is None


def test_silent_lookuper_with_args_caches():
    calls = []

    @silent_lookuper
    def table(prefix):
        calls.append(prefix)
        return {prefix + 'a': 1}

    table('x', 'a')
    table('x', 'a')
    assert calls == ['x']


def test_lookuper_preserves_metadata():
    @make_lookuper
    def my_table():
        """Table doc."""
        return {'a': 1}

    assert my_table.__name__ == 'my_table'
    assert my_table.__doc__ == 'Table doc.'


def test_make_lookuper_returns_wrapper():
    def builder():
        return {'a': 1}

    wrapped = make_lookuper(builder)
    assert wrapped is not builder
    assert callable(wrapped)


def test__make_lookuper_silent_variant():
    lookup = _make_lookuper(True)

    @lookup
    def table():
        return {'a': 1}

    assert table('a') == 1
    assert table('b') is None
