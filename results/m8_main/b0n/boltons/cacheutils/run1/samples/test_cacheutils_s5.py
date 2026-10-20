import gc
import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    make_cache_key,
    _make_cache_key,
    CachedFunction,
    CachedMethod,
    cached,
    cachedmethod,
    cachedproperty,
    ThresholdCounter,
    MinIDMap,
    DEFAULT_MAX_SIZE,
    _HashedKey,
)
from boltons.cacheutils import _MISSING, _KWARG_MARK


# ---------------------------------------------------------------------------
# LRI
# ---------------------------------------------------------------------------

class TestLRI:
    def test_defaults(self):
        c = LRI()
        assert c.max_size == DEFAULT_MAX_SIZE
        assert c.on_miss is None
        assert c.hit_count == 0
        assert c.miss_count == 0
        assert c.soft_miss_count == 0
        assert len(c) == 0

    def test_invalid_max_size(self):
        with pytest.raises(ValueError):
            LRI(max_size=0)
        with pytest.raises(ValueError):
            LRI(max_size=-1)

    def test_invalid_on_miss(self):
        with pytest.raises(TypeError):
            LRI(on_miss=5)

    def test_set_get(self):
        c = LRI(max_size=2)
        c['a'] = 'A'
        c['b'] = 'B'
        assert c['a'] == 'A'
        assert c['b'] == 'B'
        assert dict(c) == {'a': 'A', 'b': 'B'}
        assert c.hit_count == 2
        assert c.miss_count == 0

    def test_lri_evicts_least_recently_inserted(self):
        c = LRI(max_size=2)
        c['a'] = 'A'
        c['b'] = 'B'
        # Accessing 'b' should NOT change eviction order for LRI
        assert c['b'] == 'B'
        c['c'] = 'C'
        assert 'a' not in c
        assert 'b' in c
        assert 'c' in c

    def test_insert_existing_moves_to_front_no_eviction(self):
        c = LRI(max_size=2)
        c['a'] = 'A'
        c['b'] = 'B'
        c['a'] = 'A2'
        assert c['a'] == 'A2'
        assert len(c) == 2

    def test_miss_raises_without_on_miss(self):
        c = LRI()
        with pytest.raises(KeyError):
            c['x']
        assert c.miss_count == 1

    def test_on_miss(self):
        calls = []

        def on_miss(key):
            calls.append(key)
            return key.upper()

        c = LRI(on_miss=on_miss)
        assert c['a'] == 'A'
        assert calls == ['a']
        assert c['a'] == 'A'
        assert calls == ['a']

    def test_get_soft_miss(self):
        c = LRI()
        assert c.get('x') is None
        assert c.soft_miss_count == 1
        assert c.miss_count == 1
        assert c.get('x', 'D') == 'D'
        assert c.soft_miss_count == 2

    def test_get_hit(self):
        c = LRI()
        c['a'] = 'A'
        assert c.get('a') == 'A'
        assert c.soft_miss_count == 0
        assert c.hit_count == 1

    def test_setdefault(self):
        c = LRI()
        assert c.setdefault('a', 'A') == 'A'
        assert c['a'] == 'A'
        # second time returns existing, no soft miss
        sm = c.soft_miss_count
        assert c.setdefault('a', 'B') == 'A'
        assert c.soft_miss_count == sm

    def test_delitem(self):
        c = LRI(max_size=2)
        c['a'] = 'A'
        c['b'] = 'B'
        del c['a']
        assert 'a' not in c
        assert len(c) == 1
        with pytest.raises(KeyError):
            del c['a']

    def test_pop(self):
        c = LRI()
        c['a'] = 'A'
        assert c.pop('a') == 'A'
        assert 'a' not in c
        with pytest.raises(KeyError):
            c.pop('a')
        assert c.pop('a', 'D') == 'D'

    def test_popitem(self):
        c = LRI()
        c['a'] = 'A'
        item = c.popitem()
        assert item == ('a', 'A')
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = LRI(max_size=2)
        c['a'] = 'A'
        c['b'] = 'B'
        c.clear()
        assert len(c) == 0
        # After clear, should be reusable.
        c['x'] = 'X'
        assert c['x'] == 'X'
        assert len(c) == 1

    def test_copy(self):
        c = LRI(max_size=2, on_miss=None)
        c['a'] = 'A'
        c2 = c.copy()
        assert isinstance(c2, LRI)
        assert c2.max_size == 2
        assert dict(c2) == {'a': 'A'}
        # Ensure independent
        c2['b'] = 'B'
        assert 'b' not in c

    def test_values_init(self):
        c = LRI(max_size=5, values={'a': 'A', 'b': 'B'})
        assert dict(c) == {'a': 'A', 'b': 'B'}

    def test_values_init_iterable(self):
        c = LRI(max_size=5, values=[('a', 'A'), ('b', 'B')])
        assert dict(c) == {'a': 'A', 'b': 'B'}

    def test_update_dict(self):
        c = LRI()
        c.update({'a': 'A'})
        assert c['a'] == 'A'

    def test_update_kwargs(self):
        c = LRI()
        c.update(a='A', b='B')
        assert dict(c) == {'a': 'A', 'b': 'B'}

    def test_update_self_noop(self):
        c = LRI()
        c['a'] = 'A'
        c.update(c)
        assert dict(c) == {'a': 'A'}

    def test_eq(self):
        c1 = LRI()
        c1['a'] = 1
        c2 = LRI()
        c2['a'] = 1
        assert c1 == c2
        assert not (c1 != c2)
        assert c1 == {'a': 1}
        assert {'a': 1} == c1
        c2['b'] = 2
        assert c1 != c2

    def test_eq_self(self):
        c = LRI()
        assert c == c

    def test_repr(self):
        c = LRI(max_size=5)
        c['a'] = 'A'
        r = repr(c)
        assert 'LRI' in r
        assert 'max_size=5' in r
        assert "'a': 'A'" in r

    def test_repr_on_miss(self):
        def on_miss(k):
            return k
        c = LRI(on_miss=on_miss)
        assert 'on_miss=' in repr(c)
        assert 'on_miss=None' not in repr(c)


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

class TestLRU:
    def test_defaults(self):
        c = LRU()
        assert c.max_size == DEFAULT_MAX_SIZE
        assert isinstance(c, LRI)

    def test_lru_evicts_least_recently_used(self):
        c = LRU(max_size=2)
        c['a'] = 'A'
        c['b'] = 'B'
        assert c['a'] == 'A'  # touch 'a'
        c['c'] = 'C'
        assert 'b' not in c
        assert 'a' in c
        assert 'c' in c

    def test_lru_hit_count(self):
        c = LRU(max_size=2)
        c['a'] = 'A'
        assert c['a'] == 'A'
        assert c.hit_count == 1

    def test_lru_miss_with_on_miss(self):
        c = LRU(max_size=2, on_miss=lambda k: k * 2)
        assert c['x'] == 'xx'
        assert c['x'] == 'xx'
        assert c.miss_count == 1
        assert c.hit_count == 1

    def test_lru_miss_raises(self):
        c = LRU()
        with pytest.raises(KeyError):
            c['x']
        assert c.miss_count == 1

    def test_lru_get_soft_miss(self):
        c = LRU()
        assert c.get('a') is None
        assert c.soft_miss_count == 1


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

class TestMakeCacheKey:
    def test_empty(self):
        k = make_cache_key((), {})
        assert isinstance(k, _HashedKey)
        assert list(k) == []

    def test_single_fasttype(self):
        assert make_cache_key(('a',), {}) == 'a'
        assert make_cache_key((5,), {}) == 5
        assert make_cache_key((None,), {}) is None

    def test_single_non_fasttype(self):
        x = (1, 2)
        k = make_cache_key((x,), {})
        assert isinstance(k, _HashedKey)
        assert list(k) == [x]

    def test_multiple_args(self):
        k = make_cache_key(('a', 'b'), {})
        assert isinstance(k, _HashedKey)
        assert list(k) == ['a', 'b']

    def test_kwargs_sorted(self):
        k = make_cache_key((), {'c': ('d'), 'a': 1})
        assert isinstance(k, _HashedKey)
        items = list(k)
        assert items[0] is _KWARG_MARK
        assert items[1:] == [('a', 1), ('c', 'd')]

    def test_typed(self):
        k = make_cache_key((1,), {}, typed=True)
        assert isinstance(k, _HashedKey)
        assert list(k) == [1, int]
        k2 = make_cache_key((1.0,), {}, typed=True)
        assert list(k2) == [1.0, float]
        assert hash(k) != hash(k2)

    def test_typed_kwargs(self):
        k = make_cache_key((), {'a': 1}, typed=True)
        items = list(k)
        assert items[0] is _KWARG_MARK
        assert ('a', 1) in items
        assert int in items

    def test_hashed_key_hash(self):
        k = _HashedKey([1, 2, 3])
        assert hash(k) == hash((1, 2, 3))
        assert k.hash_value == hash((1, 2, 3))

    def test_hashed_key_repr(self):
        k = _HashedKey([1, 2])
        assert '_HashedKey' in repr(k)
        assert '[1, 2]' in repr(k)

    def test_alias(self):
        assert _make_cache_key is make_cache_key


# ---------------------------------------------------------------------------
# cached / CachedFunction
# ---------------------------------------------------------------------------

class TestCached:
    def test_basic(self):
        calls = []

        @cached({})
        def f(x):
            calls.append(x)
            return x * 2

        assert f(3) == 6
        assert f(3) == 6
        assert calls == [3]

    def test_cached_with_lru(self):
        cache = LRU()

        @cached(cache)
        def f(x):
            return x.lower()

        assert f("ABC") == 'abc'
        assert len(cache) == 1

    def test_cached_typed(self):
        calls = []

        @cached({}, typed=True)
        def f(x):
            calls.append(x)
            return x

        assert f(3) == 3
        assert f(3.0) == 3.0
        assert len(calls) == 2

    def test_cached_not_typed(self):
        calls = []

        @cached({})
        def f(x):
            calls.append(x)
            return x

        assert f(3) == 3
        assert f(3.0) == 3.0
        assert len(calls) == 1

    def test_cached_callable_cache(self):
        store = {}

        @cached(lambda: store)
        def f(x):
            return x * 2

        assert f(4) == 8
        assert len(store) == 1

    def test_cached_multiple_args(self):
        calls = []

        @cached({})
        def f(a, b):
            calls.append((a, b))
            return a + b

        assert f(1, 2) == 3
        assert f(1, 2) == 3
        assert calls == [(1, 2)]

    def test_cached_kwargs_order_invariant(self):
        calls = []

        @cached({})
        def f(**kw):
            calls.append(kw)
            return sum(kw.values())

        assert f(a=1, b=2) == 3
        assert f(b=2, a=1) == 3
        assert len(calls) == 1

    def test_cached_custom_key(self):
        calls = []

        def key_func(args, kwargs, typed=False):
            return 'fixed'

        @cached({}, key=key_func)
        def f(x):
            calls.append(x)
            return x

        assert f(1) == 1
        assert f(2) == 1
        assert calls == [1]

    def test_cached_scoped_false(self):
        cache = {}

        @cached(cache, scoped=False)
        def f1(x):
            return ('f1', x)

        @cached(cache, scoped=False)
        def f2(x):
            return ('f2', x)

        # With scoped=False, both use same key; f2 sees f1's value.
        assert f1(1) == ('f1', 1)
        assert f2(1) == ('f1', 1)


class TestCachedFunctionRepr:
    def test_repr_default(self):
        def f():
            pass
        cf = CachedFunction(f, {})
        r = repr(cf)
        assert 'CachedFunction' in r
        assert 'func=' in r

    def test_repr_typed(self):
        def f():
            pass
        cf = CachedFunction(f, {}, typed=True)
        assert 'typed=True' in repr(cf)

    def test_repr_not_scoped(self):
        def f():
            pass
        cf = CachedFunction(f, {}, scoped=False)
        assert 'scoped=False' in repr(cf)

    def test_invalid_cache(self):
        with pytest.raises(TypeError):
            CachedFunction(lambda: None, object())

    def test_callable_cache_invalid(self):
        # object without __getitem__/__setitem__
        with pytest.raises(TypeError):
            CachedFunction(lambda: None, [1, 2, 3] and 5)


# ---------------------------------------------------------------------------
# cachedmethod / CachedMethod
# ---------------------------------------------------------------------------

class TestCachedMethod:
    def test_basic(self):
        class Foo:
            def __init__(self):
                self.cache = {}
                self.calls = 0

            @cachedmethod('cache')
            def compute(self, x):
                self.calls += 1
                return x * 2

        f = Foo()
        assert f.compute(3) == 6
        assert f.compute(3) == 6
        assert f.calls == 1
        assert len(f.cache) == 1

    def test_multiple_instances_isolated(self):
        class Foo:
            def __init__(self):
                self.cache = {}

            @cachedmethod('cache')
            def compute(self, x):
                return x + 1

        a = Foo()
        b = Foo()
        assert a.compute(1) == 2
        assert b.compute(1) == 2
        assert len(a.cache) == 1
        assert len(b.cache) == 1

    def test_cache_as_dict_object(self):
        cache = {}

        class Foo:
            @cachedmethod(cache)
            def compute(self, x):
                return x * 3

        f = Foo()
        assert f.compute(2) == 6
        assert f.compute(2) == 6
        assert len(cache) == 1

    def test_cache_as_callable(self):
        cache = {}

        def get_cache(obj):
            return cache

        class Foo:
            @cachedmethod(get_cache)
            def compute(self, x):
                return x * 4

        f = Foo()
        assert f.compute(2) == 8
        assert len(cache) == 1

    def test_access_class_returns_cachedmethod(self):
        class Foo:
            @cachedmethod('cache')
            def compute(self, x):
                return x

        assert isinstance(Foo.compute, CachedMethod)

    def test_repr_unbound(self):
        class Foo:
            @cachedmethod('cache')
            def compute(self, x):
                return x

        r = repr(Foo.__dict__['compute'])
        assert 'CachedMethod' in r

    def test_repr_bound(self):
        class Foo:
            def __init__(self):
                self.cache = {}

            @cachedmethod('cache')
            def compute(self, x):
                return x

        f = Foo()
        r = repr(f.compute)
        assert 'bound_to' in r

    def test_invalid_cache(self):
        with pytest.raises(TypeError):
            CachedMethod(lambda self: None, 5)

    def test_isabstractmethod(self):
        class Foo:
            @cachedmethod('cache')
            @property
            def _abstract(self):
                pass

        # just checks attribute exists; actual abstractness depends on func
        assert hasattr(Foo.__dict__['_abstract'], '__isabstractmethod__')


# ---------------------------------------------------------------------------
# cachedproperty
# ---------------------------------------------------------------------------

class TestCachedProperty:
    def test_basic(self):
        calls = []

        class Foo:
            @cachedproperty
            def value(self):
                calls.append(1)
                return 42

        f = Foo()
        assert f.value == 42
        assert f.value == 42
        assert len(calls) == 1
        assert f.__dict__['value'] == 42

    def test_class_access(self):
        class Foo:
            @cachedproperty
            def value(self):
                return 1

        assert isinstance(Foo.value, cachedproperty)

    def test_clear_via_delattr(self):
        calls = []

        class Foo:
            @cachedproperty
            def value(self):
                calls.append(1)
                return len(calls)

        f = Foo()
        assert f.value == 1
        del f.value
        assert f.value == 2

    def test_doc(self):
        class Foo:
            @cachedproperty
            def value(self):
                """My doc."""
                return 1

        assert Foo.__dict__['value'].__doc__ == "My doc."

    def test_repr(self):
        class Foo:
            @cachedproperty
            def value(self):
                return 1

        r = repr(Foo.__dict__['value'])
        assert 'cachedproperty' in r


# ---------------------------------------------------------------------------
# ThresholdCounter
# ---------------------------------------------------------------------------

class TestThresholdCounter:
    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            ThresholdCounter(threshold=0)
        with pytest.raises(ValueError):
            ThresholdCounter(threshold=1)
        with pytest.raises(ValueError):
            ThresholdCounter(threshold=-0.1)

    def test_default_threshold(self):
        tc = ThresholdCounter()
        assert tc.threshold == 0.001

    def test_add_and_get(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add(1)
        tc.add(1)
        tc.add(2)
        assert tc.get(1) == 2
        assert tc.get(2) == 1
        assert tc.get(3) == 0
        assert tc.get(3, -1) == -1
        assert tc.total == 3

    def test_getitem(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')
        assert tc['a'] == 1
        with pytest.raises(KeyError):
            tc['b']

    def test_contains_len(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')
        assert 'a' in tc
        assert 'b' not in tc
        assert len(tc) == 1

    def test_keys_values_items(self):
        tc = ThresholdCounter(threshold=0.01)
        tc.add('a')
        tc.add('a')
        tc.add('b')
        assert sorted(tc.keys()) == ['a', 'b']
        assert sorted(tc.values()) == [1, 2]
        assert sorted(tc.items()) == [('a', 2), ('b', 1)]

    def test_iterkeys_itervalues_iteritems(self):
        tc = ThresholdCounter(threshold=0.01)
        tc.add('a')
        tc.add('b')
        assert sorted(tc.iterkeys()) == ['a', 'b']
        assert sorted(tc.itervalues()) == [1, 1]
        assert sorted(tc.iteritems()) == [('a', 1), ('b', 1)]

    def test_elements(self):
        tc = ThresholdCounter(threshold=0.01)
        tc.add('a')
        tc.add('a')
        tc.add('b')
        elems = sorted(tc.elements())
        assert elems == ['a', 'a', 'b']

    def test_most_common(self):
        tc = ThresholdCounter(threshold=0.01)
        tc.update(['a'] * 3 + ['b'] * 1 + ['c'] * 2)
        mc = tc.most_common()
        assert mc[0] == ('a', 3)
        assert mc[1] == ('c', 2)
        assert mc[2] == ('b', 1)

        assert tc.most_common(2) == [('a', 3), ('c', 2)]
        assert tc.most_common(0) == []
        assert tc.most_common(-1) == []
        assert len(tc.most_common(10)) == 3

    def test_update_iterable_and_mapping(self):
        tc = ThresholdCounter(threshold=0.01)
        tc.update(['a', 'b'])
        assert tc.get('a') == 1
        assert tc.get('b') == 1
        tc.update({'a': 3})
        assert tc.get('a') == 4

    def test_update_kwargs(self):
        tc = ThresholdCounter(threshold=0.01)
        tc.update(None, a=2)
        assert tc.get('a') == 2

    def test_compaction(self):
        # threshold such that _thresh_count == 10
        tc = ThresholdCounter(threshold=0.1)
        for _ in range(10):
            tc.add('x')
        # after 10 adds, bucket 2; 'x' count 10, stored[1]==0 -> 10 > 1 keep
        assert tc.get('x') == 10
        # Now add a new key 10 times to trigger another compaction
        tc.add('y')
        assert tc.get('y') == 1

    def test_doc_example(self):
        tc = ThresholdCounter(threshold=0.1)
        tc.add(1)
        assert tc.items() == [(1, 1)]
        tc.update([2] * 10)
        assert tc.get(1) == 0
        tc.add(5)
        assert 5 in tc
        assert len(list(tc.elements())) == 11

    def test_get_common_count(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')
        tc.add('a')
        tc.add('b')
        assert tc.get_common_count() == 3
        assert tc.get_uncommon_count() == 0

    def test_get_commonality(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')
        tc.add('a')
        tc.add('b')
        assert tc.get_commonality() == 1.0

    def test_commonality_empty(self):
        tc = ThresholdCounter()
        assert tc.get_common_count() == 0
        assert tc.total == 0


# ---------------------------------------------------------------------------
# MinIDMap
# ---------------------------------------------------------------------------

class TestMinIDMap:
    def test_basic(self):
        m = MinIDMap()
        a = object()
        i = m.get(a)
        assert i == 0
        assert m.get(a) == 0
        assert a in m
        assert len(m) == 1

    def test_multiple(self):
        m = MinIDMap()
        objs = [object() for _ in range(5)]
        ids = [m.get(o) for o in objs]
        assert ids == [0, 1, 2, 3, 4]
        assert len(m) == 5

    def test_drop(self):
        m = MinIDMap()
        a = object()
        b = object()
        m.get(a)
        m.get(b)
        m.drop(a)
        assert a not in m
        assert len(m) == 1

    def test_drop_and_reuse(self):
        m = MinIDMap()
        a = object()
        b = object()
        m.get(a)  # 0
        m.get(b)  # 1
        m.drop(a)  # free [0]
        c = object()
        idx = m.get(c)
        assert idx == 0

    def test_iter_and_iteritems(self):
        m = MinIDMap()
        a = object()
        b = object()
        m.get(a)
        m.get(b)
        items = list(m.iteritems())
        assert len(items) == 2
        keys_set = set(k for k, _ in items)
        assert a in keys_set
        assert b in keys_set

    def test_len(self):
        m = MinIDMap()
        assert len(m) == 0
        a = object()
        m.get(a)
        assert len(m) == 1

    def test_gc_cleans(self, capsys):
        m = MinIDMap()
        a = object()
        idx = m.get(a)
        assert idx == 0
        del a
        gc.collect()
        # After cleanup, ref should be removed from ref_map.
        assert len(m.ref_map) == 0
        # _clean prints the freed id; capture and check.
        out = capsys.readouterr().out
        assert '0' in out


# ---------------------------------------------------------------------------
# Edge cases / locks
# ---------------------------------------------------------------------------

def test_lri_lock_attribute():
    c = LRI()
    # RLock is available on normal CPython.
    assert hasattr(c, '_lock')


def test_eviction_boundary():
    # Fill exactly to max_size, ensure no premature eviction
    c = LRI(max_size=3)
    for i in range(3):
        c[i] = i
    assert len(c) == 3
    c[3] = 3
    assert len(c) == 3
    assert 0 not in c
    assert 3 in c
