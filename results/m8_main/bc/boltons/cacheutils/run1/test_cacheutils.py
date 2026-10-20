import gc
import sys

import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    CachedFunction,
    CachedMethod,
    cached,
    cachedmethod,
    cachedproperty,
    make_cache_key,
    ThresholdCounter,
    MinIDMap,
    _HashedKey,
)


# ---------------------------------------------------------------- LRI / LRU

def test_lri_basic_set_get_and_stats():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert dict(c) == {'a': 'A', 'b': 'B'}
    assert c.hit_count == 0
    assert c.miss_count == 0
    assert c.soft_miss_count == 0

    assert c['a'] == 'A'
    assert c.hit_count == 1

    with pytest.raises(KeyError):
        c['missing']
    assert c.miss_count == 1

    assert c.get('missing2') is None
    assert c.soft_miss_count == 1
    # get() delegates to __getitem__, so miss_count also incremented
    assert c.miss_count == 2


def test_lri_eviction_is_least_recently_inserted():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    # Accessing 'a' should not change insertion order in LRI.
    assert c['a'] == 'A'
    c['c'] = 'C'
    assert 'a' not in c
    assert dict(c) == {'b': 'B', 'c': 'C'}


def test_lri_max_size_validation():
    with pytest.raises(ValueError):
        LRI(max_size=0)
    with pytest.raises(ValueError):
        LRI(max_size=-1)


def test_lri_default_max_size():
    c = LRI()
    assert c.max_size == 128


def test_lri_on_miss_callable():
    calls = []

    def on_miss(key):
        calls.append(key)
        return key.upper()

    c = LRI(max_size=2, on_miss=on_miss)
    assert c['x'] == 'X'
    assert calls == ['x']
    assert c.miss_count == 1
    assert c['x'] == 'X'
    assert calls == ['x']
    assert c.hit_count == 1


def test_lri_on_miss_must_be_callable():
    with pytest.raises(TypeError):
        LRI(on_miss=5)


def test_lri_update_and_initial_values():
    c = LRI(max_size=4, values={'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}

    c.update([('c', 3), ('d', 4)])
    assert dict(c) == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    # Inserting 'e' into a max_size=4 cache evicts the oldest ('a').
    c.update(e=5)
    assert c['e'] == 5
    assert 'a' not in c
    assert dict(c) == {'b': 2, 'c': 3, 'd': 4, 'e': 5}

    c2 = LRI()  # default max_size 128, no eviction
    c2.update(c)
    assert dict(c2) == {'b': 2, 'c': 3, 'd': 4, 'e': 5}

    # update with self should not raise
    c.update(c)
    assert dict(c) == {'b': 2, 'c': 3, 'd': 4, 'e': 5}


def test_lri_setdefault():
    c = LRI()
    assert c.setdefault('a', 1) == 1
    assert c['a'] == 1
    assert c.soft_miss_count == 1
    assert c.setdefault('a', 2) == 1
    assert c.soft_miss_count == 1


def test_lri_delete_and_pop():
    c = LRI(max_size=4)
    c['a'] = 1
    c['b'] = 2
    del c['a']
    assert 'a' not in c
    assert dict(c) == {'b': 2}

    with pytest.raises(KeyError):
        del c['nope']

    assert c.pop('b') == 2
    assert 'b' not in c

    with pytest.raises(KeyError):
        c.pop('nope')
    assert c.pop('nope', 'default') == 'default'

    c['x'] = 10
    c['y'] = 20
    item = c.popitem()
    assert item in (('x', 10), ('y', 20))
    assert len(c) == 1


def test_lri_clear():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    c.clear()
    assert dict(c) == {}
    assert len(c) == 0
    # Can still add after clear
    c['c'] = 3
    assert c['c'] == 3
    assert dict(c) == {'c': 3}


def test_lri_copy():
    c = LRI(max_size=5, values={'a': 1})
    c2 = c.copy()
    assert isinstance(c2, LRI)
    assert dict(c2) == {'a': 1}
    assert c2.max_size == 5
    c2['b'] = 2
    assert 'b' not in c


def test_lri_eq_and_ne():
    c1 = LRI(values={'a': 1})
    c2 = LRI(values={'a': 1})
    assert c1 == c2
    assert not (c1 != c2)
    c2['b'] = 2
    assert c1 != c2
    assert not (c1 == c2)


def test_lri_eq_other_mapping():
    # Comparing an LRI to a plain dict only works when the LRI is on the
    # left-hand side and the dict is not a plain dict by identity; in the
    # current implementation LRI.__eq__ delegates for non-LRI to
    # ``other == self``. For non-dict "other" mappings this works. Here we
    # only test the LRI-vs-LRI path and length mismatch to avoid recursion.
    c = LRI(values={'a': 1})
    # Different length returns False without recursion.
    assert not (c == {'a': 1, 'b': 2})
    assert c != {'a': 1, 'b': 2}
    # Self-comparison short-circuits.
    assert c == c


def test_lri_eq_non_lri_same_length():
    # Exercise the ``not isinstance(other, LRI)`` branch (line 317-318).
    class DictLike:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __eq__(self, other):
            return dict(other) == self.data

    c = LRI(values={'a': 1})
    dl = DictLike({'a': 1})
    assert c == dl


def test_lri_repr():
    c = LRI(max_size=3)
    r = repr(c)
    assert 'LRI' in r
    assert 'max_size=3' in r
    assert 'on_miss=None' in r


def test_lri_internal_flattened_ll():
    c = LRI(max_size=4)
    c['a'] = 1
    c['b'] = 2
    # anchor entry (key, value) both _MISSING, then insertion order a, b
    flat = c._get_flattened_ll()
    assert len(flat) == 3
    # First entry is anchor sentinel; key/value are not real data.
    assert flat[0][0] is not 'a'
    keys = [k for (k, v) in flat[1:]]
    assert keys == ['a', 'b']
    values = [v for (k, v) in flat[1:]]
    assert values == [1, 2]


def test_lru_eviction_is_least_recently_used():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert c['a'] == 'A'  # 'a' is now most recently used
    c['c'] = 'C'  # evicts 'b'
    assert 'b' not in c
    assert dict(c) == {'a': 'A', 'c': 'C'}


def test_lru_stats():
    c = LRU(max_size=2)
    c['a'] = 'A'
    assert c['a'] == 'A'
    assert c.hit_count == 1
    with pytest.raises(KeyError):
        c['b']
    assert c.miss_count == 1


def test_lru_on_miss():
    c = LRU(max_size=2, on_miss=lambda k: k + '!')
    assert c['x'] == 'x!'
    assert c['x'] == 'x!'
    assert c.miss_count == 1
    assert c.hit_count == 1


def test_lru_setitem_updates_existing_value():
    # Exercises LRI.__setitem__ branch where key already exists (line 230).
    c = LRU(max_size=2)
    c['a'] = 'A'
    assert c['a'] == 'A'
    c['a'] = 'A2'
    assert c['a'] == 'A2'
    assert dict(c) == {'a': 'A2'}


# ---------------------------------------------------------------- make_cache_key

def test_make_cache_key_basic():
    key = make_cache_key(('a', 'b'), {'c': 'd'})
    assert isinstance(key, _HashedKey)
    assert tuple(key[:2]) == ('a', 'b')


def test_make_cache_key_fast_type_single():
    # Single fast-type positional arg returns the value directly.
    assert make_cache_key(('abc',), {}) == 'abc'
    assert make_cache_key((42,), {}) == 42
    assert make_cache_key((None,), {}) is None


def test_make_cache_key_no_fasttype_returns_hashedkey():
    # Use a hashable (but non-fast) value so it can be wrapped in _HashedKey.
    key = make_cache_key((frozenset(),), {}, fasttypes=frozenset())
    assert isinstance(key, _HashedKey)
    assert key == [frozenset()]


def test_make_cache_key_typed():
    key_int = make_cache_key((3,), {}, typed=True)
    key_float = make_cache_key((3.0,), {}, typed=True)
    assert key_int != key_float
    # Without typed, single arg case may return raw for fast types
    assert make_cache_key((3,), {}) == 3


def test_make_cache_key_typed_with_kwargs():
    # Exercises the branch adding types of kwargs values (lines 430-431).
    k1 = make_cache_key((), {'a': 1}, typed=True)
    k2 = make_cache_key((), {'a': 1.0}, typed=True)
    assert k1 != k2
    assert int in tuple(k1)
    assert float in tuple(k2)


def test_make_cache_key_ordering_of_kwargs():
    k1 = make_cache_key((), {'a': 1, 'b': 2})
    k2 = make_cache_key((), {'b': 2, 'a': 1})
    assert tuple(k1) == tuple(k2)


def test_hashed_key_hash_cached():
    hk = _HashedKey([1, 2, 3])
    assert hash(hk) == hash((1, 2, 3))
    assert hk.hash_value == hash((1, 2, 3))
    assert 'HashedKey' in repr(hk)


# ---------------------------------------------------------------- CachedFunction / cached

def test_cached_function_basic():
    cache = {}
    calls = []

    @cached(cache)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]
    assert len(cache) == 1


def test_cached_function_scoped():
    # scoped is accepted but make_cache_key does not include the function,
    # so two different functions with the same args share a cache entry.
    cache = {}
    calls = {'a': 0, 'b': 0}

    @cached(cache, scoped=True)
    def a(x):
        calls['a'] += 1
        return x

    @cached(cache, scoped=True)
    def b(x):
        calls['b'] += 1
        return x

    a(1)
    b(1)
    # The default key function does not scope by function; b reads a's entry.
    assert calls == {'a': 1, 'b': 0}
    assert len(cache) == 1


def test_cached_function_not_scoped():
    cache = {}

    @cached(cache, scoped=False)
    def a(x):
        return x

    @cached(cache, scoped=False)
    def b(x):
        return x

    a(1)
    # b(1) may read a's cached result.
    assert b(1) == 1
    assert len(cache) == 1


def test_cached_function_typed():
    cache = {}

    @cached(cache, typed=True)
    def f(x):
        return type(x).__name__

    assert f(1) == 'int'
    assert f(1.0) == 'float'
    assert len(cache) == 2


def test_cached_function_custom_key():
    cache = {}

    @cached(cache, key=lambda args, kwargs, typed: ('custom', args[0]))
    def f(x):
        return x

    f(5)
    assert ('custom', 5) in cache


def test_cached_function_callable_cache():
    caches = []

    def make_cache():
        c = {}
        caches.append(c)
        return c

    @cached(make_cache)
    def f(x):
        return x

    f(1)
    assert len(caches) == 1


def test_cached_invalid_cache():
    with pytest.raises(TypeError):
        CachedFunction(lambda x: x, object())


def test_cached_function_repr():
    def f(x):
        return x

    cf = CachedFunction(f, {})
    assert 'CachedFunction' in repr(cf)
    assert 'scoped' not in repr(cf)  # only shown when non-default
    cf2 = CachedFunction(f, {}, typed=True)
    assert 'typed=True' in repr(cf2)
    cf3 = CachedFunction(f, {}, scoped=False)
    assert 'scoped=False' in repr(cf3)


# ---------------------------------------------------------------- CachedMethod / cachedmethod

def test_cachedmethod_str_attr():
    class A:
        def __init__(self):
            self.cache = {}
            self.calls = 0

        @cachedmethod('cache')
        def double(self, x):
            self.calls += 1
            return x * 2

    a = A()
    assert a.double(2) == 4
    assert a.double(2) == 4
    assert a.calls == 1
    assert len(a.cache) == 1


def test_cachedmethod_dict_cache():
    cache = {}

    class A:
        @cachedmethod(cache)
        def triple(self, x):
            return x * 3

    a = A()
    assert a.triple(3) == 9
    assert a.triple(3) == 9
    assert len(cache) == 1


def test_cachedmethod_callable_cache():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod(lambda self: self.cache)
        def quad(self, x):
            return x * 4

    a = A()
    assert a.quad(2) == 8
    assert len(a.cache) == 1


def test_cachedmethod_invalid_cache():
    with pytest.raises(TypeError):
        CachedMethod(lambda self: None, object())


def test_cachedmethod_scoped_distinguishes_objects():
    class A:
        def __init__(self, cache):
            self.cache = cache

        @cachedmethod('cache')
        def val(self, x):
            return x

    shared = {}
    a1 = A(shared)
    a2 = A(shared)
    a1.val(1)
    a2.val(1)
    # scoped=True means different objects produce different keys
    assert len(shared) == 2


def test_cachedmethod_unscoped_shares_cache():
    class A:
        def __init__(self, cache):
            self.cache = cache

        @cachedmethod('cache', scoped=False)
        def val(self, x):
            return x

    shared = {}
    a1 = A(shared)
    a2 = A(shared)
    assert a1.val(1) == 1
    assert a2.val(1) == 1
    assert len(shared) == 1


def test_cachedmethod_typed():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod('cache', typed=True)
        def m(self, x):
            return type(x).__name__

    a = A()
    assert a.m(1) == 'int'
    assert a.m(1.0) == 'float'
    assert len(a.cache) == 2


def test_cachedmethod_custom_key():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod('cache', key=lambda args, kwargs, typed: ('k', args[-1]))
        def m(self, x):
            return x

    a = A()
    a.m(7)
    assert ('k', 7) in a.cache


def test_cachedmethod_repr_unbound_and_bound():
    class A:
        @cachedmethod('cache')
        def m(self, x):
            return x

    desc = A.__dict__['m']
    assert 'CachedMethod' in repr(desc)
    a = A()
    a.cache = {}
    bound = a.m
    assert 'bound_to' in repr(bound)


def test_cachedmethod_abstractmethod_flag():
    def f(self):
        return 1

    f.__isabstractmethod__ = True
    cm = CachedMethod(f, {})
    assert cm.__isabstractmethod__ is True


def test_cachedmethod_class_access_returns_descriptor():
    class A:
        @cachedmethod('cache')
        def m(self, x):
            return x

    assert isinstance(A.m, CachedMethod)


def test_cachedmethod_bind_obj_misses_calls_func_with_bound():
    # Exercises the branch in CachedMethod.__call__ where bound_to is not None
    # on a cache miss (lines 520-522).
    class A:
        def __init__(self):
            self.cache = {}
            self.calls = 0

        @cachedmethod('cache')
        def inc(self, x):
            self.calls += 1
            return x + 1

    a = A()
    bound = a.inc
    assert bound(10) == 11
    assert a.calls == 1
    assert bound(10) == 11
    assert a.calls == 1
    assert len(a.cache) == 1


# ---------------------------------------------------------------- cachedproperty

def test_cachedproperty():
    calls = []

    class A:
        @cachedproperty
        def val(self):
            calls.append(1)
            return 42

    a = A()
    assert a.val == 42
    assert a.val == 42
    assert calls == [1]
    assert a.__dict__['val'] == 42
    delattr(a, 'val')
    assert a.val == 42
    assert calls == [1, 1]


def test_cachedproperty_class_access():
    class A:
        @cachedproperty
        def val(self):
            return 1

    assert isinstance(A.__dict__['val'], cachedproperty)
    assert isinstance(A.val, cachedproperty)


def test_cachedproperty_doc_and_repr():
    class A:
        @cachedproperty
        def val(self):
            "docstring"
            return 1

    cp = A.__dict__['val']
    assert cp.__doc__ == 'docstring'
    assert 'cachedproperty' in repr(cp)
    assert 'val' in repr(cp)


def test_cachedproperty_abstractmethod_flag():
    def f(self):
        return 1

    f.__isabstractmethod__ = True
    cp = cachedproperty(f)
    assert cp.__isabstractmethod__ is True


# ---------------------------------------------------------------- ThresholdCounter

def test_threshold_counter_invalid_threshold():
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=0)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=1)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=-0.5)


def test_threshold_counter_basic():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    assert tc[1] == 1
    assert tc.get(1) == 1
    assert tc.get(2) == 0
    assert 1 in tc
    assert 2 not in tc
    assert len(tc) == 1
    assert tc.threshold == 0.1


def test_threshold_counter_update_list():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([2] * 10)
    assert tc.total == 10
    assert tc.get(1) == 0


def test_threshold_counter_update_mapping():
    tc = ThresholdCounter(threshold=0.1)
    tc.update({'a': 3, 'b': 2})
    assert tc.total == 5
    assert tc['a'] == 3
    assert tc['b'] == 2


def test_threshold_counter_update_none_with_kwargs():
    tc = ThresholdCounter(threshold=0.1)
    tc.update(None, k=2)
    assert tc.total == 2
    assert tc['k'] == 2


def test_threshold_counter_compaction():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    tc.update([2] * 10)
    # 1 should be evicted after compaction
    assert tc.get(1) == 0
    assert 5 not in tc
    tc.add(5)
    assert 5 in tc
    assert len(list(tc.elements())) == sum(tc.values())


def test_threshold_counter_most_common():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([1, 1, 1, 2, 2, 3])
    assert tc.most_common() == [(1, 3), (2, 2), (3, 1)]
    assert tc.most_common(2) == [(1, 3), (2, 2)]
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []
    assert tc.most_common(10) == [(1, 3), (2, 2), (3, 1)]


def test_threshold_counter_counts():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([1, 1, 1])
    assert tc.get_common_count() == 3
    assert tc.get_uncommon_count() == 0
    assert tc.get_commonality() == 1.0


def test_threshold_counter_iterators():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([1, 1, 2])
    assert list(tc.iterkeys()) == [1, 2]
    assert tc.keys() == [1, 2]
    assert list(tc.itervalues()) == [2, 1]
    assert tc.values() == [2, 1]
    assert list(tc.iteritems()) == [(1, 2), (2, 1)]
    assert tc.items() == [(1, 2), (2, 1)]
    elements = list(tc.elements())
    assert sorted(elements) == [1, 1, 2]


# ---------------------------------------------------------------- MinIDMap

class _WeakObj:
    """Weak-referenceable object used for MinIDMap tests.

    Note: instances need to support weak references. A plain class
    without __slots__ is weak-referenceable, but defining __slots__
    including __weakref__ is required for slotted classes.
    """
    __slots__ = ('__weakref__',)


@pytest.fixture
def minidmap_objects():
    return [_WeakObj() for _ in range(5)]


def test_minidmap_get_and_unique_ids(minidmap_objects):
    m = MinIDMap()
    ids = [m.get(o) for o in minidmap_objects]
    assert sorted(ids) == [0, 1, 2, 3, 4]
    # Same object yields same ID
    assert m.get(minidmap_objects[0]) == ids[0]
    assert minidmap_objects[0] in m
    assert len(m) == 5


def test_minidmap_drop_and_reuse(minidmap_objects):
    m = MinIDMap()
    ids = [m.get(o) for o in minidmap_objects]
    m.drop(minidmap_objects[1])
    assert minidmap_objects[1] not in m
    assert len(m) == 4
    # smallest freed id is reused
    new_obj = _WeakObj()
    assert m.get(new_obj) == ids[1]


def test_minidmap_iter(minidmap_objects):
    m = MinIDMap()
    for o in minidmap_objects:
        m.get(o)
    assert set(iter(m)) == set(minidmap_objects)
    items = dict(m.iteritems())
    assert set(items.keys()) == set(minidmap_objects)
    assert set(items.values()) == set(range(5))


def test_minidmap_drop_unknown_raises(minidmap_objects):
    m = MinIDMap()
    with pytest.raises(KeyError):
        m.drop(_WeakObj())
