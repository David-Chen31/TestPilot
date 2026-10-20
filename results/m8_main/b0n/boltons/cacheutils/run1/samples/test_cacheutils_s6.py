import gc
import sys
import weakref
import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    DEFAULT_MAX_SIZE,
    make_cache_key,
    CachedFunction,
    CachedMethod,
    cached,
    cachedmethod,
    cachedproperty,
    ThresholdCounter,
    MinIDMap,
    _MISSING,
    _KWARG_MARK,
    _make_cache_key,
)


# ---------------------------------------------------------------------------
# LRI
# ---------------------------------------------------------------------------

def test_lri_init_defaults():
    c = LRI()
    assert c.max_size == DEFAULT_MAX_SIZE
    assert c.on_miss is None
    assert c.hit_count == 0
    assert c.miss_count == 0
    assert c.soft_miss_count == 0
    assert len(c) == 0


def test_lri_init_invalid_max_size():
    with pytest.raises(ValueError):
        LRI(max_size=0)
    with pytest.raises(ValueError):
        LRI(max_size=-1)


def test_lri_init_invalid_on_miss():
    with pytest.raises(TypeError):
        LRI(on_miss=5)


def test_lri_init_values_dict():
    c = LRI(values={'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}


def test_lri_init_values_list_of_pairs():
    c = LRI(values=[('a', 1), ('b', 2)])
    assert dict(c) == {'a': 1, 'b': 2}


def test_lri_basic_set_get():
    c = LRI()
    c['a'] = 1
    assert c['a'] == 1
    assert c.hit_count == 1
    assert c.miss_count == 0


def test_lri_miss_raises():
    c = LRI()
    with pytest.raises(KeyError):
        c['missing']
    assert c.miss_count == 1
    assert c.hit_count == 0


def test_lri_on_miss():
    c = LRI(on_miss=lambda k: k * 2)
    assert c[3] == 6
    assert c[3] == 6
    assert c.miss_count == 1
    assert c.hit_count == 1
    assert dict(c) == {3: 6}


def test_lri_get_default():
    c = LRI()
    assert c.get('nope') is None
    assert c.soft_miss_count == 1
    assert c.get('nope', 42) == 42
    assert c.soft_miss_count == 2


def test_lri_get_hit():
    c = LRI()
    c['a'] = 1
    assert c.get('a') == 1
    assert c.hit_count == 1
    assert c.soft_miss_count == 0


def test_lri_eviction_lri_order():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    # accessing a does not change insertion order for LRI
    assert c['a'] == 'A'
    c['c'] = 'C'
    assert 'a' not in c
    assert c['b'] == 'B'
    assert c['c'] == 'C'
    assert len(c) == 2


def test_lri_doctest_like():
    cap_cache = LRI(max_size=2)
    cap_cache['a'], cap_cache['b'] = 'A', 'B'
    assert dict(cap_cache) == {'a': 'A', 'b': 'B'}
    assert [cap_cache['b'] for i in range(3)][0] == 'B'
    cap_cache['c'] = 'C'
    assert cap_cache.get('a') is None
    assert (cap_cache.hit_count, cap_cache.miss_count,
            cap_cache.soft_miss_count) == (3, 1, 1)


def test_lri_update_existing_key_keeps_position():
    c = LRI(max_size=2)
    c['a'] = 1
    c['b'] = 2
    c['a'] = 3
    assert c['a'] == 3
    # a was updated, still at front (most recently inserted/updated)
    c['c'] = 4
    assert 'b' not in c
    assert dict(c) == {'a': 3, 'c': 4}


def test_lri_delitem():
    c = LRI()
    c['a'] = 1
    del c['a']
    assert 'a' not in c
    assert c._link_lookup == {}


def test_lri_pop():
    c = LRI()
    c['a'] = 1
    assert c.pop('a') == 1
    assert 'a' not in c
    assert c.pop('missing', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        c.pop('missing')


def test_lri_pop_bypasses_stats():
    c = LRI()
    c['a'] = 1
    c.pop('a')
    c.pop('b', None)
    assert c.hit_count == 0
    assert c.miss_count == 0
    assert c.soft_miss_count == 0


def test_lri_popitem():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    item = c.popitem()
    assert item == ('b', 2)
    assert 'b' not in c
    assert c._link_lookup != {}


def test_lri_clear():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    c.clear()
    assert len(c) == 0
    assert c._link_lookup == {}


def test_lri_copy():
    c = LRI(max_size=5)
    c['a'] = 1
    c2 = c.copy()
    assert isinstance(c2, LRI)
    assert c2.max_size == 5
    assert dict(c2) == {'a': 1}
    # Independent copy
    c2['b'] = 2
    assert 'b' not in c


def test_lri_setdefault():
    c = LRI()
    assert c.setdefault('a', 10) == 10
    assert c.soft_miss_count == 1
    assert c.setdefault('a', 20) == 10
    assert c['a'] == 10


def test_lri_update_kwargs():
    c = LRI()
    c.update({'a': 1}, b=2)
    assert dict(c) == {'a': 1, 'b': 2}


def test_lri_eq():
    c1 = LRI(values={'a': 1})
    c2 = LRI(values={'a': 1})
    assert c1 == c2
    assert not (c1 != c2)
    assert c1 == {'a': 1}
    assert c1 != {'a': 2}
    assert c1 != {'a': 1, 'b': 2}


def test_lri_repr():
    c = LRI(max_size=3, on_miss=None, values={'a': 1})
    r = repr(c)
    assert r.startswith('LRI(')
    assert 'max_size=3' in r
    assert 'on_miss=None' in r
    assert "'a': 1" in r


def test_lri_get_flattened_ll():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    flat = c._get_flattened_ll()
    # anchor is at end, with _MISSING key/value
    keys = [k for (k, v) in flat if k is not _MISSING]
    assert keys == ['a', 'b']


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

def test_lru_basic():
    c = LRU()
    c['a'] = 1
    c['b'] = 2
    assert c['a'] == 1
    assert c.hit_count == 1


def test_lru_doctest_like():
    cap_cache = LRU(max_size=2)
    cap_cache['a'], cap_cache['b'] = 'A', 'B'
    assert dict(cap_cache) == {'a': 'A', 'b': 'B'}
    assert [cap_cache['b'] for i in range(3)][0] == 'B'
    cap_cache['c'] = 'C'
    assert cap_cache.get('a') is None
    assert (cap_cache.hit_count, cap_cache.miss_count,
            cap_cache.soft_miss_count) == (3, 1, 1)


def test_lru_eviction_lru_order():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    # Access a, making b the least recently used
    assert c['a'] == 'A'
    c['c'] = 'C'
    assert 'b' not in c
    assert c['a'] == 'A'
    assert c['c'] == 'C'


def test_lru_on_miss():
    c = LRU(on_miss=lambda k: k + 1)
    assert c[5] == 6
    assert c[5] == 6
    assert c.miss_count == 1
    assert c.hit_count == 1


def test_lru_miss_raises():
    c = LRU()
    with pytest.raises(KeyError):
        c['x']
    assert c.miss_count == 1


def test_lru_is_lri_subclass():
    assert issubclass(LRU, LRI)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_basic():
    key = make_cache_key(('a', 'b'), {'c': 'd'})
    assert tuple(key) == ('a', 'b', _KWARG_MARK, ('c', 'd'))


def test_make_cache_key_no_kwargs():
    key = make_cache_key(('a', 'b'), {})
    assert tuple(key) == ('a', 'b')


def test_make_cache_key_single_fasttype():
    assert make_cache_key((5,), {}) == 5
    assert make_cache_key(('abc',), {}) == 'abc'
    assert make_cache_key((None,), {}) == None  # noqa: E711


def test_make_cache_key_single_slow_type():
    key = make_cache_key(([1, 2],), {})
    # list is not hashable but make_cache_key wraps it into _HashedKey
    # actually _HashedKey tries to hash it... but let's use a hashable slow type
    # Skip this case since unhashable would raise; test with tuple instead.
    key = make_cache_key(((1, 2),), {})
    assert list(key) == [(1, 2)]
    assert hash(key) == hash(((1, 2),))


def test_make_cache_key_typed():
    k1 = make_cache_key((3,), {}, typed=True)
    k2 = make_cache_key((3.0,), {}, typed=True)
    assert list(k1) == [3, int]
    assert list(k2) == [3.0, float]
    assert k1 != k2


def test_make_cache_key_untyped_equates_int_float():
    k1 = make_cache_key((3,), {})
    k2 = make_cache_key((3.0), {}) if False else make_cache_key((3.0,), {})
    # When untyped and single fasttype, returns the raw value
    assert k1 == 3
    assert k2 == 3.0
    assert k1 == k2


def test_make_cache_key_hash_dedup():
    key = make_cache_key(('a',), {'b': 1})
    h = hash(key)
    # _HashedKey caches hash; calling again returns same value
    assert hash(key) == h


def test_make_cache_key_typed_kwargs():
    k1 = make_cache_key((), {'x': 1}, typed=True)
    k2 = make_cache_key((), {'x': 1.0}, typed=True)

    def strip_types(kk):
        return [v for v in kk if not isinstance(v, type)]
    assert strip_types(k1) == strip_types(k2)
    assert k1 != k2


def test_make_cache_key_alias():
    assert _make_cache_key is make_cache_key


def test_hashed_key_repr():
    key = make_cache_key(('a', 'b'), {'x': 1})
    assert repr(key).startswith('_HashedKey(')


# ---------------------------------------------------------------------------
# CachedFunction / cached
# ---------------------------------------------------------------------------

def test_cached_function_basic():
    cache = {}
    call_count = [0]

    @cached(cache)
    def f(x):
        call_count[0] += 1
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert call_count[0] == 1
    assert len(cache) == 1


def test_cached_repr():
    def f(x):
        return x

    cache = {}
    cf = CachedFunction(f, cache)
    r = repr(cf)
    assert 'CachedFunction' in r
    assert 'func=' in r


def test_cached_function_callable_cache():
    store = {}

    def get_cache():
        return store

    @cached(get_cache)
    def f(x):
        return x + 1
    assert f(1) == 2
    assert f(1) == 2
    assert len(store) == 1


def test_cached_function_invalid_cache():
    with pytest.raises(TypeError):
        @cached(5)
        def f(x):
            return x


def test_cached_different_functions_dont_collide():
    cache = {}

    @cached(cache)
    def f(x):
        return ('f', x)

    @cached(cache)
    def g(x):
        return ('g', x)

    assert f(1) == ('f', 1)
    assert g(1) == ('g', 1)


def test_cached_scoped_false_shares():
    cache = {}

    @cached(cache, scoped=False)
    def f(x):
        return ('f', x)

    @cached(cache, scoped=False)
    def g(x):
        return ('g', x)

    # Since scoped=False and key func doesn't include the func,
    # g should read f's cached value for the same args
    assert f(1) == ('f', 1)
    assert g(1) == ('f', 1)


def test_cached_typed():
    cache = {}

    @cached(cache, typed=True)
    def f(x):
        return x

    f(3)
    f(3.0)
    assert len(cache) == 2


def test_cached_custom_key():
    cache = {}
    seen = []

    def keyfunc(args, kwargs, typed=False):
        return ('custom',) + tuple(args)

    @cached(cache, key=keyfunc)
    def f(x):
        return x * 10

    assert f(2) == 20
    assert f(2) == 20
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# CachedMethod / cachedmethod
# ---------------------------------------------------------------------------

def test_cachedmethod_basic():
    class Lowerer:
        def __init__(self):
            self.cache = LRI()

        @cachedmethod('cache')
        def lower(self, text):
            return text.lower()

    obj = Lowerer()
    assert obj.lower('HELLO') == 'hello'
    assert len(obj.cache) == 1


def test_cachedmethod_callable_cache():
    class A:
        def __init__(self):
            self.cache = {}

        def get_cache(self):
            return self.cache

        @cachedmethod(lambda self: self.cache)
        def double(self, x):
            return x * 2

    a = A()
    assert a.double(5) == 10
    assert a.double(5) == 10
    assert len(a.cache) == 1


def test_cachedmethod_direct_cache_object():
    cache = {}

    class A:
        @cachedmethod(cache)
        def double(self, x):
            return x * 2

    a = A()
    assert a.double(4) == 8
    assert len(cache) == 1


def test_cachedmethod_invalid_cache():
    with pytest.raises(TypeError):
        cachedmethod(5)


def test_cachedmethod_different_instances_dont_collide():
    cache = {}

    class A:
        @cachedmethod(cache)
        def val(self, x):
            return id(self)

    a = A()
    b = A()
    assert a.val(1) == b.val(1) is False or a.val(1) != b.val(1)


def test_cachedmethod_repr_unbound():
    class A:
        @cachedmethod('cache')
        def method(self, x):
            return x

    cm = A.__dict__['method']
    r = repr(cm)
    assert 'CachedMethod' in r
    assert 'bound_to' not in r


def test_cachedmethod_repr_bound():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod('cache')
        def method(self, x):
            return x

    a = A()
    bound = a.method
    r = repr(bound)
    assert 'CachedMethod' in r
    assert 'bound_to' in r


def test_cachedmethod_isabstractmethod():
    from abc import abstractmethod

    class A:
        @cachedmethod('cache')
        @abstractmethod
        def meth(self):
            pass

    assert A.__dict__['meth'].__isabstractmethod__ is True


# ---------------------------------------------------------------------------
# cachedproperty
# ---------------------------------------------------------------------------

def test_cachedproperty_cached():
    call_count = [0]

    class A:
        @cachedproperty
        def value(self):
            call_count[0] += 1
            return 42

    a = A()
    assert a.value == 42
    assert a.value == 42
    assert call_count[0] == 1
    assert a.__dict__['value'] == 42


def test_cachedproperty_clearable():
    class A:
        @cachedproperty
        def value(self):
            return object()

    a = A()
    v1 = a.value
    del a.value
    v2 = a.value
    assert v1 is not v2


def test_cachedproperty_class_access():
    class A:
        @cachedproperty
        def value(self):
            """docstring"""
            return 1

    assert isinstance(A.__dict__['value'], cachedproperty)
    assert A.value.__doc__ == 'docstring'


def test_cachedproperty_repr():
    class A:
        @cachedproperty
        def value(self):
            return 1

    r = repr(A.__dict__['value'])
    assert 'cachedproperty' in r
    assert 'func=' in r


# ---------------------------------------------------------------------------
# ThresholdCounter
# ---------------------------------------------------------------------------

def test_threshold_counter_init_invalid():
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=0)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=1)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=-0.5)


def test_threshold_counter_threshold_prop():
    tc = ThresholdCounter(threshold=0.1)
    assert tc.threshold == 0.1


def test_threshold_counter_doctest_like():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    tc.update([2] * 10)
    assert tc.get(1) == 0
    tc.add(5)
    assert 5 in tc
    assert len(list(tc.elements())) == 11


def test_threshold_counter_len_contains():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')
    assert 'a' in tc
    assert 'b' not in tc
    assert len(tc) == 1


def test_threshold_counter_getitem():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')
    assert tc['a'] == 1
    with pytest.raises(KeyError):
        tc['b']


def test_threshold_counter_get_default():
    tc = ThresholdCounter(threshold=0.5)
    assert tc.get('missing') == 0
    assert tc.get('missing', -1) == -1


def test_threshold_counter_keys_values_items():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')
    tc.add('b')
    assert sorted(tc.keys()) == ['a', 'b']
    assert sorted(tc.values()) == [1, 1]
    assert sorted(tc.items()) == [('a', 1), ('b', 1)]


def test_threshold_counter_iterkeys_itervalues_iteritems():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')
    tc.add('b')
    assert sorted(tc.iterkeys()) == ['a', 'b']
    assert sorted(tc.itervalues()) == [1, 1]
    assert sorted(tc.iteritems()) == [('a', 1), ('b', 1)]


def test_threshold_counter_most_common():
    tc = ThresholdCounter(threshold=0.5)
    for _ in range(3):
        tc.add('a')
    tc.add('b')
    assert tc.most_common() == [('a', 3), ('b', 1)]
    assert tc.most_common(1) == [('a', 3)]
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []


def test_threshold_counter_common_uncommon():
    tc = ThresholdCounter(threshold=0.5)
    for _ in range(5):
        tc.add('a')
    for _ in range(5):
        tc.add('b')
    common = tc.get_common_count()
    uncommon = tc.get_uncommon_count()
    assert common + uncommon == tc.total
    assert 0.0 <= tc.get_commonality() <= 1.0


def test_threshold_counter_update_mapping():
    tc = ThresholdCounter(threshold=0.5)
    tc.update({'a': 3, 'b': 2})
    assert tc['a'] == 3
    assert tc['b'] == 2
    assert tc.total == 5


def test_threshold_counter_update_kwargs():
    tc = ThresholdCounter(threshold=0.5)
    tc.update(None, a=2, b=1)
    assert tc['a'] == 2
    assert tc['b'] == 1


def test_threshold_counter_elements():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')
    tc.add('b')
    tc.add('b')
    elems = sorted(tc.elements())
    assert elems == ['a', 'b', 'b']


# ---------------------------------------------------------------------------
# MinIDMap
# ---------------------------------------------------------------------------

class _RefObject:
    pass


def test_minidmap_basic():
    m = MinIDMap()
    o1 = _RefObject()
    o2 = _RefObject()
    id1 = m.get(o1)
    id2 = m.get(o2)
    assert id1 == 0
    assert id2 == 1
    assert m.get(o1) == 0
    assert len(m) == 2
    assert o1 in m
    assert o2 in m


def test_minidmap_free_reuse():
    m = MinIDMap()
    o1 = _RefObject()
    o2 = _RefObject()
    id1 = m.get(o1)
    id2 = m.get(o2)
    assert id1 == 0
    assert id2 == 1
    m.drop(o1)
    assert o1 not in m
    o3 = _RefObject()
    assert m.get(o3) == 0
    assert m.get(o2) == 1


def test_minidmap_iteritems():
    m = MinIDMap()
    o1 = _RefObject()
    o2 = _RefObject()
    m.get(o1)
    m.get(o2)
    items = dict(m.iteritems())
    assert items[o1] == 0
    assert items[o2] == 1


def test_minidmap_iter():
    m = MinIDMap()
    o1 = _RefObject()
    m.get(o1)
    assert list(iter(m)) == [o1]


def test_minidmap_weakref_cleanup():
    m = MinIDMap()
    o1 = _RefObject()
    o2 = _RefObject()
    id1 = m.get(o1)
    id2 = m.get(o2)
    assert id1 == 0
    assert id2 == 1
    del o1
    gc.collect()
    # The freed id should be reusable
    assert len(m) == 1
    o3 = _RefObject()
    assert m.get(o3) == id1
