import gc
import sys
import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    make_cache_key,
    _HashedKey,
    _make_cache_key,
    CachedFunction,
    CachedMethod,
    cached,
    cachedmethod,
    cachedproperty,
    ThresholdCounter,
    MinIDMap,
)


# ---------------------------------------------------------------------------
# LRI
# ---------------------------------------------------------------------------

def test_lri_basic():
    c = LRI(max_size=2)
    c['a'], c['b'] = 'A', 'B'
    assert dict(c) == {'a': 'A', 'b': 'B'}
    assert c['b'] == 'B'
    c['c'] = 'C'
    assert c.get('a') is None
    assert dict(c) == {'b': 'B', 'c': 'C'}
    assert c.hit_count == 1
    assert c.miss_count == 0
    assert c.soft_miss_count == 1


def test_lri_least_recently_inserted():
    c = LRI(max_size=2)
    c['a'] = 1
    c['b'] = 2
    # accessing 'a' should NOT update recency in LRI
    assert c['a'] == 1
    c['c'] = 3
    # 'a' was inserted first, so evicted
    assert 'a' not in c
    assert c['b'] == 2
    assert c['c'] == 3


def test_lri_invalid_max_size():
    with pytest.raises(ValueError):
        LRI(max_size=0)
    with pytest.raises(ValueError):
        LRI(max_size=-1)


def test_lri_invalid_on_miss():
    with pytest.raises(TypeError):
        LRI(on_miss=42)


def test_lri_on_miss():
    calls = []

    def on_miss(key):
        calls.append(key)
        return key * 2

    c = LRI(on_miss=on_miss)
    assert c['x'] == 'xx'
    assert calls == ['x']
    assert c['x'] == 'xx'
    assert calls == ['x']
    assert c.hit_count == 1
    assert c.miss_count == 1


def test_lri_getitem_raises_without_on_miss():
    c = LRI()
    with pytest.raises(KeyError):
        c['missing']
    assert c.miss_count == 1


def test_lri_soft_miss():
    c = LRI()
    assert c.get('nope') is None
    assert c.get('nope', 'dflt') == 'dflt'
    assert c.soft_miss_count == 2


def test_lri_setdefault():
    c = LRI()
    assert c.setdefault('a', 1) == 1
    assert c.setdefault('a', 2) == 1
    assert c['a'] == 1
    assert c.soft_miss_count == 1


def test_lri_delitem():
    c = LRI()
    c['a'] = 1
    del c['a']
    assert 'a' not in c
    with pytest.raises(KeyError):
        del c['a']


def test_lri_pop():
    c = LRI()
    c['a'] = 1
    assert c.pop('a') == 1
    assert 'a' not in c
    with pytest.raises(KeyError):
        c.pop('a')
    assert c.pop('a', 'd') == 'd'


def test_lri_popitem():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    item = c.popitem()
    assert item in (('a', 1), ('b', 2))
    assert len(c) == 1
    with pytest.raises(KeyError):
        c.clear()
        c.popitem()


def test_lri_clear():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    c.clear()
    assert len(c) == 0
    assert dict(c) == {}
    c['x'] = 10
    assert c['x'] == 10


def test_lri_copy():
    c = LRI(max_size=5)
    c['a'] = 1
    copy = c.copy()
    assert copy is not c
    assert dict(copy) == {'a': 1}
    assert copy.max_size == 5
    copy['b'] = 2
    assert 'b' not in c


def test_lri_update():
    c = LRI()
    c.update({'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}
    c.update([('c', 3)])
    assert c['c'] == 3
    c.update(d=4)
    assert c['d'] == 4


def test_lri_update_self():
    c = LRI()
    c['a'] = 1
    c.update(c)
    assert dict(c) == {'a': 1}


def test_lri_eq():
    c = LRI()
    c['a'] = 1
    other = LRI()
    other['a'] = 1
    assert c == other
    assert c == {'a': 1}
    assert {'a': 1} == c
    other['b'] = 2
    assert c != other
    assert not (c == other)
    assert c == c


def test_lri_repr():
    c = LRI(max_size=3)
    c['a'] = 1
    r = repr(c)
    assert 'LRI' in r
    assert 'max_size=3' in r
    assert 'on_miss=None' in r


def test_lri_init_values():
    c = LRI(max_size=5, values={'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

def test_lru_basic():
    c = LRU(max_size=2)
    c['a'], c['b'] = 'A', 'B'
    assert dict(c) == {'a': 'A', 'b': 'B'}
    assert c['b'] == 'B'
    c['c'] = 'C'
    assert c.get('a') is None
    assert dict(c) == {'b': 'B', 'c': 'C'}
    assert c.hit_count == 1
    assert c.miss_count == 0
    assert c.soft_miss_count == 1


def test_lru_least_recently_used():
    c = LRU(max_size=2)
    c['a'] = 1
    c['b'] = 2
    # accessing 'a' should update recency in LRU
    assert c['a'] == 1
    c['c'] = 3
    # 'b' is now least recently used
    assert 'b' not in c
    assert c['a'] == 1
    assert c['c'] == 3


def test_lru_on_miss():
    c = LRU(on_miss=lambda k: k * 2)
    assert c['x'] == 'xx'
    assert c['x'] == 'xx'
    assert c.miss_count == 1
    assert c.hit_count == 1


def test_lru_getitem_raises():
    c = LRU()
    with pytest.raises(KeyError):
        c['missing']
    assert c.miss_count == 1


def test_lru_is_lri_subclass():
    assert issubclass(LRU, LRI)
    c = LRU()
    assert isinstance(c, dict)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_basic():
    key = make_cache_key(('a', 'b'), {'c': 'd'})
    assert isinstance(key, _HashedKey)
    assert list(key[:-1]) == ['a', 'b']
    # last but one should be the kwarg mark
    assert key[-2] is not None


def test_make_cache_key_single_fast_type():
    assert make_cache_key((3,), {}) == 3
    assert make_cache_key(('s',), {}) == 's'
    assert make_cache_key((None,), {}) is None
    assert make_cache_key((frozenset([1, 2]),), {}) == frozenset([1, 2])


def test_make_cache_key_single_non_fast_type():
    key = make_cache_key(([1, 2],), {})
    assert isinstance(key, _HashedKey)
    assert list(key) == [[1, 2]]


def test_make_cache_key_typed():
    k1 = make_cache_key((3,), {}, typed=True)
    k2 = make_cache_key((3.0,), {}, typed=True)
    assert k1 != k2
    assert isinstance(k1, _HashedKey)
    assert isinstance(k2, _HashedKey)


def test_make_cache_key_typed_kwargs():
    k1 = make_cache_key((), {'x': 3}, typed=True)
    k2 = make_cache_key((), {'x': 3.0}, typed=True)
    assert k1 != k2


def test_make_cache_key_repr():
    key = make_cache_key(('a', 'b'), {'c': 'd'})
    r = repr(key)
    assert r.startswith('_HashedKey(')


def test_make_cache_key_hash_stable():
    key = make_cache_key(('a', 'b'), {'c': 'd'})
    assert hash(key) == hash(tuple(key))


def test_make_cache_key_alias():
    assert _make_cache_key is make_cache_key


def test_hashed_key():
    hk = _HashedKey([1, 2, 3])
    assert hk == [1, 2, 3]
    assert hk.hash_value == hash((1, 2, 3))
    assert hash(hk) == hk.hash_value


# ---------------------------------------------------------------------------
# CachedFunction / cached
# ---------------------------------------------------------------------------

def test_cached_function_basic():
    cache = {}
    calls = []

    @cached(cache)
    def add(a, b):
        calls.append((a, b))
        return a + b

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert len(calls) == 1
    assert len(cache) == 1


def test_cached_invalid_cache():
    with pytest.raises(TypeError):
        CachedFunction(lambda: None, 42)


def test_cached_callable_cache():
    cache = {}

    def get_cache():
        return cache

    calls = []

    @cached(get_cache)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert len(calls) == 1


def test_cached_scoped():
    cache = {}

    @cached(cache)
    def f(x):
        return x

    @cached(cache)
    def g(x):
        return x * 10

    assert f(1) == 1
    assert g(1) == 10
    assert len(cache) == 2


def test_cached_unscoped():
    cache = {}

    @cached(cache, scoped=False)
    def f(x):
        return x

    @cached(cache, scoped=False)
    def g(x):
        return x * 10

    assert f(1) == 1
    # g reads f's cache value because scoped is disabled
    assert g(1) == 1
    assert len(cache) == 1


def test_cached_typed():
    cache = {}

    @cached(cache, typed=True)
    def f(x):
        return x

    assert f(3) == 3
    assert f(3.0) == 3.0
    assert len(cache) == 2


def test_cached_untyped_collides():
    cache = {}

    @cached(cache)
    def f(x):
        return x

    assert f(3) == 3
    assert f(3.0) == 3
    assert len(cache) == 1


def test_cached_custom_key():
    cache = {}

    def key_func(args, kwargs, typed=False):
        return args[0]

    @cached(cache, key=key_func)
    def f(x):
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert cache == {3: 6}


def test_cached_function_repr():
    @cached({})
    def f(x):
        return x

    r = repr(f)
    assert 'CachedFunction' in r

    @cached({}, scoped=False)
    def g(x):
        return x

    r = repr(g)
    assert 'scoped=False' in r


# ---------------------------------------------------------------------------
# CachedMethod / cachedmethod
# ---------------------------------------------------------------------------

class _MethodHolder:
    def __init__(self):
        self.cache = LRI()
        self.calls = []


def test_cachedmethod_attr_name():
    class Holder:
        def __init__(self):
            self.cache = {}
            self.calls = []

        @cachedmethod('cache')
        def double(self, x):
            self.calls.append(x)
            return x * 2

    h = Holder()
    assert h.double(2) == 4
    assert h.double(2) == 4
    assert h.calls == [2]
    assert len(h.cache) == 1


def test_cachedmethod_invalid():
    with pytest.raises(TypeError):
        CachedMethod(lambda self: None, 42)


def test_cachedmethod_callable_cache():
    cache = {}

    def get_cache(obj):
        return cache

    class Holder:
        @cachedmethod(get_cache)
        def triple(self, x):
            return x * 3

    h = Holder()
    assert h.triple(2) == 6
    assert h.triple(2) == 6
    assert len(cache) == 1


def test_cachedmethod_mapping_cache():
    cache = {}

    class Holder:
        @cachedmethod(cache)
        def quad(self, x):
            return x * 4

    h = Holder()
    assert h.quad(2) == 8
    assert h.quad(2) == 8
    assert len(cache) == 1


def test_cachedmethod_scoped():
    cache = {}

    class Holder:
        @cachedmethod(cache)
        def a(self, x):
            return x

        @cachedmethod(cache)
        def b(self, x):
            return x * 100

    h = Holder()
    assert h.a(1) == 1
    assert h.b(1) == 100
    assert len(cache) == 2


def test_cachedmethod_unscoped():
    cache = {}

    class Holder:
        @cachedmethod(cache, scoped=False)
        def a(self, x):
            return x

        @cachedmethod(cache, scoped=False)
        def b(self, x):
            return x * 100

    h = Holder()
    assert h.a(1) == 1
    assert h.b(1) == 1
    assert len(cache) == 1


def test_cachedmethod_descriptor_class_access():
    cache = {}

    class Holder:
        @cachedmethod(cache)
        def method(self, x):
            return x

    assert isinstance(Holder.method, CachedMethod)


def test_cachedmethod_repr():
    cache = {}

    class Holder:
        @cachedmethod(cache)
        def method(self, x):
            return x

    h = Holder()
    h.method(1)
    r = repr(h.method)
    assert 'CachedMethod' in r
    assert 'bound_to' in r


def test_cachedmethod_abstractmethod_flag():
    class Holder:
        @cachedmethod({})
        def method(self):
            return 1

    cm = Holder.__dict__['method']
    assert hasattr(cm, '__isabstractmethod__')


# ---------------------------------------------------------------------------
# cachedproperty
# ---------------------------------------------------------------------------

def test_cachedproperty():
    calls = []

    class Holder:
        @cachedproperty
        def value(self):
            calls.append(1)
            return 42

    h = Holder()
    assert h.value == 42
    assert h.value == 42
    assert len(calls) == 1
    assert h.__dict__['value'] == 42


def test_cachedproperty_class_access():
    class Holder:
        @cachedproperty
        def value(self):
            return 42

    assert isinstance(Holder.value, cachedproperty)


def test_cachedproperty_deletable():
    calls = []

    class Holder:
        @cachedproperty
        def value(self):
            calls.append(1)
            return 42

    h = Holder()
    assert h.value == 42
    del h.value
    assert h.value == 42
    assert len(calls) == 2


def test_cachedproperty_repr():
    class Holder:
        @cachedproperty
        def value(self):
            return 42

    assert 'cachedproperty' in repr(Holder.value)


def test_cachedproperty_doc():
    class Holder:
        @cachedproperty
        def value(self):
            """the doc"""
            return 42

    assert Holder.value.__doc__ == 'the doc'


# ---------------------------------------------------------------------------
# ThresholdCounter
# ---------------------------------------------------------------------------

def test_threshold_counter_invalid():
    with pytest.raises(ValueError):
        ThresholdCounter(0)
    with pytest.raises(ValueError):
        ThresholdCounter(1)
    with pytest.raises(ValueError):
        ThresholdCounter(-0.5)


def test_threshold_counter_basic():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    assert tc.total == 1
    assert tc[1] == 1


def test_threshold_counter_threshold_property():
    tc = ThresholdCounter(threshold=0.1)
    assert tc.threshold == 0.1


def test_threshold_counter_update_iterable():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([2] * 10)
    assert tc.total == 10
    assert tc.get(2) == 10
    assert tc.get(1) == 0


def test_threshold_counter_update_mapping():
    tc = ThresholdCounter(threshold=0.1)
    tc.update({3: 5})
    assert tc.total == 5
    assert tc[3] == 5


def test_threshold_counter_update_kwargs():
    tc = ThresholdCounter(threshold=0.1)
    tc.update(None, a=3)
    assert tc.total == 3


def test_threshold_counter_compaction():
    tc = ThresholdCounter(threshold=0.5)  # thresh_count = 2
    tc.add(1)
    tc.add(1)
    # Now total == 2, compaction triggered
    assert tc.total == 2
    assert tc[1] == 2


def test_threshold_counter_eviction():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')  # total 1, a count 1
    tc.add('b')  # total 2 -> compaction: a count 1 > 1? no -> a removed
    assert 'a' not in tc
    assert 'b' in tc


def test_threshold_counter_elements():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([1, 1, 2])
    elements = sorted(tc.elements())
    assert elements == [1, 1, 2]


def test_threshold_counter_most_common():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([1, 1, 1, 2, 2, 3])
    mc = tc.most_common()
    assert mc[0] == (1, 3)
    assert mc[1] == (2, 2)
    assert mc[2] == (3, 1)
    assert tc.most_common(2) == [(1, 3), (2, 2)]
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []
    assert tc.most_common(10) == mc


def test_threshold_counter_common_uncommon():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([1, 1, 1])
    assert tc.get_common_count() == 3
    assert tc.get_uncommon_count() == 0
    assert tc.get_commonality() == 1.0


def test_threshold_counter_iterkeys_values_items():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([1, 2, 2])
    assert set(tc.keys()) == {1, 2}
    assert sorted(tc.values()) == [1, 2]
    assert sorted(tc.items()) == [(1, 1), (2, 2)]


def test_threshold_counter_len_contains():
    tc = ThresholdCounter(threshold=0.1)
    tc.add('x')
    assert len(tc) == 1
    assert 'x' in tc
    assert 'y' not in tc


def test_threshold_counter_get_default():
    tc = ThresholdCounter(threshold=0.1)
    assert tc.get('nope') == 0
    assert tc.get('nope', -1) == -1


# ---------------------------------------------------------------------------
# MinIDMap
# ---------------------------------------------------------------------------

class _Obj:
    pass


def test_min_id_map_basic():
    m = MinIDMap()
    a = _Obj()
    b = _Obj()
    ida = m.get(a)
    idb = m.get(b)
    assert ida != idb
    assert m.get(a) == ida
    assert len(m) == 2
    assert a in m
    assert b in m


def test_min_id_map_drop_and_reuse():
    m = MinIDMap()
    a = _Obj()
    b = _Obj()
    ida = m.get(a)
    idb = m.get(b)
    m.drop(a)
    assert a not in m
    assert len(m) == 1
    c = _Obj()
    idc = m.get(c)
    # should reuse the freed smallest id
    assert idc == ida
    assert idb in (ida, idc, m.get(b))


def test_min_id_map_iter():
    m = MinIDMap()
    a = _Obj()
    b = _Obj()
    m.get(a)
    m.get(b)
    keys = list(m)
    assert len(keys) == 2


def test_min_id_map_iteritems():
    m = MinIDMap()
    a = _Obj()
    ida = m.get(a)
    items = dict(m.iteritems())
    assert items[a] == ida


def test_min_id_map_len():
    m = MinIDMap()
    assert len(m) == 0
    a = _Obj()
    m.get(a)
    assert len(m) == 1
