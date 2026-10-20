import gc
import pytest
import weakref

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
    _MISSING,
    _KWARG_MARK,
)


# ---------------------------------------------------------------------------
# LRI
# ---------------------------------------------------------------------------

def test_lri_basic_setget():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert dict(c) == {'a': 'A', 'b': 'B'}
    assert c['a'] == 'A'
    assert c['b'] == 'B'


def test_lri_eviction_by_insertion_order():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    # accessing 'a' does not change insertion order for LRI
    _ = c['a']
    c['c'] = 'C'
    assert 'a' not in c
    assert dict(c) == {'b': 'B', 'c': 'C'}


def test_lri_setitem_existing_updates_value():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['a'] = 'AA'
    assert c['a'] == 'AA'
    assert len(c) == 1


def test_lri_invalid_max_size():
    with pytest.raises(ValueError):
        LRI(max_size=0)
    with pytest.raises(ValueError):
        LRI(max_size=-1)


def test_lri_on_miss_not_callable():
    with pytest.raises(TypeError):
        LRI(on_miss=42)


def test_lri_on_miss_used():
    calls = []

    def on_miss(key):
        calls.append(key)
        return key * 2

    c = LRI(on_miss=on_miss)
    assert c['x'] == 'xx'
    assert calls == ['x']
    assert c.miss_count == 1
    # second access is a hit
    assert c['x'] == 'xx'
    assert c.hit_count == 1
    assert calls == ['x']


def test_lri_getitem_miss_no_on_miss():
    c = LRI()
    with pytest.raises(KeyError):
        c['nope']
    assert c.miss_count == 1
    assert c.hit_count == 0


def test_lri_get_soft_miss():
    c = LRI()
    assert c.get('nope') is None
    assert c.soft_miss_count == 1
    assert c.miss_count == 1
    c['x'] = 1
    assert c.get('x') == 1
    assert c.hit_count == 1


def test_lri_get_default_value():
    c = LRI()
    assert c.get('nope', 'dflt') == 'dflt'


def test_lri_delitem():
    c = LRI()
    c['a'] = 'A'
    del c['a']
    assert 'a' not in c
    with pytest.raises(KeyError):
        del c['a']


def test_lri_pop():
    c = LRI()
    c['a'] = 'A'
    assert c.pop('a') == 'A'
    assert 'a' not in c
    assert c.pop('b', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        c.pop('b')


def test_lri_pop_does_not_count_hit():
    c = LRI()
    c['a'] = 'A'
    c.pop('a')
    assert c.hit_count == 0
    assert c.miss_count == 0


def test_lri_popitem():
    c = LRI()
    c['a'] = 'A'
    c['b'] = 'B'
    item = c.popitem()
    assert item in (('a', 'A'), ('b', 'B'))
    assert len(c) == 1


def test_lri_clear():
    c = LRI()
    c['a'] = 'A'
    c.clear()
    assert dict(c) == {}
    assert c._get_flattened_ll() == [(_MISSING, _MISSING)]
    c['b'] = 'B'
    assert c['b'] == 'B'


def test_lri_copy():
    c = LRI(max_size=5)
    c['a'] = 'A'
    c2 = c.copy()
    assert isinstance(c2, LRI)
    assert c2.max_size == 5
    assert dict(c2) == {'a': 'A'}
    c2['b'] = 'B'
    assert 'b' not in c


def test_lri_setdefault():
    c = LRI()
    assert c.setdefault('a', 'A') == 'A'
    assert c.soft_miss_count == 1
    assert c.setdefault('a', 'B') == 'A'
    assert c.soft_miss_count == 1
    assert c.hit_count == 1


def test_lri_update_from_dict_and_iterable():
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
    c1 = LRI()
    c1['a'] = 1
    c2 = LRI()
    c2['a'] = 1
    assert c1 == c2
    assert c1 == {'a': 1}
    assert {'a': 1} == c1
    assert not (c1 != c2)
    c3 = LRI()
    c3['a'] = 2
    assert c1 != c3
    assert not (c1 == c3)
    # different lengths
    c1['b'] = 2
    assert c1 != c2


def test_lri_eq_not_mapping_size_error_branch():
    c = LRI()
    c['a'] = 1
    # object with len that differs
    assert (c == [1, 2]) is False


def test_lri_repr():
    c = LRI(max_size=10)
    c['a'] = 1
    r = repr(c)
    assert r.startswith('LRI(max_size=10')
    assert "'a': 1" in r


def test_lri_init_with_values():
    c = LRI(values={'a': 1})
    assert c['a'] == 1


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

def test_lru_getitem_moves_to_front():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    _ = c['a']  # moves a to front
    c['c'] = 'C'  # should evict b
    assert 'a' in c
    assert 'b' not in c
    assert 'c' in c


def test_lru_stats():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    for _ in range(3):
        c['b']
    c['c'] = 'C'
    c.get('a')
    assert c.hit_count == 3
    assert c.miss_count >= 1
    assert c.soft_miss_count >= 1


def test_lru_on_miss():
    c = LRU(on_miss=lambda k: k.upper())
    assert c['x'] == 'X'
    assert c.miss_count == 1
    assert c['x'] == 'X'
    assert c.hit_count == 1


def test_lru_is_subclass():
    assert issubclass(LRU, LRI)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_simple_args():
    key = make_cache_key(('a', 'b'), {})
    assert isinstance(key, _HashedKey)
    assert tuple(key) == ('a', 'b')


def test_make_cache_key_kwargs():
    key = make_cache_key(('a', 'b'), {'c': 'd'})
    assert tuple(key) == ('a', 'b', _KWARG_MARK, ('c', 'd'))


def test_make_cache_key_fast_type_single():
    assert make_cache_key((5,), {}) == 5
    assert make_cache_key(('x',), {}) == 'x'
    assert make_cache_key((None,), {}) is None


def test_make_cache_key_typed():
    k1 = make_cache_key((3,), {}, typed=True)
    k2 = make_cache_key((3.0,), {}, typed=True)
    assert k1 != k2
    assert type(k1) is _HashedKey
    # without typed, single fast type int returns the int
    assert make_cache_key((3,), {}, typed=False) == 3


def test_make_cache_key_typed_kwargs():
    k = make_cache_key((), {'x': 1}, typed=True)
    assert _KWARG_MARK in tuple(k)
    assert int in tuple(k)


def test_make_cache_key_hash_once():
    k = make_cache_key(('a', 'b'), {})
    assert hash(k) == k.hash_value


def test_hashed_key_repr():
    k = _HashedKey([1, 2])
    assert repr(k) == '_HashedKey([1, 2])'


def test_make_cache_key_alias():
    assert _make_cache_key is make_cache_key


# ---------------------------------------------------------------------------
# CachedFunction / cached
# ---------------------------------------------------------------------------

def test_cached_function_basic():
    calls = []

    @cached({})
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]


def test_cached_function_with_lru_cache():
    cache = LRU()
    f = CachedFunction(lambda x: x + 1, cache)
    assert f(1) == 2
    assert f(1) == 2
    assert len(cache) == 1


def test_cached_callable_cache():
    storages = []

    def get_cache():
        d = {}
        storages.append(d)
        return d

    f = CachedFunction(lambda x: x, get_cache)
    assert f(1) == 1
    assert len(storages) == 1


def test_cached_invalid_cache():
    with pytest.raises(TypeError):
        CachedFunction(lambda x: x, 42)


def test_cached_typed():
    calls = []

    @cached({}, typed=True)
    def f(x):
        calls.append(x)
        return x

    f(3)
    f(3.0)
    assert len(calls) == 2


def test_cached_custom_key():
    def key(args, kwargs, typed=False):
        return ('custom',) + args

    f = CachedFunction(lambda x: x, {}, key=key)
    assert f(5) == 5


def test_cached_function_repr():
    def f(x):
        return x

    cf = CachedFunction(f, {})
    assert 'CachedFunction(func=' in repr(cf)
    cf_typed = CachedFunction(f, {}, typed=True)
    assert 'typed=True' in repr(cf_typed)
    cf_unscoped = CachedFunction(f, {}, scoped=False)
    assert 'scoped=False' in repr(cf_unscoped)


# ---------------------------------------------------------------------------
# CachedMethod / cachedmethod
# ---------------------------------------------------------------------------

class _MethodHost:
    def __init__(self, cache=None):
        self.cache = cache if cache is not None else {}
        self.calls = 0

    @cachedmethod('cache')
    def compute(self, x):
        self.calls += 1
        return x * 10


def test_cachedmethod_attr_name():
    h = _MethodHost()
    assert h.compute(2) == 20
    assert h.compute(2) == 20
    assert h.calls == 1
    assert len(h.cache) == 1


def test_cachedmethod_scoped_per_instance():
    h1 = _MethodHost()
    h2 = _MethodHost()
    assert h1.compute(3) == 30
    assert h2.compute(3) == 30
    assert h1.calls == 1
    assert h2.calls == 1


def test_cachedmethod_dict_cache():
    cache = {}

    class H:
        @cachedmethod(cache)
        def m(self, x):
            return x + 1

    h = H()
    assert h.m(1) == 2
    assert h.m(1) == 2
    assert len(cache) == 1


def test_cachedmethod_callable_cache():
    def make():
        return {}

    class H:
        @cachedmethod(make)
        def m(self, x):
            return x

    h = H()
    assert h.m(1) == 1


def test_cachedmethod_invalid_cache():
    with pytest.raises(TypeError):
        CachedMethod(lambda self: None, 42)


def test_cachedmethod_access_on_class():
    assert isinstance(_MethodHost.__dict__['compute'], CachedMethod)
    cm = _MethodHost.compute
    assert cm.bound_to is None
    assert 'CachedMethod' in repr(cm)


def test_cachedmethod_unscoped():
    cache = {}

    class H:
        @cachedmethod(cache, scoped=False)
        def m(self, x):
            return x

    h1 = H()
    h2 = H()
    h1.m(1)
    h2.m(1)
    # With scoped=False, obj not in key
    assert len(cache) == 1


def test_cachedmethod_typed():
    cache = {}

    class H:
        @cachedmethod(cache, typed=True)
        def m(self, x):
            return x

    h = H()
    h.m(3)
    h.m(3.0)
    assert len(cache) == 2


def test_cachedmethod_repr_bound():
    h = _MethodHost()
    bound = h.compute
    r = repr(bound)
    assert 'bound_to=' in r


def test_cachedmethod_abstractmethod_flag():
    def f(self):
        pass
    f.__isabstractmethod__ = True
    cm = CachedMethod(f, {})
    assert cm.__isabstractmethod__ is True


def test_cachedmethod_custom_key():
    cache = {}

    def key(args, kwargs, typed=False):
        return ('k',) + tuple(a for a in args if not hasattr(a, '__dict__'))

    class H:
        @cachedmethod(cache, key=key)
        def m(self, x):
            return x

    h = H()
    assert h.m(6) == 6


# ---------------------------------------------------------------------------
# cachedproperty
# ---------------------------------------------------------------------------

def test_cachedproperty_basic():
    calls = []

    class C:
        @cachedproperty
        def val(self):
            calls.append(1)
            return 42

    c = C()
    assert c.val == 42
    assert c.val == 42
    assert calls == [1]
    assert c.__dict__['val'] == 42


def test_cachedproperty_class_access():
    class C:
        @cachedproperty
        def val(self):
            "docstring"
            return 1

    assert isinstance(C.val, cachedproperty)
    assert C.val.__doc__ == "docstring"
    assert repr(C.val).startswith('<cachedproperty func=')


def test_cachedproperty_clearable():
    class C:
        @cachedproperty
        def val(self):
            return object()

    c = C()
    v = c.val
    del c.val
    assert c.val is not v


def test_cachedproperty_abstractmethod():
    def f(self):
        pass
    f.__isabstractmethod__ = True
    cp = cachedproperty(f)
    assert cp.__isabstractmethod__ is True


# ---------------------------------------------------------------------------
# ThresholdCounter
# ---------------------------------------------------------------------------

def test_threshold_counter_invalid_threshold():
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=0)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=1)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=-0.5)


def test_threshold_counter_threshold_property():
    tc = ThresholdCounter(threshold=0.1)
    assert tc.threshold == 0.1


def test_threshold_counter_add_and_get():
    tc = ThresholdCounter(threshold=0.1)
    tc.add('a')
    tc.add('a')
    tc.add('b')
    assert tc['a'] == 2
    assert tc['b'] == 1
    assert tc.get('a') == 2
    assert tc.get('z') == 0
    assert tc.get('z', -1) == -1
    assert tc.total == 3


def test_threshold_counter_compaction():
    tc = ThresholdCounter(threshold=0.1)
    for _ in range(10):
        tc.add('a')
    # a should survive compaction
    assert 'a' in tc
    tc.add('b')
    assert 'a' in tc


def test_threshold_counter_contains_and_len():
    tc = ThresholdCounter(threshold=0.5)
    tc.add(1)
    assert 1 in tc
    assert 2 not in tc
    assert len(tc) == 1


def test_threshold_counter_iterators():
    tc = ThresholdCounter(threshold=0.5)
    tc.update([1, 1, 2])
    assert set(tc.keys()) == {1, 2}
    assert sorted(tc.values()) == [1, 2]
    items = dict(tc.items())
    assert items == {1: 2, 2: 1}
    assert set(tc.iterkeys()) == {1, 2}
    assert sorted(tc.itervalues()) == [1, 2]
    assert dict(tc.iteritems()) == {1: 2, 2: 1}


def test_threshold_counter_elements():
    tc = ThresholdCounter(threshold=0.5)
    tc.update([1, 1, 2])
    elems = sorted(tc.elements())
    assert elems == [1, 1, 2]


def test_threshold_counter_most_common():
    tc = ThresholdCounter(threshold=0.05)
    tc.update([1, 1, 1, 2, 2, 3])
    assert tc.most_common(1) == [(1, 3)]
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []
    mc = tc.most_common()
    assert mc[0] == (1, 3)
    assert len(mc) == 3


def test_threshold_counter_update_mapping():
    tc = ThresholdCounter(threshold=0.1)
    tc.update({1: 3, 2: 1})
    assert tc[1] == 3
    assert tc[2] == 1


def test_threshold_counter_update_kwargs():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([1])
    tc.update(iterable=None, extra=1)
    assert tc.get('extra') == 1


def test_threshold_counter_common_uncommon():
    tc = ThresholdCounter(threshold=0.2)
    tc.update([1, 1, 1])
    assert tc.get_common_count() == 3
    assert tc.get_uncommon_count() == 0
    assert tc.get_commonality() == 1.0


# ---------------------------------------------------------------------------
# MinIDMap
# ---------------------------------------------------------------------------

class _WeakObj:
    pass


def test_minidmap_basic():
    m = MinIDMap()
    a = _WeakObj()
    b = _WeakObj()
    ida = m.get(a)
    idb = m.get(b)
    assert ida == 0
    assert idb == 1
    assert m.get(a) == 0
    assert a in m
    assert len(m) == 2
    assert dict(m.iteritems()) == {a: 0, b: 1}
    assert set(iter(m)) == {a, b}


def test_minidmap_drop_and_reuse():
    m = MinIDMap()
    a = _WeakObj()
    b = _WeakObj()
    ida = m.get(a)
    m.get(b)
    m.drop(a)
    assert a not in m
    assert len(m) == 1
    c = _WeakObj()
    assert m.get(c) == ida


def test_minidmap_weakref_cleanup():
    m = MinIDMap()
    a = _WeakObj()
    ida = m.get(a)
    ref = weakref.ref(a)
    del a
    gc.collect()
    assert ref() is None
    # freed id should be reusable
    b = _WeakObj()
    assert m.get(b) == ida


def test_minidmap_get_smallest_free():
    m = MinIDMap()
    objs = [_WeakObj() for _ in range(3)]
    for o in objs:
        m.get(o)
    m.drop(objs[1])
    m.drop(objs[0])
    new = _WeakObj()
    assert m.get(new) == 0
