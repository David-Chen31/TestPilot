import pytest

from boltons.cacheutils import (
    LRI, LRU, make_cache_key, _HashedKey,
    CachedFunction, CachedMethod, cached, cachedmethod, cachedproperty,
    ThresholdCounter, MinIDMap, _MISSING,
)


# ---------------- LRI ----------------

def test_lri_basic_insertion_and_get():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert dict(c) == {'a': 'A', 'b': 'B'}
    assert c['a'] == 'A'
    assert c['b'] == 'B'
    assert c.hit_count == 2
    assert c.miss_count == 0


def test_lri_lri_eviction_order():
    # LRI: least recently inserted is evicted (not reads)
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert c['a'] == 'A'  # reading does not refresh
    c['c'] = 'C'
    assert dict(c) == {'b': 'B', 'c': 'C'}
    assert 'a' not in c
    assert c.get('a') is None
    assert c.soft_miss_count == 1


def test_lri_reinsert_moves_to_front():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    c['a'] = 'A2'  # re-set counts as recent insertion
    c['c'] = 'C'
    assert dict(c) == {'a': 'A2', 'c': 'C'}
    assert 'b' not in c


def test_lri_invalid_max_size():
    with pytest.raises(ValueError):
        LRI(max_size=0)
    with pytest.raises(ValueError):
        LRI(max_size=-5)


def test_lri_on_miss_invalid():
    with pytest.raises(TypeError):
        LRI(on_miss="not callable")


def test_lri_on_miss_used():
    calls = []
    def on_miss(key):
        calls.append(key)
        return key * 2
    c = LRI(on_miss=on_miss)
    assert c['x'] == 'xx'
    assert calls == ['x']
    assert c.miss_count == 1
    assert c['x'] == 'xx'
    assert c.miss_count == 1
    assert c.hit_count == 1


def test_lri_getitem_miss_raises_without_on_miss():
    c = LRI()
    with pytest.raises(KeyError):
        c['nope']
    assert c.miss_count == 1


def test_lri_delitem_and_pop():
    c = LRI(max_size=3)
    c['a'] = 1
    c['b'] = 2
    del c['a']
    assert 'a' not in c
    assert c.pop('b') == 2
    assert 'b' not in c
    assert c.pop('missing', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        c.pop('missing')


def test_lri_popitem():
    c = LRI(max_size=3)
    c['a'] = 1
    c['b'] = 2
    k, v = c.popitem()
    assert (k, v) == ('b', 2)
    assert 'b' not in c
    assert len(c) == 1


def test_lri_clear():
    c = LRI(max_size=3)
    c['a'] = 1
    c.clear()
    assert len(c) == 0
    assert dict(c) == {}
    # internal LL reset
    assert c._get_flattened_ll() == [(_MISSING, _MISSING)]
    # can still add
    c['x'] = 1
    assert c['x'] == 1


def test_lri_setdefault():
    c = LRI()
    assert c.setdefault('a', 5) == 5
    assert c['a'] == 5
    assert c.setdefault('a', 99) == 5
    assert c.soft_miss_count == 1


def test_lri_copy():
    c = LRI(max_size=2)
    c['a'] = 1
    c['b'] = 2
    c2 = c.copy()
    assert isinstance(c2, LRI)
    assert c2.max_size == 2
    assert dict(c2) == {'a': 1, 'b': 2}


def test_lri_update_variants():
    c = LRI()
    c.update({'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}
    c.update([('c', 3)])
    assert c['c'] == 3
    c.update(d=4)
    assert c['d'] == 4
    c.update(c)  # self-update should be no-op
    assert dict(c) == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


def test_lri_eq_ne():
    c1 = LRI()
    c1['a'] = 1
    c2 = LRI()
    c2['a'] = 1
    assert c1 == c2
    assert not (c1 != c2)
    assert c1 == {'a': 1}
    c2['b'] = 2
    assert c1 != c2
    assert c1 != {'a': 1, 'b': 2}


def test_lri_repr():
    c = LRI(max_size=5, on_miss=None)
    r = repr(c)
    assert r.startswith('LRI(')
    assert 'max_size=5' in r
    assert 'on_miss=None' in r


# ---------------- LRU ----------------

def test_lru_eviction_by_use():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert c['a'] == 'A'  # refreshes a
    c['c'] = 'C'  # should evict b
    assert 'a' in c
    assert 'b' not in c
    assert dict(c) == {'a': 'A', 'c': 'C'}


def test_lru_stats():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    c['b']
    c['c'] = 'C'
    assert c.get('a') is None
    assert c.hit_count == 1
    assert c.miss_count == 1
    assert c.soft_miss_count == 1


def test_lru_on_miss():
    c = LRU(on_miss=lambda k: k.upper())
    assert c['x'] == 'X'
    assert c['x'] == 'X'
    assert c.hit_count == 1
    assert c.miss_count == 1


def test_lru_is_lri_subclass():
    assert issubclass(LRU, LRI)


# ---------------- make_cache_key ----------------

def test_make_cache_key_basic():
    k = make_cache_key(('a', 'b'), {'c': 'd'})
    assert tuple(k) == ('a', 'b', make_cache_key.__defaults__[1], ('c', 'd'))


def test_make_cache_key_no_kwargs_returns_hashed_key():
    k = make_cache_key(('a', 'b'), {})
    assert isinstance(k, _HashedKey)
    assert tuple(k) == ('a', 'b')


def test_make_cache_key_fasttype_single_arg_returned_unwrapped():
    assert make_cache_key((5,), {}) == 5
    assert make_cache_key(('s',), {}) == 's'
    assert make_cache_key((None,), {}) is None


def test_make_cache_key_typed_separates_types():
    k1 = make_cache_key((3,), {}, typed=True)
    k2 = make_cache_key((3.0,), {}, typed=True)
    assert k1 != k2
    k3 = make_cache_key((3,), {}, typed=False)
    k4 = make_cache_key((3,), {}, typed=False)
    assert k3 == k4


def test_make_cache_key_typed_kwargs():
    k = make_cache_key((), {'a': 1}, typed=True)
    assert isinstance(k, _HashedKey)
    # contains kwarg mark, ('a', 1), and type
    assert int in k


def test_hashed_key_hash_called_once():
    class Tracker:
        def __init__(self, v):
            self.v = v
            self.hashes = 0
        def __hash__(self):
            self.hashes += 1
            return hash(self.v)
        def __eq__(self, other):
            return isinstance(other, Tracker) and self.v == other.v

    t = Tracker(1)
    hk = _HashedKey([t])
    assert t.hashes == 1
    hash(hk)
    hash(hk)
    assert t.hashes == 1


def test_hashed_key_repr():
    hk = _HashedKey([1, 2])
    assert repr(hk) == '_HashedKey([1, 2])'


# ---------------- CachedFunction / cached ----------------

def test_cached_function_def():
    calls = []
    @cached({})
    def fn(x):
        calls.append(x)
        return x * 2
    assert fn(3) == 6
    assert fn(3) == 6
    assert calls == [3]


def test_cached_invalid_cache():
    with pytest.raises(TypeError):
        CachedFunction(lambda x: x, object())


def test_cached_callable_cache():
    store = {}
    @cached(lambda: store)
    def fn(x):
        return x + 1
    assert fn(1) == 2
    assert len(store) == 1


def test_cached_typed():
    @cached({}, typed=True)
    def fn(x):
        return type(x).__name__
    assert fn(1) == 'int'
    assert fn(1.0) == 'float'


def test_cached_scoped_repr():
    def f(): pass
    cf = CachedFunction(f, {})
    assert repr(cf) == f'CachedFunction(func={f!r})'
    cf2 = CachedFunction(f, {}, typed=True)
    assert 'typed=True' in repr(cf2)
    cf3 = CachedFunction(f, {}, scoped=False)
    assert 'scoped=False' in repr(cf3)


def test_cached_custom_key():
    def key_func(args, kwargs, typed=False):
        return (args[0] % 2,)
    @cached({}, key=key_func)
    def fn(x):
        return x
    assert fn(2) == 2
    # same key parity -> returns cached 2 for even
    assert fn(4) == 2


# ---------------- CachedMethod / cachedmethod ----------------

def test_cachedmethod_basic():
    class Lowerer:
        def __init__(self):
            self.cache = LRI()
            self.calls = 0

        @cachedmethod('cache')
        def lower(self, text):
            self.calls += 1
            return text.lower()

    obj = Lowerer()
    assert obj.lower('ABC') == 'abc'
    assert obj.lower('ABC') == 'abc'
    assert obj.calls == 1
    assert len(obj.cache) == 1


def test_cachedmethod_invalid_cache():
    with pytest.raises(TypeError):
        CachedMethod(lambda self: None, object())


def test_cachedmethod_unbound_call():
    def method(self, x):
        return x * 3
    cm = CachedMethod(method, {})
    class Dummy:
        pass
    d = Dummy()
    # calling unbound: args[0] is the "obj"
    assert cm(d, 2) == 6


def test_cachedmethod_get_via_class_returns_descriptor():
    class Holder:
        @cachedmethod('cache')
        def m(self, x):
            return x

    assert isinstance(Holder.__dict__['m'], CachedMethod)
    assert isinstance(Holder.m, CachedMethod)


def test_cachedmethod_repr():
    def f(self):
        return 1
    cm = CachedMethod(f, {})
    r = repr(cm)
    assert r.startswith('CachedMethod(')
    assert 'scoped=True' in r


def test_cachedmethod_isabstractmethod_propagates():
    def f(self):
        return 1
    f.__isabstractmethod__ = True
    cm = CachedMethod(f, {})
    assert cm.__isabstractmethod__ is True


def test_cachedmethod_callable_cache_and_scoped_false():
    shared = {}
    class T:
        @cachedmethod(lambda obj: shared, scoped=False)
        def m(self, x):
            return x * 2
    t = T()
    assert t.m(2) == 4
    assert t.m(2) == 4
    assert len(shared) == 1


# ---------------- cachedproperty ----------------

def test_cachedproperty():
    calls = []
    class Obj:
        def __init__(self):
            self.x = 5

        @cachedproperty
        def doubled(self):
            calls.append(1)
            return self.x * 2

    o = Obj()
    assert o.doubled == 10
    assert o.doubled == 10
    assert len(calls) == 1
    assert 'doubled' in o.__dict__


def test_cachedproperty_reset_via_delattr():
    class Obj:
        @cachedproperty
        def val(self):
            return 42
    o = Obj()
    assert o.val == 42
    del o.val
    assert o.val == 42


def test_cachedproperty_class_access():
    class Obj:
        @cachedproperty
        def val(self):
            return 1
    assert isinstance(Obj.val, cachedproperty)


def test_cachedproperty_doc_and_repr():
    class Obj:
        @cachedproperty
        def val(self):
            """My doc"""
            return 1
    assert Obj.val.__doc__ == "My doc"
    assert 'cachedproperty' in repr(Obj.val)


def test_cachedproperty_isabstractmethod():
    class Obj:
        @cachedproperty
        def val(self):
            return 1
    assert Obj.val.__isabstractmethod__ is False


# ---------------- ThresholdCounter ----------------

def test_threshold_counter_invalid():
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
    assert tc.total == 1


def test_threshold_counter_compaction():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    tc.update([2] * 10)
    assert tc.get(1) == 0
    assert 2 in tc
    assert tc.get(2) == 10
    tc.add(5)
    assert 5 in tc


def test_threshold_counter_elements():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    tc.update([2] * 10)
    tc.add(5)
    assert len(list(tc.elements())) == 11


def test_threshold_counter_most_common():
    tc = ThresholdCounter(threshold=0.5)
    tc.update(['a', 'a', 'a', 'b', 'b', 'c'])
    assert tc.most_common(1) == [('a', 3)]
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []
    all_common = tc.most_common()
    assert all_common[0] == ('a', 3)
    assert tc.most_common(100) == all_common


def test_threshold_counter_common_uncommon():
    tc = ThresholdCounter(threshold=0.5)
    tc.update(['a', 'a', 'b'])
    common = tc.get_common_count()
    uncommon = tc.get_uncommon_count()
    assert common + uncommon == 3
    assert 0.0 <= tc.get_commonality() <= 1.0


def test_threshold_counter_keys_values_items():
    tc = ThresholdCounter(threshold=0.5)
    tc.update(['a', 'a', 'b'])
    assert sorted(tc.keys()) == ['a', 'b']
    assert sorted(tc.values()) == [1, 2]
    assert dict(tc.items()) == {'a': 2, 'b': 1}
    assert len(tc) == 2
    assert tc.get('missing') == 0
    assert tc.get('missing', 7) == 7


def test_threshold_counter_update_mapping():
    tc = ThresholdCounter(threshold=0.5)
    tc.update({'a': 3})
    assert tc['a'] == 3
    tc.update([('b', 1)][0:0])  # no-op iterable
    tc.update(['b'], c=2)
    assert tc['b'] == 1
    assert tc['c'] == 2


def test_threshold_counter_threshold_property():
    tc = ThresholdCounter(threshold=0.25)
    assert tc.threshold == 0.25


def test_threshold_counter_getitem_missing():
    tc = ThresholdCounter()
    with pytest.raises(KeyError):
        tc['nope']


# ---------------- MinIDMap ----------------

def test_min_idmap_basic():
    m = MinIDMap()
    a = object()
    b = object()
    ia = m.get(a)
    ib = m.get(b)
    assert ia == 0
    assert ib == 1
    assert m.get(a) == ia  # stable
    assert a in m
    assert b in m
    assert len(m) == 2


def test_min_idmap_drop_and_reuse():
    m = MinIDMap()
    a = object()
    b = object()
    c = object()
    ia = m.get(a)
    ib = m.get(b)
    m.drop(a)
    assert a not in m
    assert len(m) == 1
    # The freed id should be reused (smallest free)
    ic = m.get(c)
    assert ic == ia
    assert ic != ib


def test_min_idmap_iter():
    m = MinIDMap()
    a = object()
    b = object()
    m.get(a)
    m.get(b)
    assert set(m) == {a, b}
    items = dict(m.iteritems())
    assert items[a] == 0
    assert items[b] == 1


def test_min_idmap_weakref_cleanup(capsys):
    import gc
    m = MinIDMap()

    class Obj:
        pass

    a = Obj()
    ia = m.get(a)
    ref = weakref.ref(a)
    del a
    gc.collect()
    # after cleanup the size should decrease eventually; free should have id
    assert ia in m.free or len(m) == 0


import weakref  # noqa: E402  (used in test_min_idmap_weakref_cleanup)
