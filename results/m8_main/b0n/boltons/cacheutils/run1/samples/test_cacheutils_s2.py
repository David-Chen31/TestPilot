import gc
import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    make_cache_key,
    _make_cache_key,
    _HashedKey,
    _KWARG_MARK,
    _MISSING,
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

def test_lri_basic_set_get():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert dict(c) == {'a': 'A', 'b': 'B'}
    assert c['a'] == 'A'
    assert c['b'] == 'B'


def test_lri_eviction_by_insertion():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    # access 'a' does not save it from LRI eviction
    c['a']
    c['c'] = 'C'
    assert 'a' not in c
    assert 'b' in c
    assert 'c' in c


def test_lri_docstring_stats():
    cap_cache = LRI(max_size=2)
    cap_cache['a'], cap_cache['b'] = 'A', 'B'
    for i in range(3):
        assert cap_cache['b'] == 'B'
    cap_cache['c'] = 'C'
    assert cap_cache.get('a') is None
    assert (cap_cache.hit_count, cap_cache.miss_count,
            cap_cache.soft_miss_count) == (3, 1, 1)


def test_lri_invalid_max_size():
    with pytest.raises(ValueError):
        LRI(max_size=0)
    with pytest.raises(ValueError):
        LRI(max_size=-1)


def test_lri_invalid_on_miss():
    with pytest.raises(TypeError):
        LRI(on_miss=5)


def test_lri_on_miss():
    c = LRI(max_size=4, on_miss=lambda k: k * 2)
    assert c['x'] == 'xx'
    assert c['x'] == 'xx'
    assert c.miss_count == 1
    assert c.hit_count == 1


def test_lri_on_miss_raises():
    c = LRI()
    with pytest.raises(KeyError):
        c['nope']
    assert c.miss_count == 1


def test_lri_get_default():
    c = LRI()
    assert c.get('x') is None
    assert c.get('x', 42) == 42
    assert c.soft_miss_count == 2


def test_lri_setdefault():
    c = LRI()
    assert c.setdefault('a', 5) == 5
    assert c['a'] == 5
    assert c.soft_miss_count == 1
    assert c.setdefault('a', 99) == 5


def test_lri_delitem():
    c = LRI()
    c['a'] = 1
    del c['a']
    assert 'a' not in c
    with pytest.raises(KeyError):
        del c['a']


def test_lri_pop_and_popitem():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    assert c.pop('a') == 1
    assert 'a' not in c
    with pytest.raises(KeyError):
        c.pop('a')
    assert c.pop('a', 'default') == 'default'
    # remaining item
    key, value = c.popitem()
    assert key == 'b'
    assert value == 2
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


def test_lri_clear():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    c.clear()
    assert len(c) == 0
    assert list(c._get_flattened_ll()) == [(_MISSING, _MISSING)]
    # still usable
    c['x'] = 10
    assert c['x'] == 10


def test_lri_copy():
    c = LRI(max_size=3)
    c['a'] = 1
    c2 = c.copy()
    assert isinstance(c2, LRI)
    assert c2.max_size == 3
    assert dict(c2) == {'a': 1}
    c2['b'] = 2
    assert 'b' not in c


def test_lri_update_variants():
    c = LRI()
    c.update({'a': 1})
    assert c['a'] == 1
    c.update([('b', 2)])
    assert c['b'] == 2
    c.update(c=3)
    assert c['c'] == 3


def test_lri_init_values():
    c = LRI(max_size=3, values={'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}


def test_lri_eq_ne():
    a = LRI(values={'a': 1})
    b = LRI(values={'a': 1})
    assert a == b
    assert not (a != b)
    assert a == {'a': 1}
    b['b'] = 2
    assert a != b
    assert a != 'notadict'


def test_lri_repr():
    c = LRI(max_size=5)
    c['a'] = 1
    r = repr(c)
    assert r.startswith('LRI(max_size=5, on_miss=None, values=')
    assert "'a': 1" in r


def test_lri_setitem_update_existing():
    c = LRI(max_size=2)
    c['a'] = 1
    c['a'] = 2
    assert c['a'] == 2
    assert len(c) == 1


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

def test_lru_access_protects_from_eviction():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    c['a']
    c['c'] = 'C'
    assert 'a' in c
    assert 'b' not in c
    assert 'c' in c
    assert c['a'] == 'A'


def test_lru_is_lri_subclass():
    assert issubclass(LRU, LRI)


def test_lru_stats():
    c = LRU(max_size=2)
    c['a'] = 1
    assert c['a'] == 1
    with pytest.raises(KeyError):
        c['missing']
    assert c.hit_count == 1
    assert c.miss_count == 1


def test_lru_on_miss():
    c = LRU(max_size=2, on_miss=lambda k: k + '!')
    assert c['x'] == 'x!'
    assert c['x'] == 'x!'
    assert c.miss_count == 1
    assert c.hit_count == 1


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_docstring():
    key = make_cache_key(('a', 'b'), {'c': 'd'})
    assert tuple(key) == ('a', 'b', _KWARG_MARK, ('c', 'd'))


def test_make_cache_key_fasttype_single_arg():
    # single str/int/None/frozenset arg returns raw arg
    assert make_cache_key(('a',), {}) == 'a'
    assert make_cache_key((3,), {}) == 3
    assert make_cache_key((None,), {}) is None
    assert make_cache_key((frozenset([1, 2]),), {}) == frozenset([1, 2])


def test_make_cache_key_returns_hashed_key():
    key = make_cache_key((['x'],), {})  # list is not a fasttype
    assert isinstance(key, _HashedKey)
    # multiple fast args -> _HashedKey
    key2 = make_cache_key(('a', 'b'), {})
    assert isinstance(key2, _HashedKey)
    assert tuple(key2) == ('a', 'b')


def test_make_cache_key_hash_cached():
    key = make_cache_key(('a', 'b'), {})
    assert key.__hash__() == hash(tuple(key))
    assert key.hash_value == hash(('a', 'b'))


def test_make_cache_key_kwargs_sorted():
    k1 = make_cache_key((), {'a': 1, 'b': 2})
    k2 = make_cache_key((), {'b': 2, 'a': 1})
    assert list(k1) == list(k2)


def test_make_cache_key_typed():
    k_int = make_cache_key((3,), {}, typed=True)
    k_float = make_cache_key((3.0,), {}, typed=True)
    assert list(k_int) != list(k_float)
    # typed forces _HashedKey even for single fasttype
    assert isinstance(k_int, _HashedKey)


def test_hashed_key_repr():
    key = _HashedKey([1, 2])
    assert repr(key) == '_HashedKey([1, 2])'


def test_make_cache_key_backwards_compat():
    assert _make_cache_key is make_cache_key


# ---------------------------------------------------------------------------
# CachedFunction / cached
# ---------------------------------------------------------------------------

def test_cached_function_basic():
    cache = {}
    calls = []

    @cached(cache)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]
    assert len(cache) == 1


def test_cached_with_lru():
    cache = LRU(max_size=2)

    @cached(cache)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert f(1) == 2
    assert len(cache) == 1


def test_cached_callable_cache():
    store = {}
    calls = []

    def get_cache():
        return store

    @cached(get_cache)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1]
    assert 1 in store or len(store) == 1


def test_cached_invalid_cache():
    with pytest.raises(TypeError):
        CachedFunction(lambda x: x, cache=5)


def test_cached_scoped():
    shared = {}
    calls = []

    @cached(shared, scoped=True)
    def f(x):
        calls.append(('f', x))
        return x

    @cached(shared, scoped=True)
    def g(x):
        calls.append(('g', x))
        return x

    f(1)
    g(1)
    f(1)
    g(1)
    # scoped: f and g have separate keys
    assert calls == [('f', 1), ('g', 1)]
    assert len(shared) == 2


def test_cached_unscoped_shared():
    shared = {}
    calls = []

    @cached(shared, scoped=False)
    def f(x):
        calls.append(('f', x))
        return x

    @cached(shared, scoped=False)
    def g(x):
        calls.append(('g', x))
        return x

    assert f(1) == 1
    assert g(1) == 1  # hits f's cache
    assert calls == [('f', 1)]


def test_cached_typed():
    cache = {}

    @cached(cache, typed=True)
    def f(x):
        return x

    assert f(3) == 3
    assert f(3.0) == 3.0
    assert len(cache) == 2


def test_cached_custom_key():
    cache = {}

    def key_func(args, kwargs, typed=False):
        return args[0]

    @cached(cache, key=key_func)
    def f(x):
        return x

    assert f(5) == 5
    assert f(5) == 5
    assert cache == {5: 5}


def test_cached_function_repr():
    def f(x):
        return x

    cf = CachedFunction(f, {})
    r = repr(cf)
    assert r.startswith('CachedFunction(func=')
    assert 'scoped' not in r

    cf2 = CachedFunction(f, {}, scoped=False)
    r2 = repr(cf2)
    assert 'scoped=False' in r2
    assert 'typed=False' in r2

    cf3 = CachedFunction(f, {}, typed=True)
    assert 'typed=True' in repr(cf3)


def test_cached_function_kwargs():
    cache = {}
    calls = []

    @cached(cache)
    def f(a, b=1):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    assert calls == [(1, 2)]


# ---------------------------------------------------------------------------
# CachedMethod / cachedmethod
# ---------------------------------------------------------------------------

class _Lowerer:
    def __init__(self):
        self.cache = LRI()
        self.calls = []

    @cachedmethod('cache')
    def lower(self, text):
        self.calls.append(text)
        return text.lower()


def test_cachedmethod_basic():
    lo = _Lowerer()
    assert lo.lower('ABC') == 'abc'
    assert lo.lower('ABC') == 'abc'
    assert lo.calls == ['ABC']
    assert len(lo.cache) == 1


def test_cachedmethod_attribute_str():
    lo = _Lowerer()
    m = lo.lower
    assert isinstance(m, CachedMethod)
    assert m.bound_to is lo


def test_cachedmethod_method_no_cache_hit_on_other_instance():
    a = _Lowerer()
    b = _Lowerer()
    assert a.lower('X') == 'x'
    assert b.lower('X') == 'x'
    assert a.calls == ['X']
    assert b.calls == ['X']


def test_cachedmethod_invalid_cache():
    with pytest.raises(TypeError):
        CachedMethod(lambda self: None, cache=5)


def test_cachedmethod_unscoped():
    class C:
        def __init__(self):
            self.cache = {}
            self.calls = []

        @cachedmethod(lambda self: self.cache, scoped=False)
        def method(self, x):
            self.calls.append(x)
            return x

    c = C()
    assert c.method(1) == 1
    assert c.method(1) == 1
    assert c.calls == [1]


def test_cachedmethod_class_access():
    m = _Lowerer.lower
    assert isinstance(m, CachedMethod)
    assert m.bound_to is None


def test_cachedmethod_call_unbound():
    lo = _Lowerer()
    m = _Lowerer.lower
    # unbound: first arg is obj
    assert m(lo, 'ABC') == 'abc'
    assert m(lo, 'ABC') == 'abc'
    assert lo.calls == ['ABC']


def test_cachedmethod_repr_bound_and_unbound():
    lo = _Lowerer()
    m = lo.lower
    assert repr(m).startswith('<CachedMethod func=')
    assert 'bound_to=' in repr(m)

    unbound = _Lowerer.lower
    r = repr(unbound)
    assert r.startswith('CachedMethod(func=')
    assert 'bound_to=' not in r


def test_cachedmethod_isabstractmethod():
    class C:
        def __init__(self):
            self.cache = {}

        @cachedmethod('cache')
        def method(self, x):
            return x

    m = C.method
    assert m.__isabstractmethod__ is False


# ---------------------------------------------------------------------------
# cachedproperty
# ---------------------------------------------------------------------------

def test_cachedproperty():
    calls = []

    class C:
        @cachedproperty
        def value(self):
            calls.append(1)
            return 42

    c = C()
    assert c.value == 42
    assert c.value == 42
    assert calls == [1]
    # stored in instance __dict__
    assert c.__dict__['value'] == 42
    # clearable
    del c.__dict__['value']
    assert c.value == 42
    assert calls == [1, 1]


def test_cachedproperty_class_access():
    class C:
        @cachedproperty
        def value(self):
            """docstring"""
            return 1

    cp = C.value
    assert isinstance(cp, cachedproperty)
    assert cp.__doc__ == 'docstring'
    assert 'cachedproperty' in repr(cp)


def test_cachedproperty_isabstractmethod():
    class C:
        @cachedproperty
        def value(self):
            return 1

    assert C.value.__isabstractmethod__ is False


# ---------------------------------------------------------------------------
# ThresholdCounter
# ---------------------------------------------------------------------------

def test_threshold_counter_invalid():
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=0)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=1)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=-0.5)


def test_threshold_counter_docstring():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    tc.update([2] * 10)
    assert tc.get(1) == 0
    tc.add(5)
    assert 5 in tc
    assert len(list(tc.elements())) == 11


def test_threshold_counter_add_and_get():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')
    tc.add('a')
    tc.add('b')
    assert tc['a'] == 2
    assert tc['b'] == 1
    assert tc.get('z') == 0
    assert tc.get('z', 99) == 99


def test_threshold_counter_len_contains_keys():
    tc = ThresholdCounter(threshold=0.5)
    tc.add(1)
    tc.add(2)
    assert len(tc) == 2
    assert 1 in tc
    assert 3 not in tc
    assert set(tc.keys()) == {1, 2}
    assert sorted(tc.values()) == [1, 1]
    assert sorted(tc.items()) == [(1, 1), (2, 1)]


def test_threshold_counter_iterkeys_values_items():
    tc = ThresholdCounter(threshold=0.5)
    tc.update([1, 1, 2])
    assert set(tc.iterkeys()) == {1, 2}
    assert sorted(tc.itervalues()) == [1, 2]
    assert dict(tc.iteritems()) == {1: 2, 2: 1}


def test_threshold_counter_update_mapping_and_kwargs():
    tc = ThresholdCounter(threshold=0.5)
    tc.update({'a': 3, 'b': 1})
    assert tc['a'] == 3
    assert tc['b'] == 1
    tc.update(None, c=2)
    assert tc['c'] == 2


def test_threshold_counter_most_common():
    tc = ThresholdCounter(threshold=0.5)
    tc.update([1, 1, 1, 2, 2, 3])
    mc = tc.most_common()
    assert mc[0] == (1, 3)
    assert tc.most_common(1) == [(1, 3)]
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []
    assert tc.most_common(100) == mc


def test_threshold_counter_commonality():
    tc = ThresholdCounter(threshold=0.5)
    tc.update([1, 1, 1, 1])
    assert tc.get_common_count() == 4
    assert tc.get_uncommon_count() == 0
    assert tc.get_commonality() == 1.0


def test_threshold_counter_threshold_property():
    tc = ThresholdCounter(threshold=0.25)
    assert tc.threshold == 0.25


def test_threshold_counter_elements():
    tc = ThresholdCounter(threshold=0.5)
    tc.update([1, 1, 2])
    elements = sorted(tc.elements())
    assert elements == [1, 1, 2]


# ---------------------------------------------------------------------------
# MinIDMap
# ---------------------------------------------------------------------------

class _Obj:
    pass


def test_min_id_map_basic():
    m = MinIDMap()
    a = _Obj()
    b = _Obj()
    ia = m.get(a)
    ib = m.get(b)
    assert ia == 0
    assert ib == 1
    assert m.get(a) == 0
    assert a in m
    assert b in m
    assert len(m) == 2


def test_min_id_map_iter_items():
    m = MinIDMap()
    a = _Obj()
    b = _Obj()
    m.get(a)
    m.get(b)
    items = dict(m.iteritems())
    assert items[a] == 0
    assert items[b] == 1
    assert set(m) == {a, b}


def test_min_id_map_drop_reuses_id():
    m = MinIDMap()
    a = _Obj()
    b = _Obj()
    m.get(a)  # 0
    m.get(b)  # 1
    m.drop(a)
    assert a not in m
    assert len(m) == 1
    c = _Obj()
    assert m.get(c) == 0


def test_min_id_map_clean_on_gc(capsys):
    m = MinIDMap()
    a = _Obj()
    m.get(a)  # 0
    b = _Obj()
    m.get(b)  # 1
    del a
    gc.collect()
    capsys.readouterr()  # swallow the print from _clean
    c = _Obj()
    # 0 was freed by cleanup
    assert m.get(c) == 0
