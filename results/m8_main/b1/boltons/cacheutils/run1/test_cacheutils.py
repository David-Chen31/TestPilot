import gc
import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    make_cache_key,
    _make_cache_key,
    _HashedKey,
    CachedFunction,
    CachedMethod,
    cached,
    cachedmethod,
    cachedproperty,
    ThresholdCounter,
    MinIDMap,
    DEFAULT_MAX_SIZE,
    _KWARG_MARK,
    _MISSING,
)


# ---------------------------------------------------------------------------
# LRI
# ---------------------------------------------------------------------------

def test_lri_basic_set_get():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    assert c['a'] == 1
    assert c['b'] == 2
    assert dict(c) == {'a': 1, 'b': 2}


def test_lri_invalid_max_size():
    with pytest.raises(ValueError):
        LRI(max_size=0)
    with pytest.raises(ValueError):
        LRI(max_size=-5)


def test_lri_invalid_on_miss():
    with pytest.raises(TypeError):
        LRI(on_miss=42)


def test_lri_default_max_size():
    c = LRI()
    assert c.max_size == DEFAULT_MAX_SIZE
    assert c.hit_count == 0
    assert c.miss_count == 0
    assert c.soft_miss_count == 0


def test_lri_eviction_least_recently_inserted():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert c['a'] == 'A'
    c['c'] = 'C'
    assert 'a' not in c
    assert dict(c) == {'b': 'B', 'c': 'C'}


def test_lri_update_insertion_order_eviction():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    c['a'] = 'A2'
    c['c'] = 'C'
    assert 'b' not in c
    assert c['a'] == 'A2'
    assert c['c'] == 'C'


def test_lri_missing_raises_keyerror_and_counts():
    c = LRI()
    with pytest.raises(KeyError):
        c['nope']
    assert c.miss_count == 1
    assert c.hit_count == 0


def test_lri_on_miss():
    c = LRI(on_miss=lambda k: k.upper())
    assert c['a'] == 'A'
    assert c['a'] == 'A'
    assert c.miss_count == 1
    assert c.hit_count == 1
    assert 'a' in c


def test_lri_get_and_soft_miss():
    c = LRI()
    c['a'] = 1
    assert c.get('a') == 1
    assert c.get('b') is None
    assert c.get('b', 42) == 42
    assert c.hit_count == 1
    # each get() miss goes through __getitem__ incrementing miss_count,
    # and get() also increments soft_miss_count
    assert c.miss_count == 2
    assert c.soft_miss_count == 2


def test_lri_getitem_does_not_move_in_lri():
    c = LRI(max_size=2)
    c['a'] = 1
    c['b'] = 2
    c['a']
    c['c'] = 3
    assert 'a' not in c
    assert 'b' in c


def test_lri_delete():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    del c['a']
    assert 'a' not in c
    assert dict(c) == {'b': 2}
    with pytest.raises(KeyError):
        del c['a']


def test_lri_pop():
    c = LRI()
    c['a'] = 1
    assert c.pop('a') == 1
    assert 'a' not in c
    assert c.pop('a', 'default') == 'default'
    with pytest.raises(KeyError):
        c.pop('a')


def test_lri_pop_no_stat_counts():
    c = LRI()
    c['a'] = 1
    c.pop('a')
    c.pop('missing', None)
    assert c.hit_count == 0
    assert c.miss_count == 0
    assert c.soft_miss_count == 0


def test_lri_popitem():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    k, v = c.popitem()
    assert (k, v) in [('a', 1), ('b', 2)]
    assert len(c) == 1


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
    c['b'] = 2
    c2 = c.copy()
    assert isinstance(c2, LRI)
    assert c2.max_size == 5
    assert dict(c2) == {'a': 1, 'b': 2}
    c2['c'] = 3
    assert 'c' not in c


def test_lri_setdefault():
    c = LRI()
    assert c.setdefault('a', 1) == 1
    assert c.setdefault('a', 10) == 1
    assert c['a'] == 1
    assert c.soft_miss_count == 1


def test_lri_update_mapping_and_iterable():
    c = LRI()
    c.update({'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}
    c.update([('c', 3), ('d', 4)])
    assert dict(c) == {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    c.update(e=5)
    assert c['e'] == 5


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
    # Comparing an LRI against a plain dict currently recurses infinitely
    # (LRI.__eq__ calls ``other == self`` which dispatches back to
    # LRI.__eq__). Assert that the recursion actually happens.
    with pytest.raises(RecursionError):
        c1 == {'a': 1}
    assert c1 == c1
    assert (c1 != c2) is False
    c2['b'] = 2
    assert c1 != c2


def test_lri_repr():
    c = LRI(max_size=3, on_miss=None)
    c['a'] = 1
    r = repr(c)
    assert 'LRI' in r
    assert 'max_size=3' in r
    assert "'a': 1" in r


def test_lri_init_with_values():
    c = LRI(max_size=5, values={'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}
    assert c.max_size == 5


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

def test_lru_basic():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert c['a'] == 'A'
    c['c'] = 'C'
    assert 'b' not in c
    assert 'a' in c
    assert 'c' in c


def test_lru_eviction_least_recently_used():
    c = LRU(max_size=2)
    c['a'] = 1
    c['b'] = 2
    c['a']
    c['c'] = 3
    assert 'b' not in c


def test_lru_miss_and_on_miss():
    c = LRU(max_size=2, on_miss=lambda k: k * 2)
    assert c['x'] == 'xx'
    assert c.miss_count == 1
    assert c['x'] == 'xx'
    assert c.hit_count == 1


def test_lru_get():
    c = LRU()
    c['a'] = 1
    assert c.get('a') == 1
    assert c.get('zzz') is None
    assert c.soft_miss_count == 1


def test_lru_stats():
    c = LRU(max_size=2)
    c['a'], c['b'] = 'A', 'B'
    [c['b'] for _ in range(3)]
    c['c'] = 'C'
    assert c.get('a') is None
    assert (c.hit_count, c.miss_count, c.soft_miss_count) == (3, 1, 1)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_basic():
    k = make_cache_key(('a', 'b'), {'c': 'd'})
    assert tuple(k) == ('a', 'b', _KWARG_MARK, ('c', 'd'))


def test_make_cache_key_no_kwargs():
    k = make_cache_key((), {})
    assert isinstance(k, _HashedKey)
    assert tuple(k) == ()


def test_make_cache_key_single_fasttype():
    assert make_cache_key((5,), {}) == 5
    assert make_cache_key(('hello',), {}) == 'hello'
    assert make_cache_key((None,), {}) is None


def test_make_cache_key_single_slowtype():
    class Foo:
        def __hash__(self):
            return 1
    f = Foo()
    k = make_cache_key((f,), {})
    assert isinstance(k, _HashedKey)
    assert k[0] is f


def test_make_cache_key_typed():
    k1 = make_cache_key((3,), {}, typed=True)
    k2 = make_cache_key((3.0,), {}, typed=True)
    assert k1 != k2
    assert hash(k1) != hash(k2) or k1 != k2


def test_make_cache_key_typed_same_type_equal():
    k1 = make_cache_key((3,), {}, typed=True)
    k2 = make_cache_key((3,), {}, typed=True)
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_make_cache_key_typed_with_kwargs():
    k1 = make_cache_key((), {'a': 1}, typed=True)
    k2 = make_cache_key((), {'a': 1.0}, typed=True)
    assert k1 != k2


def test_make_cache_key_kwargs_sorted():
    k1 = make_cache_key((), {'a': 1, 'b': 2})
    k2 = make_cache_key((), {'b': 2, 'a': 1})
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_make_cache_key_backwards_compat_alias():
    assert _make_cache_key is make_cache_key


def test_hashed_key_repr():
    hk = _HashedKey([1, 2])
    assert 'HashedKey' in repr(hk) or '_HashedKey' in repr(hk)
    assert '1' in repr(hk) and '2' in repr(hk)


def test_hashed_key_hash_cached():
    hk = _HashedKey((1, 2))
    assert hk.hash_value == hash((1, 2))
    assert hash(hk) == hash((1, 2))


# ---------------------------------------------------------------------------
# CachedFunction / cached
# ---------------------------------------------------------------------------

def test_cached_function_basic():
    calls = []

    @cached({})
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_cached_with_lru_shared_cache():
    cache = LRU()
    calls = []

    @cached(cache)
    def f(x):
        calls.append(x)
        return x + 1

    assert f(1) == 2
    assert f(1) == 2
    assert len(cache) == 1
    assert calls == [1]


def test_cached_scoped_true_shares_key_between_functions():
    # Note: CachedFunction does not include the function itself in the key,
    # so two functions with the same arguments share cache entries.
    cache = {}
    calls = []

    @cached(cache)
    def f(x):
        calls.append(('f', x))
        return x

    @cached(cache)
    def g(x):
        calls.append(('g', x))
        return x

    f(1)
    g(1)
    assert calls == [('f', 1)]


def test_cached_scoped_false_shares():
    cache = {}
    calls = []

    @cached(cache, scoped=False)
    def f(x):
        calls.append(('f', x))
        return x

    @cached(cache, scoped=False)
    def g(x):
        calls.append(('g', x))
        return x

    assert f(1) == 1
    assert g(1) == 1
    assert calls == [('f', 1)]


def test_cached_with_cache_callable():
    store = {}

    def get_cache():
        return store

    @cached(get_cache)
    def f(x):
        return x * 3

    assert f(2) == 6
    assert len(store) == 1


def test_cached_typed_option():
    cache = {}
    calls = []

    @cached(cache, typed=True)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1.0)
    assert len(calls) == 2


def test_cached_invalid_cache():
    with pytest.raises(TypeError):
        @cached(42)
        def f(x):
            return x


def test_cached_function_repr():
    def f(x):
        return x

    cf = CachedFunction(f, {}, typed=False, scoped=True)
    r = repr(cf)
    assert 'CachedFunction' in r
    cf2 = CachedFunction(f, {}, typed=True)
    assert 'typed=True' in repr(cf2)
    cf3 = CachedFunction(f, {}, scoped=False)
    assert 'scoped=False' in repr(cf3)


def test_cached_custom_key():
    cache = {}

    @cached(cache, key=lambda args, kwargs, typed: args[0])
    def f(x):
        return x * 10

    assert f(5) == 50
    assert f(5) == 50
    assert 5 in cache


def test_cached_function_keyerror_from_body_propagates():
    @cached({})
    def f():
        raise KeyError('inside')

    with pytest.raises(KeyError):
        f()


# ---------------------------------------------------------------------------
# CachedMethod / cachedmethod
# ---------------------------------------------------------------------------

def test_cachedmethod_attribute_name():
    class Lowerer:
        def __init__(self):
            self.cache = LRI()

        @cachedmethod('cache')
        def lower(self, text):
            return text.lower()

    lo = Lowerer()
    assert lo.lower('ABC') == 'abc'
    assert len(lo.cache) == 1
    assert lo.lower('ABC') == 'abc'
    assert len(lo.cache) == 1


def test_cachedmethod_dict_cache():
    cache = {}

    class Cls:
        @cachedmethod(cache)
        def double(self, x):
            return x * 2

    c = Cls()
    assert c.double(2) == 4
    assert len(cache) == 1


def test_cachedmethod_scoped_true_separates_instances():
    cache = {}

    class Cls:
        @cachedmethod(cache, scoped=True)
        def f(self, x):
            return x

    a = Cls()
    b = Cls()
    a.f(1)
    b.f(1)
    assert len(cache) == 2


def test_cachedmethod_scoped_false_shares():
    cache = {}
    calls = []

    class Cls:
        @cachedmethod(cache, scoped=False)
        def f(self, x):
            calls.append(x)
            return x

    a = Cls()
    b = Cls()
    assert a.f(1) == 1
    assert b.f(1) == 1
    assert calls == [1]


def test_cachedmethod_typed_option():
    cache = {}
    calls = []

    class Cls:
        @cachedmethod(cache, typed=True)
        def f(self, x):
            calls.append(x)
            return x

    c = Cls()
    c.f(1)
    c.f(1.0)
    assert len(calls) == 2


def test_cachedmethod_callable_cache():
    store = {}

    def make_cache(obj):
        return store

    class Cls:
        @cachedmethod(make_cache)
        def f(self, x):
            return x

    c = Cls()
    assert c.f(5) == 5
    assert len(store) == 1


def test_cachedmethod_unbound_descriptor_returns_self():
    def f(self):
        return 1

    cm = CachedMethod(f, {})
    assert cm.__get__(None) is cm


def test_cachedmethod_repr():
    def f(self):
        return 1

    cm = CachedMethod(f, {})
    r = repr(cm)
    assert 'CachedMethod' in r
    assert 'bound_to' not in r

    class Cls:
        method = CachedMethod(f, {})

    inst = Cls()
    bound = inst.method
    assert 'bound_to' in repr(bound)


def test_cachedmethod_abstractmethod_flag():
    def f(self):
        return 1

    f.__isabstractmethod__ = True
    cm = CachedMethod(f, {})
    assert cm.__isabstractmethod__ is True


# ---------------------------------------------------------------------------
# cachedproperty
# ---------------------------------------------------------------------------

def test_cachedproperty_basic():
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


def test_cachedproperty_clearable_via_delattr():
    calls = []

    class Foo:
        @cachedproperty
        def value(self):
            calls.append(1)
            return len(calls)

    f = Foo()
    assert f.value == 1
    del f.__dict__['value']
    assert f.value == 2


def test_cachedproperty_class_access_returns_descriptor():
    class Foo:
        @cachedproperty
        def value(self):
            return 1

    assert isinstance(Foo.value, cachedproperty)


def test_cachedproperty_repr():
    class Foo:
        @cachedproperty
        def value(self):
            return 1

    r = repr(Foo.value)
    assert 'cachedproperty' in r
    assert 'value' in r


def test_cachedproperty_doc():
    class Foo:
        @cachedproperty
        def value(self):
            "the docs"
            return 1

    assert Foo.value.__doc__ == 'the docs'


# ---------------------------------------------------------------------------
# ThresholdCounter
# ---------------------------------------------------------------------------

def test_thresholdcounter_invalid_threshold():
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=0)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=1)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=-0.5)


def test_thresholdcounter_basic():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    tc.update([2] * 10)
    assert tc.get(1) == 0
    tc.add(5)
    assert 5 in tc
    assert len(list(tc.elements())) == 11


def test_thresholdcounter_threshold_property():
    tc = ThresholdCounter(threshold=0.5)
    assert tc.threshold == 0.5


def test_thresholdcounter_get_default():
    tc = ThresholdCounter()
    assert tc.get('nope') == 0
    assert tc.get('nope', 99) == 99


def test_thresholdcounter_keys_values_items():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')
    tc.add('a')
    tc.add('b')
    assert tc.keys() == ['a', 'b'] or set(tc.keys()) == {'a', 'b'}
    assert set(tc.values()) == {2, 1}
    assert set(tc.items()) == {('a', 2), ('b', 1)}
    assert list(tc.iterkeys()) == tc.keys()
    assert list(tc.itervalues()) == tc.values()
    assert list(tc.iteritems()) == tc.items()


def test_thresholdcounter_most_common_empty_after_compaction():
    # With threshold=0.5, all items get culled during compaction
    tc = ThresholdCounter(threshold=0.5)
    tc.update(['a', 'a', 'a', 'b', 'b', 'c'])
    assert tc.most_common() == []
    assert tc.most_common(1) == []
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []


def test_thresholdcounter_most_common_small_threshold():
    tc = ThresholdCounter(threshold=0.1)
    tc.update(['a', 'a', 'a', 'b', 'b', 'c'])
    mc = tc.most_common()
    assert ('a', 3) in mc
    assert tc.most_common(1) in ([('a', 3)], [('b', 2)])
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []


def test_thresholdcounter_len_contains():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('x')
    assert len(tc) == 1
    assert 'x' in tc
    assert 'y' not in tc


def test_thresholdcounter_getitem_missing():
    tc = ThresholdCounter()
    with pytest.raises(KeyError):
        tc['missing']


def test_thresholdcounter_common_uncommon_counts():
    tc = ThresholdCounter(threshold=0.5)
    tc.update(['a', 'a', 'a'])
    assert tc.total == 3
    assert tc.get_common_count() == 3
    assert tc.get_uncommon_count() == 0
    assert tc.get_commonality() == pytest.approx(1.0)


def test_thresholdcounter_update_mapping():
    # threshold=0.5 -> _thresh_count=2, compaction every 2 additions.
    # Feeding {'a': 3, 'b': 2} iterates a first (3 adds), then b (2 adds).
    # During add#4 (b's first add) a compaction fires; at that point
    # 'b' has sum(v) == 1 + 1 == 2 which is NOT > _cur_bucket(2), so 'b'
    # is evicted. Its subsequent add re-introduces it at count 1.
    tc = ThresholdCounter(threshold=0.5)
    tc.update({'a': 3, 'b': 2})
    assert tc['a'] == 3
    assert 'b' in tc
    assert tc['b'] == 1
    assert tc.total == 5


def test_thresholdcounter_update_kwargs():
    tc = ThresholdCounter(threshold=0.5)
    tc.update(None, a=2, b=3)
    assert 'a' not in tc
    assert tc['b'] == 3
    assert tc.total == 5


def test_thresholdcounter_compaction_removes_rare():
    tc = ThresholdCounter(threshold=0.5)
    # _thresh_count = 2
    tc.add('rare')
    tc.add('common')
    # After 2 additions, compaction happens; both count 1 not > cur_bucket(1)
    assert 'rare' not in tc
    assert 'common' not in tc
    assert len(tc) == 0


def test_thresholdcounter_elements_multiple():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')
    tc.add('a')
    elements = list(tc.elements())
    assert elements == ['a', 'a']


# ---------------------------------------------------------------------------
# MinIDMap
# ---------------------------------------------------------------------------

def test_minidmap_basic():
    class Obj:
        pass

    m = MinIDMap()
    a = Obj()
    b = Obj()
    ida = m.get(a)
    idb = m.get(b)
    assert ida == 0
    assert idb == 1
    assert m.get(a) == 0
    assert m.get(b) == 1


def test_minidmap_contains_and_len_and_iter():
    class Obj:
        pass

    m = MinIDMap()
    a = Obj()
    b = Obj()
    m.get(a)
    m.get(b)
    assert a in m
    assert b in m
    assert len(m) == 2
    assert set(iter(m)) == {a, b}


def test_minidmap_drop_and_reuse():
    class Obj:
        pass

    m = MinIDMap()
    a = Obj()
    b = Obj()
    ida = m.get(a)
    idb = m.get(b)
    assert (ida, idb) == (0, 1)
    m.drop(a)
    assert a not in m
    assert len(m) == 1
    c = Obj()
    idc = m.get(c)
    assert idc == 0


def test_minidmap_drop_missing():
    class Obj:
        pass

    m = MinIDMap()
    a = Obj()
    with pytest.raises(KeyError):
        m.drop(a)


def test_minidmap_iteritems():
    class Obj:
        pass

    m = MinIDMap()
    a = Obj()
    b = Obj()
    m.get(a)
    m.get(b)
    items = dict(m.iteritems())
    assert items[a] == 0
    assert items[b] == 1


def test_minidmap_cleanup_on_gc(capsys):
    class Obj:
        pass

    m = MinIDMap()
    a = Obj()
    ida = m.get(a)
    assert ida == 0
    del a
    gc.collect()
    # _clean prints the freed id (side effect of the current implementation)
    captured = capsys.readouterr()
    assert '0' in captured.out
    assert len(m) == 0
    b = Obj()
    assert m.get(b) == 0
