import gc
import weakref

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


# ---------------------------------------------------------------------------
# LRI / LRU
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [LRI, LRU])
def test_init_bad_max_size(cls):
    with pytest.raises(ValueError):
        cls(max_size=0)
    with pytest.raises(ValueError):
        cls(max_size=-5)


@pytest.mark.parametrize("cls", [LRI, LRU])
def test_init_bad_on_miss(cls):
    with pytest.raises(TypeError):
        cls(on_miss=5)


@pytest.mark.parametrize("cls", [LRI, LRU])
def test_init_values(cls):
    c = cls(max_size=5, values={'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}
    assert c.max_size == 5
    assert c.hit_count == 0
    assert c.miss_count == 0
    assert c.soft_miss_count == 0


@pytest.mark.parametrize("cls", [LRI, LRU])
def test_get_set_and_counters(cls):
    c = cls()
    c['a'] = 1
    assert c['a'] == 1
    assert c.hit_count == 1
    with pytest.raises(KeyError):
        c['b']
    assert c.miss_count == 1


def test_lri_evicts_least_recently_inserted():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    # reading doesn't change insertion order for LRI
    assert c['a'] == 'A'
    c['c'] = 'C'
    assert 'a' not in c
    assert dict(c) == {'b': 'B', 'c': 'C'}


def test_lru_evicts_least_recently_used():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    # reading a makes b the least recently used
    assert c['a'] == 'A'
    c['c'] = 'C'
    assert 'b' not in c
    assert dict(c) == {'a': 'A', 'c': 'C'}


def test_lri_overwrite_existing_value():
    c = LRI(max_size=2)
    c['a'] = 1
    c['a'] = 2
    assert c['a'] == 2
    assert len(c) == 1


def test_on_miss():
    calls = []

    def on_miss(key):
        calls.append(key)
        return key * 2

    for cls in (LRI, LRU):
        c = cls(on_miss=on_miss)
        assert c['x'] == 'xx'
        assert 'x' in c
        assert calls == ['x']
        assert c.miss_count == 1


def test_get_soft_miss():
    c = LRI()
    assert c.get('nope') is None
    assert c.get('nope2', 'dflt') == 'dflt'
    assert c.soft_miss_count == 2
    # get hits don't affect soft_miss_count
    c['k'] = 'v'
    assert c.get('k') == 'v'
    assert c.soft_miss_count == 2
    assert c.hit_count == 1


def test_setdefault():
    c = LRI()
    assert c.setdefault('a', 5) == 5
    assert c['a'] == 5
    assert c.soft_miss_count == 1
    assert c.setdefault('a', 99) == 5
    assert c.soft_miss_count == 1


def test_delitem_and_pop_and_popitem():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    del c['a']
    assert 'a' not in c
    assert dict(c) == {'b': 2}

    assert c.pop('b') == 2
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.pop('missing')
    assert c.pop('missing', 'default') == 'default'

    c['x'] = 10
    c['y'] = 20
    item = c.popitem()
    assert item in (('x', 10), ('y', 20))
    assert len(c) == 1


def test_clear_resets():
    c = LRI()
    c['a'] = 1
    c.clear()
    assert dict(c) == {}
    assert len(c) == 0
    c['b'] = 2
    assert c['b'] == 2


def test_copy():
    c = LRI(max_size=5)
    c['a'] = 1
    c2 = c.copy()
    assert isinstance(c2, LRI)
    assert dict(c2) == {'a': 1}
    assert c2.max_size == 5
    # independent copy
    c2['b'] = 2
    assert 'b' not in c


def test_update():
    c = LRI()
    c.update({'a': 1}, b=2)
    assert dict(c) == {'a': 1, 'b': 2}
    c.update([('c', 3)])
    assert c['c'] == 3
    # update from self should not blow up
    c.update(c)
    assert dict(c) == {'a': 1, 'b': 2, 'c': 3}


def test_eq_ne():
    a = LRI()
    a['x'] = 1
    b = LRI()
    b['x'] = 1
    assert a == b
    assert not (a != b)
    assert a == {'x': 1}
    assert a != {'x': 2}
    b['y'] = 2
    assert a != b


def test_repr():
    c = LRI(max_size=3)
    c['a'] = 1
    r = repr(c)
    assert 'LRI' in r
    assert 'max_size=3' in r
    assert "'a': 1" in r


def test_flattened_ll_order():
    c = LRI(max_size=10)
    c['a'] = 1
    c['b'] = 2
    # flattened list includes anchor (MISSING) at ends; keys in order
    flat = c._get_flattened_ll()
    keys = [entry[0] for entry in flat]
    # 'a' then 'b'
    idx_a = keys.index('a')
    idx_b = keys.index('b')
    assert idx_a < idx_b


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_basic():
    key = make_cache_key(('a', 'b'), {'c': 'd'})
    assert tuple(key) == ('a', 'b', key[-2], ('c', 'd'))


def test_make_cache_key_fast_type_single_arg():
    # single arg of fasttype returned unwrapped
    assert make_cache_key((5,), {}) == 5
    assert make_cache_key(('hello',), {}) == 'hello'


def test_make_cache_key_non_fast_type():
    key = make_cache_key(([1, 2],), {})
    assert isinstance(key, _HashedKey)
    assert list(key) == [[1, 2]]


def test_make_cache_key_typed():
    k1 = make_cache_key((3,), {}, typed=True)
    k2 = make_cache_key((3.0,), {}, typed=True)
    assert k1 != k2


def test_make_cache_key_untyped_equality():
    k1 = make_cache_key((3,), {}, typed=False)
    k2 = make_cache_key((3.0,), {}, typed=False)
    assert hash(k1) == hash(k2)
    assert k1 == k2


def test_make_cache_key_kwargs_sorted():
    k = make_cache_key((), {'b': 2, 'a': 1})
    # sorted items after kwarg_mark
    assert tuple(k)[1:] == (('a', 1), ('b', 2))


def test_hashed_key_hash_cached():
    hk = _HashedKey([1, 2, 3])
    assert hash(hk) == hash((1, 2, 3))
    assert 'hash_value' in _HashedKey.__slots__ or hasattr(hk, 'hash_value')
    assert repr(hk).startswith('_HashedKey(')


# ---------------------------------------------------------------------------
# CachedFunction / cached
# ---------------------------------------------------------------------------

def test_cached_basic():
    calls = []

    @cached({})
    def add(a, b):
        calls.append((a, b))
        return a + b

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_cached_uses_lru():
    cache = LRU()
    calls = []

    @cached(cache)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]
    assert len(cache) == 1


def test_cached_callable_cache():
    store = {}

    @cached(lambda: store)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert len(store) == 1


def test_cached_typed():
    calls = []

    @cached({}, typed=True)
    def f(x):
        calls.append(x)
        return x

    f(3)
    f(3.0)
    assert len(calls) == 2


def test_cached_key_func():
    calls = []

    def key_func(args, kwargs, typed=False):
        return make_cache_key(args, {}, typed=False)

    @cached({}, key=key_func)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert len(calls) == 1


def test_cached_false_scoped_repr():
    def f(x):
        return x

    cf = CachedFunction(f, {}, scoped=False)
    r = repr(cf)
    assert 'scoped=False' in r


def test_cachedfunction_bad_cache():
    with pytest.raises(TypeError):
        CachedFunction(lambda x: x, 123)


def test_cached_function_repr():
    def f(x):
        return x

    cf = CachedFunction(f, {})
    r = repr(cf)
    assert 'CachedFunction' in r
    assert 'func=' in r


# ---------------------------------------------------------------------------
# CachedMethod / cachedmethod
# ---------------------------------------------------------------------------

def test_cachedmethod_attr_name():
    class Foo:
        def __init__(self):
            self.cache = LRI()

        @cachedmethod('cache')
        def bar(self, x):
            return x * 10

    foo = Foo()
    assert foo.bar(2) == 20
    assert foo.bar(2) == 20
    assert len(foo.cache) == 1


def test_cachedmethod_dict_cache():
    cache = {}

    class Foo:
        @cachedmethod(cache)
        def bar(self, x):
            return x + 1

    foo = Foo()
    assert foo.bar(1) == 2
    assert foo.bar(1) == 2
    assert len(cache) == 1


def test_cachedmethod_callable_cache():
    class Foo:
        def __init__(self):
            self.cache = {}

        def get_cache(self, obj):
            return obj.cache

        @cachedmethod(get_cache)
        def bar(self, x):
            return x

    foo = Foo()
    assert foo.bar(5) == 5
    assert len(foo.cache) == 1


def test_cachedmethod_bad_cache():
    with pytest.raises(TypeError):
        CachedMethod(lambda self: None, 123)


def test_cachedmethod_unbound_access():
    class Foo:
        @cachedmethod('cache')
        def bar(self, x):
            return x

    # accessing on the class returns the descriptor itself
    assert isinstance(Foo.__dict__['bar'], CachedMethod)


def test_cachedmethod_scoped_false_shares_cache():
    cache = {}

    class Foo:
        @cachedmethod(cache, scoped=False)
        def a(self, x):
            return ('a', x)

        @cachedmethod(cache, scoped=False)
        def b(self, x):
            return ('b', x)

    foo = Foo()
    foo.a(1)
    # shared cache means b reuses a's entry for same key
    assert foo.b(1) == ('a', 1)


def test_cachedmethod_repr():
    def method(self, x):
        return x

    cm = CachedMethod(method, 'cache')
    r = repr(cm)
    assert 'CachedMethod' in r
    assert 'scoped=True' in r


def test_cachedmethod_isabstractmethod():
    def method(self):
        pass

    method.__isabstractmethod__ = True
    cm = CachedMethod(method, 'cache')
    assert cm.__isabstractmethod__ is True


# ---------------------------------------------------------------------------
# cachedproperty
# ---------------------------------------------------------------------------

def test_cachedproperty():
    calls = []

    class Foo:
        @cachedproperty
        def val(self):
            calls.append(1)
            return 42

    foo = Foo()
    assert foo.val == 42
    assert foo.val == 42
    assert len(calls) == 1
    # stored in instance __dict__
    assert foo.__dict__['val'] == 42


def test_cachedproperty_delattr_clears():
    calls = []

    class Foo:
        @cachedproperty
        def val(self):
            calls.append(1)
            return len(calls)

    foo = Foo()
    assert foo.val == 1
    del foo.val
    assert foo.val == 2


def test_cachedproperty_class_access():
    class Foo:
        @cachedproperty
        def val(self):
            return 1

    assert isinstance(Foo.val, cachedproperty)


def test_cachedproperty_doc():
    class Foo:
        @cachedproperty
        def val(self):
            "my doc"
            return 1

    assert Foo.__dict__['val'].__doc__ == "my doc"


def test_cachedproperty_repr():
    class Foo:
        @cachedproperty
        def val(self):
            return 1

    r = repr(Foo.__dict__['val'])
    assert 'cachedproperty' in r
    assert 'func=' in r


# ---------------------------------------------------------------------------
# ThresholdCounter
# ---------------------------------------------------------------------------

def test_threshold_counter_bad_threshold():
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=0)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=1)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=-0.5)


def test_threshold_counter_basic():
    tc = ThresholdCounter(threshold=0.5)
    assert tc.threshold == 0.5
    tc.add(1)
    assert tc[1] == 1
    assert tc.get(1) == 1
    assert tc.get(99) == 0
    assert 1 in tc
    assert 99 not in tc
    assert len(tc) == 1


def test_threshold_counter_compaction():
    tc = ThresholdCounter(threshold=0.5)  # thresh_count = 2
    tc.add('x')
    tc.add('y')
    # after two adds compaction occurs; both dropped (count 1 <= bucket)
    assert len(tc) == 0
    assert tc.total == 2


def test_threshold_counter_keeps_common():
    tc = ThresholdCounter(threshold=0.5)
    for _ in range(2):
        tc.add('common')
    tc.add('rare')
    tc.add('rare')
    # 'common' count >= bucket survives compaction
    assert 'common' in tc


def test_threshold_counter_update_iterable():
    tc = ThresholdCounter(threshold=0.01)
    tc.update([1, 1, 2])
    assert tc[1] == 2
    assert tc[2] == 1
    assert tc.total == 3


def test_threshold_counter_update_mapping():
    tc = ThresholdCounter(threshold=0.01)
    tc.update({'a': 3, 'b': 1})
    assert tc['a'] == 3
    assert tc['b'] == 1
    assert tc.total == 4


def test_threshold_counter_update_kwargs():
    tc = ThresholdCounter(threshold=0.01)
    tc.update(None, x=2)
    assert tc['x'] == 2


def test_threshold_counter_most_common():
    tc = ThresholdCounter(threshold=0.001)
    tc.update({'a': 3, 'b': 5, 'c': 1})
    mc = tc.most_common()
    assert mc[0] == ('b', 5)
    assert mc[1] == ('a', 3)
    assert mc[2] == ('c', 1)
    assert tc.most_common(2) == mc[:2]
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []


def test_threshold_counter_elements():
    tc = ThresholdCounter(threshold=0.001)
    tc.update({'a': 2, 'b': 1})
    elements = list(tc.elements())
    assert elements.count('a') == 2
    assert elements.count('b') == 1
    assert len(elements) == 3


def test_threshold_counter_counts():
    tc = ThresholdCounter(threshold=0.001)
    tc.update({'a': 3, 'b': 2})
    assert tc.get_common_count() == 5
    assert tc.get_uncommon_count() == 0
    assert tc.get_commonality() == 1.0


def test_threshold_counter_keys_values_items():
    tc = ThresholdCounter(threshold=0.001)
    tc.update({'a': 1, 'b': 2})
    assert sorted(tc.keys()) == ['a', 'b']
    assert sorted(tc.values()) == [1, 2]
    assert sorted(tc.items()) == [('a', 1), ('b', 2)]
    assert sorted(tc.iterkeys()) == ['a', 'b']
    assert sorted(tc.itervalues()) == [1, 2]
    assert sorted(tc.iteritems()) == [('a', 1), ('b', 2)]


# ---------------------------------------------------------------------------
# MinIDMap
# ---------------------------------------------------------------------------

class _Obj:
    pass


def test_minidmap_basic():
    m = MinIDMap()
    a = _Obj()
    b = _Obj()
    ia = m.get(a)
    ib = m.get(b)
    assert ia == 0
    assert ib != ia
    assert a in m
    assert len(m) == 2
    assert m.get(a) == ia  # idempotent


def test_minidmap_drop_reuses_ids():
    m = MinIDMap()
    a = _Obj()
    b = _Obj()
    ia = m.get(a)
    m.get(b)
    m.drop(a)
    assert a not in m
    c = _Obj()
    # freed id reused (smallest available)
    assert m.get(c) == ia


def test_minidmap_iter_and_items():
    m = MinIDMap()
    a = _Obj()
    b = _Obj()
    m.get(a)
    m.get(b)
    assert set(id(k) for k in m) <= {id(a), id(b)}
    items = dict(m.iteritems())
    assert len(items) == 2


def test_minidmap_weakref_cleanup():
    m = MinIDMap()
    a = _Obj()
    ia = m.get(a)
    del a
    gc.collect()
    # after the object is collected, its id is freed
    c = _Obj()
    assert m.get(c) == ia
