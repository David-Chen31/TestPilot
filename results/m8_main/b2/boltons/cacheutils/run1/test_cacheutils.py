import gc
import weakref

import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    make_cache_key,
    CachedFunction,
    CachedMethod,
    cached,
    cachedmethod,
    cachedproperty,
    ThresholdCounter,
    MinIDMap,
    _HashedKey,
    _KWARG_MARK,
)


class _Weaky:
    """A weakref-able object usable as a key for MinIDMap."""


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
    assert c.hit_count == 2
    assert c.miss_count == 0


def test_lri_eviction_least_recently_inserted():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    # Accessing 'b' does NOT refresh insertion order for LRI
    assert c['b'] == 'B'
    c['c'] = 'C'
    # 'a' was inserted first -> evicted
    assert 'a' not in c
    assert dict(c) == {'b': 'B', 'c': 'C'}
    assert c.max_size == 2


def test_lri_update_on_existing_key():
    c = LRI(max_size=2)
    c['a'] = 'A'
    c['a'] = 'AA'
    assert c['a'] == 'AA'
    assert len(c) == 1


def test_lri_max_size_invalid():
    with pytest.raises(ValueError):
        LRI(max_size=0)
    with pytest.raises(ValueError):
        LRI(max_size=-1)


def test_lri_on_miss_non_callable():
    with pytest.raises(TypeError):
        LRI(on_miss=42)


def test_lri_on_miss_callable():
    calls = []

    def on_miss(key):
        calls.append(key)
        return key.upper()

    c = LRI(on_miss=on_miss)
    assert c['foo'] == 'FOO'
    assert calls == ['foo']
    assert c.miss_count == 1
    # Second access is a hit
    assert c['foo'] == 'FOO'
    assert c.hit_count == 1
    assert calls == ['foo']


def test_lri_getitem_missing_raises():
    c = LRI()
    with pytest.raises(KeyError):
        c['missing']
    assert c.miss_count == 1


def test_lri_get_default():
    c = LRI()
    assert c.get('missing') is None
    assert c.get('missing', 'dflt') == 'dflt'
    assert c.soft_miss_count == 2
    c['present'] = 1
    assert c.get('present') == 1
    assert c.hit_count == 1


def test_lri_delitem():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    del c['a']
    assert 'a' not in c
    assert dict(c) == {'b': 2}
    assert 'a' not in c._link_lookup
    with pytest.raises(KeyError):
        del c['a']


def test_lri_pop():
    c = LRI()
    c['a'] = 1
    assert c.pop('a') == 1
    assert 'a' not in c
    assert 'a' not in c._link_lookup
    with pytest.raises(KeyError):
        c.pop('a')
    assert c.pop('a', 'default') == 'default'


def test_lri_popitem():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    item = c.popitem()
    assert item in (('a', 1), ('b', 2))
    assert item[0] not in c
    assert item[0] not in c._link_lookup


def test_lri_clear():
    c = LRI()
    c['a'] = 1
    c['b'] = 2
    c.clear()
    assert dict(c) == {}
    assert c._link_lookup == {}


def test_lri_copy():
    c = LRI(max_size=5)
    c['a'] = 1
    c2 = c.copy()
    assert isinstance(c2, LRI)
    assert c2.max_size == 5
    assert dict(c2) == {'a': 1}
    # Modifying copy shouldn't affect original
    c2['b'] = 2
    assert 'b' not in c


def test_lri_setdefault():
    c = LRI()
    assert c.setdefault('a', 10) == 10
    assert c['a'] == 10
    assert c.soft_miss_count == 1
    assert c.setdefault('a', 20) == 10
    assert c.soft_miss_count == 1  # hit path


def test_lri_update():
    c = LRI()
    c.update({'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}
    c.update([('c', 3)])
    assert dict(c) == {'a': 1, 'b': 2, 'c': 3}
    c.update(d=4)
    assert dict(c) == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


def test_lri_update_self_noop():
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
    assert not (c1 != c2)
    c2['b'] = 2
    assert c1 != c2
    c3 = LRI()
    c3['a'] = 2
    assert c1 != c3
    # Comparing to a plain dict triggers infinite recursion in this
    # implementation (dict.__eq__ dispatches back to LRI.__eq__). Verify
    # that behavior explicitly rather than asserting equality.
    with pytest.raises(RecursionError):
        c1 == {'a': 1}
    # Identity comparison is safe and returns True
    assert c1 == c1


def test_lri_init_values():
    c = LRI(values={'a': 1, 'b': 2})
    assert dict(c) == {'a': 1, 'b': 2}


def test_lri_repr():
    c = LRI(max_size=3)
    r = repr(c)
    assert 'LRI' in r
    assert 'max_size=3' in r
    assert 'on_miss=None' in r


# --- private linked-list helpers (exercise to catch mutants) --------------

def test_lri_get_flattened_ll():
    c = LRI(max_size=3)
    c['a'] = 1
    c['b'] = 2
    # anchor is first entry with _MISSING key/value
    flat = c._get_flattened_ll()
    # Last entry is the anchor whose key/value are _MISSING (not the real key).
    keys = [k for k, v in flat]
    values = [v for k, v in flat]
    assert keys[-1] is not 'a'
    assert values[-1] is not 1
    # Non-anchor entries are in insertion order a, b
    assert ('a', 1) in flat
    assert ('b', 2) in flat
    # The list contains exactly one extra anchor entry.
    assert len(flat) == 3
    # Order: anchor, then a, then b (anchor first)
    assert flat[1] == ('a', 1)
    assert flat[2] == ('b', 2)


def test_lri_print_ll(capsys):
    c = LRI(max_size=3)
    c['a'] = 1
    c._print_ll()
    out = capsys.readouterr().out
    # The print uses '***' lines and key/value lines.
    lines = out.splitlines()
    assert lines[0] == '***'
    assert lines[-1] == '***'
    # Body should include the entry for 'a' printed with its key and value
    body = lines[1:-1]
    # first body line is anchor with _MISSING, second is 'a' 1
    assert any(line.strip() == 'a 1' for line in body)


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

def test_lru_eviction_least_recently_used():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['b'] = 'B'
    assert c['a'] == 'A'  # refresh 'a'
    c['c'] = 'C'
    # 'b' is least recently used
    assert 'b' not in c
    assert dict(c) == {'a': 'A', 'c': 'C'}


def test_lru_hit_miss_counts():
    c = LRU(max_size=2)
    c['a'] = 'A'
    c['a']
    c['a']
    assert c.hit_count == 2
    with pytest.raises(KeyError):
        c['x']
    assert c.miss_count == 1


def test_lru_on_miss():
    def on_miss(k):
        return k * 2

    c = LRU(on_miss=on_miss)
    assert c['a'] == 'aa'
    assert c.miss_count == 1
    assert c['a'] == 'aa'
    assert c.hit_count == 1


def test_lru_is_lri_subclass():
    assert issubclass(LRU, LRI)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_basic():
    key = make_cache_key(('a', 'b'), {})
    assert isinstance(key, _HashedKey)
    assert list(key) == ['a', 'b']


def test_make_cache_key_with_kwargs():
    key = make_cache_key(('a',), {'c': 'd'})
    # Layout: ['a', <KWARG_MARK>, ('c', 'd')]
    assert list(key)[0] == 'a'
    assert list(key)[1] is _KWARG_MARK
    assert list(key)[2] == ('c', 'd')
    assert list(key) == ['a', _KWARG_MARK, ('c', 'd')]


def test_make_cache_key_kwargs_sorted():
    k1 = make_cache_key((), {'b': 2, 'a': 1})
    k2 = make_cache_key((), {'a': 1, 'b': 2})
    assert list(k1) == list(k2)
    assert list(k1) == [_KWARG_MARK, ('a', 1), ('b', 2)]


def test_make_cache_key_single_fasttype():
    # single int returns int directly
    assert make_cache_key((3,), {}) == 3
    assert make_cache_key(('x',), {}) == 'x'
    assert make_cache_key((None,), {}) is None
    assert make_cache_key((frozenset([1]),), {}) == frozenset([1])


def test_make_cache_key_typed():
    k_int = make_cache_key((3,), {}, typed=True)
    k_float = make_cache_key((3.0,), {}, typed=True)
    assert list(k_int) != list(k_float)


def test_make_cache_key_untyped_single_fasttype_vs_wrapped():
    # Untyped: a single fasttype (int) is returned bare, while a float
    # (not a fasttype) gets wrapped in a _HashedKey, so they are not equal.
    k_int = make_cache_key((3,), {}, typed=False)
    k_float = make_cache_key((3.0,), {}, typed=False)
    assert k_int == 3
    assert isinstance(k_float, _HashedKey)
    assert list(k_float) == [3.0]
    assert k_int != k_float


def test_make_cache_key_untyped_non_fasttype_equal():
    # Two wrappers with equal lists compare equal.
    k1 = make_cache_key((3.0,), {}, typed=False)
    k2 = make_cache_key((3.0,), {}, typed=False)
    assert k1 == k2


def test_make_cache_key_typed_with_kwargs():
    # typed=True appends types of args and kwarg values
    k = make_cache_key((3,), {'x': 'y'}, typed=True)
    lst = list(k)
    # ['3', KWARG_MARK, ('x','y'), int, str]
    assert 3 in lst
    assert _KWARG_MARK in lst
    assert ('x', 'y') in lst
    assert int in lst
    assert str in lst
    assert len(lst) == 5


def test_make_cache_key_custom_kwarg_mark():
    mark = object()
    k = make_cache_key((), {'a': 1}, kwarg_mark=mark)
    lst = list(k)
    assert lst[0] is mark
    assert lst[1] == ('a', 1)


def test_hashed_key_repr():
    hk = _HashedKey(['a', 'b'])
    assert repr(hk) == "_HashedKey(['a', 'b'])"


def test_hashed_key_hashing():
    hk = _HashedKey(['a', 'b'])
    # hash_value computed at init, and __hash__ returns it
    assert hk.hash_value == hash(('a', 'b'))
    assert hash(hk) == hk.hash_value
    # Also equal-lists produce equal hashes
    hk2 = _HashedKey(['a', 'b'])
    assert hash(hk) == hash(hk2)


# ---------------------------------------------------------------------------
# CachedFunction / cached
# ---------------------------------------------------------------------------

def test_cached_function():
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


def test_cached_function_shared_cache_scoped_not_keyed_by_func():
    # CachedFunction does not incorporate the function/scope in the key,
    # so two differently-behaving functions sharing a cache collide.
    cache = {}

    @cached(cache, scoped=True)
    def f(x):
        return x

    @cached(cache, scoped=True)
    def g(x):
        return x + 1

    assert f(1) == 1
    # g(1) reads f's cached value for the same key
    assert g(1) == 1
    assert len(cache) == 1


def test_cached_invalid_cache():
    # The TypeError is raised when the returned decorator is applied.
    with pytest.raises(TypeError):
        cached(42)(lambda x: x)


def test_cached_typed():
    cache = {}

    @cached(cache, typed=True)
    def f(x):
        return x

    # With typed=True on a single arg, the key includes type info via
    # the wrapped _HashedKey path (for ints). Ensure no collision.
    assert f(3) == 3
    assert f(3.0) == 3.0
    assert len(cache) == 2


def test_cached_callable_cache():
    store = {}

    def get_cache():
        return store

    @cached(get_cache)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert len(store) == 1


def test_cached_custom_key():
    cache = {}

    def key_func(args, kwargs, typed=False):
        return ('custom',) + args

    @cached(cache, key=key_func)
    def f(x, y):
        return x + y

    assert f(1, 2) == 3
    assert ('custom', 1, 2) in cache


def test_cached_function_repr():
    def f(x):
        return x

    cf = CachedFunction(f, {}, scoped=True, typed=False)
    assert repr(cf) == f"CachedFunction(func={f!r})"

    cf2 = CachedFunction(f, {}, scoped=False, typed=False)
    r2 = repr(cf2)
    assert 'scoped=False' in r2

    cf3 = CachedFunction(f, {}, scoped=True, typed=True)
    r3 = repr(cf3)
    assert 'typed=True' in r3


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

    obj = Lowerer()
    assert obj.lower('HELLO') == 'hello'
    assert obj.lower('HELLO') == 'hello'
    assert len(obj.cache) == 1


def test_cachedmethod_dict_cache():
    cache = {}

    class Foo:
        @cachedmethod(cache)
        def val(self, x):
            return x + 1

    obj = Foo()
    assert obj.val(1) == 2
    assert obj.val(1) == 2
    assert len(cache) == 1


def test_cachedmethod_scoped_false():
    cache = {}

    class Foo:
        @cachedmethod(cache, scoped=False)
        def val(self, x):
            return x

    a = Foo()
    b = Foo()
    assert a.val(1) == 1
    # scoped=False means obj not in key, so b reads a's cache
    assert b.val(1) == 1
    assert len(cache) == 1


def test_cachedmethod_invalid_cache():
    with pytest.raises(TypeError):
        cachedmethod(42)(lambda self, x: x)


def test_cachedmethod_repr_unbound():
    def f(self, x):
        return x

    cm = CachedMethod(f, {}, scoped=True, typed=False)
    r = repr(cm)
    assert 'CachedMethod' in r
    assert 'scoped=True' in r
    assert 'typed=False' in r


def test_cachedmethod_get_descriptor_returns_self():
    def f(self, x):
        return x

    cm = CachedMethod(f, {}, scoped=True, typed=False)
    # Accessing via class (obj=None) returns the descriptor
    assert CachedMethod.__get__(cm, None, None) is cm


def test_cachedmethod_binding_repr_and_call():
    cache = {}

    class Foo:
        @cachedmethod(cache)
        def val(self, x):
            return x * 10

    obj = Foo()
    bound = obj.val
    # bound_to should be the object
    assert bound.bound_to is obj
    r = repr(bound)
    assert 'bound_to' in r
    assert obj.val(2) == 20


def test_cachedmethod_isabstractmethod():
    class Foo:
        @cachedmethod({})
        def method(self):
            raise NotImplementedError

    # If no __isabstractmethod__, it should be False
    result = getattr(Foo.method, '__isabstractmethod__', False)
    assert result is False


def test_cachedmethod_call_unbound():
    # Calling a CachedMethod directly (no binding) uses args[0] as obj
    cache = {}

    class Foo:
        def __init__(self):
            self.value = 5

    def method(self, x):
        return self.value + x

    cm = CachedMethod(method, cache)
    obj = Foo()
    assert cm(obj, 1) == 6
    assert cm(obj, 1) == 6
    assert len(cache) == 1


def test_cachedmethod_typed():
    cache = {}

    class Foo:
        @cachedmethod(cache, typed=True)
        def val(self, x):
            return x

    obj = Foo()
    assert obj.val(3) == 3
    assert obj.val(3.0) == 3.0
    assert len(cache) == 2


def test_cachedmethod_custom_key():
    cache = {}

    def key_func(args, kwargs, typed=False):
        return ('K',) + args

    class Foo:
        @cachedmethod(cache, key=key_func)
        def val(self, x):
            return x

    obj = Foo()
    assert obj.val(5) == 5
    assert ('K', obj, Foo.val.func, 5) in cache


def test_cachedmethod_callable_cache():
    store = {}

    class Foo:
        @cachedmethod(lambda obj: store)
        def val(self, x):
            return x + 1

    obj = Foo()
    assert obj.val(1) == 2
    assert len(store) == 1


# ---------------------------------------------------------------------------
# cachedproperty
# ---------------------------------------------------------------------------

def test_cachedproperty():
    calls = []

    class Foo:
        @cachedproperty
        def bar(self):
            calls.append(1)
            return 42

    obj = Foo()
    assert obj.bar == 42
    assert calls == [1]
    assert obj.bar == 42
    assert calls == [1]
    # Stored in __dict__
    assert obj.__dict__['bar'] == 42


def test_cachedproperty_can_be_deleted():
    class Foo:
        @cachedproperty
        def bar(self):
            return 42

    obj = Foo()
    assert obj.bar == 42
    del obj.bar
    assert 'bar' not in obj.__dict__
    assert obj.bar == 42


def test_cachedproperty_class_access():
    class Foo:
        @cachedproperty
        def bar(self):
            return 42

    assert isinstance(Foo.bar, cachedproperty)


def test_cachedproperty_repr():
    def f(self):
        pass

    cp = cachedproperty(f)
    r = repr(cp)
    assert 'cachedproperty' in r
    assert 'func=' in r


def test_cachedproperty_doc():
    class Foo:
        @cachedproperty
        def bar(self):
            """Bar docs."""
            return 1

    assert Foo.bar.__doc__ == "Bar docs."


def test_cachedproperty_isabstractmethod_flag():
    def f(self):
        pass

    cp = cachedproperty(f)
    assert cp.__isabstractmethod__ is False


# ---------------------------------------------------------------------------
# ThresholdCounter
# ---------------------------------------------------------------------------

def test_thresholdcounter_basic():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    assert tc.total == 1


def test_thresholdcounter_doctest_behavior():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    tc.update([2] * 10)
    assert tc.get(1) == 0
    tc.add(5)
    assert 5 in tc
    assert len(list(tc.elements())) == 11


def test_thresholdcounter_invalid_threshold():
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=0)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=1)
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=-0.5)


def test_thresholdcounter_threshold_property():
    tc = ThresholdCounter(threshold=0.05)
    assert tc.threshold == 0.05


def test_thresholdcounter_most_common():
    tc = ThresholdCounter(threshold=0.001)
    tc.add('a')
    tc.add('a')
    tc.add('b')
    assert tc.most_common() == [('a', 2), ('b', 1)]
    assert tc.most_common(1) == [('a', 2)]
    assert tc.most_common(0) == []
    assert tc.most_common(-1) == []
    # n larger than available
    assert len(tc.most_common(10)) == 2


def test_thresholdcounter_common_uncommon_counts():
    tc = ThresholdCounter(threshold=0.5)
    tc.add('a')
    assert tc.get_common_count() == 1
    assert tc.get_uncommon_count() == 0
    assert tc.get_commonality() == 1.0


def test_thresholdcounter_accessors():
    tc = ThresholdCounter(threshold=0.001)
    tc.add('a')
    tc.add('b')
    assert tc['a'] == 1
    assert len(tc) == 2
    assert 'a' in tc
    assert 'z' not in tc
    assert sorted(tc.keys()) == ['a', 'b']
    assert sorted(tc.values()) == [1, 1]
    assert sorted(tc.items()) == [('a', 1), ('b', 1)]
    assert tc.get('a') == 1
    assert tc.get('z') == 0
    assert tc.get('z', -1) == -1


def test_thresholdcounter_missing_key():
    tc = ThresholdCounter(threshold=0.001)
    with pytest.raises(KeyError):
        tc['missing']


def test_thresholdcounter_update_mapping():
    tc = ThresholdCounter(threshold=0.001)
    tc.update({'x': 3})
    assert tc['x'] == 3
    assert tc.total == 3


def test_thresholdcounter_update_kwargs():
    tc = ThresholdCounter(threshold=0.001)
    tc.update(None, a=2)
    assert tc['a'] == 2


def test_thresholdcounter_iterkeys_itervalues():
    tc = ThresholdCounter(threshold=0.001)
    tc.add('a')
    tc.add('b')
    assert sorted(tc.iterkeys()) == ['a', 'b']
    assert sorted(tc.itervalues()) == [1, 1]
    assert sorted(tc.iteritems()) == [('a', 1), ('b', 1)]


def test_thresholdcounter_elements_and_uncommon():
    tc = ThresholdCounter(threshold=0.5)
    # thresh_count = int(1/0.5) = 2. After 2 adds, cur_bucket advances.
    tc.add('x')
    tc.add('x')
    # total=2, bucket 1 -> 2 advances to cur_bucket=2, keeps x (count 2 > 1)
    assert tc.get('x') == 2
    # Add many y's; eventually some y's get culled
    for _ in range(6):
        tc.add('y')
    # x has count 2, y likely partially culled
    common = tc.get_common_count()
    uncommon = tc.get_uncommon_count()
    assert common + uncommon == tc.total


# ---------------------------------------------------------------------------
# MinIDMap
# ---------------------------------------------------------------------------

def test_minidmap_basic():
    m = MinIDMap()
    a = _Weaky()
    assert m.get(a) == 0
    assert m.get(a) == 0
    assert len(m) == 1
    assert a in m


def test_minidmap_multiple():
    m = MinIDMap()
    a = _Weaky()
    b = _Weaky()
    id_a = m.get(a)
    id_b = m.get(b)
    assert id_a != id_b
    assert len(m) == 2


def test_minidmap_drop():
    m = MinIDMap()
    a = _Weaky()
    b = _Weaky()
    id_a = m.get(a)
    m.get(b)
    m.drop(a)
    assert a not in m
    assert len(m) == 1
    # New object should reuse freed smallest ID
    c = _Weaky()
    assert m.get(c) == id_a


def test_minidmap_iter():
    m = MinIDMap()
    a = _Weaky()
    b = _Weaky()
    m.get(a)
    m.get(b)
    items = dict(m.iteritems())
    assert len(items) == 2
    assert set(items.keys()) == {a, b}
    assert sorted(items.values()) == [0, 1]


def test_minidmap_contains_and_iter():
    m = MinIDMap()
    a = _Weaky()
    m.get(a)
    assert a in m
    assert list(m) == [a]


def test_minidmap_weakref_cleanup(capsys):
    m = MinIDMap()
    a = _Weaky()
    m.get(a)
    id_a = m.get(a)
    del a
    gc.collect()
    # The cleanup callback prints the freed ID; drain stdout.
    capsys.readouterr()
    # After cleanup, the freed ID should be reusable
    b = _Weaky()
    assert m.get(b) == id_a


def test_minidmap_clean_callback_prints(capsys):
    # _clean prints the freed ID
    m = MinIDMap()
    a = _Weaky()
    m.get(a)
    del a
    out = capsys.readouterr().out
    assert out.strip() == '0'
    # freed id back in free list
    assert 0 in m.free
