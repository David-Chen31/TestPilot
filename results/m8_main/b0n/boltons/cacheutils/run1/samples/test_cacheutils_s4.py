import gc
import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    make_cache_key,
    _HashedKey,
    CachedFunction,
    CachedMethod,
    cached,
    cachedmethod,
    cachedproperty,
    ThresholdCounter,
    MinIDMap,
    _KWARG_MARK,
)


# ---------------------------------------------------------------------------
# LRI
# ---------------------------------------------------------------------------

class TestLRI:
    def test_init_invalid_max_size(self):
        with pytest.raises(ValueError):
            LRI(max_size=0)
        with pytest.raises(ValueError):
            LRI(max_size=-5)

    def test_init_invalid_on_miss(self):
        with pytest.raises(TypeError):
            LRI(on_miss=5)

    def test_init_defaults(self):
        c = LRI()
        assert c.max_size == 128
        assert c.on_miss is None
        assert len(c) == 0
        assert c.hit_count == 0
        assert c.miss_count == 0
        assert c.soft_miss_count == 0

    def test_basic_set_get(self):
        c = LRI(max_size=3)
        c['a'] = 1
        c['b'] = 2
        assert c['a'] == 1
        assert c['b'] == 2
        assert c.hit_count == 2
        assert c.miss_count == 0

    def test_init_with_values(self):
        c = LRI(values={'a': 1, 'b': 2})
        assert dict(c) == {'a': 1, 'b': 2}

    def test_init_with_pairs(self):
        c = LRI(values=[('a', 1), ('b', 2)])
        assert dict(c) == {'a': 1, 'b': 2}

    def test_eviction_least_recently_inserted(self):
        c = LRI(max_size=2)
        c['a'] = 'A'
        c['b'] = 'B'
        # accessing 'a' does not protect it in LRI (insertion order)
        assert c['a'] == 'A'
        c['c'] = 'C'
        assert 'a' not in c
        assert dict(c) == {'b': 'B', 'c': 'C'}

    def test_getitem_hit_miss_stats(self):
        c = LRI()
        c['a'] = 1
        assert c['a'] == 1
        assert c.hit_count == 1
        with pytest.raises(KeyError):
            c['missing']
        assert c.miss_count == 1

    def test_on_miss(self):
        calls = []

        def om(key):
            calls.append(key)
            return key * 2

        c = LRI(on_miss=om)
        assert c['x'] == 'xx'
        assert calls == ['x']
        # Now cached
        assert c['x'] == 'xx'
        assert calls == ['x']
        assert c.hit_count == 1
        assert c.miss_count == 1

    def test_get_default(self):
        c = LRI()
        c['a'] = 1
        assert c.get('a') == 1
        assert c.get('b') is None
        assert c.get('b', 'dflt') == 'dflt'
        assert c.soft_miss_count == 2  # both b lookups missing

    def test_delitem(self):
        c = LRI()
        c['a'] = 1
        del c['a']
        assert 'a' not in c
        with pytest.raises(KeyError):
            del c['a']

    def test_pop(self):
        c = LRI()
        c['a'] = 1
        assert c.pop('a') == 1
        assert 'a' not in c
        with pytest.raises(KeyError):
            c.pop('a')
        assert c.pop('a', 'dflt') == 'dflt'
        # pop bypasses hit/miss counters
        assert c.hit_count == 0
        assert c.miss_count == 0
        assert c.soft_miss_count == 0

    def test_popitem(self):
        c = LRI()
        c['a'] = 1
        c['b'] = 2
        key, value = c.popitem()
        # popitem on dict returns last inserted (LIFO) in modern Python
        assert (key, value) == ('b', 2)
        assert 'b' not in c
        assert c.popitem() == ('a', 1)

    def test_clear(self):
        c = LRI()
        c['a'] = 1
        c['b'] = 2
        c.clear()
        assert len(c) == 0
        assert dict(c) == {}
        # should still be usable
        c['x'] = 10
        assert c['x'] == 10

    def test_copy(self):
        c = LRI(max_size=5)
        c['a'] = 1
        c['b'] = 2
        c2 = c.copy()
        assert isinstance(c2, LRI)
        assert c2.max_size == 5
        assert dict(c2) == {'a': 1, 'b': 2}
        c2['c'] = 3
        assert 'c' not in c

    def test_setdefault(self):
        c = LRI()
        assert c.setdefault('a', 10) == 10
        assert c.soft_miss_count == 1
        assert c.setdefault('a', 20) == 10
        assert c['a'] == 10

    def test_setdefault_default_none(self):
        c = LRI()
        assert c.setdefault('a') is None
        assert 'a' in c

    def test_update_mapping(self):
        c = LRI()
        c.update({'a': 1, 'b': 2})
        assert dict(c) == {'a': 1, 'b': 2}

    def test_update_pairs(self):
        c = LRI()
        c.update([('a', 1), ('b', 2)])
        assert dict(c) == {'a': 1, 'b': 2}

    def test_update_kwargs(self):
        c = LRI()
        c.update(a=1, b=2)
        assert dict(c) == {'a': 1, 'b': 2}

    def test_update_self_is_noop(self):
        c = LRI()
        c['a'] = 1
        c.update(c)
        assert dict(c) == {'a': 1}

    def test_eq(self):
        c1 = LRI()
        c2 = LRI()
        c1['a'] = 1
        c2['a'] = 1
        assert c1 == c2
        assert not (c1 != c2)

        c2['b'] = 2
        assert c1 != c2

        assert c1 == {'a': 1}
        assert c1 != {'a': 2}

    def test_eq_self(self):
        c = LRI()
        c['a'] = 1
        assert c == c

    def test_repr(self):
        c = LRI(max_size=5)
        c['a'] = 1
        r = repr(c)
        assert r.startswith('LRI(')
        assert 'max_size=5' in r
        assert 'on_miss=None' in r
        assert "'a': 1" in r

    def test_overwrite_existing(self):
        c = LRI(max_size=2)
        c['a'] = 1
        c['a'] = 2
        assert c['a'] == 2
        assert len(c) == 1

    def test_overwrite_moves_to_front(self):
        c = LRI(max_size=2)
        c['a'] = 1
        c['b'] = 2
        c['a'] = 10  # reinsertion moves 'a' to newest
        c['c'] = 3  # should evict 'b'
        assert 'a' in c
        assert 'b' not in c
        assert 'c' in c


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

class TestLRU:
    def test_is_lri_subclass(self):
        assert issubclass(LRU, LRI)

    def test_lru_eviction(self):
        c = LRU(max_size=2)
        c['a'] = 'A'
        c['b'] = 'B'
        assert c['a'] == 'A'  # move 'a' to front
        c['c'] = 'C'  # evict 'b'
        assert 'a' in c
        assert 'b' not in c
        assert 'c' in c

    def test_lru_stats(self):
        c = LRU(max_size=2)
        c['a'] = 'A'
        c['b'] = 'B'
        assert c['b'] == 'B'
        assert c['b'] == 'B'
        assert c.hit_count == 2
        with pytest.raises(KeyError):
            c['nope']
        assert c.miss_count == 1

    def test_lru_on_miss(self):
        c = LRU(on_miss=lambda k: k.upper())
        assert c['x'] == 'X'
        assert c['x'] == 'X'
        assert c.hit_count == 1
        assert c.miss_count == 1

    def test_lru_get(self):
        c = LRU()
        c['a'] = 1
        assert c.get('a') == 1
        assert c.get('b') is None
        assert c.soft_miss_count == 1


# ---------------------------------------------------------------------------
# make_cache_key / _HashedKey
# ---------------------------------------------------------------------------

class TestMakeCacheKey:
    def test_simple_args(self):
        key = make_cache_key(('a', 'b'), {})
        assert isinstance(key, _HashedKey)
        assert tuple(key) == ('a', 'b')

    def test_with_kwargs(self):
        key = make_cache_key(('a',), {'c': 'd'})
        assert tuple(key) == ('a', _KWARG_MARK, ('c', 'd'))

    def test_kwargs_sorted(self):
        key = make_cache_key((), {'b': 2, 'a': 1})
        assert tuple(key) == (_KWARG_MARK, ('a', 1), ('b', 2))

    def test_single_fasttype_unwrapped(self):
        assert make_cache_key(('a',), {}) == 'a'
        assert make_cache_key((3,), {}) == 3
        assert make_cache_key((None,), {}) is None

    def test_single_non_fasttype_wrapped(self):
        key = make_cache_key(([1, 2],), {})
        assert isinstance(key, _HashedKey)
        assert tuple(key) == ([1, 2],)

    def test_typed(self):
        k1 = make_cache_key((3,), {}, typed=True)
        k2 = make_cache_key((3.0,), {}, typed=True)
        assert k1 != k2
        # Unwrapped for single fasttype? typed keeps last type -> not unwrapped
        assert isinstance(k1, _HashedKey)

    def test_typed_with_kwargs(self):
        k = make_cache_key((), {'a': 1}, typed=True)
        assert tuple(k) == (_KWARG_MARK, ('a', 1), int)

    def test_hashed_key_hash(self):
        hk = _HashedKey((1, 2, 3))
        assert hash(hk) == hash((1, 2, 3))
        assert hk.hash_value == hash((1, 2, 3))

    def test_hashed_key_repr(self):
        hk = _HashedKey((1, 2))
        assert repr(hk) == '_HashedKey([1, 2])'

    def test_hashed_key_equality(self):
        assert _HashedKey((1, 2)) == _HashedKey((1, 2))
        assert _HashedKey((1, 2)) != _HashedKey((1, 3))


# ---------------------------------------------------------------------------
# CachedFunction / cached
# ---------------------------------------------------------------------------

class TestCachedFunction:
    def test_basic_caching(self):
        calls = []

        def func(x):
            calls.append(x)
            return x * 2

        cache = {}
        cf = CachedFunction(func, cache)
        assert cf(3) == 6
        assert cf(3) == 6
        assert calls == [3]
        assert len(cache) == 1

    def test_callable_cache(self):
        cache = {}

        def getter():
            return cache

        cf = CachedFunction(lambda x: x + 1, getter)
        assert cf(1) == 2
        assert len(cache) == 1

    def test_invalid_cache(self):
        with pytest.raises(TypeError):
            CachedFunction(lambda x: x, 5)

    def test_custom_key(self):
        def key_func(args, kwargs, typed=False):
            return args[0]

        cache = {}
        cf = CachedFunction(lambda x: x, cache, key=key_func)
        assert cf('a') == 'a'
        assert 'a' in cache

    def test_repr(self):
        def fn(x):
            return x

        cf = CachedFunction(fn, {})
        r = repr(cf)
        assert r.startswith('CachedFunction(')
        assert 'func=' in r

    def test_repr_typed(self):
        cf = CachedFunction(lambda x: x, {}, typed=True)
        r = repr(cf)
        assert 'typed=True' in r

    def test_decorator(self):
        cache = {}

        @cached(cache)
        def add(a, b):
            return a + b

        assert add(1, 2) == 3
        assert add(1, 2) == 3
        assert len(cache) == 1

    def test_decorator_scoped_default(self):
        cache = {}

        @cached(cache)
        def f1(x):
            return ('f1', x)

        @cached(cache)
        def f2(x):
            return ('f2', x)

        assert f1(1) == ('f1', 1)
        assert f2(1) == ('f2', 1)
        assert len(cache) == 2

    def test_decorator_typed(self):
        cache = {}

        @cached(cache, typed=True)
        def ident(x):
            return x

        assert ident(3) == 3
        assert ident(3.0) == 3.0
        assert len(cache) == 2

    def test_decorator_untyped(self):
        cache = {}

        @cached(cache, typed=False)
        def ident2(x):
            return x

        # 1 and 1.0 hash equal and compare equal, so same key
        ident2(1)
        n = len(cache)
        ident2(1.0)
        assert len(cache) == n


# ---------------------------------------------------------------------------
# CachedMethod / cachedmethod
# ---------------------------------------------------------------------------

class TestCachedMethod:
    def test_basic_method_cache(self):
        class Foo:
            def __init__(self):
                self.cache = {}
                self.calls = 0

            @cachedmethod(lambda self: self.cache)
            def compute(self, x):
                self.calls += 1
                return x * 10

        f = Foo()
        assert f.compute(2) == 20
        assert f.compute(2) == 20
        assert f.calls == 1
        assert len(f.cache) == 1

    def test_cachedmethod_string_attr(self):
        class Foo:
            def __init__(self):
                self.cache = LRI()
                self.calls = 0

            @cachedmethod('cache')
            def compute(self, x):
                self.calls += 1
                return x + 1

        f = Foo()
        assert f.compute(1) == 2
        assert f.compute(1) == 2
        assert f.calls == 1

    def test_cachedmethod_dict_cache(self):
        cache = {}

        class Foo:
            @cachedmethod(cache)
            def compute(self, x):
                return x * 2

        f = Foo()
        assert f.compute(3) == 6
        assert len(cache) == 1

    def test_cachedmethod_invalid_cache(self):
        with pytest.raises(TypeError):
            CachedMethod(lambda self: None, 5)

    def test_cachedmethod_scoped_default_isolation(self):
        # Different instances should not share scoped cache entries
        cache = {}

        class Foo:
            @cachedmethod(cache)
            def get_val(self, x):
                return id(self), x

        f1 = Foo()
        f2 = Foo()
        r1 = f1.get_val(1)
        r2 = f2.get_val(1)
        assert r1 != r2

    def test_cachedmethod_scoped_false(self):
        cache = {}

        class Foo:
            @cachedmethod(cache, scoped=False)
            def get_val(self, x):
                return x + 1

        f1 = Foo()
        f2 = Foo()
        assert f1.get_val(5) == 6
        assert f2.get_val(5) == 6
        # scoped=False means shared cache across instances -> one entry
        assert len(cache) == 1

    def test_cachedmethod_repr(self):
        def method(self, x):
            return x

        cm = CachedMethod(method, {})
        r = repr(cm)
        assert r.startswith('CachedMethod(')

    def test_cachedmethod_descriptor_returns_self_on_class(self):
        class Foo:
            @cachedmethod({})
            def m(self):
                return 1

        # Accessing on the class returns the descriptor itself
        assert isinstance(Foo.__dict__['m'], CachedMethod)

    def test_cachedmethod_abstractmethod_attr(self):
        def method(self):
            pass
        method.__isabstractmethod__ = True
        cm = CachedMethod(method, {})
        assert cm.__isabstractmethod__ is True

    def test_cachedmethod_typed(self):
        cache = {}

        class Foo:
            @cachedmethod(cache, typed=True)
            def ident(self, x):
                return x

        f = Foo()
        assert f.ident(3) == 3
        assert f.ident(3.0) == 3.0


# ---------------------------------------------------------------------------
# cachedproperty
# ---------------------------------------------------------------------------

class TestCachedProperty:
    def test_cached_property_basic(self):
        calls = []

        class Foo:
            @cachedproperty
            def val(self):
                calls.append(1)
                return 42

        f = Foo()
        assert f.val == 42
        assert f.val == 42
        assert len(calls) == 1

    def test_cached_property_stored_in_dict(self):
        class Foo:
            @cachedproperty
            def val(self):
                return 'x'

        f = Foo()
        assert 'val' not in f.__dict__
        assert f.val == 'x'
        assert f.__dict__['val'] == 'x'

    def test_cached_property_clearable(self):
        calls = []

        class Foo:
            @cachedproperty
            def val(self):
                calls.append(1)
                return len(calls)

        f = Foo()
        assert f.val == 1
        del f.val
        assert f.val == 2

    def test_cached_property_class_access_returns_descriptor(self):
        class Foo:
            @cachedproperty
            def val(self):
                return 1

        assert isinstance(Foo.val, cachedproperty)
        assert repr(Foo.val).startswith('<cachedproperty func=')

    def test_cached_property_abstractmethod_attr(self):
        class Foo:
            @cachedproperty
            def val(self):
                return 1

        # default __isabstractmethod__ is False when not set
        assert Foo.__dict__['val'].__isabstractmethod__ is False


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
            ThresholdCounter(threshold=-0.5)

    def test_default_threshold(self):
        tc = ThresholdCounter()
        assert tc.threshold == 0.001

    def test_basic_add(self):
        tc = ThresholdCounter(threshold=0.1)
        tc.add(1)
        assert tc.total == 1
        assert tc[1] == 1
        assert 1 in tc
        assert len(tc) == 1

    def test_add_repeat(self):
        tc = ThresholdCounter(threshold=0.1)
        for _ in range(5):
            tc.add('a')
        assert tc['a'] == 5
        assert tc.total == 5

    def test_getitem_missing(self):
        tc = ThresholdCounter(threshold=0.1)
        with pytest.raises(KeyError):
            tc['missing']

    def test_get(self):
        tc = ThresholdCounter(threshold=0.1)
        tc.add('a')
        assert tc.get('a') == 1
        assert tc.get('b') == 0
        assert tc.get('b', -1) == -1

    def test_len_and_contains(self):
        tc = ThresholdCounter(threshold=0.1)
        assert len(tc) == 0
        assert 'a' not in tc
        tc.add('a')
        assert len(tc) == 1
        assert 'a' in tc

    def test_keys_values_items(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')
        tc.add('a')
        tc.add('b')
        assert tc.keys() == ['a', 'b']
        assert sorted(tc.values()) == [1, 2]
        assert sorted(tc.items()) == [('a', 2), ('b', 1)]

    def test_iter_keys_values_items(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')
        tc.add('b')
        assert sorted(tc.iterkeys()) == ['a', 'b']
        assert sorted(tc.itervalues()) == [1, 1]
        assert sorted(tc.iteritems()) == [('a', 1), ('b', 1)]

    def test_elements(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')
        tc.add('a')
        tc.add('b')
        elements = list(tc.elements())
        assert elements.count('a') == 2
        assert elements.count('b') == 1
        assert len(elements) == 3

    def test_most_common(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')
        tc.add('a')
        tc.add('b')
        mc = tc.most_common()
        assert mc[0] == ('a', 2)

    def test_most_common_n(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')
        tc.add('a')
        tc.add('b')
        assert tc.most_common(1) == [('a', 2)]
        assert tc.most_common(0) == []
        assert tc.most_common(-1) == []
        assert len(tc.most_common(100)) == 2

    def test_get_common_count_and_uncommon(self):
        tc = ThresholdCounter(threshold=0.5)
        for _ in range(3):
            tc.add('a')
        assert tc.get_common_count() == 3
        assert tc.get_uncommon_count() == 0
        assert tc.get_commonality() == 1.0

    def test_compaction(self):
        tc = ThresholdCounter(threshold=0.5)  # thresh_count = 2
        tc.add('a')  # total 1
        tc.add('b')  # total 2 -> compaction, cur_bucket -> 2
        # both had counts [1, 0], sum = 1, not > 1? cur_bucket was 1 during compaction
        # After: cur_bucket == 2
        assert tc._cur_bucket == 2

    def test_compaction_prunes_low_counts(self):
        tc = ThresholdCounter(threshold=0.5)
        tc.add('a')  # bucket 0
        tc.add('b')  # total 2 -> compaction; cur_bucket 1->2
        # a: [1, 0] sum=1 not > 1 => pruned
        # b: [1, 0] sum=1 not > 1 => pruned
        assert len(tc) == 0
        assert tc.total == 2
        assert tc.get_uncommon_count() == 2

    def test_update_iterable(self):
        tc = ThresholdCounter(threshold=0.1)
        tc.update([1, 1, 2])
        assert tc[1] == 2
        assert tc[2] == 1
        assert tc.total == 3

    def test_update_mapping(self):
        tc = ThresholdCounter(threshold=0.1)
        tc.update({'a': 3, 'b': 1})
        assert tc['a'] == 3
        assert tc['b'] == 1
        assert tc.total == 4

    def test_update_kwargs(self):
        tc = ThresholdCounter(threshold=0.1)
        tc.update(None, x=2)
        assert tc['x'] == 2
        assert tc.total == 2

    def test_update_mapping_and_kwargs(self):
        tc = ThresholdCounter(threshold=0.1)
        tc.update({'a': 1}, b=2)
        assert tc['a'] == 1
        assert tc['b'] == 2


# ---------------------------------------------------------------------------
# MinIDMap
# ---------------------------------------------------------------------------

class TestMinIDMap:
    def test_get_assigns_ids(self):
        m = MinIDMap()
        a = object()
        b = object()
        ida = m.get(a)
        idb = m.get(b)
        assert ida == 0
        assert idb == 1
        assert m.get(a) == ida
        assert m.get(b) == idb

    def test_len_and_contains(self):
        m = MinIDMap()
        a = object()
        assert len(m) == 0
        assert a not in m
        m.get(a)
        assert len(m) == 1
        assert a in m

    def test_drop(self):
        m = MinIDMap()
        a = object()
        b = object()
        ida = m.get(a)
        idb = m.get(b)
        m.drop(a)
        assert a not in m
        assert len(m) == 1
        # dropped id is reused
        c = object()
        assert m.get(c) == ida
        assert m.get(b) == idb

    def test_drop_invalid(self):
        m = MinIDMap()
        a = object()
        with pytest.raises(KeyError):
            m.drop(a)

    def test_iter(self):
        m = MinIDMap()
        a = object()
        b = object()
        m.get(a)
        m.get(b)
        assert set(m) == {a, b}

    def test_iteritems(self):
        m = MinIDMap()
        a = object()
        ida = m.get(a)
        items = list(m.iteritems())
        assert items == [(a, ida)]

    def test_weakref_cleanup(self, capsys):
        m = MinIDMap()
        a = object()
        ida = m.get(a)
        assert ida == 0
        del a
        gc.collect()
        # _clean prints the freed id
        captured = capsys.readouterr()
        assert str(ida) in captured.out
        # freed id should be reused
        b = object()
        assert m.get(b) == ida

    def test_ids_smallest_available(self):
        m = MinIDMap()
        a = object()
        b = object()
        c = object()
        m.get(a)  # 0
        m.get(b)  # 1
        m.get(c)  # 2
        m.drop(b)  # free 1
        d = object()
        assert m.get(d) == 1
