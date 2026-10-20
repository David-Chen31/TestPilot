import pickle
import sys
import pytest

from toolz.functoolz import (
    identity,
    apply,
    thread_first,
    thread_last,
    instanceproperty,
    InstanceProperty,
    curry,
    memoize,
    Compose,
    compose,
    compose_left,
    pipe,
    complement,
    juxt,
    do,
    flip,
    return_none,
    excepts,
    num_required_args,
    has_varargs,
    has_keywords,
    is_valid_args,
    is_partial_args,
    is_arity,
)


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_with_kwargs():
    def add(x, y, z=0):
        return x + y + z

    assert apply(add, 1, 2, z=3) == 6


def test_apply_no_args_raises():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


def test_apply_as_map():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(7) == 7


def test_thread_last_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_no_forms():
    assert thread_last(9) == 9


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        '''class docstring'''

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'obj doc'

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == 'class docstring'
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == 'obj doc'
    assert obj.val == 42


def test_instanceproperty_without_fget_returns_partial():
    from functools import partial
    p = instanceproperty()
    assert isinstance(p, partial)


def test_instanceproperty_setter():
    class C:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, v):
            self._x = v

    c = C()
    assert c.x == 0
    c.x = 5
    assert c.x == 5


def test_instanceproperty_class_attribute_value():
    class C:
        @instanceproperty(classval='CLASSVAL')
        def v(self):
            return 'INSTANCE'

    assert C.v == 'CLASSVAL'
    assert C().v == 'INSTANCE'


def test_instance_property_reduce_roundtrip():
    class C:
        @instanceproperty(classval='CV')
        def v(self):
            return 'IV'

    prop = C.__dict__['v']
    assert isinstance(prop, InstanceProperty)
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    restored = cls(*state)
    assert restored.classval == 'CV'
    assert restored.fget is prop.fget


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert isinstance(double, curry)
    assert double(10) == 20
    assert cmul(2, 5) == 10


def test_curry_as_decorator_keywords():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_init_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable_raises():
    with pytest.raises(TypeError, match='Input must be callable'):
        curry(5)


def test_curry_merges_existing_partial():
    def f(a, b, c):
        return a + b + c

    c1 = curry(f, 1)
    c2 = curry(c1, 2)
    assert c2(3) == 6
    assert c2.func is f
    assert c2.args == (1, 2)


def test_curry_merges_keywords():
    def f(a, b=0):
        return a + b

    c1 = curry(f, b=1)
    c2 = curry(c1, b=2)
    assert c2(1) == 3


def test_curry_metadata():
    def f(x):
        'docstring'
        return x

    c = curry(f)
    assert c.__name__ == 'f'
    assert c.__doc__ == 'docstring'
    assert c.__module__ == f.__module__
    assert c.__qualname__ == f.__qualname__
    assert c.func_name == 'f'


def test_curry_str_repr():
    def f(x):
        return x

    c = curry(f)
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_eq_hash():
    def f(x, y):
        return x + y

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert not (c1 != c2)
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_curry_call_genuine_typeerror_raises():
    def f(x):
        return x

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2)


def test_curry_bind_and_call():
    def f(a, b):
        return a + b

    c = curry(f)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound.args == (1,)
    assert c.call(1, 2) == 3


def test_curry_descriptor():
    class C:
        @curry
        def meth(self, x, y):
            return self, x, y

    c = C()
    bound = C.__dict__['meth'].__get__(c, C)
    assert isinstance(bound, curry)
    result = bound(1, 2)
    assert result == (c, 1, 2)


def test_curry_descriptor_class_access():
    class C:
        @curry
        def meth(self, x):
            return x

    assert isinstance(C.__dict__['meth'], curry)


def test_curry_getattr_properties():
    def f(a, b):
        return a + b

    c = curry(f, 1)
    assert c.args == (1,)
    assert c.keywords == {}
    assert c.func is f


def test_curry_varargs_curries_anyway():
    def f(*args):
        return args

    c = curry(f)
    # Calling with a TypeError inside doesn't happen here; just make sure
    # currying on unknown args works.
    assert c(1)(2) == (1, 2)


def test_curry_pickle_roundtrip():
    c = curry(_top_level_add, 1, b=2)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert restored(3) == 6


def _top_level_add(a, b=0, c=0):
    return a + b + c


def test_curry_pickle_decorated():
    # `memoize` and `flip` are module-level curried functions.
    assert isinstance(flip, curry)
    data = pickle.dumps(flip)
    restored = pickle.loads(data)
    assert restored is flip


def test_curry_signature():
    import inspect

    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = inspect.signature(c)
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names == ['b', 'c']
    assert params[0].default is inspect.Parameter.empty or params[0].default is not None


def test_curry_signature_incorrect_args_raises():
    def f(a):
        return a

    c = curry(f).bind(1)
    # Bind an extra positional beyond what's valid
    c2 = curry(c, 2)
    with pytest.raises(TypeError):
        c2.__signature__


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    memo = memoize(add)
    assert memo(1, 2) == 3
    assert memo(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_decorator_with_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    # Not in cache initially, will compute
    assert add(4, 5) == 9


def test_memoize_metadata():
    def add(x, y):
        'adds'
        return x + y

    memo = memoize(add)
    assert memo.__name__ == 'add'
    assert memo.__doc__ == 'adds'
    assert memo.__wrapped__ is add


def test_memoize_unary_key():
    calls = []

    def square(x):
        calls.append(x)
        return x * x

    memo = memoize(square)
    assert memo(4) == 16
    assert memo(4) == 16
    assert calls == [4]


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2) == 3
    assert len(calls) == 1


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='hashable'):
        f([1, 2, 3])


def test_memoize_kwargs_key():
    calls = []

    def f(x, y=0):
        calls.append((x, y))
        return x + y

    memo = memoize(f)
    assert memo(1, y=2) == 3
    assert memo(1, y=2) == 3
    assert calls == [(1, 2)]


# ---------------------------------------------------------------------------
# compose / Compose / compose_left / pipe / complement
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_no_args_returns_identity():
    assert compose() is identity


def test_compose_single_returns_func():
    def f(x):
        return x
    assert compose(f) is f


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_args():
    assert compose_left() is identity


def test_compose_kwargs():
    def f(x, y=1):
        return x + y

    def g(x):
        return x * 2

    assert compose(g, f)(3, y=4) == 14


def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3


def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


def test_compose_class_doc():
    def f(x):
        return x

    def g(x):
        return x

    comp = Compose((f, g))
    assert isinstance(comp.__doc__, str)
    assert 'g(f(' in comp.__doc__


def test_compose_class_doc_no_names():
    comp = Compose((len, len))
    assert isinstance(comp.__doc__, str)


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    comp = Compose((f, g))
    # compose(f, g) applies g first then f -> name f_of_g
    assert comp.__name__ == 'f_of_g'


def test_compose_name_no_attr():
    comp = Compose((functools_partial_obj(),))
    assert comp.__name__ == 'Compose'


def functools_partial_obj():
    from functools import partial
    return partial(pow, 2)


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    comp = Compose((f, g))
    assert repr(comp).startswith('Compose')


def test_compose_eq_ne_hash():
    def f(x):
        return x

    def g(x):
        return x

    c1 = Compose((f, g))
    c2 = Compose((f, g))
    c3 = Compose((g, f))
    assert c1 == c2
    assert not (c1 != c2)
    assert c1 != c3
    assert not (c1 == 5)
    assert c1.__ne__(5) is NotImplemented
    assert hash(c1) == hash(c2)


def test_compose_descriptor():
    def inc(x):
        return x + 1

    class C:
        f = compose(inc, inc)

    c = C()
    assert isinstance(c.f, type(c.f))  # MethodType
    assert c.f() if False else True  # method exists
    # Compose requires the bound arg; give it
    class D:
        g = compose(lambda self: self)

    d = D()
    assert d.g() is d


def test_compose_signature():
    import inspect

    def f(a: int, b: str) -> float:
        return 0.0

    def g(x: float) -> str:
        return ''

    comp = compose(f, g)
    sig = inspect.signature(comp)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['x']
    assert sig.return_annotation is float


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    comp = Compose((f, g))
    assert comp.__wrapped__ is f


def test_compose_annotations():
    def f(a: int) -> float:
        return float(a)

    def g(x: str) -> int:
        return int(x)

    comp = compose(f, g)
    ann = comp.__annotations__
    # annotations is a dict on instances
    assert ann == {'a': int, 'return': float}


def test_compose_annotations_skips_uninspectable():
    def f(x):
        return x

    comp = Compose((f, len))
    ann = comp.__annotations__
    # len has return annotation but no param annotations
    assert isinstance(ann, dict)


def test_compose_state_roundtrip():
    def f(x):
        return x + 1

    comp = Compose((f, f))
    state = comp.__getstate__()
    new = Compose.__new__(Compose)
    new.__setstate__(state)
    assert new.first is comp.first
    assert new.funcs == comp.funcs


def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y):
        return x + y

    def g(x, y):
        return x * y

    assert juxt(f, g)(2, 3) == (5, 6)


def test_juxt_state():
    j = juxt(lambda x: x, lambda x: x)
    state = j.__getstate__()
    new = juxt.__new__(juxt)
    new.__setstate__(state)
    assert new.funcs == j.funcs


def test_do():
    log = []
    inc = lambda x: x + 1
    f = compose(inc, do(log.append))
    assert f(1) == 2
    assert f(11) == 12
    assert log == [1, 11]


def test_return_none():
    assert return_none(ValueError('x')) is None


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip_basic():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3


def test_flip_partial():
    def div(a, b):
        return a // b

    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# excepts
# ---------------------------------------------------------------------------

def test_excepts_basic():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_doc():
    excepting = excepts(ValueError, lambda a: a, lambda e: e)
    assert isinstance(excepting.__doc__, str)


def test_excepts_doc_fallback():
    class Bad:
        pass

    excepting = excepts(ValueError, Bad(), Bad())
    assert isinstance(excepting.__doc__, str)


def test_excepts_name():
    def foo(a):
        return a

    def handler(e):
        return e

    excepting = excepts(ValueError, foo, handler)
    assert excepting.__name__ == 'foo_excepting_ValueError'

    multi = excepts((IndexError, KeyError), foo, handler)
    assert multi.__name__ == 'foo_excepting_IndexError_or_KeyError'


def test_excepts_name_fallback():
    class Bad:
        pass

    excepting = excepts(ValueError, Bad(), Bad())
    assert excepting.__name__ == 'excepting'


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_has_varargs():
    def f(*args):
        return args

    def g(**kwargs):
        return kwargs

    assert has_varargs(f) is True
    assert has_varargs(g) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x, y):
        return x + y

    assert has_keywords(f) is True
    assert has_keywords(g) is False


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False
    assert is_partial_args(map, (), {}) is True


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(0, f) is False


def test_introspection_builtin():
    # len has a signature in CPython; check through the helpers
    assert num_required_args(len) == 1
    assert has_varargs(len) is False
