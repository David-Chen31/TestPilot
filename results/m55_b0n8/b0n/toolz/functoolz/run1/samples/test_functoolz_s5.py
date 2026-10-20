import inspect
import pickle
import sys
from functools import partial

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
from toolz.utils import no_default


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10
    assert apply(double, x=4) == 8


def test_apply_no_args_raises():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


def test_apply_kwargs():
    def add(x, y=1):
        return x + y

    assert apply(add, 3, y=10) == 13


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4
    assert thread_first(1) == 1


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_last_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4
    assert thread_last(1) == 1


def test_thread_last_tuples():
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

    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'An object docstring'

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == 'The class docstring'
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == 'An object docstring'
    assert obj.val == 42


def test_instanceproperty_is_property_subclass():
    assert issubclass(InstanceProperty, property)


def test_instanceproperty_setter_deleter():
    class C:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    c = C()
    assert c.x == 0
    c.x = 5
    assert c.x == 5


def test_instanceproperty_reduce():
    class C:
        @instanceproperty(classval='cv')
        def x(self):
            return 1

    prop = C.__dict__['x']
    restored, state = prop.__reduce__()
    assert restored is InstanceProperty
    assert state[4] == 'cv'


def test_instanceproperty_partial_when_no_fget():
    p = instanceproperty(classval=7)
    assert isinstance(p, partial)


# ---------------------------------------------------------------------------
# curry: basics
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    mul = curry(mul)
    double = mul(2)
    assert double(10) == 20
    assert mul(2, 3) == 6
    assert mul(2)(3) == 6


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_requires_callable():
    with pytest.raises(TypeError, match='Input must be callable'):
        curry(5)


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_attributes():
    def foo(x, y):
        """my doc"""
        return x + y

    c = curry(foo)
    assert c.__name__ == 'foo'
    assert c.__doc__ == 'my doc'
    assert c.func is foo
    assert c.func_name == 'foo'
    assert c.args == ()
    assert c.keywords == {}


def test_curry_call_and_bind():
    def add3(a, b, c):
        return a + b + c

    c = curry(add3, 1)
    assert c.call(2, 3) == 6
    b = c.bind(2)
    assert isinstance(b, curry)
    assert b(3) == 6


def test_curry_uses_partial_unpacking():
    def add(x, y):
        return x + y

    c1 = curry(add, 1)
    c2 = curry(c1, 2)
    assert c2() == 3
    assert c2.func is add
    assert c2.args == (1, 2)


def test_curry_partial_unpacking():
    def add(x, y):
        return x + y

    p = partial(add, 1)
    c = curry(p, 2)
    assert c() == 3
    assert c.func is add


def test_curry_equality_hash_repr_str():
    def add(x, y):
        return x + y

    c1 = curry(add, 1)
    c2 = curry(add, 1)
    c3 = curry(add, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert repr(c1) == repr(add)
    assert str(c1) == str(add)
    assert c1 != 5


def test_curry_genuine_typeerror_not_curried():
    def f(x):
        return x + 'a'

    c = curry(f)
    with pytest.raises(TypeError):
        c(1)


def test_curry_raises_typeerror_for_bad_args():
    def add(x, y):
        return x + y

    c = curry(add)
    with pytest.raises(TypeError):
        c(1, 2, 3, 4)


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ['b', 'c']
    assert sig.parameters['b'].default is no_default


def test_curry_signature_incorrect_args():
    def f(a):
        return a

    c = curry(f, 1, 2)
    with pytest.raises(TypeError, match='incorrect arguments'):
        c.__signature__


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x, y):
            return self.val + x + y

        val = 100

    c = C()
    b = C.__dict__['method'].__get__(c, C)
    assert isinstance(b, curry)
    assert b(1, 2) == 103
    assert C.__dict__['method'].__get__(None, C) is C.__dict__['method']


def test_curry_varargs_always_curries():
    calls = []

    def deco(*args, **kwargs):
        calls.append((args, kwargs))
        return 'called'

    c = curry(deco)
    # Non-error call returns directly.
    assert c(1, 2) == 'called'


def test_curry_pickle_roundtrip():
    for func in (compose, curry, memoize):
        assert pickle.loads(pickle.dumps(func)) is func


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_unary():
    calls = []

    @memoize
    def sq(x):
        calls.append(x)
        return x * x

    assert sq(3) == 9
    assert sq(3) == 9
    assert calls == [3]


def test_memoize_with_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        raise AssertionError('should be cached')

    assert add(1, 2) == 3


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert calls == [(1, 2)]


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='must be hashable'):
        f([1, 2, 3])


def test_memoize_kwargs_cache():
    calls = []

    @memoize
    def add(x, y=0):
        calls.append((x, y))
        return x + y

    assert add(1, y=2) == 3
    assert add(1, y=2) == 3
    assert len(calls) == 1


def test_memoize_dunder():
    def f(x):
        """docstring"""
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'docstring'
    assert m.__wrapped__ is f


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_no_args():
    assert compose() is identity


def test_compose_single():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_order():
    inc = lambda i: i + 1
    f = compose(str, inc)
    assert f(3) == '4'
    assert isinstance(f, Compose)


def test_compose_left_order():
    inc = lambda i: i + 1
    f = compose_left(inc, str)
    assert f(3) == '4'


def test_compose_left_no_args():
    assert compose_left() is identity


def test_compose_call_kwargs():
    def f(x, y):
        return x + y

    def g(v):
        return v * 2

    c = compose(g, f)
    assert c(1, y=2) == 6


def test_compose_name():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert c.__name__ == 'foo_of_bar'


def test_compose_repr():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    r = repr(c)
    assert r.startswith('Compose(')
    assert 'foo' in r and 'bar' in r


def test_compose_eq_ne_hash():
    def foo(x):
        return x

    def bar(x):
        return x

    c1 = compose(foo, bar)
    c2 = compose(foo, bar)
    c3 = compose(bar, foo)
    assert c1 == c2
    assert not (c1 != c2)
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1.__eq__(5) is NotImplemented
    assert c1.__ne__(5) is NotImplemented


def test_compose_doc():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert 'foo' in c.__doc__ and 'bar' in c.__doc__
    assert c.__doc__.startswith('lambda')


def test_compose_doc_no_dunder_name():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__doc__ == 'A composition of functions'


def test_compose_name_no_dunder_name():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__name__ == 'Compose'


def test_compose_signature():
    def f(x: int) -> str:
        return str(x)

    def g(y: int) -> int:
        return y

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation is str


def test_compose_descriptor():
    class C:
        @compose(lambda self: self.val, lambda self: self.val + 1)
        def val_plus(self, _extra=None):
            pass

        val = 10

    c = C()
    # Simpler: test __get__ behavior directly.
    comp = Compose([lambda x: x + 1, lambda x: x])
    bound = comp.__get__(c, C)
    assert bound(5) == 6
    assert comp.__get__(None, C) is comp


def test_compose_getstate_setstate():
    def foo(x):
        return x

    c = compose(foo, foo)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2(3) == 3


def test_compose_wrapped():
    def foo(x):
        return x

    c = compose(foo, foo)
    assert c.__wrapped__ is foo


def test_compose_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(y: int) -> float:
        return float(y)

    c = compose(f, g)
    ann = c._combined_annotations()
    assert ann.get('x') is int
    assert ann.get('return') is str


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(5) == 5


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------

def test_juxt():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y):
        return x + y

    def g(x, y):
        return x * y

    assert juxt(f, g)(2, y=3) == (5, 6)


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(1) == (2,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError()) is None


def test_excepts():
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


def test_excepts_attributes():
    def func(x):
        """func doc"""
        return x

    def handler(e):
        """handler doc"""
        return 0

    e = excepts(ValueError, func, handler)
    assert e.exc is ValueError
    assert e.func is func
    assert e.handler is handler


def test_excepts_doc():
    def func(x):
        return x

    def handler(e):
        return 0

    e = excepts(ValueError, func, handler)
    assert 'func' in e.__doc__
    assert 'ValueError' in e.__doc__


def test_excepts_doc_tuple():
    def func(x):
        return x

    e = excepts((ValueError, KeyError), func)
    assert '(ValueError, KeyError)' in e.__doc__


def test_excepts_class_doc():
    e = excepts(ValueError, lambda x: x)
    # Access via class -> classval is the class docstring
    assert excepts.__dict__['__doc__'].classval is excepts.__doc__


def test_excepts_name():
    def func(x):
        return x

    e = excepts(ValueError, func)
    assert e.__name__ == 'func_excepting_ValueError'


def test_excepts_name_tuple():
    def func(x):
        return x

    e = excepts((ValueError, KeyError), func)
    assert e.__name__ == 'func_excepting_ValueError_or_KeyError'


def test_excepts_name_fallback():
    e = excepts(ValueError, 5)
    assert e.__name__ == 'excepting'


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    def g(*args, **kwargs):
        pass

    assert num_required_args(f) == 2
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

    def g(x):
        return x

    assert has_keywords(f) is True
    assert has_keywords(g) is False


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(add, (1, 2), {'x': 1}) is False


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    def h(*args):
        return args

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False


def test_is_arity_variants():
    def f(x, y):
        return x + y

    def g(**kwargs):
        return kwargs

    assert is_arity(2, f) is True
    assert is_arity(1, f) is False
    assert is_arity(0, g) is False


def test_is_valid_args_map():
    # Builtin with known signature registry fallback.
    assert is_valid_args(map, (), {}) is False
    assert is_partial_args(map, (), {}) is True
