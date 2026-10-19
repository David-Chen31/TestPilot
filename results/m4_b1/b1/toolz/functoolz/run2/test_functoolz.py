import pickle
import sys
import inspect
import pytest

from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left, pipe,
    complement, juxt, do, flip, return_none, excepts,
    num_required_args, has_varargs, has_keywords, is_valid_args,
    is_partial_args, is_arity,
)


# ---------------------------------------------------------------- basics

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3
    assert apply(lambda x, y=10: x + y, 1, y=5) == 6
    with pytest.raises(TypeError):
        apply()


# ------------------------------------------------------------ threading

def test_thread_first():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, inc, double) == 4
    assert thread_first(1, (add, 4), (pow_, 2)) == 25
    assert thread_first(5) == 5


def test_thread_last():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, inc, double) == 4
    assert thread_last(1, (add, 4), (pow_, 2)) == 32
    assert thread_last(5) == 5


# ------------------------------------------------------ instanceproperty

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


def test_instanceproperty_setter():
    class C:
        def __init__(self):
            self._x = 1

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, v):
            self._x = v

    c = C()
    assert c.x == 1
    c.x = 5
    assert c.x == 5


def test_instanceproperty_partial_like():
    # called with no fget -> returns partial
    factory = instanceproperty(classval=99)
    assert callable(factory)

    class C:
        pass

    p = factory(lambda self: 7)
    assert isinstance(p, InstanceProperty)
    assert p.classval == 99


def test_instanceproperty_pickle():
    # Local functions cannot be pickled; test the __reduce__ shape instead.
    fget = lambda self: 42
    ip = InstanceProperty(fget=fget, classval=5)
    reducer, state = ip.__reduce__()
    assert reducer is InstanceProperty
    assert state[0] is fget
    assert state[1] is None
    assert state[2] is None
    assert state[4] == 5


# ----------------------------------------------------------------- curry

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    assert cmul(2, 10) == 20
    assert cmul(2)(10) == 20
    double = cmul(2)
    assert double(10) == 20
    assert double(3) == 6


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5
    assert f(2, 3, a=2) == 10


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(3)


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_attributes():
    def myfunc(x, y):
        """my docstring"""
        return x + y

    c = curry(myfunc, 1)
    assert c.func is myfunc
    assert c.args == (1,)
    assert c.keywords == {}
    assert c.__name__ == 'myfunc'
    assert c.__doc__ == 'my docstring'
    assert c.func_name == 'myfunc'
    assert str(c) == str(myfunc)
    assert repr(c) == repr(myfunc)


def test_curry_from_partial_like():
    from functools import partial

    def add(x, y, z):
        return x + y + z

    p = partial(add, 1, 2)
    c = curry(p, 3)
    # args are merged: 1, 2, 3; add(1, 2, 3) == 6
    assert c.args == (1, 2, 3)
    assert c.keywords == {}
    assert c() == 6


def test_curry_partial_kwargs_merge():
    from functools import partial

    def f(a, b, c=0):
        return a + b + c

    p = partial(f, c=10)
    c = curry(p, a=1)
    assert c.keywords == {'c': 10, 'a': 1}


def test_curry_eq_hash():
    def f(x, y):
        return x + y

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert a != 42


def test_curry_bind():
    def f(x, y):
        return x + y

    c = curry(f)
    b = c.bind(1)
    assert isinstance(b, curry)
    assert b(2) == 3


def test_curry_call_method():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    assert c.call(2) == 3


def test_curry_descriptor():
    class C:
        def __init__(self, base):
            self.base = base

        @curry
        def add(self, x):
            return self.base + x

    c = C(10)
    assert c.add(5) == 15
    # Unbound access
    assert isinstance(C.__dict__['add'], curry)


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    # 'a' removed (bound), 'b' and 'c' remain
    assert [p.name for p in params] == ['b', 'c']


def test_curry_str():
    def f(x):
        return x

    c = curry(f)
    assert str(c) == str(f)


def test_curry_reduce():
    def module_level_fn(x, y):
        return x + y

    c = curry(module_level_fn, 1)
    # Pickle-ability might fail for local functions; test class structure only
    assert callable(c)


# --------------------------------------------------------------- memoize

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    madd = memoize(add)
    assert madd(1, 2) == 3
    assert madd(1, 2) == 3
    assert len(calls) == 1
    assert madd(2, 3) == 5
    assert len(calls) == 2


def test_memoize_unary():
    calls = []

    def f(x):
        calls.append(x)
        return x * 2

    mf = memoize(f)
    assert mf(3) == 6
    assert mf(3) == 6
    assert calls == [3]


def test_memoize_with_cache():
    cache = {(1, 2): 3}

    def add(x, y):
        return x + y

    madd = memoize(add, cache=cache)
    assert madd(1, 2) == 3
    # cache dict mutated
    assert (1, 2) in cache


def test_memoize_custom_key():
    calls = []

    def add(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    madd = memoize(add, key=lambda args, kwargs: args)
    assert madd(1, 2, verbose=True) == 3
    assert madd(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_kwargs_distinct():
    def f(x, y=0):
        return x + y

    mf = memoize(f)
    assert mf(1, y=2) == 3
    assert mf(1, y=3) == 4


def test_memoize_unhashable():
    def f(x):
        return x

    mf = memoize(f)
    with pytest.raises(TypeError):
        mf([1, 2, 3])


def test_memoize_wrapped():
    def f(x):
        return x

    mf = memoize(f)
    assert mf.__wrapped__ is f
    assert mf.__name__ == 'f'


def test_memoize_decorator_form():
    @memoize
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3


def test_memoize_decorator_with_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100


# --------------------------------------------------------------- compose

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    f = lambda x: x + 1
    assert compose(f) is f


def test_compose_multiple():
    f = lambda x: x + 1
    g = lambda x: x * 2
    h = lambda x: x - 3
    assert compose(f, g, h)(10) == (10 - 3) * 2 + 1


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_multiple():
    f = lambda x: x + 1
    g = lambda x: x * 2
    assert compose_left(f, g)(3) == 8


def test_compose_left_empty():
    assert compose_left() is identity


def test_compose_obj():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    assert isinstance(c, Compose)
    assert c.first is g
    assert c.funcs == (f,)


def test_compose_doc():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(inc, double)
    assert c.__doc__ == 'lambda *args, **kwargs: inc(double(*args, **kwargs))'


def test_compose_name():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(inc, double)
    assert c.__name__ == 'inc_of_double'


def test_compose_repr():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(inc, double)
    assert 'Compose' in repr(c)


def test_compose_eq():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    assert compose(inc, double) == compose(inc, double)
    assert compose(inc, double) != compose(double, inc)
    # not a Compose
    assert Compose.__eq__(compose(inc, double), 42) is NotImplemented


def test_compose_hash():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c1 = compose(inc, double)
    c2 = compose(inc, double)
    assert hash(c1) == hash(c2)


def test_compose_getstate_setstate():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(inc, double)
    state = c.__getstate__()
    # state is (first, funcs) = (double, (inc,))
    assert state == (double, (inc,))
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2.first is double
    assert c2.funcs == (inc,)


def test_compose_pickle():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    # Functions defined here are not picklable by name; only check state
    c = compose(inc, double)
    assert c.__getstate__() == (double, (inc,))


def test_compose_descriptor():
    def transform(x):
        return x * 2

    class C:
        def __init__(self, base):
            self.base = base

        method = compose(lambda x: x + 1, lambda x: x)

    c = C(10)
    # Compose binds like a method
    assert C.__dict__['method'].__get__(None, C) is C.__dict__['method']


def test_compose_signature():
    def f(a: int, b: int) -> int:
        return a + b

    def g(x) -> str:
        return str(x)

    c = compose(g, f)
    sig = c.__signature__
    assert list(sig.parameters.keys()) == ['a', 'b']
    assert sig.return_annotation is str


def test_compose_annotations():
    def f(a: int, b: str) -> int:
        return 1

    def g(x: float) -> str:
        return ''

    c = compose(g, f)
    ann = c.__annotations__
    assert ann['a'] is int
    assert ann['b'] is str
    assert ann['return'] is str


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(g, f)
    assert c.__wrapped__ is f


# ------------------------------------------------------------------ pipe

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(5) == 5
    assert pipe(1, lambda x: x + 1, lambda x: x * 3) == 6


# ----------------------------------------------------------- complement

def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------- juxt

def test_juxt():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_funcs_attr():
    inc = lambda x: x + 1
    j = juxt(inc)
    assert j.funcs == (inc,)


def test_juxt_kwargs():
    f = lambda x, y=0: x + y
    g = lambda x, y=0: x * y
    assert juxt(f, g)(3, y=4) == (7, 12)


def test_juxt_pickle():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    assert state == (inc,)
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2.funcs == (inc,)


# ------------------------------------------------------------------ do

def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


def test_do_composed():
    log = []
    inc = lambda x: x + 1
    from toolz import compose as compose_
    from toolz.curried import do as cdo
    f = compose_(inc, cdo(log.append))
    assert f(1) == 2
    assert f(11) == 12
    assert log == [1, 11]


# ---------------------------------------------------------------- flip

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


# ------------------------------------------------------------ return_none

def test_return_none():
    assert return_none(ValueError('x')) is None
    assert return_none(None) is None


# -------------------------------------------------------------- excepts

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
    def myfunc(x):
        """docstring for myfunc"""
        return x

    def myhandler(e):
        """handler doc"""
        return None

    e = excepts(ValueError, myfunc, myhandler)
    assert 'valueerror' in e.__doc__.lower() or 'ValueError' in e.__doc__
    assert 'myfunc' in e.__doc__
    assert 'myhandler' in e.__doc__


def test_excepts_name():
    e = excepts(ValueError, lambda x: x)
    assert e.__name__.endswith('_excepting_ValueError')


def test_excepts_name_tuple():
    e = excepts((IndexError, KeyError), lambda x: x)
    assert e.__name__.endswith('_excepting_IndexError_or_KeyError')


def test_excepts_name_default():
    class Weird:
        pass

    # func without __name__
    e = excepts(ValueError, Weird())
    assert e.__name__ == 'excepting'


# ------------------------------------- introspection helper functions

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

    def h(x):
        return x

    assert has_varargs(f) is True
    assert has_varargs(g) is False
    assert has_varargs(h) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x, *args, **kwargs):
        pass

    def h(x):
        return x

    assert has_keywords(f) is True
    assert has_keywords(g) is True
    assert has_keywords(h) is False


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(add, (1, 2, 3), {}) is False


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


def test_is_arity_two():
    def f(x, y):
        return x + y

    assert is_arity(2, f) is True
    assert is_arity(1, f) is False


# ------------------------------------- curry interacting with introspection

def test_curry_arity():
    def f(x, y):
        return x + y

    c = curry(f)
    # `curry` has a computed __signature__; is_arity consults the underlying
    # signature introspection and reports False for curry objects.
    assert is_arity(2, c) is False


def test_curry_should_curry_true_on_missing():
    def f(a, b, c):
        return a + b + c

    c = curry(f)
    # Calling with too few args should curry, not raise
    partial_c = c(1)
    assert isinstance(partial_c, curry)
    assert partial_c(2)(3) == 6


def test_curry_varargs_curries():
    def f(*args, **kwargs):
        if len(args) < 2:
            raise TypeError("need 2")
        return args[0] + args[1]

    c = curry(f)
    result = c(1)
    # Because function has varargs, TypeError -> curry anyway
    assert isinstance(result, curry)


def test_curry_raises_on_genuine_type_error():
    def f(x):
        return x + "not compatible"

    c = curry(f)
    with pytest.raises(TypeError):
        c(1)


def test_curry_invalid_args_raises():
    def f(x, y):
        return x + y

    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__
