import pickle
import inspect
import sys
import pytest

from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left, pipe,
    complement, juxt, do, flip, return_none, excepts, num_required_args,
    has_varargs, has_keywords, is_valid_args, is_partial_args, is_arity,
)


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


def test_apply():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3
    assert apply(lambda x, a=0: x + a, 2, a=3) == 5


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_first(1, inc, double) == 4
    add = lambda x, y: x + y
    pow_ = lambda x, y: x ** y
    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_last(1, inc, double) == 4
    add = lambda x, y: x + y
    pow_ = lambda x, y: x ** y
    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    inc = lambda x: x + 1
    iseven = lambda x: x % 2 == 0
    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_no_forms():
    assert thread_last(5) == 5


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_classval_and_instance():
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


def test_instanceproperty_returns_partial_when_no_fget():
    from functools import partial
    result = instanceproperty()
    assert isinstance(result, partial)


def test_InstanceProperty_reduce():
    def fget(self):
        return 1

    prop = InstanceProperty(fget=fget, classval='CLASS')
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    assert state[0] is fget
    assert state[4] == 'CLASS'


def test_instanceproperty_is_property_subclass():
    assert issubclass(InstanceProperty, property)


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic_partial_application():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert isinstance(double, curry)
    assert double(10) == 20
    assert cmul(2, 5) == 10


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_unpacks_partial_like():
    def add(x, y, z):
        return x + y + z

    c1 = curry(add, 1)
    c2 = curry(c1, 2)
    assert c2(3) == 6
    assert c2.args == (1, 2)


def test_curry_attributes():
    def foo(x):
        "foo doc"
        return x

    c = curry(foo)
    assert c.__name__ == 'foo'
    assert c.__doc__ == 'foo doc'
    assert c.func is foo
    assert c.func_name == 'foo'
    assert c.args == ()
    assert c.keywords == {}
    assert str(c) == str(foo)
    assert repr(c) == repr(foo)


def test_curry_call_raises_genuine_typeerror():
    def f(x):
        return x + 1

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2)


def test_curry_bind_and_call():
    def add(x, y):
        return x + y

    c = curry(add)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert c.call(1, 2) == 3


def test_curry_equality_hash():
    def add(x, y):
        return x + y

    a = curry(add, 1)
    b = curry(add, 1)
    c = curry(add, 2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_curry_descriptor():
    class C:
        def __init__(self):
            self.v = 10

        @curry
        def method(self, x):
            return self.v + x

    c = C()
    assert c.method(5) == 15
    assert C.method(5) is not None


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = inspect.signature(c)
    params = list(sig.parameters)
    assert params == ['y', 'z']


def test_curry_signature_incorrect_args():
    def f(x):
        return x

    c = curry(f)
    c._partial = __import__('functools').partial(f, 1, 2)
    with pytest.raises(TypeError):
        inspect.signature(c)


def test_curry_pickle_roundtrip():
    from toolz.functoolz import _module_level_curried_add
    c = _module_level_curried_add
    restored = pickle.loads(pickle.dumps(c))
    assert restored(1)(2) == 3


def test_curry_pickle_partial():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


def test_curry_get_returns_self_on_class():
    def add(x, y):
        return x + y

    c = curry(add)
    assert c.__get__(None, object) is c


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(add)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert len(calls) == 1


def test_memoize_decorator():
    @memoize
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add.__name__ == 'add'
    assert add.__wrapped__.__name__ == 'add'


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add(2, 3) == 5


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_unary():
    calls = []

    @memoize
    def double(x):
        calls.append(x)
        return 2 * x

    assert double(2) == 4
    assert double(2) == 4
    assert len(calls) == 1


def test_memoize_unhashable():
    @memoize
    def add(x, y):
        return x + y

    with pytest.raises(TypeError):
        add([1], [2])


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_empty_returns_identity():
    assert compose() is identity


def test_compose_single():
    f = lambda x: x + 1
    assert compose(f) is f


def test_compose_multiple():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity


def test_compose_object_properties():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    assert isinstance(c, Compose)
    assert c(3) == 7  # f(g(3)) -> f(6) -> 7
    assert c.first is g
    assert c.funcs == (f,)
    assert c.__name__ == 'f_of_g'


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert 'lambda *args, **kwargs:' in c.__doc__


def test_compose_eq_hash():
    def f(x):
        return x

    def g(x):
        return x

    c1 = compose(f, g)
    c2 = compose(f, g)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert (c1 != c2) is False
    assert (c1 == 3) is False


def test_compose_get_descriptor():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)

    class C:
        method = c

    obj = C()
    assert C.method is c
    assert obj.method(3) == 7


def test_compose_signature():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(f, g)
    sig = inspect.signature(c)
    assert sig.return_annotation is str


def test_compose_annotations():
    def f(x: int) -> float:
        return float(x)

    def g(y: int) -> int:
        return y

    c = compose(f, g)
    assert c.__annotations__ == {'y': int, 'return': float}


def test_compose_pickle():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    assert c.__getstate__() == (g, (f,))
    c2 = Compose.__new__(Compose)
    c2.__setstate__(c.__getstate__())
    assert c2(3) == 7


# ---------------------------------------------------------------------------
# pipe / complement
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(5) == 5


def test_complement():
    iseven = lambda n: n % 2 == 0
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
    f = lambda x, y=0: x + y
    g = lambda x, y=0: x * y
    assert juxt(f, g)(3, y=4) == (7, 12)


def test_juxt_pickle():
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
    result = do(log.append, 5)
    assert result == 5
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
    data = [1, 'a', 'b', 2, 1.5, 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError()) is None


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
    excepting = excepts(ValueError, lambda a: a, lambda e: -1)
    assert 'ValueError' in excepting.__doc__
    assert excepting.__name__.startswith('<')


def test_excepts_name():
    def f(a):
        return a

    def h(e):
        return -1

    excepting = excepts(ValueError, f, h)
    assert excepting.__name__ == 'f_excepting_ValueError'

    excepting2 = excepts((IndexError, KeyError), f, h)
    assert excepting2.__name__ == 'f_excepting_IndexError_or_KeyError'


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


def test_is_arity_builtin():
    # map takes at least 1 arg but is variadic
    assert is_arity(1, map) is False


# ---------------------------------------------------------------------------
# module-level helper used for pickling tests
# ---------------------------------------------------------------------------

from toolz.functoolz import curry as _curry


@_curry
def _module_level_curried_add(x, y):
    return x + y
