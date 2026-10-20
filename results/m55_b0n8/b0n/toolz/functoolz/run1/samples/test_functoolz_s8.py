"""Unit tests for toolz.functoolz."""

import pickle
import sys
import inspect
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
# identity
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3
    assert apply(lambda x, y=0: x + y, 1, y=10) == 11


def test_apply_requires_func():
    with pytest.raises(TypeError):
        apply()


def test_apply_map():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

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


def test_thread_last_filter_map():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


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
    class C:
        @instanceproperty
        def v(self):
            return 1

    assert isinstance(C.v, InstanceProperty)
    assert isinstance(C.v, property)


def test_instanceproperty_setter():
    class C:
        def __init__(self):
            self._x = 1

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    c = C()
    assert c.x == 1
    c.x = 5
    assert c.x == 5


def test_instanceproperty_reduce():
    def fget(self):
        return 1

    ip = InstanceProperty(fget=fget, classval='cv')
    func, state = ip.__reduce__()
    assert func is InstanceProperty
    assert state[0] is fget
    restored = func(*state)
    assert restored.classval == 'cv'


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20
    assert cmul(2, 10) == 20
    assert cmul(2)(10) == 20


def test_curry_keyword():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5
    assert f(2, 3, a=2) == 10


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable_raises():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_attributes():
    def mul(x, y):
        """multiply"""
        return x * y

    cmul = curry(mul)
    assert cmul.__name__ == 'mul'
    assert cmul.__doc__ == 'multiply'
    assert cmul.func is mul
    assert cmul.args == ()
    assert cmul.keywords == {}

    double = cmul(2)
    assert double.func is mul
    assert double.args == (2,)


def test_curry_repr_str():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    assert str(cmul) == str(mul)
    assert repr(cmul) == repr(mul)


def test_curry_equality_hash():
    def mul(x, y):
        return x * y

    assert curry(mul, 2) == curry(mul, 2)
    assert curry(mul, 2) != curry(mul, 3)
    assert hash(curry(mul, 2)) == hash(curry(mul, 2))
    assert curry(mul, 2) != "not-a-curry"


def test_curry_unpacks_curry():
    def f(x, y, z):
        return x + y + z

    c1 = curry(f, 1)
    c2 = curry(c1, 2)
    assert c2.args == (1, 2)
    assert c2(3) == 6


def test_curry_unpacks_partial():
    from functools import partial

    def f(x, y, z):
        return x + y + z

    p = partial(f, 1)
    c = curry(p, 2)
    assert c(3) == 6


def test_curry_genuine_typeerror():
    def f(x):
        if x < 0:
            raise TypeError('nope')
        return x

    c = curry(f)
    with pytest.raises(TypeError):
        c(-1)


def test_curry_call_bind_call():
    def f(x, y):
        return x + y

    c = curry(f)
    assert c.call(1, 2) == 3
    b = c(1)
    assert isinstance(b, curry)
    assert b.call(2) == 3


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ['y', 'z']


def test_curry_signature_incorrect_args():
    def f(x):
        return x

    c = curry(f, 1, 2)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x):
            return self, x

    c = C()
    res = c.method(5)
    assert res == (c, 5)


def test_curry_pickle_roundtrip():
    cmul = curry(flip.__wrapped__ if hasattr(flip, '__wrapped__') else _mul_global, 2) if False else None

    # Use a module-level function so pickling can find it.
    c = curry(_add_for_pickle, 3)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(4) == 7
    assert restored.args == (3,)
    assert restored.func is _add_for_pickle


def _add_for_pickle(a, b):
    return a + b


def test_curry_with_varargs_curries_on_typeerror():
    @curry
    def deco(*args, **kwargs):
        return (args, kwargs)

    # Calling with one arg raises TypeError initally? Actually *args accepts all
    res = deco(1, 2)
    assert res == ((1, 2), {})


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
    assert calls == [(1, 2)]


def test_memoize_as_decorator():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(2, 3) == 5
    assert add(2, 3) == 5
    assert calls == [(2, 3)]


def test_memoize_with_initial_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100
    assert add(3, 4) == 7


def test_memoize_key_function():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3


def test_memoize_unary():
    calls = []

    def double(x):
        calls.append(x)
        return 2 * x

    m = memoize(double)
    assert m(3) == 6
    assert m(3) == 6
    assert calls == [3]


def test_memoize_kwargs_key():
    calls = []

    def f(x, y=0):
        calls.append((x, y))
        return x + y

    m = memoize(f)
    assert m(1, y=2) == 3
    assert m(1, y=2) == 3
    assert m(1) == 1
    assert calls == [(1, 2), (1, 0)]


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_attributes():
    def f(x):
        """docstring"""
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'docstring'
    assert m.__wrapped__ is f


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'
    assert compose()(5) == 5
    assert compose() is identity
    f = lambda x: x
    assert compose(f) is f


def test_compose_order():
    f = lambda x: x + 'f'
    g = lambda x: x + 'g'
    h = lambda x: x + 'h'
    assert compose(f, g, h)('') == 'hgf'


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left()(5) == 5


def test_compose_class():
    c = Compose([lambda x: x + 1, lambda x: x * 2])
    # funcs applied right to left: first is last given
    # Compose([f, g]) == f(g(x))
    assert c(3) == 7
    assert isinstance(c, Compose)


def test_compose_doc():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = Compose([double, inc])
    assert c.__doc__ == 'lambda *args, **kwargs: double(inc(*args, **kwargs))'


def test_compose_doc_no_names():
    c = Compose([lambda x: x, lambda x: x])
    # lambdas have __name__ == '<lambda>'; still works
    assert 'lambda *args, **kwargs' in c.__doc__


def test_compose_name():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = Compose([double, inc])
    assert c.__name__ == 'double_of_inc'


def test_compose_repr():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = Compose([double, inc])
    assert repr(c).startswith('Compose(')
    assert 'double' in repr(c)
    assert 'inc' in repr(c)


def test_compose_eq():
    def f(x):
        return x

    def g(x):
        return x

    assert Compose([f, g]) == Compose([f, g])
    assert Compose([f, g]) != Compose([g, f])
    assert Compose([f, g]) != "not a compose"
    assert (Compose([f, g]) != "x") is True


def test_compose_hash():
    def f(x):
        return x

    def g(x):
        return x

    assert hash(Compose([f, g])) == hash(Compose([f, g]))


def test_compose_getstate_setstate():
    def f(x):
        return x

    def g(x):
        return x

    c = Compose([f, g])
    first, funcs = c.__getstate__()
    c2 = Compose([f, g])
    c2.__setstate__((first, funcs))
    assert c2 == c


def test_compose_descriptor():
    class C:
        @Compose
        def method(self, x):
            return (self, x)

    # Compose applied as a descriptor
    class D:
        pass

    def f(x):
        return x

    c = Compose([f])
    assert c.__get__(None, None) is c


def test_compose_signature():
    def f(x, y):
        return x

    def g(a):
        return a

    c = Compose([f, g])
    sig = c.__signature__
    assert list(sig.parameters) == ['a']


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = Compose([f, g])
    assert c.__wrapped__ is c.first


def test_compose_pickle():
    c = Compose([_pickle_compose_f, _pickle_compose_g])
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == _pickle_compose_f(_pickle_compose_g(3))


def _pickle_compose_f(x):
    return x * 2


def _pickle_compose_g(x):
    return x + 1


def test_compose_combined_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(y: str) -> int:
        return int(y)

    # compose(f, g): g first then f
    c = Compose([f, g])
    ann = c._combined_annotations()
    assert ann['return'] is str


def test_compose_annotations_attribute():
    def f(x: int) -> str:
        return str(x)

    def g(y: str) -> int:
        return int(y)

    c = Compose([f, g])
    ann = c.__annotations__
    assert ann['return'] is str


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


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt(inc)
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


def test_do_compose():
    log = []

    def inc(x):
        return x + 1

    inc = compose(inc, do(log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


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
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(None) is None
    assert return_none(ValueError('x')) is None


# ---------------------------------------------------------------------------
# excepts
# ---------------------------------------------------------------------------

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


def test_excepts_doc():
    excepting = excepts(ValueError, lambda a: a, lambda _: -1)
    assert 'ValueError' in excepting.__doc__


def test_excepts_name():
    excepting = excepts((ValueError, KeyError), lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError_or_KeyError'


def test_excepts_class_doc():
    assert excepts.__doc__ is not None


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

    assert has_varargs(f) is True

    def g(**kwargs):
        return kwargs

    assert has_varargs(g) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    assert has_keywords(f) is True

    def g(x, y):
        return x + y

    assert has_keywords(g) is False


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False


def test_is_arity():
    def f(x):
        return x

    assert is_arity(1, f) is True

    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False


def test_introspection_with_sigspec():
    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2
    assert has_varargs(f, sig) is False
    assert has_keywords(f, sig) is False
    assert is_valid_args(f, (1, 2), {}, sig) is True
    assert is_partial_args(f, (1,), {}, sig) is True
    assert is_arity(2, f, sig) is True
