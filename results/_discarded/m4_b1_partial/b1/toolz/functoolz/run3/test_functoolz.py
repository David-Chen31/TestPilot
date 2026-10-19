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


def test_apply_kwargs():
    def f(a, b, c=0):
        return a + b + c

    assert apply(f, 1, 2, c=3) == 6


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_plain():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(42) == 42


def test_thread_last_plain():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


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

def test_instanceproperty():
    class MyClass:
        """Class docstring."""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return "An object docstring"

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == "Class docstring."
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == "An object docstring"
    assert obj.val == 42


def test_instanceproperty_with_setter():
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


def test_instance_property_is_property():
    assert isinstance(InstanceProperty(lambda self: 1), property)


def test_instanceproperty_partial_no_fget():
    p = instanceproperty()
    assert isinstance(p, partial)


def test_instanceproperty_reduce():
    class C:
        @instanceproperty
        def val(self):
            return 1

    prop = C.__dict__['val']
    func, state = prop.__reduce__()
    assert func is InstanceProperty
    assert state[0] is prop.fget


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


def test_curry_all_at_once():
    def mul(x, y):
        return x * y

    assert curry(mul)(2, 10) == 20


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_requires_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_attributes():
    def f(x):
        """docstring"""
        return x

    c = curry(f)
    assert c.func is f
    assert c.args == ()
    assert c.keywords == {}
    assert c.__name__ == 'f'
    assert c.__doc__ == 'docstring'


def test_curry_bind_and_call():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    assert c(2) == 3
    bound = c.bind(3)
    assert bound() == 4


def test_curry_call_method():
    def add(x, y):
        return x + y

    c = curry(add)
    with pytest.raises(TypeError):
        c.call(1)
    assert c.call(1, 2) == 3


def test_curry_genuine_typeerror():
    def f(x):
        raise TypeError("inner")

    c = curry(f)
    with pytest.raises(TypeError):
        c(1)


def test_curry_equality_hash():
    def add(x, y):
        return x + y

    a = curry(add, 1)
    b = curry(add, 1)
    c = curry(add, 2)
    assert a == b
    assert a != c
    assert not (a == 5)
    assert hash(a) == hash(b)


def test_curry_str_repr():
    def add(x, y):
        return x + y

    c = curry(add)
    assert str(c) == str(add)
    assert repr(c) == repr(add)


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters)
    assert 'x' not in params
    assert 'y' in params


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
            return x + 1

    c = C()
    assert c.method(2) == 3
    assert C.method.func is C.__dict__['method'].func


def test_curry_partial_unpacking():
    def add(x, y, z):
        return x + y + z

    p = partial(add, 1)
    c = curry(p, 2)
    assert c(3) == 6


def test_curry_pickle():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    # Local functions cannot be pickled by reference; use a module-level one.
    assert c(2) == 3


def test_curry_varargs_function():
    def f(*args):
        return sum(args)

    c = curry(f)
    assert c(1) == 1
    assert c(1, 2) == 3


def test_curry_func_name():
    def add(x, y):
        return x + y

    c = curry(add)
    assert c.func_name == 'add'


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


def test_memoize_with_cache_kwarg():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_kwargs_distinct():
    calls = []

    @memoize
    def f(x, *, verbose=False):
        calls.append((x, verbose))
        return x

    f(1, verbose=False)
    f(1, verbose=True)
    assert len(calls) == 2


def test_memoize_unary():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def f(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    f(1, 2, verbose=True)
    f(1, 2, verbose=False)
    assert len(calls) == 1


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_wrapped_and_doc():
    def f(x):
        """doc"""
        return x

    m = memoize(f)
    assert m.__wrapped__ is f
    assert m.__doc__ == 'doc'
    assert m.__name__ == 'f'


# ---------------------------------------------------------------------------
# Compose / compose / compose_left / pipe
# ---------------------------------------------------------------------------

def test_compose_zero():
    assert compose() is identity


def test_compose_one():
    def f(x):
        return x + 1

    assert compose(f) is f


def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_order():
    f = lambda x: x + 1
    g = lambda x: x * 2
    # compose(f, g)(x) == f(g(x))
    assert compose(f, g)(3) == 7


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_order():
    f = lambda x: x + 1
    g = lambda x: x * 2
    assert compose_left(f, g)(3) == 8


def test_compose_kwargs():
    def f(a, b, c=0):
        return a + b + c

    comp = compose(lambda x: x * 2, f)
    assert comp(1, 2, c=3) == 12


def test_compose_class_attributes():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    assert isinstance(c, Compose)
    assert c.first is g
    assert c.funcs == (f,)
    assert c.__wrapped__ is g


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert 'f(g(*args, **kwargs))' in c.__doc__


def test_compose_doc_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__doc__ == 'A composition of functions'
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert repr(c).startswith('Compose(')


def test_compose_eq():
    def f(x):
        return x

    def g(x):
        return x

    assert compose(f, g) == compose(f, g)
    assert not (compose(f, g) != compose(f, g))
    assert compose(f, g) != compose(g, f)
    assert (compose(f, g) == object()) is False


def test_compose_hash():
    def f(x):
        return x

    c = compose(f, f)
    assert isinstance(hash(c), int)


def test_compose_descriptor():
    # A Compose used as a class attribute behaves like a function descriptor:
    # accessing it on an instance binds the instance as the first argument to
    # the first-applied (rightmost) function.
    class C:
        method = compose(str, lambda self, x: x + 1)

    c = C()
    assert c.method(1) == '2'
    # The bound callable is a MethodType wrapping the Compose.
    bound = c.method
    assert isinstance(bound, type(bound))
    assert bound.__self__ is c
    # The unbound access returns the Compose object itself.
    assert C.method is C.__dict__['method']


def test_compose_signature():
    def f(x: int, y: int) -> int:
        return x + y

    def g(a: int) -> int:
        return a

    c = compose(g, f)
    sig = c.__signature__
    assert list(sig.parameters) == ['x', 'y']
    assert sig.return_annotation is int


def test_compose_getstate_setstate():
    def f(x):
        return x

    c = compose(f, f)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2.first is c.first
    assert c2.funcs == c.funcs


def test_compose_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(y: int) -> int:
        return y

    c = compose(f, g)
    ann = c._combined_annotations()
    assert ann.get('return') is str


def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3


# ---------------------------------------------------------------------------
# complement / juxt / do / flip
# ---------------------------------------------------------------------------

def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


def test_juxt():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_single_callable():
    inc = lambda x: x + 1
    j = juxt(inc)
    assert j.funcs == (inc,)
    assert j(1) == (2,)


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2.funcs == j.funcs


def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


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


def test_excepts_multiple_default_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_name():
    def f(x):
        return x

    e = excepts(ValueError, f)
    assert e.__name__ == 'f_excepting_ValueError'


def test_excepts_name_tuple():
    def f(x):
        return x

    e = excepts((ValueError, KeyError), f)
    assert e.__name__ == 'f_excepting_ValueError_or_KeyError'


def test_excepts_doc():
    def f(x):
        """f doc"""
        return x

    e = excepts(ValueError, f)
    doc = e.__doc__
    assert 'f doc' in doc
    assert 'ValueError' in doc


def test_excepts_doc_no_name():
    class NotCallable:
        pass

    e = excepts(ValueError, NotCallable())
    assert e.__doc__ == excepts.__doc__


def test_return_none():
    assert return_none(ValueError()) is None


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

    assert has_keywords(f) is True

    def g(x):
        return x

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

    assert is_arity(1, f) is True

    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False

    def h(*args):
        return args

    assert is_arity(1, h) is False


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


# ---------------------------------------------------------------------------
# pickle round-trips for module-level functions
# ---------------------------------------------------------------------------

def _module_level_add(x, y):
    return x + y


def _module_level_inc(x):
    return x + 1


def test_curry_pickle_module_level():
    c = curry(_module_level_add, 1)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert restored(2) == 3


def test_compose_pickle():
    # compose applies right-to-left; give the first function a single arg.
    c = compose(str, _module_level_inc)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert restored(1) == '2'
