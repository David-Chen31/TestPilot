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
    assert apply(lambda x: x * 2, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3


def test_apply_kwargs():
    def f(a, b=0):
        return a + b
    assert apply(f, 1, b=2) == 3


def test_apply_no_args():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_simple():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert thread_first(1, inc, double) == 4


def test_thread_first_tuple_forms():
    add = lambda x, y: x + y
    powf = lambda x, y: x ** y
    assert thread_first(1, (add, 4), (powf, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_simple():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert thread_last(1, inc, double) == 4


def test_thread_last_tuple_forms():
    add = lambda x, y: x + y
    powf = lambda x, y: x ** y
    assert thread_last(1, (add, 4), (powf, 2)) == 32


def test_thread_last_map_filter():
    inc = lambda x: x + 1
    iseven = lambda x: x % 2 == 0
    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_no_forms():
    assert thread_last(5) == 5


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_classattr():
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


def test_instanceproperty_setter_getter():
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


def test_instanceproperty_reduce():
    class C:
        @instanceproperty
        def val(self):
            return 42

    prop = C.__dict__['val']
    assert isinstance(prop, InstanceProperty)
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    fget, fset, fdel, doc, classval = state
    assert fget is prop.fget
    assert classval is None


def test_isinstance_property_class():
    class C:
        @instanceproperty
        def val(self):
            return 1

    assert isinstance(C.__dict__['val'], InstanceProperty)
    assert issubclass(InstanceProperty, property)


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


def test_curry_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5
    assert f(2, 3, a=2) == 10


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_attributes():
    def myfunc(x):
        "My docstring"
        return x
    c = curry(myfunc)
    assert c.__name__ == 'myfunc'
    assert c.__doc__ == 'My docstring'
    assert c.func is myfunc
    assert c.func_name == 'myfunc'
    assert c.args == ()
    assert c.keywords == {}
    assert str(c) == str(myfunc)
    assert repr(c) == repr(myfunc)


def test_curry_partial_of_partial():
    def add3(a, b, c):
        return a + b + c
    c1 = curry(add3, 1)
    c2 = curry(c1, 2)
    assert c2(3) == 6


def test_curry_hash_eq():
    def f(x, y):
        return x + y
    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_curry_bind_and_call():
    def f(x, y):
        return x + y
    c = curry(f)
    bound = c.bind(1)
    assert bound(2) == 3
    with pytest.raises(TypeError):
        c.call(1)


def test_curry_type_error_not_curried():
    def f(x):
        return x + 1
    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2)


def test_curry_get_descriptor():
    @curry
    def add(x, y):
        return x + y

    class C:
        method = add

    c = C()
    bound = c.method
    assert isinstance(bound, curry)
    assert bound(3) == 3 + 3


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c
    c = curry(f, 1)
    sig = c.__signature__
    assert 'a' not in sig.parameters
    assert 'b' in sig.parameters
    assert 'c' in sig.parameters


def test_curry_signature_incorrect_args():
    def f(a, b):
        return a + b
    c = curry(f, 1)
    bad = curry.__new__(curry)
    # construct manually to bypass args checks
    bad._partial = partial(lambda x: x)  # force weird
    # Simpler: too many args makes bind_partial fail
    with pytest.raises(TypeError):
        curry(f, 1, 2, 3)()


def test_curry_varargs_curry_anyway():
    @curry
    def f(*args, **kwargs):
        return (args, kwargs)
    # With varargs, invalid calls become curry objects
    result = f(1)
    assert result == ((1,), {})
    result2 = f(1, 2)
    assert result2 == ((1, 2), {})


def test_curry_call_matches_kwargs():
    @curry
    def f(a, b, c):
        return a + b + c
    assert f(1)(2)(3) == 6
    assert f(1, 2, 3) == 6
    assert f(a=1)(b=2)(c=3) == 6


def test_curry_exception_genuine_type_error():
    @curry
    def f(a, b):
        return a + b
    c = curry(f, 1)
    with pytest.raises(TypeError):
        c.call(None)  # actually let's check call with insufficient


def test_curry_pickle():
    def f(x, y):
        return x + y
    c = curry(f, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


def test_curry_should_curry_no_curry():
    def f(a, b):
        return a + b
    c = curry(f)
    # too many args -> TypeError, cannot be made valid
    with pytest.raises(TypeError):
        c(1, 2, 3)


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    madd = memoize(add)
    assert madd(1, 2) == 3
    assert madd(1, 2) == 3
    assert calls == [(1, 2)]
    assert madd.__name__ == 'add'
    assert madd.__wrapped__ is add


def test_memoize_custom_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_key_function():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, verbose=True) == 3
    assert calls == [(1, 2)]


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x
    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_unary():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_memoize_docs():
    @memoize
    def f(x):
        "docstring"
        return x

    assert f.__doc__ == 'docstring'


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    f = lambda x: x + 1
    assert compose(f) is f


def test_compose_basic():
    inc = lambda x: x + 1
    c = compose(str, inc)
    assert c(3) == '4'
    assert isinstance(c, Compose)


def test_compose_left():
    inc = lambda x: x + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity


def test_compose_multiple_args():
    def add(x, y):
        return x + y
    c = compose(str, add)
    assert c(1, 2) == '3'


def test_compose_name():
    def f(x):
        return x
    def g(x):
        return x
    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_repr():
    def f(x):
        return x
    def g(x):
        return x
    c = compose(f, g)
    assert repr(c) == f'Compose{(f, g)!r}'


def test_compose_eq():
    f = lambda x: x + 1
    g = lambda x: x * 2
    c1 = compose(f, g)
    c2 = compose(f, g)
    assert c1 == c2
    assert not (c1 != c2)


def test_compose_hash():
    f = lambda x: x + 1
    g = lambda x: x * 2
    c = compose(f, g)
    assert hash(c) == hash(f) ^ hash(g)


def test_compose_doc():
    def f(x):
        return x
    def g(x):
        return x
    c = compose(f, g)
    assert 'f(g(' in c.__doc__


def test_compose_doc_no_names():
    c = Compose([1, 2])
    assert c.__doc__ == 'A composition of functions'


def test_compose_signature():
    def f(x: int) -> float:
        return float(x)
    def g(y: str) -> int:
        return int(y)
    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation == float


def test_compose_get_descriptor():
    def f(x):
        return x
    def g(x):
        return x
    c = compose(f, g)

    class C:
        method = c

    inst = C()
    bound = inst.method
    assert bound(5) == 5


def test_compose_wrapped():
    def f(x):
        return x
    def g(x):
        return x
    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_annotations():
    def f(x: int) -> float:
        return float(x)
    def g(y: str) -> int:
        return int(y)
    c = compose(f, g)
    ann = c.__annotations__
    assert ann['x'] == int
    assert ann['return'] == float


def test_compose_pickles():
    f = lambda x: x + 1
    g = lambda x: x * 2
    c = compose(f, g)
    # lambdas can't be pickled; test getstate/setstate
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2.first == c.first
    assert c2.funcs == c.funcs


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_empty():
    assert pipe(5) == 5


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

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


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y
    def g(x, y=0):
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


def test_flip_curried():
    def div(a, b):
        return a // b
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
    assert 'except' in excepting.__doc__


def test_excepts_name():
    excepting = excepts(ValueError, lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_tuple():
    def f(x):
        return x
    def h(e):
        return None
    excepting = excepts((ValueError, KeyError), f, h)
    assert excepting.__name__ == 'f_excepting_ValueError_or_KeyError'


def test_excepts_class_doc():
    assert excepts.__doc__ is not None


def test_excepts_doc_attribute_error():
    excepting = excepts(ValueError, 5)
    assert excepting.__doc__ == excepts.__doc__


# ---------------------------------------------------------------------------
# introspection functions
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


def test_num_required_args_builtin():
    assert num_required_args(dict) == 0


def test_is_arity_builtin():
    # len is arity 1
    assert is_arity(1, len) is True
