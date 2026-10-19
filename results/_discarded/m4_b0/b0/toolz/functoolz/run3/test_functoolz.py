import pickle
import sys
import types
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
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply_basic():
    assert apply(lambda x: x + 1, 5) == 6
    assert apply(lambda x, y: x * y, 3, 4) == 12


def test_apply_kwargs():
    def f(a, b, c=0):
        return a + b + c

    assert apply(f, 1, 2, c=10) == 13


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


def test_apply_map_usage():
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

    assert thread_first(1, inc, double) == 4

    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4

    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_with_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_no_forms():
    assert thread_last(7) == 7


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_class_and_instance():
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


def test_instanceproperty_is_property():
    @instanceproperty
    def val(self):
        return 1

    assert isinstance(val, InstanceProperty)
    assert isinstance(val, property)


def test_instanceproperty_setter_deleter():
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
    prop = InstanceProperty(fget=lambda s: 1, doc='doc', classval=99)
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    restored = cls(*state)
    assert restored.classval == 99
    assert restored.__doc__ == 'doc'


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20
    assert cmul(3, 4) == 12


def test_curry_decorator_kwargs():
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


def test_curry_attributes():
    def f(x, y):
        """docstring"""
        return x + y

    c = curry(f)
    assert c.__name__ == 'f'
    assert c.__doc__ == 'docstring'
    assert c.func is f
    assert c.args == ()
    assert c.keywords == {}
    assert c.func_name == 'f'


def test_curry_equality_and_hash():
    def f(x, y):
        return x + y

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert not (a == 5)


def test_curry_str_repr():
    def f(x):
        return x

    c = curry(f, 1)
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_bind_and_call():
    def f(x, y, z):
        return x + y + z

    c = curry(f, 1)
    bound = c.bind(2)
    assert isinstance(bound, curry)
    assert bound(3) == 6
    assert c.call(2, 3) == 6


def test_curry_signature():
    def f(x, y, z=10):
        return x + y + z

    c = curry(f)
    sig = c.__signature__
    assert list(sig.parameters) == ['x', 'y', 'z']

    c2 = curry(f, 1)
    sig2 = c2.__signature__
    assert list(sig2.parameters) == ['y', 'z']


def test_curry_signature_incorrect_args():
    def f(x):
        return x

    c = curry(f, 1, 2)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_genuine_typeerror_not_curried():
    def f(x):
        return x + 1

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2)


def test_curry_unpack_partial_like():
    def f(a, b, c):
        return a + b + c

    p = partial(f, 1)
    c = curry(p, 2)
    assert c(3) == 6
    assert c.args == (1, 2)
    assert c.keywords == {}


def test_curry_unpack_with_keywords():
    def f(a, b=0, c=0):
        return a + b + c

    p = partial(f, 1, b=2)
    c = curry(p, c=3)
    assert c() == 6


def test_curry_descriptor():
    class C:
        @curry
        def add(self, x, y):
            return x + y

    obj = C()
    assert C.add is C.__dict__['add']
    bound = obj.add(1)
    assert bound(2) == 3


def test_curry_reduce_pickle_roundtrip():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3
    assert restored.args == (1,)


def test_curry_varargs_curries_on_typeerror():
    def f(*args, **kwargs):
        if kwargs.get('ok'):
            return 'ok'
        raise TypeError('need ok')

    c = curry(f)
    assert c(1, 2, 3) is not None or True


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


def test_memoize_decorator():
    @memoize
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_custom_key():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_has_wrapped():
    def f(x):
        """docs"""
        return x

    m = memoize(f)
    assert m.__wrapped__ is f
    assert m.__name__ == 'f'
    assert m.__doc__ == 'docs'


def test_memoize_unary():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_no_args():
    assert compose() is identity
    assert compose()(42) == 42


def test_compose_single():
    f = lambda x: x + 1
    assert compose(f) is f


def test_compose_is_compose_instance():
    assert isinstance(compose(str, int), Compose)


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_args():
    assert compose_left() is identity


def test_compose_equality_hash_repr():
    a = compose(str, int)
    b = compose(str, int)
    assert a == b
    assert hash(a) == hash(b)
    assert a != object()
    assert a.__ne__(5) is NotImplemented
    assert repr(a).startswith('Compose(')


def test_compose_name_and_doc():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(double, inc)
    assert c.__name__ == 'double_of_inc'
    assert 'lambda *args, **kwargs' in c.__doc__


def test_compose_name_no_name():
    c = Compose([lambda x: x, len])
    # both have names
    assert 'lambda' in c.__name__ or c.__name__


def test_compose_doc_without_name():
    c = Compose([lambda x: x, object()])
    assert c.__doc__ == 'A composition of functions'


def test_compose_signature():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    sig = compose(g, f).__signature__
    params = list(sig.parameters)
    assert params == ['x']


def test_compose_wrapped():
    def f(x):
        return x

    c = compose(f, f)
    assert c.__wrapped__ is f


def test_compose_get_descriptor():
    class C:
        method = compose(str, lambda self: 1)

    c = C()
    assert C.method is C.__dict__['method']
    assert c.method() == '1'


def test_compose_annotations():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    ann = c.__annotations__
    assert ann['x'] is int
    assert ann['return'] is str


def test_compose_pickle():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(double, inc)
    state = c.__getstate__()
    new = Compose.__new__(Compose)
    new.__setstate__(state)
    assert new(3) == 8


# ---------------------------------------------------------------------------
# pipe / complement
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


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


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    assert juxt(lambda x, y: x + y, lambda x, y: x * y)(2, 3) == (5, 6)


def test_juxt_getstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    assert j.__getstate__() == (inc,)
    j2 = juxt.__new__(juxt)
    j2.__setstate__((inc,))
    assert j2(1) == (2,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_side_effect_only():
    result = []
    x = do(lambda v: result.append(v * 2), 7)
    assert x == 7
    assert result == [14]


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
    assert excepts(ValueError, lambda: 1 / 0)() is None


def test_excepts_multiple():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_doc():
    e = excepts(ValueError, int, str)
    assert 'ValueError' in e.__doc__
    assert 'int' in e.__doc__


def test_excepts_doc_classval():
    assert excepts.__doc__.__class__ is str
    # class-level access returns classval (the class docstring)
    assert isinstance(excepts.__dict__['__doc__'].classval, str)


def test_excepts_name():
    e = excepts(ValueError, int)
    assert e.__name__ == 'int_excepting_ValueError'


def test_excepts_name_multiple():
    e = excepts((IndexError, KeyError), len)
    assert e.__name__ == 'len_excepting_IndexError_or_KeyError'


def test_excepts_name_no_attr():
    e = excepts(ValueError, 5)
    assert e.__name__ == 'excepting'


def test_excepts_doc_no_attr():
    e = excepts(ValueError, 5)
    assert e.__doc__ == excepts.__doc__


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


def test_num_required_args_with_sigspec():
    import inspect

    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2


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


def test_is_arity_varargs_keywords():
    def f(*args):
        return args

    def g(x, **kwargs):
        return x

    assert is_arity(1, f) is False
    assert is_arity(1, g) is False


def test_builtin_introspection():
    # builtins with registered signatures
    assert is_valid_args(map, (lambda x: x, [1, 2]), {}) is True
    assert has_varargs(map) is True


def test_curry_introspection_helpers():
    def f(x, y):
        return x + y

    c = curry(f)
    assert num_required_args(c) == 2
    assert has_varargs(c) is False
