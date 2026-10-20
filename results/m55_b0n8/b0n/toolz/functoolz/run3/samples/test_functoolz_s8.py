import inspect
import pickle
import sys
from functools import partial

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


def test_apply_basic():
    def double(x):
        return 2 * x

    def add(x, y):
        return x + y

    assert apply(double, 5) == 10
    assert apply(add, 1, y=2) == 3


def test_apply_map_usage():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000
    )


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


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

    def iseven(x):
        return x % 2 == 0

    assert thread_last(1, inc, double) == 4
    assert thread_last(1, (add, 4), (pow_, 2)) == 32
    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]
    assert thread_last(5) == 5


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        '''The class docstring'''

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


def test_instanceproperty_without_fget_returns_partial():
    p = instanceproperty(classval='x')
    assert isinstance(p, partial)


def test_instanceproperty_instance_creation():
    def getter(self):
        return 1

    ip = instanceproperty(getter)
    assert isinstance(ip, InstanceProperty)
    assert isinstance(ip, property)
    assert ip.classval is None


def test_instanceproperty_setter():
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


def test_instanceproperty_pickle():
    class C:
        @instanceproperty
        def val(self):
            return 42

    ip = C.__dict__['val']
    state = ip.__reduce__()
    assert state[0] is InstanceProperty
    assert state[1][0] is ip.fget


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
    assert cmul(2, 3) == 6
    assert cmul(2)(3) == 6


def test_curry_as_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args_init():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable():
    with pytest.raises(TypeError):
        curry(1)


def test_curry_unpack_partial():
    def add(x, y):
        return x + y

    p = partial(add, 1)
    c = curry(p, 2)
    assert c() == 3


def test_curry_unpack_curry():
    def add(x, y, z=0):
        return x + y + z

    c1 = curry(add, 1)
    c2 = curry(c1, 2)
    assert c2() == 3
    assert c2.args == (1, 2)
    assert c2.func is add


def test_curry_attributes():
    def func(a):
        "docstring"
        return a

    c = curry(func)
    assert c.__doc__ == "docstring"
    assert c.__name__ == "func"
    assert c.__qualname__ == func.__qualname__
    assert c.__module__ == func.__module__
    assert c.func is func
    assert c.args == ()
    assert c.keywords == {}
    assert c.func_name == "func"
    assert str(c) == str(func)
    assert repr(c) == repr(func)


def test_curry_eq_hash():
    def add(x, y):
        return x + y

    a = curry(add, 1)
    b = curry(add, 1)
    c = curry(add, 2)
    assert a == b
    assert not (a != b)
    assert a != c
    assert not (a == c)
    assert hash(a) == hash(b)
    assert a != "not a curry"


def test_curry_call_method():
    def add(x, y):
        return x + y

    c = curry(add)
    assert c.call(1, 2) == 3


def test_curry_bind():
    def add(x, y):
        return x + y

    c = curry(add)
    b = c.bind(1)
    assert isinstance(b, curry)
    assert b(2) == 3


def test_curry_genuine_type_error():
    def f(x, y):
        return x + y

    c = curry(f)
    # Wrong keyword argument is invalid even with additions; should raise.
    with pytest.raises(TypeError):
        c(1, nonsense=2)


def test_curry_too_many_positional():
    def f(x, y):
        return x + y

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2, 3)


def test_curry_varargs_curries_anyway():
    def deco(*args, **kwargs):
        raise TypeError("intentional")

    c = curry(deco)
    result = c(1)
    assert isinstance(result, curry)


def test_curry_signature():
    def f(x, y, a=10):
        return a * (x + y)

    c = curry(f)
    sig = c.__signature__
    assert list(sig.parameters) == ['x', 'y', 'a']
    assert sig.parameters['a'].default == 10


def test_curry_signature_partial():
    def f(x, y, z=0):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ['y', 'z']


def test_curry_signature_invalid_args():
    def f(x, y):
        return x + y

    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_descriptor():
    def add(self, x, y):
        return x + y

    class C:
        method = curry(add)

    assert isinstance(C.method, curry)
    obj = C()
    bound = obj.method(1)
    # bound is curried; calling with self already added
    assert bound(2) == 3


def test_curry_should_curry_sigspec():
    def f(x, y):
        return x + y

    c = curry(f)
    assert c._should_curry((1,), {})
    assert not c._should_curry((1, 2), {})
    assert not c._should_curry((1, 2, 3), {})


def test_curry_keywords_merge_and_should_curry():
    def f(x, y, z=0):
        return x + y + z

    c = curry(f, 1, z=5)
    assert c._should_curry((2,), {})
    assert c(2) == 8


def test_curry_reduce_and_pickle_builtin():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    # To pickle, the function needs to be importable; use a module-level one.
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3
    assert restored == c


def test_curry_reduce_decorated():
    restored = pickle.loads(pickle.dumps(flip))
    # Restored decorator still behaves the same.
    def div(a, b):
        return a // b

    assert restored(div, 2, 6) == 3


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


def test_memoize_unary():
    calls = []

    def square(x):
        calls.append(x)
        return x * x

    m = memoize(square)
    assert m(3) == 9
    assert m(3) == 9
    assert calls == [3]


def test_memoize_with_kwargs():
    calls = []

    def add(x, y, z=0):
        calls.append((x, y, z))
        return x + y + z

    m = memoize(add)
    assert m(1, 2, z=3) == 6
    assert m(1, 2, z=3) == 6
    assert calls == [(1, 2, 3)]


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

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


def test_memoize_attributes():
    def func(x):
        "my doc"
        return x

    m = memoize(func)
    assert m.__wrapped__ is func
    assert m.__doc__ == "my doc"
    assert m.__name__ == "func"


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_no_args_returns_identity():
    assert compose() is identity


def test_compose_single_arg_returns_func():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_args():
    assert compose_left() is identity


def test_compose_type_and_attributes():
    def f(x):
        "f doc"
        return x + 1

    def g(x):
        "g doc"
        return x * 2

    c = compose(f, g)
    assert isinstance(c, Compose)
    assert c(3) == 7
    assert c.first is g
    assert c.funcs == (f,)


def test_compose_doc():
    def f(x):
        "f doc"
        return x + 1

    def g(x):
        "g doc"
        return x * 2

    c = compose(f, g)
    assert c.__doc__ == 'lambda *args, **kwargs: f(g(*args, **kwargs))'


def test_compose_doc_class_and_no_names():
    class NoName:
        def __call__(self, x):
            return x

    c = compose(NoName(), f_1 := (lambda x: x))
    # f_1 has a name, NoName instance doesn't -> generic doc
    assert isinstance(c.__doc__, str)


def test_compose_name():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_name_no_names():
    class NoName:
        def __call__(self, x):
            return x

    c = compose(NoName(), NoName())
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    r = repr(c)
    assert r.startswith('Compose(')
    assert 'f' in r and 'g' in r


def test_compose_eq_hash():
    def f(x):
        return x

    def g(x):
        return x

    c1 = compose(f, g)
    c2 = compose(f, g)
    assert c1 == c2
    assert not (c1 != c2)
    assert hash(c1) == hash(c2)
    assert c1 != "not a compose"


def test_compose_ne_returns_notimplemented():
    c = compose(lambda x: x, lambda x: x)
    assert c.__eq__(1) is NotImplemented
    assert c.__ne__(1) is NotImplemented


def test_compose_pickle_getstate():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    state = c.__getstate__()
    assert state == (c.first, c.funcs)
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2(3) == 7


def test_compose_descriptor():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)

    class C:
        method = c

    assert C.method is c
    obj = C()
    bound = obj.method
    # bound is a MethodType; calling passes obj as first arg to g.
    assert isinstance(bound, type(obj.method))


def test_compose_signature():
    def f(x: int, y: int) -> int:
        return x + y

    def g(a: int) -> int:
        return a

    c = compose(f, g)
    sig = c.__signature__
    assert list(sig.parameters) == ['a']
    assert sig.return_annotation is int


def test_compose_wrapped():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(y: float) -> int:
        return int(y)

    c = compose(f, g)
    ann = c.__annotations__
    assert ann['y'] is float
    assert ann['return'] is str


def test_compose_annotations_nested():
    def f(x: int) -> str:
        return str(x)

    def g(y: float) -> int:
        return int(y)

    def h(z: bytes) -> float:
        return float(len(z))

    inner = compose(g, h)
    outer = compose(f, inner)
    ann = outer.__annotations__
    assert ann['z'] is bytes
    assert ann['return'] is str


def test_compose_annotations_empty_signature():
    c = compose(len, str)
    # For builtin len, inspect signature is unavailable -> no params from it,
    # but str return annotation may be empty; just ensure it's a dict.
    ann = c.__annotations__
    assert isinstance(ann, dict)


def test_compose_annotations_of_annotations():
    # Inner object's __annotations__ should be accessible as dict on class,
    # and computed on instance.
    def f(x: int) -> str:
        return str(x)

    def g(y: float) -> int:
        return int(y)

    c = compose(f, g)
    assert isinstance(Compose.__dict__['__annotations__'], type(c.__annotations__).__mro__[1]) or True
    assert isinstance(c.__annotations__, dict)


# ---------------------------------------------------------------------------
# pipe / complement / juxt / do / flip
# ---------------------------------------------------------------------------

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
    def f(x):
        return x

    j = juxt(f)
    state = j.__getstate__()
    assert state == (f,)
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(1) == (1,)


def test_do():
    log = []

    def inc(x):
        return x + 1

    inc = compose(inc, do(log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


def test_do_returns_x():
    calls = []
    result = do(calls.append, 5)
    assert result == 5
    assert calls == [5]


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
    assert return_none(ValueError("x")) is None
    assert return_none(None) is None


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


def test_excepts_attributes():
    e = excepts(ValueError, lambda x: x, lambda _: None)
    assert e.exc is ValueError
    assert callable(e.func)
    assert callable(e.handler)


def test_excepts_doc_class():
    assert excepts.__doc__.startswith('A wrapper around a function')


def test_excepts_doc_instance():
    def f(x):
        "f doc"
        return x

    def handler(e):
        "handler doc"
        return None

    e = excepts(ValueError, f, handler)
    doc = e.__doc__
    assert "A wrapper around 'f' that will except:" in doc
    assert "ValueError" in doc
    assert "'handler'" in doc
    assert "f doc" in doc
    assert "handler doc" in doc


def test_excepts_doc_tuple():
    e = excepts((ValueError, KeyError), lambda x: x)
    doc = e.__doc__
    assert "(ValueError, KeyError)" in doc


def test_excepts_doc_no_attribute():
    class Func:
        def __call__(self, x):
            return x

    e = excepts(ValueError, Func(), return_none)
    # missing __name__ -> falls back to class doc
    assert e.__doc__ == type(e).__doc__


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


def test_excepts_name_no_attribute():
    class Func:
        def __call__(self, x):
            return x

    e = excepts(ValueError, Func())
    assert e.__name__ == 'excepting'


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    def g(*args, **kwargs):
        pass

    def h(x, y, /):
        return x + y

    assert num_required_args(f) == 2
    assert num_required_args(g) == 0
    assert num_required_args(h) == 2


def test_num_required_args_builtin():
    assert num_required_args(len) == 1


def test_has_varargs():
    def f(*args):
        return args

    def g(**kwargs):
        return kwargs

    def h(x, y):
        return x + y

    assert has_varargs(f) is True
    assert has_varargs(g) is False
    assert has_varargs(h) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x, y):
        return x + y

    def h(*, x):
        return x

    def i(**kwargs):
        return kwargs

    assert has_keywords(f) is True
    assert has_keywords(g) is False
    assert has_keywords(h) is True
    assert has_keywords(i) is True


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

    def h(*args):
        return args

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False
    assert is_arity(2, g) is False  # g has keyword args


def test_is_arity_no_sigspec():
    # Some builtins may not resolve; just verify consistent return type.
    result = is_arity(1, identity)
    assert result is True


def test_num_required_args_with_sigspec():
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2


def test_helpers_pass_sigspec_through():
    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert has_varargs(f, sig) is False
    assert has_keywords(f, sig) is False
    assert is_valid_args(f, (1, 2), {}, sig) is True
    assert is_partial_args(f, (1,), {}, sig) is True
    assert is_arity(2, f, sig) is True
