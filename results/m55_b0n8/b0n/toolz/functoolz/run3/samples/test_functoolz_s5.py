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


def test_apply():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert apply(double, 5) == 10
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000
    )


def test_apply_kwargs():
    def f(a, b=0):
        return a + b

    assert apply(f, 1, b=2) == 3


def test_apply_no_args_raises():
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
    # no forms -> identical value
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


def test_thread_last_map_filter():
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


def test_instanceproperty_with_decorator_none():
    # instanceproperty() with no fget returns a partial-like
    ip = instanceproperty()
    assert callable(ip)


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
    assert C.x is None


def test_instance_property_is_property():
    def fget(self):
        return 1

    ip = InstanceProperty(fget=fget, classval='cv')
    assert isinstance(ip, property)
    assert ip.classval == 'cv'


def test_instance_property_reduce():
    def fget(self):
        return 1

    ip = InstanceProperty(fget=fget, classval='cv')
    factory, state = ip.__reduce__()
    assert factory is InstanceProperty
    restored = factory(*state)
    assert restored.classval == 'cv'
    assert restored.fget is fget


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20
    assert double(3) == 6


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable_raises():
    with pytest.raises(TypeError):
        curry(3)


def test_curry_attributes():
    def f(x, y):
        "docstring here"
        return x + y

    cf = curry(f)
    assert cf.func is f
    assert cf.args == ()
    assert cf.keywords == {}
    assert cf.func_name == 'f'
    assert cf.__name__ == 'f'
    assert cf.__doc__ == 'docstring here'
    assert str(cf) == str(f)
    assert repr(cf) == repr(f)


def test_curry_flattening():
    def f(x, y, z):
        return x + y + z

    c1 = curry(f, 1)
    c2 = curry(c1, 2)
    assert c2.func is f
    assert c2.args == (1, 2)
    assert c2(3) == 6


def test_curry_kwargs_merging():
    def f(x, y=0, z=0):
        return x + y + z

    c1 = curry(f, y=1)
    c2 = curry(c1, z=2)
    assert c2.args == ()
    assert c2.keywords == {'y': 1, 'z': 2}
    assert c2(10) == 13


def test_curry_functools_partial_input():
    def f(x, y):
        return x + y

    p = partial(f, 1)
    c = curry(p)
    assert c.func is f
    assert c.args == (1,)
    assert c(2) == 3


def test_curry_genuine_typeerror_raised():
    def f(x):
        return x + 'a'

    cf = curry(f)
    with pytest.raises(TypeError):
        cf(1)


def test_curry_too_many_args_raises():
    def f(x, y):
        return x + y

    cf = curry(f)
    with pytest.raises(TypeError):
        cf(1, 2, 3)


def test_curry_call_method():
    def f(x, y):
        return x + y

    cf = curry(f, 1)
    assert cf.call(2) == 3


def test_curry_bind_method():
    def f(x, y):
        return x + y

    cf = curry(f)
    bound = cf.bind(1)
    assert isinstance(bound, curry)
    assert bound(2) == 3


def test_curry_equality_and_hash():
    def f(x, y):
        return x + y

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert not (a != b)
    assert a != c
    assert hash(a) == hash(b)
    assert a != "not a curry"


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x, y):
            return self.value + x + y

        value = 10

    assert isinstance(C.__dict__['method'], curry)
    # access on class returns the curry itself
    assert isinstance(C.method, curry)
    obj = C()
    bound = obj.method
    assert isinstance(bound, curry)
    assert bound(1)(2) == 13


def test_curry_signature():
    import inspect

    def f(a, b, c=3):
        return a + b + c

    cf = curry(f, 1)
    sig = inspect.signature(cf)
    params = list(sig.parameters)
    assert params == ['b', 'c']


def test_curry_signature_incorrect_args():
    def f(x):
        return x

    cf = curry(f)
    # manually corrupt partial args
    object.__setattr__ if False else None
    cf._partial = partial(f, 1, 2)
    with pytest.raises(TypeError):
        cf.__signature__


def test_curry_get_signature_with_keywords():
    import inspect

    def f(a, b=2, c=3):
        return a + b + c

    cf = curry(f, 1, c=10)
    sig = inspect.signature(cf)
    params = sig.parameters
    assert list(params) == ['b', 'c']


def test_curry_pickle_roundtrip():
    def f(x, y):
        return x + y

    cf = curry(f, 1)
    assert cf(2) == 3
    restored = pickle.loads(pickle.dumps(cf))
    assert restored(2) == 3
    assert restored.func is f
    assert restored.args == (1,)


def test_curry_reduce_decorated():
    # module-level curried function, memoize is decorated
    state = memoize.__reduce__()
    assert callable(state[0])
    restored = pickle.loads(pickle.dumps(memoize))
    assert isinstance(restored, curry)


def test_curry_varargs_currying():
    def f(*args, **kwargs):
        return (args, kwargs)

    cf = curry(f)
    # Currying on varargs triggers bind
    result = cf(1, 2, 3)
    assert result == ((1, 2, 3), {})


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


def test_memoize_binary_cache():
    calls = []

    def f(x, y):
        calls.append(x)
        return x * y

    m = memoize(f, cache={(1, 2): 100})
    assert m(1, 2) == 100
    assert calls == []


def test_memoize_decorator():
    @memoize
    def f(x, y):
        return x + y

    assert f(1, 2) == 3


def test_memoize_cache_kwarg():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_unary():
    calls = []

    def f(x):
        calls.append(x)
        return x * 2

    m = memoize(f)
    assert m(3) == 6
    assert m(3) == 6
    assert calls == [3]


def test_memoize_key_function():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3


def test_memoize_kwargs():
    calls = []

    def f(x, y=0):
        calls.append((x, y))
        return x + y

    m = memoize(f)
    assert m(1, y=2) == 3
    assert m(1, y=2) == 3
    assert calls == [(1, 2)]


def test_memoize_unhashable_raises():
    def f(x):
        return x

    m = memoize(f)
    with pytest.raises(TypeError):
        m([1, 2, 3])


def test_memoize_wrapped():
    def f(x):
        return x

    m = memoize(f)
    assert m.__wrapped__ is f
    assert m.__name__ == 'f'


# ---------------------------------------------------------------------------
# compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_no_args():
    assert compose() is identity
    assert compose()(5) == 5


def test_compose_single():
    def f(x):
        return x + 1

    assert compose(f) is f


def test_compose_multiple():
    f = lambda x: x * 2
    g = lambda x: x + 3
    h = lambda x: x - 1
    # compose(f, g, h)(x) == f(g(h(x)))
    assert compose(f, g, h)(10) == ((10 - 1) + 3) * 2


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_args():
    assert compose_left() is identity


def test_compose_left_multiple():
    f = lambda x: x * 2
    g = lambda x: x + 3
    # compose_left(f, g)(x) == g(f(x))
    assert compose_left(f, g)(4) == 4 * 2 + 3


def test_compose_kwargs_passed_to_first():
    def f(a, b):
        return a - b

    def g(x):
        return x * 10

    # h(g(x)) style. compose(g, f)(a=.., b=..)
    c = compose(g, f)
    assert c(5, 3) == 20


def test_compose_equality():
    f = lambda x: x + 1
    g = lambda x: x * 2
    c1 = compose(f, g)
    c2 = compose(f, g)
    c3 = compose(g, f)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1.__eq__("not compose") is NotImplemented


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    r = repr(c)
    assert r.startswith('Compose')
    assert 'f' in r and 'g' in r


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert isinstance(c.__doc__, str)
    assert 'f(g(' in c.__doc__


def test_compose_doc_no_names():
    c = compose(object(), object())
    assert c.__doc__ == 'A composition of functions'


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_name_no_names():
    c = compose(object(), object())
    assert c.__name__ == 'Compose'


def test_compose_signature():
    import inspect

    def f(a: int, b: int) -> int:
        return a + b

    def g(x: int) -> str:
        return str(x)

    c = compose(f, g)
    sig = inspect.signature(c)
    # parameter annotations from first applied (g), return from last (f)
    assert sig.return_annotation is int


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_descriptor():
    class C:
        def add(self, x):
            return self.offset + x

        offset = 10

        op = compose(lambda x: x * 2, add)

    obj = C()
    assert obj.op(3) == (10 + 3) * 2
    assert isinstance(C.op, Compose)


def test_compose_pickle():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(5) == c(5)


def test_compose_annotations():
    def f(a: int, b: int) -> str:
        return str(a + b)

    def g(x: int) -> int:
        return x * 2

    c = compose(f, g)
    ann = c.__annotations__
    assert ann['x'] is int
    assert ann['return'] is str
    assert 'b' not in ann


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


def test_pipe_single():
    assert pipe(5, lambda x: x + 1) == 6


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
    def f(a, b=0):
        return a + b

    def g(a, b=0):
        return a - b

    assert juxt(f, g)(5, b=2) == (7, 3)


def test_juxt_empty():
    assert juxt()(1) == ()


def test_juxt_pickle():
    inc = lambda x: x + 1

    j = juxt(inc)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(1) == (2,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    inc = compose(lambda x: x + 1, do(log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


def test_do_returns_input():
    seen = []
    obj = object()
    assert do(seen.append, obj) is obj
    assert seen == [obj]


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


def test_excepts_no_matching_exception():
    e = excepts(ValueError, lambda x: x)
    with pytest.raises(TypeError):
        e(object())


def test_excepts_name():
    e = excepts(ValueError, lambda a: a)
    assert e.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_tuple():
    e = excepts((IndexError, KeyError), lambda a: a)
    assert e.__name__ == '<lambda>_excepting_IndexError_or_KeyError'


def test_excepts_name_no_attr():
    e = excepts(ValueError, object())
    assert e.__name__ == 'excepting'


def test_excepts_doc():
    e = excepts(ValueError, lambda a: a)
    assert isinstance(e.__doc__, str)


def test_excepts_doc_no_attr():
    e = excepts(ValueError, object())
    assert e.__doc__ == excepts.__doc__


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def _f(x, y, z=3):
    return x + y + z


def test_num_required_args():
    assert num_required_args(_f) == 2

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

    def g(x):
        return x

    assert has_keywords(f) is True
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

    def g(x, y=1):
        return x + y

    def h(*args):
        return args

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False


def test_is_arity_builtin():
    # map is a builtin with a registered signature
    assert is_arity(1, map) is False


def test_is_valid_args_builtin_map():
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args_builtin_map():
    assert is_partial_args(map, (), {}) is True
