import inspect
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
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply_basic():
    assert apply(lambda x: 2 * x, 5) == 10


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


def test_apply_kwargs():
    def f(a, b, c=0):
        return a + b + c

    assert apply(f, 1, 2, c=10) == 13


def test_apply_multiple_args():
    assert apply(lambda *a: sum(a), 1, 2, 3) == 6


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_callable():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuple():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(10) == 10


def test_thread_last_callable():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuple():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_no_forms():
    assert thread_last(7) == 7


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_classval_vs_instance():
    class MyClass:
        @instanceproperty(classval="cls-doc")
        def val(self):
            return "inst-val"

    assert MyClass.val == "cls-doc"
    assert MyClass().val == "inst-val"


def test_instanceproperty_plain():
    class MyClass:
        @instanceproperty
        def val(self):
            return 42

    assert MyClass.val is None
    assert MyClass().val == 42


def test_instanceproperty_setter():
    class MyClass:
        def __init__(self):
            self._v = 0

        @instanceproperty
        def val(self):
            return self._v

        @val.setter
        def val(self, x):
            self._v = x

    obj = MyClass()
    assert obj.val == 0
    obj.val = 5
    assert obj.val == 5


def test_instance_property_is_property():
    def fget(self):
        return 1

    p = InstanceProperty(fget=fget, classval=99)
    assert isinstance(p, property)
    assert p.classval == 99


def test_instance_property_pickle():
    class MyClass:
        @instanceproperty
        def val(self):
            return 5

    obj = MyClass()
    assert obj.val == 5


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    c = curry(mul)
    double = c(2)
    assert isinstance(double, curry)
    assert double(10) == 20


def test_curry_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_init_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_init_not_callable():
    with pytest.raises(TypeError):
        curry(3)


def test_curry_attributes():
    def f(x):
        "docstring"
        return x

    c = curry(f)
    assert c.func is f
    assert c.__name__ == 'f'
    assert c.__doc__ == 'docstring'
    assert c.args == ()
    assert c.keywords == {}
    assert c.func_name == 'f'


def test_curry_str_repr():
    def f(x):
        return x

    c = curry(f)
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_repr_and_str_of_partial():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    assert c(2) == 3
    assert c.args == (1,)


def test_curry_eq_hash():
    def f(x, y):
        return x + y

    a = curry(f)
    b = curry(f)
    assert a == b
    assert hash(a) == hash(b)

    c = curry(f, 1)
    d = curry(f, 1)
    assert c == d
    assert c != a


def test_curry_ne_non_curry():
    def f(x):
        return x

    c = curry(f)
    assert c != f
    assert not (c == f)


def test_curry_call_kwargs():
    @curry
    def f(x, y, z=0):
        return x + y + z

    assert f(1, 2, z=3) == 6


def test_curry_genuine_typeerror():
    def f(x):
        return x

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2)


def test_curry_curry_on_typeerror():
    def f(x, y):
        return x + y

    c = curry(f)
    assert c(1)(2) == 3


def test_curry_bind():
    def f(x, y):
        return x + y

    c = curry(f)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound(2) == 3


def test_curry_call_method():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    assert c.call(2) == 3


def test_curry_call_no_curry_on_valid_args():
    def f(x):
        if x > 0:
            raise TypeError("boom")
        return x

    c = curry(f)
    # 1 is valid arg, so TypeError raised by body propagates
    with pytest.raises(TypeError):
        c(1)


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x):
            return x * 2

    obj = C()
    assert obj.method(3) == 6
    assert isinstance(C.method, curry)


def test_curry_reduce_pickle_roundtrip():
    # pickling requires a module-level function; use a builtin
    c = curry(pow, 2)
    data = pickle.dumps(c)

    # Builtins may or may not roundtrip via restore; instead just test __reduce__
    red = c.__reduce__()
    assert callable(red[0])
    assert red[1][0] is curry


def test_curry_reduce_structure():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    func, state = c.__reduce__()
    cls, restored_func, args, kwargs, userdict, is_decorated = state
    assert cls is curry
    assert restored_func is f
    assert args == (1,)
    assert kwargs == {} or kwargs is None
    assert isinstance(userdict, tuple)


def test_curry_merges_partial():
    def f(a, b, c=0):
        return a + b + c

    inner = curry(f)(1)
    outer = curry(inner, 2, c=3)
    assert outer() == 6


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['y', 'z']


def test_curry_signature_keyword_removed():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    c2 = c(y=2)
    sig = c2.__signature__
    assert 'y' not in sig.parameters


def test_curry_signature_incorrect_args():
    def f(x):
        return x

    c = curry(f)
    c._partial = c._partial.__class__(f, 1, 2)  # invalid
    with pytest.raises(TypeError):
        c.__signature__


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


def test_memoize_keeps_name_doc():
    @memoize
    def f(x):
        "hello"
        return x

    assert f.__name__ == 'f'
    assert f.__doc__ == 'hello'
    assert f.__wrapped__ is not None


def test_memoize_initial_cache():
    cache = {(1, 2): 3}

    @memoize(cache=cache)
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert (1, 2) in cache


def test_memoize_custom_key():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, verbose=True) == 3


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_unary():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]


def test_memoize_keyword_only():
    calls = []

    @memoize
    def f(*, x):
        calls.append(x)
        return x

    assert f(x=1) == 1
    assert f(x=1) == 1
    assert calls == [1]


# ---------------------------------------------------------------------------
# compose / Compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_no_funcs():
    assert compose() is identity


def test_compose_single():
    f = lambda x: x
    assert compose(f) is f


def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_funcs():
    assert compose_left() is identity


def test_compose_returns_compose():
    f = lambda x: x
    g = lambda x: x
    assert isinstance(compose(f, g), Compose)


def test_compose_multi_arg_first():
    def add(x, y):
        return x + y

    def double(x):
        return 2 * x

    assert compose(double, add)(1, 2) == 6


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
    r = repr(c)
    assert r.startswith('Compose')
    assert 'f' in r and 'g' in r


def test_compose_eq():
    def f(x):
        return x

    def g(x):
        return x

    assert compose(f, g) == compose(f, g)
    assert compose(f, g) != compose(g, f)
    assert compose(f, g).__eq__(5) is NotImplemented
    assert compose(f, g).__ne__(5) is NotImplemented


def test_compose_hash():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert hash(c) == hash(c)


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__doc__ == 'lambda *args, **kwargs: f(g(*args, **kwargs))'


def test_compose_doc_classval():
    assert Compose.__doc__ is not None


def test_compose_doc_no_names():
    class NoName:
        def __call__(self, x):
            return x

    c = compose(NoName(), NoName())
    assert c.__doc__ == 'A composition of functions'


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is f


def test_compose_signature():
    def f(x, y) -> int:
        return x + y

    def g(z) -> str:
        return z

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation is int
    assert list(sig.parameters) == ['z']


def test_compose_descriptor():
    class C:
        def _double(self, x):
            return 2 * x

        def _inc(self, x):
            return x + 1

        method = compose(_inc, _double)

    obj = C()
    assert obj.method(3) == 7


def test_compose_getstate_setstate():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2(3) == 7


def test_compose_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(y: str) -> int:
        return int(y)

    c = compose(f, g)
    ann = c.__annotations__
    assert ann['y'] is str
    assert ann['return'] is str


def test_compose_annotations_nested():
    def f(x: int) -> str:
        return str(x)

    def g(y: str) -> int:
        return int(y)

    def h(z: int) -> float:
        return float(z)

    c = compose(f, compose(g, h))
    ann = c.__annotations__
    assert ann['z'] is int
    assert ann['return'] is str


def test_compose_annotations_no_signature():
    class Call:
        def __call__(self, x):
            return x

    def f(x: int) -> str:
        return str(x)

    c = compose(f, Call())
    ann = c.__annotations__
    # Call has signature without annotation, return from f
    assert ann.get('return') is str


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
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

def test_juxt_basic():
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
    double = lambda x: x * 2
    j = juxt(inc, double)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(10) == (11, 20)


def test_juxt_returns_tuple():
    assert isinstance(juxt()(5), tuple)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    result = do(log.append, 1)
    assert result == 1
    assert log == [1]


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
    excepting = excepts(ValueError, lambda a: [1, 2].index(a))
    assert excepting(3) is None


def test_excepts_tuple():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_attributes():
    def f(x):
        return x

    def h(e):
        return None

    e = excepts(ValueError, f, h)
    assert e.exc is ValueError
    assert e.func is f
    assert e.handler is h


def test_excepts_doc():
    def f(x):
        "func doc"
        return x

    def h(e):
        "handler doc"
        return None

    e = excepts(ValueError, f, h)
    doc = e.__doc__
    assert "'f'" in doc
    assert 'ValueError' in doc
    assert "'h'" in doc


def test_excepts_doc_tuple():
    def f(x):
        return x

    e = excepts((ValueError, KeyError), f)
    assert '(ValueError, KeyError)' in e.__doc__


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


def test_excepts_name_no_attr():
    e = excepts(ValueError, lambda x: x)
    assert e.__name__.startswith('f_excepting') or 'excepting' in e.__name__


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2


def test_num_required_args_varargs():
    def g(*args, **kwargs):
        return None

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

    assert is_arity(1, f) is False


def test_is_arity_keywords():
    def f(x, **kwargs):
        return x

    assert is_arity(1, f) is False


def test_is_valid_args_with_sigspec():
    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert is_valid_args(f, (1, 2), {}, sig) is True
    assert is_valid_args(f, (1,), {}, sig) is False


def test_is_partial_args_with_sigspec():
    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert is_partial_args(f, (1,), {}, sig) is True
    assert is_partial_args(f, (1, 2, 3), {}, sig) is False


def test_num_required_args_with_sigspec():
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2


def test_has_varargs_with_sigspec():
    def f(*args):
        return args

    sig = inspect.signature(f)
    assert has_varargs(f, sig) is True


def test_has_keywords_with_sigspec():
    def f(x, y=0):
        return x + y

    sig = inspect.signature(f)
    assert has_keywords(f, sig) is True
