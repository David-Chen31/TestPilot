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


def test_apply_basic():
    assert apply(lambda x: 2 * x, 5) == 10


def test_apply_kwargs():
    def f(a, b=10):
        return a + b

    assert apply(f, 1) == 11
    assert apply(f, 1, b=2) == 3


def test_apply_no_args_raises():
    with pytest.raises(TypeError, match="func argument is required"):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_plain():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuple_form():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_first_identity_empty():
    assert thread_first(42) == 42


def test_thread_last_plain():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuple_form():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


def test_thread_last_identity_empty():
    assert thread_last(99) == 99


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_class_access():
    class C:
        """Class doc"""

        @instanceproperty(classval="CLASSVAL")
        def val(self):
            return "instance"

    assert C.val == "CLASSVAL"
    assert C().val == "instance"


def test_instanceproperty_default_classval_none():
    class C:
        @instanceproperty
        def val(self):
            return 42

    assert C.val is None
    assert C().val == 42


def test_instanceproperty_delayed_decoration():
    def getter(self):
        return 7

    prop = instanceproperty(fset=None, classval="CV")
    prop2 = prop(getter)
    assert isinstance(prop2, InstanceProperty)
    assert prop2.classval == "CV"

    class C:
        val = prop2

    assert C.val == "CV"
    assert C().val == 7


def test_instanceproperty_with_setter_and_deleter():
    class C:
        def __init__(self):
            self._v = 0

        @instanceproperty
        def val(self):
            return self._v

        @val.setter
        def val(self, v):
            self._v = v

    c = C()
    assert c.val == 0
    c.val = 5
    assert c.val == 5


def test_instanceproperty_reduce_roundtrip():
    class C:
        @instanceproperty(classval="X")
        def val(self):
            return 1

    prop = C.__dict__['val']
    restored = pickle.loads(pickle.dumps(prop))
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == "X"
    assert restored.__doc__ == prop.__doc__


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20


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
    with pytest.raises(TypeError, match="Input must be callable"):
        curry(3)


def test_curry_str_repr():
    def mul(x, y):
        return x * y

    cm = curry(mul)
    assert str(cm) == str(mul)
    assert repr(cm) == repr(mul)


def test_curry_call_forces():
    @curry
    def f(x, y):
        return x + y

    partial_f = f(1)
    assert partial_f.call(2) == 3


def test_curry_bind():
    @curry
    def f(x, y, z):
        return x + y + z

    g = f(1)
    h = g.bind(2)
    assert h(3) == 6


def test_curry_genuine_typeerror_raised():
    @curry
    def f(x):
        return int(x)

    with pytest.raises(ValueError):
        f("notanint")


def test_curry_too_many_args_raises():
    @curry
    def f(x, y):
        return x + y

    with pytest.raises(TypeError):
        f(1, 2, 3)


def test_curry_eq_hash():
    def f(x, y):
        return x + y

    c1 = curry(f)(1)
    c2 = curry(f)(1)
    c3 = curry(f)(2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)


def test_curry_neq_other_type():
    @curry
    def f(x):
        return x

    assert f != 42


def test_curry_attributes():
    def f(x):
        "docstring"
        return x

    c = curry(f)
    assert c.__doc__ == "docstring"
    assert c.__name__ == "f"
    assert c.func is f
    assert c.args == ()
    assert c.keywords == {}


def test_curry_from_partial():
    from functools import partial

    def f(a, b, c):
        return a + b + c

    p = partial(f, 1, b=2)
    c = curry(p, 3)
    assert c() == 6


def test_curry_getattr_binding():
    class C:
        @curry
        def method(self, x):
            return self, x

    c = C()
    bound = c.method
    assert bound(5) == (c, 5)


def test_curry_class_access_returns_self():
    class C:
        @curry
        def method(self, x):
            return x

    assert isinstance(C.__dict__['method'], curry) or isinstance(C.method, curry)


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f)(1)
    sig = c.__signature__
    params = list(sig.parameters)
    assert params == ['b', 'c']
    assert sig.parameters['c'].default == 3


def test_curry_signature_incorrect_args():
    def f(a):
        return a

    c = curry(f)
    c._partial = __import__('functools').partial(f, 1, 2)
    with pytest.raises(TypeError, match="incorrect arguments"):
        c.__signature__


def test_curry_reduce_roundtrip():
    def f(x, y):
        return x + y

    c = curry(f)(1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]
    assert f.__name__ == 'f'


def test_memoize_with_cache():
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
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match="hashable"):
        f([1, 2, 3])


def test_memoize_wrapped():
    @memoize
    def f(x):
        return x

    assert f.__wrapped__ is not None
    assert f.__doc__ == f.__wrapped__.__doc__


def test_memoize_kwargs_key():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert calls == [(1, 2)]


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_order():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    f = lambda x: x
    assert compose(f) is f


def test_compose_left_order():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_empty():
    assert compose_left() is identity


def test_compose_multiple_args():
    f = compose(lambda x, y: x + y, lambda x, y: (x * 2, y * 3))
    assert f(1, 2) == 8


def test_compose_obj_type():
    inc = lambda i: i + 1
    c = compose(str, inc)
    assert isinstance(c, Compose)
    assert c(3) == '4'


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert repr(c) == 'Compose(%r)' % ((f, g),)


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


def test_compose_eq_other_type():
    c = compose(str, int)
    assert c != 42


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__doc__ == 'lambda *args, **kwargs: f(g(*args, **kwargs))'
    assert Compose.__doc__ is not None


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_signature():
    def f(x: int, y: str) -> float:
        return 0.0

    def g(a) -> str:
        return str(a)

    c = compose(f, g)
    sig = c.__signature__
    assert list(sig.parameters) == ['a']
    assert sig.return_annotation is float


def test_compose_get_descriptor():
    def f(x):
        return x * 2

    def g(x):
        return x + 1

    class C:
        method = compose(f, g)

    c = C()
    assert c.method(3) == 8
    assert isinstance(C.method, Compose)


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_annotations():
    def f(x: int) -> float:
        return 0.0

    def g(a: str) -> int:
        return int(a)

    c = compose(f, g)
    ann = c.__annotations__
    assert ann == {'a': str, 'return': float}


def test_compose_pickle():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__getstate__() == (g, (f,))
    c2 = Compose.__new__(Compose)
    c2.__setstate__((g, (f,)))
    assert c2.first is g
    assert c2.funcs == (f,)


# ---------------------------------------------------------------------------
# pipe / complement
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_empty():
    assert pipe(5) == 5


def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
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
    def f(x, y):
        return x + y

    def g(x, y):
        return x * y

    assert juxt(f, g)(2, 3) == (5, 6)


def test_juxt_pickle():
    inc = lambda x: x + 1
    j = juxt(inc)
    assert j.__getstate__() == (inc,)
    j2 = juxt.__new__(juxt)
    j2.__setstate__((inc,))
    assert j2(1) == (2,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do_returns_x():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_compose():
    log = []
    inc = lambda x: x + 1
    inc = compose(inc, do(log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip_basic():
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


def test_excepts_multiple_default_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_doc_and_name():
    def f(a):
        "f doc"
        return a

    def h(e):
        "h doc"
        return None

    e = excepts(ValueError, f, h)
    assert e.__name__ == 'f_excepting_ValueError'
    assert 'f doc' in e.__doc__
    assert 'h doc' in e.__doc__


def test_excepts_name_multiple():
    e = excepts((IndexError, KeyError), lambda a: a[0])
    assert e.__name__ == 'lambda_excepting_IndexError_or_KeyError'


def test_excepts_name_fallback():
    class NoName:
        def __call__(self, *a, **k):
            return 1

    inst = NoName()
    exc = type('MyExc', (Exception,), {})
    e = excepts((exc,), inst)
    assert e.__name__ == 'excepting'


def test_excepts_doc_classval():
    assert isinstance(excepts.__doc__, str)


# ---------------------------------------------------------------------------
# signature helpers
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

    def h(x, y):
        return x + y

    assert is_arity(1, h) is False
    assert is_arity(2, h) is True


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


def test_builtin_map_signature_helpers():
    # builtins are resolved via _signatures registry
    assert is_partial_args(map, (), {}) in (True, None)
    assert is_valid_args(map, (lambda x: x, [1]), {}) in (True, None)
