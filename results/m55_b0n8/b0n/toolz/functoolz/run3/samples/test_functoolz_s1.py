import operator
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
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_with_kwargs():
    def f(a, b, c=0):
        return a + b + c

    assert apply(f, 1, 2, c=3) == 6


def test_apply_requires_func():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def inc(x):
    return x + 1


def double(x):
    return 2 * x


def add(x, y):
    return x + y


def pow_(x, y):
    return x ** y


def test_thread_first_basic():
    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_basic():
    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    def iseven(x):
        return x % 2 == 0

    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_classval_vs_instance():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return "An object docstring"

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == "The class docstring"
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == "An object docstring"
    assert obj.val == 42


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
    obj.val = 7
    assert obj.val == 7


def test_instanceproperty_returns_partial_when_no_fget():
    p = instanceproperty(classval=5)
    assert isinstance(p, type(instanceproperty)) or callable(p)


def test_instanceproperty_pickle():
    class MyClass:
        @instanceproperty
        def val(self):
            return 42

    prop = MyClass.__dict__["val"]
    restored = pickle.loads(pickle.dumps(prop))
    assert isinstance(restored, InstanceProperty)
    assert restored.fget is prop.fget


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double_c = cmul(2)
    assert double_c(10) == 20


def test_curry_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add_ = f(a=1)
    assert add_(2, 3) == 5


def test_curry_init_requires_func():
    with pytest.raises(TypeError):
        curry()


def test_curry_init_requires_callable():
    with pytest.raises(TypeError):
        curry(1)


def test_curry_unpacks_partial_like():
    def f(a, b, c):
        return a + b + c

    c1 = curry(f, 1)
    c2 = curry(c1, 2)
    assert c2(3) == 6


def test_curry_call_forwards_typeerror():
    def f(x):
        return x + 1

    cf = curry(f)
    with pytest.raises(TypeError):
        cf("a", "b")


def test_curry_call_genuine_type_error():
    def f(x, y):
        return x + y

    cf = curry(f)
    with pytest.raises(TypeError):
        cf(1, 2, 3)


def test_curry_bind_and_call():
    def f(x, y, z):
        return x + y + z

    cf = curry(f)
    b = cf.bind(1)
    assert isinstance(b, curry)
    assert b(2, 3) == 6
    assert cf.call(1, 2, 3) == 6


def test_curry_attributes():
    def myfunc(x):
        """docstring here"""
        return x

    cf = curry(myfunc)
    assert cf.func is myfunc
    assert cf.func_name == "myfunc"
    assert cf.__doc__ == "docstring here"
    assert str(cf) == str(myfunc)
    assert repr(cf) == repr(myfunc)


def test_curry_hash_eq():
    def f(x, y):
        return x + y

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert not (a != b)
    assert a != c
    assert hash(a) == hash(b)


def test_curry_eq_non_curry():
    cf = curry(lambda x: x)
    assert cf != 5
    assert not (cf == 5)


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x, y):
            return self.value + x + y

    c = C()
    c.value = 10
    assert c.method(1, 2) == 13
    bound = c.method
    assert bound(1, 2) == 13


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    cf = curry(f, 1)
    sig = cf.__signature__
    names = list(sig.parameters)
    # 'a' was bound
    assert "a" not in names


def test_curry_signature_keyword_binding():
    def f(a, b, c=3):
        return a + b + c

    cf = curry(f)(a=1)
    sig = cf.__signature__
    # 'a' became keyword only
    assert sig.parameters["a"].kind == inspect.Parameter.KEYWORD_ONLY


def test_curry_reduce_pickle():
    def f(x, y):
        return x + y

    cf = curry(f, 1)
    restored = pickle.loads(pickle.dumps(cf))
    assert isinstance(restored, curry)
    assert restored(2) == 3


def test_curry_varargs_decorator():
    @curry
    def wrapper(*args, **kwargs):
        return (args, kwargs)

    # decorator-style: calls with *args should curry when TypeError
    # Here no TypeError should occur on call with args.
    assert wrapper(1, 2) == ((1, 2), {})


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


def test_memoize_key_function():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, verbose=True) == 3


def test_memoize_unhashable_args():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_unary_key():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_memoize_wrapped_and_doc():
    def f(x):
        """mydoc"""
        return x

    m = memoize(f)
    assert m.__wrapped__ is f
    assert m.__doc__ == "mydoc"
    assert m.__name__ == "f"


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc_ = lambda i: i + 1
    assert compose(str, inc_)(3) == "4"


def test_compose_no_funcs():
    assert compose() is identity


def test_compose_one_func():
    f = lambda x: x
    assert compose(f) is f


def test_compose_multiple_args():
    add_ = lambda x, y: x + y
    assert compose(double, add_)(1, 2) == 6


def test_compose_left():
    inc_ = lambda i: i + 1
    assert compose_left(inc_, str)(3) == "4"


def test_compose_left_no_funcs():
    assert compose_left() is identity


def test_compose_le_compose_left():
    f = lambda x: x + 1
    g = lambda x: x * 2
    assert compose(f, g)(5) == compose_left(g, f)(5)


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert "f(g(*args, **kwargs))" in c.__doc__


def test_compose_doc_no_names():
    class C:
        def __call__(self, x):
            return x

    c = compose(C(), C())
    assert c.__doc__ == "A composition of functions"


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == "f_of_g"


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    r = repr(c)
    assert r.startswith("Compose(")


def test_compose_eq_hash():
    def f(x):
        return x

    def g(x):
        return x

    c1 = compose(f, g)
    c2 = compose(f, g)
    c3 = compose(g, f)
    assert c1 == c2
    assert not (c1 != c2)
    assert c1 != c3
    assert hash(c1) == hash(c2)


def test_compose_eq_other():
    c = compose(lambda x: x, lambda x: x)
    assert c.__eq__(5) is NotImplemented
    assert c.__ne__(5) is NotImplemented


def test_compose_descriptor():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    class C:
        method = compose(f, g)

    c = C()
    assert c.method(5) == 11
    assert C.method(5) == 11


def test_compose_signature():
    def f(a, b):
        return a + b

    def g(x) -> int:
        return x

    c = compose(g, f)
    sig = c.__signature__
    assert list(sig.parameters) == ["a", "b"]


def test_compose_annotations():
    def f(a: int, b: int) -> str:
        return "x"

    def g(x: str) -> int:
        return len(x)

    c = compose(f, g)
    ann = c.__annotations__
    assert ann["a"] is int
    assert ann["b"] is int
    assert ann["return"] is str


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_pickle_state():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2.first == c.first
    assert c2.funcs == c.funcs


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    double_ = lambda i: 2 * i
    assert pipe(3, double_, str) == "6"


def test_pipe_no_funcs():
    assert pipe(3) == 3


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

def test_juxt_positional():
    inc_ = lambda x: x + 1
    double_ = lambda x: x * 2
    assert juxt(inc_, double_)(10) == (11, 20)


def test_juxt_list():
    inc_ = lambda x: x + 1
    double_ = lambda x: x * 2
    assert juxt([inc_, double_])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y

    assert juxt(f, f)(1, y=2) == (3, 3)


def test_juxt_state():
    f = lambda x: x
    j = juxt(f)
    assert j.__getstate__() == (f,)
    j2 = juxt.__new__(juxt)
    j2.__setstate__((f,))
    assert j2.funcs == (f,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do_returns_x_and_side_effect():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_composed():
    log = []
    inc_ = lambda x: x + 1
    f = compose(inc_, do(log.append))
    assert f(1) == 2
    assert f(11) == 12
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
    data = [1, "a", "b", 2, 1.5, 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError("x")) is None


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


def test_excepts_name_and_doc():
    def fn(a):
        return a

    def handler(e):
        return -1

    e = excepts(ValueError, fn, handler)
    assert e.__name__ == "fn_excepting_ValueError"
    assert "ValueError" in e.__doc__
    assert 'fn' in e.__doc__

    e2 = excepts((KeyError, IndexError), fn, handler)
    assert e2.__name__ == "fn_excepting_KeyError_or_IndexError"
    assert "(KeyError, IndexError)" in e2.__doc__


def test_excepts_name_doc_attribute_error():
    class NoName:
        def __call__(self, a):
            return a

    e = excepts(ValueError, NoName())
    assert e.__name__ == "excepting"
    # doc should fall back to class docstring without raising
    assert isinstance(e.__doc__, str)


def test_excepts_class_doc():
    assert isinstance(excepts.__dict__["__doc__"].classval, str)


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

    def h():
        pass

    assert num_required_args(h) == 0


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

    def h(**kwargs):
        return kwargs

    assert has_keywords(h) is True


def test_is_valid_args():
    def add_(x, y):
        return x + y

    assert is_valid_args(add_, (1,), {}) is False
    assert is_valid_args(add_, (1, 2), {}) is True
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args():
    def add_(x, y):
        return x + y

    assert is_partial_args(add_, (1,), {}) is True
    assert is_partial_args(add_, (1, 2), {}) is True
    assert is_partial_args(add_, (1, 2, 3), {}) is False
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


def test_is_arity_with_sigspec():
    def f(x):
        return x

    sig = inspect.signature(f)
    assert is_arity(1, f, sig) is True


def test_num_required_args_none_result():
    # A builtin which has no known signature: use something weird
    # Compose has proper signature, but let's trust num_required_args
    # works for builtins that have signatures.
    assert num_required_args(len) == 1
