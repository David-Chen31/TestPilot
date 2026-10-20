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


def _module_level_add_one(x):
    return x + 1


# ----------------------------- identity / apply -----------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


def test_apply_basic():
    assert apply(lambda x: x * 2, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3


def test_apply_kwargs():
    def f(a, b, c=0):
        return a + b + c

    assert apply(f, 1, 2, c=3) == 6


def test_apply_requires_func():
    with pytest.raises(TypeError):
        apply()


# ----------------------------- thread_first/last ----------------------------

def test_thread_first_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def pw(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pw, 2)) == 25


def test_thread_first_empty():
    assert thread_first(5) == 5


def test_thread_last_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
    def add(x, y):
        return x + y

    def pw(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pw, 2)) == 32


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_empty():
    assert thread_last(5) == 5


# ----------------------------- instanceproperty -----------------------------

def test_instanceproperty_class_attr():
    class MyClass:
        """Class doc"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return "instance doc"

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == "Class doc"
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == "instance doc"
    assert obj.val == 42


def test_instanceproperty_partial_no_fget():
    class C:
        @instanceproperty(classval="cv")
        def value(self):
            return "iv"

    assert C.value == "cv"
    assert C().value == "iv"


def test_instanceproperty_with_setter():
    class C:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, v):
            self._x = v

    c = C()
    c.x = 10
    assert c.x == 10


def test_instanceproperty_reduce():
    class C:
        @instanceproperty
        def x(self):
            return 1

    prop = C.__dict__["x"]
    assert isinstance(prop, InstanceProperty)
    restored, state = prop.__reduce__()
    assert restored is InstanceProperty
    assert state[0] is prop.fget


# ----------------------------- curry ----------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    c = curry(mul)
    assert c(2)(10) == 20
    assert c(2, 3) == 6


def test_curry_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_is_curry_identity():
    def f(a, b):
        return a + b

    c1 = curry(f, 1)
    c2 = curry(c1, 2)
    assert c2() == 3


def test_curry_attributes():
    def myfunc(a, b):
        "docs"

    c = curry(myfunc)
    assert c.func is myfunc
    assert c.args == ()
    assert c.keywords == {}
    assert c.func_name == "myfunc"
    assert c.__name__ == "myfunc"
    assert c.__doc__ == "docs"


def test_curry_str_repr():
    def myfunc(a):
        return a

    c = curry(myfunc)
    assert str(c) == str(myfunc)
    assert repr(c) == repr(myfunc)


def test_curry_equality_hash():
    def f(a, b):
        return a + b

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_curry_eq_noncurry():
    def f(a):
        return a

    assert curry(f) != f


def test_curry_signature():
    def f(a, b, c=1):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.keys())
    assert params == ["b", "c"]


def test_curry_bind_and_call():
    def f(a, b):
        return a + b

    c = curry(f)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    with pytest.raises(TypeError):
        bound.call()
    assert bound.call(2) == 3


def test_curry_get_descriptor():
    class C:
        @curry
        def method(self, x):
            return self.value + x

        def __init__(self):
            self.value = 10

    c = C()
    assert c.method(5) == 15


def test_curry_get_on_class_returns_self():
    def f(a):
        return a

    c = curry(f)
    assert c.__get__(None, object) is c


def test_curry_genuine_typeerror():
    @curry
    def f(x):
        raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        f(1)


def test_curry_varargs():
    @curry
    def f(*args):
        return args

    assert f(1) == (1,)


def test_curry_incorrect_args_signature():
    def f(a, b):
        return a + b

    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_pickle_module_level():
    # curry around a builtin-ish function pickles via reduce
    c = curry(pow, 2)
    data = pickle.dumps(c)
    c2 = pickle.loads(data)
    assert c2(3) == 8


# ----------------------------- memoize --------------------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert len(calls) == 1


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    # filled in
    assert add(2, 3) == 5


def test_memoize_unary():
    calls = []

    @memoize
    def sq(x):
        calls.append(x)
        return x * x

    assert sq(3) == 9
    assert sq(3) == 9
    assert sq(4) == 16
    assert calls == [3, 4]


def test_memoize_custom_key():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2) == 3


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2])


def test_memoize_attributes():
    def f(x):
        "docstring"
        return x

    m = memoize(f)
    assert m.__name__ == "f"
    assert m.__doc__ == "docstring"
    assert m.__wrapped__ is f


# ----------------------------- compose / Compose ----------------------------

def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    def f(x):
        return x + 1

    assert compose(f) is f


def test_compose_order():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == "4"


def test_compose_left_order():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == "4"


def test_compose_multi_arg():
    # only the first (rightmost) function receives multiple arguments;
    # subsequent functions receive the single threaded result.
    def add_then_double(x, y):
        return x + y

    inc = lambda i: i + 1
    assert compose(inc, add_then_double)(3, 4) == 8


def test_compose_class_basics():
    def f(x):
        return x + 1

    c = Compose((f, f))
    assert c(0) == 2
    assert isinstance(c, Compose)


def test_compose_doc():
    def foo(x):
        return x

    def bar(x):
        return x

    c = Compose((foo, bar))
    assert "foo" in c.__doc__
    assert "bar" in c.__doc__


def test_compose_doc_no_names():
    c = Compose((lambda x: x,))
    # single lambda in Compose keeps only first? Actually Compose requires >= funcs
    assert isinstance(c.__doc__, str)


def test_compose_name():
    def foo(x):
        return x

    def bar(x):
        return x

    c = Compose((foo, bar))
    assert c.__name__ == "foo_of_bar"


def test_compose_name_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = Compose((Callable(), Callable()))
    assert c.__name__ == "Compose"


def test_compose_repr():
    def foo(x):
        return x

    c = Compose((foo,))
    assert "Compose" in repr(c)
    assert "foo" in repr(c)


def test_compose_equality():
    def f(x):
        return x

    def g(x):
        return x

    a = Compose((f, g))
    b = Compose((f, g))
    c = Compose((g, f))
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_compose_eq_not_implemented():
    c = Compose((lambda x: x,))
    assert c.__eq__(1) is NotImplemented
    assert c.__ne__(1) is NotImplemented


def test_compose_descriptor():
    class C:
        f = Compose((lambda x: x + 1,))

        def __init__(self):
            self.value = 10

    # Compose is not set up as a method here since first isn't self, but
    # accessing on instance should produce a bound MethodType.
    inst = C()
    bound = inst.f
    assert callable(bound)


def test_compose_descriptor_class_access():
    c = Compose((lambda x: x,))
    assert c.__get__(None, object) is c


def test_compose_signature():
    def f(x, y) -> int:
        return x + y

    def g(z) -> str:
        return str(z)

    c = Compose((g, f))
    sig = c.__signature__
    assert list(sig.parameters) == ["x", "y"]
    assert sig.return_annotation is str


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = Compose((f, g))
    # __wrapped__ is the first function applied in the composition,
    # i.e. the last function in the user-supplied argument order.
    assert c.__wrapped__ is g


@pytest.mark.skipif(sys.version_info < (3, 10), reason="needs newer typing")
def test_compose_annotations():
    def f(x: int) -> bool:
        return bool(x)

    def g(y: str) -> int:
        return int(y)

    c = Compose((f, g))
    anns = c.__annotations__
    # Parameter annotations come from the first-applied function (g).
    assert anns["y"] is str
    # Return annotation comes from the last-applied function (f).
    assert anns["return"] is bool


def test_compose_pickle():
    add_one = _module_level_add_one
    c = Compose((add_one,))
    data = pickle.dumps(c)
    c2 = pickle.loads(data)
    assert c2(1) == 2


# ----------------------------- pipe / complement ----------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == "6"


def test_pipe_empty():
    assert pipe(5) == 5


def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ----------------------------- juxt -----------------------------------------

def test_juxt_args():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(a, b=0):
        return a + b

    def g(a, b=0):
        return a * b

    assert juxt(f, g)(2, b=3) == (5, 6)


def test_juxt_pickle():
    j = juxt(lambda x: x + 1)
    # lambdas don't pickle, so use module-level functions
    j2 = juxt(abs, str)
    data = pickle.dumps(j2)
    j3 = pickle.loads(data)
    assert j3(-5) == (5, "-5")


# ----------------------------- do / flip ------------------------------------

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
    data = [1, "a", "b", 2]
    result = list(filter(flip(isinstance, int), data))
    assert result == [1, 2]


# ----------------------------- return_none / excepts ------------------------

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


def test_excepts_reraise_other():
    excepting = excepts(ValueError, lambda a: a[0])
    with pytest.raises(TypeError):
        excepting(5)


def test_excepts_attributes():
    def func(a):
        "func doc"
        return a

    def handler(e):
        "handler doc"
        return None

    e = excepts(ValueError, func, handler)
    assert e.func is func
    assert e.handler is handler
    assert e.exc is ValueError
    assert e.__name__ == "func_excepting_ValueError"


def test_excepts_tuple_name():
    e = excepts((ValueError, KeyError), lambda x: x, return_none)
    assert e.__name__ == "<lambda>_excepting_ValueError_or_KeyError"


def test_excepts_name_fallback():
    class NoName:
        def __call__(self, x):
            return x

    e = excepts(ValueError, NoName())
    assert e.__name__ == "excepting"


def test_excepts_doc():
    def func(a):
        "func doc"
        return a

    def handler(e):
        "handler doc"
        return None

    e = excepts(ValueError, func, handler)
    doc = e.__doc__
    assert "func doc" in doc
    assert "handler doc" in doc
    assert "ValueError" in doc


def test_excepts_class_doc():
    assert "A wrapper around a function" in excepts.__doc__


# ----------------------------- signature helpers ----------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0

    def h(x, *, y):
        pass

    assert num_required_args(h) == 1


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

    assert is_arity(2, h) is True
    assert is_arity(1, h) is False


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


def test_is_arity_keywords():
    def f(x, **kwargs):
        return x

    assert is_arity(1, f) is False


# ----------------------------- curry with signatures -----------------------

def test_curry_signature_with_keywords():
    def f(a, b, *, c=1):
        return a + b + c

    c = curry(f, 1, c=5)
    sig = c.__signature__
    assert "b" in sig.parameters
    # c became keyword-only with default 5
    param = sig.parameters.get("c")
    if param is not None:
        assert param.default == 5


def test_curry_word_bound_signature():
    def f(a, b):
        return a + b

    c = curry(f)
    sig = c.__signature__
    assert list(sig.parameters) == ["a", "b"]
