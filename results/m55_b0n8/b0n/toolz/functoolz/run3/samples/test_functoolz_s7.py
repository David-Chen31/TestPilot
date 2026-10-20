import pickle
import copy
import inspect
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

def test_identity_returns_same_object():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


def test_apply_basic():
    assert apply(lambda x: 2 * x, 5) == 10


def test_apply_kwargs():
    def f(a, b=0):
        return a + b
    assert apply(f, 1, b=2) == 3


def test_apply_no_func_raises():
    with pytest.raises(TypeError):
        apply()


def test_apply_through_map():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000
    )


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_first_no_forms():
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


def test_thread_last_no_forms():
    assert thread_last(5) == 5


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty():
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


def test_instanceproperty_decorator_returns_instanceproperty():
    class C:
        @instanceproperty
        def x(self):
            return 1

    assert isinstance(C.x, InstanceProperty)
    assert isinstance(C.__dict__["x"], InstanceProperty)


def test_instanceproperty_partial_factory():
    # Calling without fget returns a partial-like factory
    factory = instanceproperty(classval=99)
    prop = factory(lambda self: 1)
    assert isinstance(prop, InstanceProperty)
    assert prop.classval == 99


def test_instanceproperty_reduce_roundtrip():
    class C:
        @instanceproperty(classval="cv")
        def x(self):
            return 1

    prop = C.__dict__["x"]
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    restored = cls(*state)
    assert restored.classval == "cv"


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20


def test_curry_as_decorator_kwargs():
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
        curry(1)


def test_curry_attributes():
    def f(x, y):
        "docstring"
        return x + y

    c = curry(f, 1)
    assert c.func is f
    assert c.args == (1,)
    assert c.keywords == {}
    assert c.__name__ == "f"
    assert c.__doc__ == "docstring"
    assert c.func_name == "f"


def test_curry_unpacks_partial():
    import functools

    def f(a, b, c):
        return a + b + c

    p = functools.partial(f, 1, 2)
    c = curry(p, 3)
    assert c() == 6
    assert c.func is f


def test_curry_unpacks_curry():
    def f(a, b, c):
        return a + b + c

    c1 = curry(f, 1)
    c2 = curry(c1, 2)
    assert c2(3) == 6
    assert c2.func is f
    assert c2.args == (1, 2)


def test_curry_call_gives_result():
    @curry
    def f(a, b):
        return a + b

    assert f.call(1, 2) == 3


def test_curry_does_not_curry_genuine_typeerror():
    @curry
    def f(a, b):
        return a + b

    with pytest.raises(TypeError):
        f(1, "x")


def test_curry_str_repr():
    def f(x):
        return x

    c = curry(f)
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_equality_and_hash():
    def f(x, y):
        return x + y

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert a != c
    assert a == a
    assert not (a == 5)
    assert a != 5
    assert hash(a) == hash(b)


def test_curry_get_descriptor():
    @curry
    def f(self, x):
        return self + x

    class C:
        m = f

    c = C()
    bound = c.m
    assert isinstance(bound, curry)
    assert bound(3) == c + 3


def test_curry_get_class_returns_self():
    @curry
    def f(x):
        return x

    class C:
        m = f

    assert C.m is f


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names == ["b", "c"]


def test_curry_signature_invalid_raises():
    def f(a, b):
        return a + b

    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_signature_kwonly():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1, b=2)
    sig = c.__signature__
    params = sig.parameters
    assert list(params) == ["c"]
    assert params["c"].kind == inspect.Parameter.KEYWORD_ONLY


def test_curry_pickle_roundtrip():
    c = curry(_module_level_add, 1)
    c2 = pickle.loads(pickle.dumps(c))
    assert c2(2) == 3


def _module_level_add(a, b):
    return a + b


def test_curry_with_varargs_curries_on_typeerror():
    @curry
    def f(*args, **kwargs):
        if len(args) < 2:
            raise TypeError("need two")
        return args

    partial_result = f(1)
    assert isinstance(partial_result, curry)
    assert partial_result(2) == (1, 2)


def test_curry_bind():
    def f(a, b):
        return a + b

    c = curry(f, 1)
    b = c.bind(2)
    assert isinstance(b, curry)
    assert b() == 3


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic_caching():
    calls = []
    add = memoize(lambda x, y: (calls.append((x, y)), x + y)[1])
    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_preserves_metadata():
    def f(x):
        "the doc"
        return x

    m = memoize(f)
    assert m.__name__ == "f"
    assert m.__doc__ == "the doc"
    assert m.__wrapped__ is f


def test_memoize_custom_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        raise AssertionError("should not be called")

    assert add(1, 2) == 3


def test_memoize_custom_key():
    seen = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        seen.append((x, y))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2) == 3
    assert seen == [(1, 2)]


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_unary_caches_by_arg():
    calls = []

    def double(x):
        calls.append(x)
        return 2 * x

    m = memoize(double)
    assert m(3) == 6
    assert m(3) == 6
    assert calls == [3]


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == "4"


def test_compose_no_funcs_returns_identity():
    assert compose() is identity


def test_compose_single_func_returns_it():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_multiple():
    f = lambda x: x + 1
    g = lambda x: x * 2
    h = lambda x: x - 3
    assert compose(f, g, h)(5) == ((5 - 3) * 2) + 1


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == "4"


def test_compose_left_no_funcs():
    assert compose_left() is identity


def test_compose_left_single():
    def f(x):
        return x

    assert compose_left(f) is f


def test_compose_class_direct():
    c = Compose((str, lambda x: x + 1))
    assert c(1) == "2"
    assert isinstance(c, Compose)


def test_compose_first_func_gets_all_args():
    c = compose(lambda x, y: x + y, lambda a: a * 2)
    assert c(3) == 6


def test_compose_equality():
    f = str
    g = lambda x: x + 1
    a = compose(f, g)
    b = compose(f, g)
    assert a == b
    assert a.__eq__(5) is NotImplemented
    assert a != 5


def test_compose_hash():
    f = lambda x: x
    g = lambda x: x
    c = compose(f, g)
    assert hash(c) == hash(f) ^ hash((g,))


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    r = repr(c)
    assert r.startswith("Compose")


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert "f(g(" in c.__doc__


def test_compose_doc_no_names():
    c = Compose((lambda x: x, 5 if False else (lambda y: y)))
    # should still work; just ensure __doc__ is a string
    assert isinstance(c.__doc__, str)


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == "f_of_g"


def test_compose_signature():
    def f(a: int, b: str) -> float:
        return 0.0

    def g(x: int) -> int:
        return x

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation is float


def test_compose_getstate_setstate():
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


def test_compose_descriptor_binding():
    c = compose(lambda a: a + 1, lambda a: a * 2)

    class C:
        m = c

    inst = C()
    bound = inst.m
    assert bound(3) == 7
    assert C.m is c


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_annotations():
    def f(a: int) -> str:
        return str(a)

    def g(x: int) -> int:
        return x

    c = compose(f, g)
    ann = c.__annotations__
    assert ann["a"] is int
    assert ann["return"] is str


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == "6"


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

def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list_arg():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    f = lambda a, b=0: a + b
    g = lambda a, b=0: a - b
    assert juxt(f, g)(5, b=2) == (7, 3)


def test_juxt_getstate_setstate():
    j = juxt(str, int)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2("3") == ("3", 3)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do_side_effect_and_return():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


def test_do_with_compose():
    log = []
    inc = lambda x: x + 1
    f = compose(inc, do(log.append))
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


def test_flip_curried():
    def div(a, b):
        return a // b

    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_with_isinstance():
    data = [1, "a", "b", 2, 1.5, object(), 3]
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
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert "ValueError" in excepting.__doc__
    assert "lambda" in excepting.__doc__


def test_excepts_name():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting.__name__ == "lambda_excepting_ValueError"


def test_excepts_name_tuple():
    def myfunc(a):
        return a[0]

    excepting = excepts((IndexError, KeyError), myfunc)
    assert excepting.__name__ == "myfunc_excepting_IndexError_or_KeyError"


def test_excepts_name_fallback():
    class NotAFunction:
        def __call__(self, *args):
            return 0

    excepting = excepts(ValueError, NotAFunction())
    assert excepting.__name__ == "excepting"


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2


def test_num_required_args_varargs():
    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_num_required_args_builtin_known():
    # map is known in the signatures registry
    assert num_required_args(map) == 1


def test_has_varargs():
    def f(*args):
        return args

    def g(**kwargs):
        return kwargs

    assert has_varargs(f) is True
    assert has_varargs(g) is False


def test_has_varargs_builtin():
    assert has_varargs(map) is True


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x):
        return x

    assert has_keywords(f) is True
    assert has_keywords(g) is False


def test_has_keywords_builtin():
    assert has_keywords(map) is True


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


def test_is_arity_no_signature_returns_none_or_value():
    # An object with no inspectable signature and not in registry
    class NoSig:
        pass

    result = is_arity(1, NoSig())
    # Should be either None or False; just ensure it doesn't raise
    assert result in (None, False)


def test_num_required_args_with_sigspec_param():
    def f(a, b, c=0):
        return a + b + c

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2


# ---------------------------------------------------------------------------
# instanceproperty on curry used as property
# ---------------------------------------------------------------------------

def test_curry_instanceproperty_on_class():
    class C:
        @curry
        def add(self, x, y):
            return x + y

    c = C()
    assert isinstance(C.add, curry)
    bound = c.add(1)
    assert bound(2) == 3


def test_curry_reduce_decorated():
    # curry objects pickled through head module-level function
    c = curry(_module_level_add, 1, 2)
    c2 = pickle.loads(pickle.dumps(c))
    assert c2() == 3


def test_compose_annotations_with_compose_nested():
    def f(x: int) -> str:
        return str(x)

    def g(y: int) -> int:
        return y

    inner = compose(f, g)
    outer = compose(inner, g)
    ann = outer.__annotations__
    assert ann["x"] is int
    assert ann["return"] is str


def test_memoize_no_name_attribute():
    class CallableNoName:
        def __call__(self, x):
            return x * 2

        def __doc__(self):
            pass

    f = CallableNoName()
    m = memoize(f)
    assert m(3) == 6


def test_compose_copy_picklable_functions_only():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    # Test getstate/setstate semantics used by pickle for module-level funcs
    state = c.__getstate__()
    assert state == (c.first, c.funcs)
