import inspect
import pickle
from functools import partial

import pytest

from toolz.functoolz import (
    Compose,
    InstanceProperty,
    apply,
    complement,
    compose,
    compose_left,
    curry,
    do,
    excepts,
    flip,
    has_keywords,
    has_varargs,
    identity,
    instanceproperty,
    is_arity,
    is_partial_args,
    is_valid_args,
    juxt,
    memoize,
    num_required_args,
    pipe,
    return_none,
    thread_first,
    thread_last,
)
from toolz.utils import no_default


@curry
def _module_curried_add(x, y):
    return x + y


def test_identity_returns_same_object():
    sentinel = object()

    assert identity(sentinel) is sentinel
    assert identity(3) == 3
    assert identity(None) is None


def test_apply_calls_function_with_args_and_kwargs():
    def add(a, b, c=0):
        return a + b + c

    assert apply(add, 1, 2) == 3
    assert apply(add, 1, 2, c=4) == 7
    assert apply(lambda: 42) == 42

    with pytest.raises(TypeError, match="func argument is required"):
        apply()


def test_thread_first_callables_and_tuple_forms():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    def add(x, y):
        return x + y

    def power(x, y):
        return x**y

    assert thread_first(1, inc, double) == 4
    assert thread_first(1, (add, 4), (power, 2)) == 25
    assert thread_first("unchanged") == "unchanged"


def test_thread_last_callables_and_tuple_forms():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    def add(x, y):
        return x + y

    def power(x, y):
        return x**y

    assert thread_last(1, inc, double) == 4
    assert thread_last(1, (add, 4), (power, 2)) == 32
    assert list(
        thread_last([1, 2, 3], (map, inc), (filter, lambda x: x % 2 == 0))
    ) == [2, 4]
    assert thread_last("unchanged") == "unchanged"


def test_instanceproperty_class_and_instance_access():
    class C:
        """The class docstring"""

        def __init__(self):
            self._value = 5

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return "An object docstring"

        @instanceproperty(classval="C.value")
        def value(self):
            return self._value

    assert C.__doc__ == "The class docstring"
    assert C.value == "C.value"

    obj = C()
    assert obj.__doc__ == "An object docstring"
    assert obj.value == 5


def test_instanceproperty_setter_deleter_and_reduce():
    def getx(self):
        return self._x

    def setx(self, value):
        self._x = value

    def delx(self):
        del self._x

    prop = instanceproperty(
        getx, setx, delx, doc="x property", classval="class x"
    )

    class C:
        x = prop

    obj = C()
    obj.x = 10
    assert obj.x == 10
    assert C.x == "class x"
    assert prop.__doc__ == "x property"

    del obj.x
    assert not hasattr(obj, "_x")

    assert prop.__reduce__() == (
        InstanceProperty,
        (getx, setx, delx, "x property", "class x"),
    )


def test_instanceproperty_partial_form_and_default_classval():
    def get_val(self):
        return self._value

    class C:
        def __init__(self):
            self._value = 9

        val = instanceproperty(classval="class val")(get_val)

    assert C.val == "class val"
    assert C().val == 9

    class D:
        @instanceproperty
        def val(self):
            return 1

    assert D.val is None
    assert D().val == 1


def test_curry_basic_partial_application_and_methods():
    def add(x, y):
        return x + y

    curried = curry(add)

    assert curried(1, 2) == 3

    add_two = curried(2)
    assert isinstance(add_two, curry)
    assert add_two(3) == 5

    bound = curried.bind(2)
    assert isinstance(bound, curry)
    assert bound.args == (2,)
    assert bound(3) == 5

    assert curried.call(2, 3) == 5
    with pytest.raises(TypeError):
        curried.call(2)


def test_curry_metadata_and_string_representation():
    def documented(x):
        """A documented function."""
        return x

    curried = curry(documented)

    assert curried.func is documented
    assert curried.args == ()
    assert curried.keywords == {}
    assert curried.func_name == "documented"
    assert curried.__name__ == "documented"
    assert curried.__qualname__ == documented.__qualname__
    assert curried.__module__ == documented.__module__
    assert curried.__doc__ == "A documented function."
    assert str(curried) == str(documented)
    assert repr(curried) == repr(documented)


def test_curry_init_and_call_errors():
    with pytest.raises(TypeError, match="takes at least"):
        curry()

    with pytest.raises(TypeError, match="Input must be callable"):
        curry(3)

    def boom(x):
        raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        curry(boom)(1)

    def one(x):
        return x

    with pytest.raises(TypeError):
        curry(one)(1, 2)


def test_curry_curries_variadic_functions_on_typeerror():
    def accepts_anything(*args, **kwargs):
        raise TypeError("not enough information")

    curried = curry(accepts_anything)

    result = curried(1)
    assert isinstance(result, curry)
    assert result.args == (1,)


def test_curry_merges_functools_partial_and_curry_objects():
    def add3(a, b, c):
        return a + b + c

    p = partial(add3, 1)
    curried_partial = curry(p, 2)

    assert curried_partial.func is add3
    assert curried_partial.args == (1, 2)
    assert curried_partial(3) == 6

    base = curry(add3, 1)
    merged = curry(base, 2)

    assert merged.func is add3
    assert merged.args == (1, 2)
    assert merged(3) == 6


def test_curry_keywords_equality_and_hash():
    def f(a, b=0):
        return a + b

    c1 = curry(f, a=1)
    c2 = curry(f, a=1)
    c3 = curry(f, a=2)

    assert c1 == c2
    assert c1 != c3
    assert c1 != f
    assert hash(c1) == hash(c2)
    assert c1(b=2) == 3


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    curried = curry(f, 1)
    sig = inspect.signature(curried)

    assert list(sig.parameters) == ["y", "z"]
    assert sig.parameters["y"].default is no_default
    assert sig.parameters["z"].default == 3

    keyword_curried = curry(f, x=1)
    sig2 = inspect.signature(keyword_curried)

    assert sig2.parameters["x"].default == 1
    assert sig2.parameters["x"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig2.parameters["y"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig2.parameters["z"].default == 3


def test_curry_signature_rejects_incorrect_arguments():
    def f(a, b):
        return a + b

    bad = curry(f, 1, 2, 3)

    with pytest.raises(TypeError, match="curry object has incorrect arguments"):
        inspect.signature(bad)


def test_curry_descriptor_binds_instances():
    def add(self, x):
        return self.base + x

    class C:
        base = 10
        method = curry(add)

    obj = C()

    assert obj.method(5) == 15
    assert isinstance(C.method, curry)
    assert C.method.func is add


def test_curry_pickle_roundtrip():
    curried = curry(pow, 2)
    restored = pickle.loads(pickle.dumps(curried))

    assert isinstance(restored, curry)
    assert restored.func is pow
    assert restored.args == (2,)
    assert restored(3) == 8

    assert pickle.loads(pickle.dumps(_module_curried_add)) is _module_curried_add

    partial_curried = _module_curried_add(1)
    restored_partial = pickle.loads(pickle.dumps(partial_curried))

    assert isinstance(restored_partial, curry)
    assert restored_partial.args == (1,)
    assert restored_partial(2) == 3


def test_memoize_caches_and_preserves_metadata():
    calls = []

    def add(x, y):
        """Add two numbers."""
        calls.append((x, y))
        return x + y

    cached = memoize(add)

    assert cached.__name__ == "add"
    assert cached.__doc__ == "Add two numbers."
    assert cached.__wrapped__ is add

    assert cached(1, 2) == 3
    assert cached(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_initial_cache_and_cache_updates():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    cache = {(1, 2): 100}
    cached = memoize(cache=cache)(add)

    assert cached(1, 2) == 100
    assert calls == []

    assert cached(3, 4) == 7
    assert calls == [(3, 4)]
    assert cache[(3, 4)] == 7


def test_memoize_custom_key_and_unhashable_arguments():
    calls = []

    def add_verbose(x, y, verbose=False):
        calls.append(verbose)
        return x + y

    cached = memoize(key=lambda args, kwargs: args)(add_verbose)

    assert cached(1, 2, verbose=True) == 3
    assert cached(1, 2, verbose=False) == 3
    assert calls == [True]

    def unary(x):
        return x

    unary_cached = memoize(unary)
    with pytest.raises(
        TypeError, match="Arguments to memoized function must be hashable"
    ):
        unary_cached([1, 2, 3])


def test_memoize_callable_without_name():
    class Callable:
        def __call__(self, x):
            return x + 1

    cached = memoize(Callable())

    assert cached.__name__ == "memof"
    assert cached(1) == 2
    assert cached(1) == 2


def test_compose_helpers_and_simple_composition():
    inc = lambda i: i + 1
    double = lambda i: 2 * i

    assert compose() is identity
    assert compose(inc) is inc
    assert compose(str, inc)(3) == "4"

    assert compose_left() is identity
    assert compose_left(inc) is inc
    assert compose_left(inc, str)(3) == "4"

    assert pipe(3, double, str) == "6"
    assert pipe("unchanged") == "unchanged"


def test_complement_returns_composed_predicate():
    is_even = lambda n: n % 2 == 0
    is_odd = complement(is_even)

    assert isinstance(is_odd, Compose)
    assert is_odd(2) is False
    assert is_odd(3) is True


def test_compose_behavior_introspection_and_repr():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    def add(a, b):
        return a + b

    c = compose(str, inc)

    assert c.first is inc
    assert c.funcs == (str,)
    assert c(3) == "4"
    assert compose(double, add)(1, 2) == 6

    assert c.__name__ == "str_of_inc"
    assert c.__doc__ == "lambda *args, **kwargs: str(inc(*args, **kwargs))"
    assert c.__wrapped__ is inc
    assert Compose.__wrapped__ is None
    assert repr(c) == "Compose" + repr((str, inc))


def test_compose_state_pickle_and_equality():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c1 = compose(str, inc)
    c2 = compose(str, inc)
    c3 = compose(str, double)

    state = c1.__getstate__()
    assert state == (inc, (str,))

    new = Compose.__new__(Compose)
    new.__setstate__(state)
    assert new.first is inc
    assert new.funcs == (str,)
    assert new(3) == "4"

    assert c1 == c2
    assert c1 != c3
    assert c1.__eq__(object()) is NotImplemented
    assert c1.__ne__(object()) is NotImplemented
    assert c1.__ne__(c2) is False
    assert c1.__ne__(c3) is True
    assert hash(c1) == hash(c1.first) ^ hash(c1.funcs)

    pickled = compose(str, abs)
    assert pickle.loads(pickle.dumps(pickled))(-4) == "4"


def test_compose_signature_and_annotations():
    def first(x: int) -> float:
        return float(x)

    def second(y: float) -> str:
        return str(y)

    c = compose(second, first)
    sig = inspect.signature(c)

    assert list(sig.parameters) == ["x"]
    assert sig.parameters["x"].annotation is int
    assert sig.return_annotation is str

    assert c.__annotations__ == {"x": int, "return": str}
    assert c._combined_annotations() == {"x": int, "return": str}
    assert Compose.__annotations__ == {}

    nested = compose(second, c)
    assert inspect.signature(nested).return_annotation is str
    assert nested.__annotations__ == {"x": int, "return": str}


def test_compose_callable_without_name_and_single_function_instance():
    class NoNameCallable:
        def __call__(self, x):
            return x + 1

    c = compose(str, NoNameCallable())

    assert c.__name__ == "Compose"
    assert c.__doc__ == "A composition of functions"

    def inc_single(x):
        return x + 1

    single = Compose((inc_single,))

    assert single.first is inc_single
    assert single.funcs == ()
    assert single(1) == 2
    assert single.__name__ == "inc_single"
    assert (
        single.__doc__
        == "lambda *args, **kwargs: inc_single(*args, **kwargs)"
    )


def test_compose_descriptor_binds_as_method():
    def add_base(self, x):
        return self.base + x

    def inc(x):
        return x + 1

    class C:
        method = compose(inc, add_base)

    obj = C()
    obj.base = 10

    assert obj.method(5) == 16
    assert isinstance(C.method, Compose)
    assert C.method.first is add_base


def test_juxt_calls_all_functions_and_preserves_kwargs():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    def add(a, b=0):
        return a + b

    def mul(a, b=0):
        return a * b

    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)
    assert juxt((inc, double))(10) == (11, 20)
    assert juxt(add, mul)(3, b=4) == (7, 12)

    j = juxt(inc)
    state = j.__getstate__()
    assert state == (inc,)

    restored = juxt.__new__(juxt)
    restored.__setstate__(state)
    assert restored(2) == (3,)


def test_do_returns_input_and_runs_side_effect():
    log = []

    def record(value):
        log.append(value)

    assert do(record, 5) == 5
    assert log == [5]


def test_flip_and_curried_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    assert isinstance(flip(div, 2), curry)

    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2

    assert flip(isinstance, int)(1) is True


def test_excepts_dispatches_and_default_handler():
    def index(a):
        return [1, 2].index(a)

    def handler(exc):
        return -1

    excepting = excepts(ValueError, index, handler)

    assert excepting(1) == 0
    assert excepting(3) == -1
    assert excepting.__name__ == "index_excepting_ValueError"

    def first(a):
        return a[0]

    defaulting = excepts((IndexError, KeyError), first)

    assert defaulting([]) is None
    assert defaulting([1]) == 1
    assert defaulting({}) is None
    assert defaulting({0: 1}) == 1
    assert (
        defaulting.__name__
        == "first_excepting_IndexError_or_KeyError"
    )

    with pytest.raises(IndexError):
        excepts(KeyError, first)([])


def test_excepts_custom_handler_receives_exception():
    def raise_value(x):
        raise ValueError("bad value")

    def handle(exc):
        return type(exc).__name__, str(exc)

    excepting = excepts(ValueError, raise_value, handle)

    assert excepting(1) == ("ValueError", "bad value")
    assert return_none(ValueError("boom")) is None


def test_excepts_doc_and_name_fallbacks():
    def index(a):
        """Find index."""
        return [1, 2].index(a)

    def handle(exc):
        """Return sentinel."""
        return -1

    excepting = excepts(ValueError, index, handle)

    assert excepting.__doc__.splitlines() == [
        "A wrapper around 'index' that will except:",
        "ValueError",
        "and handle any exceptions with 'handle'.",
        "",
        "Docs for 'index':",
        "Find index.",
        "",
        "Docs for 'handle':",
        "Return sentinel.",
    ]

    tuple_excepting = excepts((ValueError, KeyError), index, handle)
    assert tuple_excepting.__doc__.splitlines()[1] == "(ValueError, KeyError)"

    class NoNameCallable:
        def __call__(self, x):
            return x

    no_name = excepts(ValueError, NoNameCallable())
    assert no_name.__name__ == "excepting"
    assert no_name.__doc__ == excepts.__doc__


def test_introspection_functions_for_python_functions():
    def f(a, b, c=3):
        return a + b + c

    assert num_required_args(f) == 2
    assert has_varargs(f) is False
    assert has_keywords(f) is True
    assert is_valid_args(f, (1, 2), {}) is True
    assert is_valid_args(f, (1,), {}) is False
    assert is_valid_args(f, (1, 2, 3, 4), {}) is False
    assert is_partial_args(f, (1,), {}) is True
    assert is_partial_args(f, (1, 2, 3, 4), {}) is False

    def variadic(*args):
        return args

    assert has_varargs(variadic) is True

    def keyworded(**kwargs):
        return kwargs

    assert has_varargs(keyworded) is False
    assert has_keywords(keyworded) is True

    def kwonly(*, x):
        return x

    assert num_required_args(kwonly) == 0
    assert has_keywords(kwonly) is True
    assert is_valid_args(kwonly, (), {"x": 1}) is True
    assert is_valid_args(kwonly, (1,), {}) is False


def test_is_arity():
    def unary(x):
        return x

    def optional(x, y=1):
        return x + y

    def variadic_pos(x, *args):
        return x, args

    def no_args():
        return None

    assert is_arity(1, unary) is True
    assert is_arity(1, optional) is False
    assert is_arity(1, variadic_pos) is False
    assert is_arity(0, no_args) is True


def test_introspection_functions_use_builtin_signature_registry():
    assert is_valid_args(map, (), {}) is False
    assert is_partial_args(map, (), {}) is True
