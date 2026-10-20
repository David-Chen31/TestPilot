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
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_multiple_args():
    def add(x, y):
        return x + y

    assert apply(add, 1, 2) == 3


def test_apply_kwargs():
    def f(a, b=0):
        return a - b

    assert apply(f, 10, b=3) == 7


def test_apply_no_args_raises():
    with pytest.raises(TypeError):
        apply()


def test_apply_no_args_message():
    with pytest.raises(TypeError) as excinfo:
        apply()
    assert "func argument is required" in str(excinfo.value)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def _double(x):
    return 2 * x


def _inc(x):
    return x + 1


def _add(x, y):
    return x + y


def _pow(x, y):
    return x ** y


def test_thread_first_simple():
    assert thread_first(1, _inc, _double) == 4


def test_thread_first_tuples():
    assert thread_first(1, (_add, 4), (_pow, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_simple():
    assert thread_last(1, _inc, _double) == 4


def test_thread_last_tuples():
    assert thread_last(1, (_add, 4), (_pow, 2)) == 32


def test_thread_last_no_forms():
    assert thread_last(5) == 5


def test_thread_last_map_filter():
    def iseven(x):
        return x % 2 == 0

    result = list(thread_last([1, 2, 3], (map, _inc), (filter, iseven)))
    assert result == [2, 4]


def test_thread_first_empty_tuple_form_returns_none():
    # evalform_front returns None for non-callable, non-tuple forms.
    assert thread_first(5, 42) is None


def test_thread_last_empty_tuple_form_returns_none():
    assert thread_last(5, 42) is None


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        @instanceproperty
        def val(self):
            return 42

    assert MyClass.val is None
    assert MyClass().val == 42


def test_instanceproperty_classval():
    class MyClass:
        @instanceproperty(classval="class-value")
        def val(self):
            return "instance-value"

    assert MyClass.val == "class-value"
    assert MyClass().val == "instance-value"


def test_instanceproperty_setter():
    class MyClass:
        def __init__(self):
            self._x = 1

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    obj = MyClass()
    assert obj.x == 1
    obj.x = 10
    assert obj.x == 10


def test_instanceproperty_no_fget_returns_partial():
    p = instanceproperty(classval="cv")
    assert isinstance(p, partial)
    descriptor = p(lambda self: "inst")
    assert isinstance(descriptor, InstanceProperty)
    assert descriptor.classval == "cv"


def test_instanceproperty_reduce():
    def fget(self):
        return 1

    prop = InstanceProperty(fget=fget, classval="cv")
    new_prop, state = prop.__reduce__()
    assert new_prop is InstanceProperty
    assert state == (fget, None, None, prop.__doc__, "cv")


def test_instanceproperty_deleter():
    class MyClass:
        def __init__(self):
            self._x = 5

        @instanceproperty
        def x(self):
            return self._x

        @x.deleter
        def x(self):
            del self._x

    obj = MyClass()
    assert obj.x == 5
    del obj.x
    assert not hasattr(obj, "_x")


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_partial_application():
    def mul(x, y):
        return x * y

    c = curry(mul)
    double = c(2)
    assert isinstance(double, curry)
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
        curry(5)


def test_curry_non_callable_message():
    with pytest.raises(TypeError) as excinfo:
        curry(5)
    assert "Input must be callable" in str(excinfo.value)


def test_curry_bind():
    def add(x, y):
        return x + y

    c = curry(add)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound(2) == 3


def test_curry_call_method():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    # call() invokes without currying
    with pytest.raises(TypeError):
        c.call()


def test_curry_call_method_success():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    assert c.call(2) == 3


def test_curry_attributes():
    def add(x, y):
        """docstring here"""
        return x + y

    c = curry(add)
    assert c.func is add
    assert c.args == ()
    assert c.keywords == {}
    assert c.__doc__ == "docstring here"
    assert c.__name__ == "add"
    assert c.func_name == "add"


def test_curry_str_repr():
    def add(x, y):
        return x + y

    c = curry(add)
    assert str(c) == str(add)
    assert repr(c) == repr(add)


def test_curry_equality_and_hash():
    def add(x, y):
        return x + y

    c1 = curry(add, 1)
    c2 = curry(add, 1)
    c3 = curry(add, 2)
    assert c1 == c2
    assert not (c1 != c2)
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1.__eq__(5) is False


def test_curry_genuine_type_error_reraises():
    def f(x):
        return x.attribute_that_does_not_exist

    c = curry(f)
    with pytest.raises(AttributeError):
        c(1)


def test_curry_genuine_type_error_inside_body_reraises():
    # TypeError raised inside function body: args are valid, no varargs, so
    # curry knows the TypeError is genuine and re-raises (line 309).
    def f(x):
        return x + "not an int"

    c = curry(f)
    with pytest.raises(TypeError):
        c(1)


def test_curry_merges_existing_partial():
    def add(x, y, z=0):
        return x + y + z

    p = partial(add, 1)
    c = curry(p, 2)
    assert c(3) == 6


def test_curry_merges_partial_keywords():
    def f(a, b=0, c=0):
        return a + b + c

    p = partial(f, b=1)
    c = curry(p, c=2)
    assert c(1) == 4


def test_curry_varargs_always_curries():
    # A wrapper with signature (*args, **kwargs) that raises TypeError when it
    # receives too few real arguments: since the wrapper has *args, curry
    # cannot know the TypeError was genuine, so it curries anyway.
    def inner(x, y):
        return x + y

    def wrapper(*args, **kwargs):
        return inner(*args, **kwargs)

    c = curry(wrapper)
    partial_result = c(1)
    assert isinstance(partial_result, curry)
    assert partial_result(2) == 3


def test_curry_varargs_passthrough_when_valid():
    # *args function that does NOT raise: call succeeds, no currying occurs.
    def f(*args, **kwargs):
        return (args, kwargs)

    c = curry(f)
    result = c(1)
    assert result == ((1,), {})


def test_curry_bind_with_keywords_triggers_should_curry():
    # Calling with existing keywords triggers the `if self.keywords:` branch
    # in _should_curry, and the call also raises TypeError so currying occurs.
    def f(a, b, c, d):
        return a + b + c + d

    c = curry(f, 1)
    partial_result = c(b=2)
    assert isinstance(partial_result, curry)


def test_curry_get_on_class_returns_self():
    def method(self, x):
        return x

    c = curry(method)

    class Foo:
        m = c

    assert Foo.m is c


def test_curry_get_on_instance_binds():
    def method(self, x):
        return x + 1

    class Foo:
        m = curry(method)

    foo = Foo()
    bound = foo.m
    assert isinstance(bound, curry)
    result = bound(10)
    assert result == 11


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names == ["b", "c"]


def test_curry_signature_incorrect_args_raises():
    def f(a):
        return a

    c = curry(f, 1, 2)
    with pytest.raises(TypeError) as excinfo:
        c.__signature__
    assert "curry object has incorrect arguments" in str(excinfo.value)


def test_curry_signature_skips_varargs_param():
    def f(a, *args):
        return a

    c = curry(f, 1)
    sig = c.__signature__
    names = [p.name for p in sig.parameters.values()]
    # The *args parameter is kept, its name is "args"
    assert names == ["args"]


def test_curry_signature_var_keyword_skipped():
    def f(a, **kwargs):
        return a

    c = curry(f, 1)
    sig = c.__signature__
    names = [p.name for p in sig.parameters.values()]
    # VAR_KEYWORD param is kept in this implementation
    assert names == ["kwargs"]


def test_curry_signature_keyword_only_param_from_keywords():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1, b=2)
    sig = c.__signature__
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    # b was supplied as keyword and is retained; both b and c are listed
    assert names == ["b", "c"]
    # b becomes keyword-only
    b_param = sig.parameters["b"]
    assert b_param.kind == inspect.Parameter.KEYWORD_ONLY


def test_curry_selected_annotation_no_default_uses_no_default():
    from toolz.utils import no_default

    def f(a, b):
        return a + b

    c = curry(f, 1)
    b_param = c.__signature__.parameters["b"]
    assert b_param.default is no_default


def test_curry_pickle_roundtrip():
    # Module-level function so pickle can resolve it.
    c = curry(_add, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert isinstance(restored, curry)
    assert restored(2) == 3


def test_curry_pickle_preserves_userdict():
    c = curry(_add, 1)
    c.custom_attribute = "hello"
    restored = pickle.loads(pickle.dumps(c))
    assert restored.custom_attribute == "hello"


def test_curry_no_module_name():
    def inner(x, y):
        return x + y

    c = curry(inner)
    assert c.__name__ == "inner"


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    memoized = memoize(add)
    assert memoized(1, 2) == 3
    assert memoized(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_single_arg_uses_arg_as_key():
    calls = []

    def square(x):
        calls.append(x)
        return x * x

    memoized = memoize(square)
    assert memoized(4) == 16
    assert memoized(4) == 16
    assert calls == [4]


def test_memoize_with_initial_cache():
    def add(x, y):
        raise AssertionError("should not be called")

    memoized = memoize(add, cache={(1, 2): 99})
    assert memoized(1, 2) == 99


def test_memoize_custom_key():
    calls = []

    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    memoized = memoize(add, key=lambda args, kwargs: args)
    assert memoized(1, 2, verbose=True) == 3
    assert memoized(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_kwargs_key():
    calls = []

    def add(x, y=0):
        calls.append((x, y))
        return x + y

    memoized = memoize(add)
    assert memoized(1, y=2) == 3
    assert memoized(1, y=2) == 3
    assert len(calls) == 1


def test_memoize_unhashable_raises():
    def f(x):
        return x

    memoized = memoize(f)
    with pytest.raises(TypeError):
        memoized([1, 2, 3])


def test_memoize_unhashable_message():
    def f(x):
        return x

    memoized = memoize(f)
    with pytest.raises(TypeError) as excinfo:
        memoized([1, 2, 3])
    assert "must be hashable" in str(excinfo.value)


def test_memoize_preserves_metadata():
    def add(x, y):
        """Add two numbers"""
        return x + y

    memoized = memoize(add)
    assert memoized.__name__ == "add"
    assert memoized.__doc__ == "Add two numbers"
    assert memoized.__wrapped__ is add


def test_memoize_as_decorator():
    @memoize
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_with_cache_decorator():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100
    assert add(3, 4) == 7


def test_memoize_no_kwargs_positional_only_key():
    # A function with no keywords and not unary: key is just args.
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    memoized = memoize(add)
    assert memoized(1, 2) == 3
    assert memoized(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_metadata_without_name():
    # A callable without __name__: memof.__name__ assignment fails, caught.
    class Callable:
        def __call__(self, x):
            return x + 1

        def __doc__(self):
            return ""

    obj = Callable()
    memoized = memoize(obj)
    assert memoized(1) == 2


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == "4"


def test_compose_no_funcs_identity():
    assert compose()(42) == 42


def test_compose_single_func():
    def f(x):
        return x + 1

    assert compose(f) is f


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == "4"


def test_compose_left_no_funcs():
    assert compose_left()(5) == 5


def test_compose_kwargs():
    def f(a, b):
        return a - b

    def g(x):
        return x * 2

    assert compose(g, f)(10, 3) == 14


def test_compose_doc_and_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == "f_of_g"
    assert "f(g(*args, **kwargs))" in c.__doc__


def test_compose_no_name():
    c = compose(len, sum)
    assert c.__name__ == "len_of_sum"


def test_compose_doc_no_names():
    def f(x):
        return x

    class Callable:
        def __call__(self, x):
            return x

    obj = Callable()
    c = compose(f, obj)
    # obj has no __name__ so the fallback message is used
    assert c.__doc__ == "A composition of functions"


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    # repr shows the actual function reprs in forward order (f, g)
    assert repr(c) == "Compose(%r, %r)" % (f, g)


def test_compose_equality():
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
    assert c1.__eq__(42) is NotImplemented
    assert c1.__ne__(42) is NotImplemented


def test_compose_hash():
    def f(x):
        return x

    def g(x):
        return x

    c1 = compose(f, g)
    c2 = compose(f, g)
    assert hash(c1) == hash(c2)


def test_compose_getstate_setstate():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    state = c.__getstate__()
    new = Compose.__new__(Compose)
    new.__setstate__(state)
    assert new(5) == 5


def test_compose_signature():
    def f(a: int) -> int:
        return a

    def g(b: int) -> str:
        return str(b)

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation == int


def test_compose_get_descriptor():
    def f(x):
        return x + 1

    def g(self, x):
        return x * 2

    c = compose(f, g)

    class Foo:
        method = c

    assert Foo.method is c
    instance = Foo()
    bound = instance.method
    # f(g(instance, 5)) == f(10) == 11
    assert bound(5) == 11


def test_compose_annotations():
    def f(a: int, b: str) -> float:
        return 0.0

    def g(x: bool) -> int:
        return 1

    c = compose(f, g)
    ann = c._combined_annotations()
    assert ann["return"] == float
    assert ann["x"] is bool
    assert "a" not in ann
    assert "b" not in ann


def test_compose_combined_annotations_via_descriptor():
    def f(a: int) -> float:
        return 0.0

    def g(x: bool) -> int:
        return 1

    c = compose(f, g)
    ann = c.__annotations__
    assert ann["return"] == float
    assert ann["x"] is bool
    assert isinstance(Compose.__annotations__, dict)


def test_compose_combined_annotations_function_without_signature():
    import builtins

    def f(x) -> int:
        return 1

    class NoSig:
        def __call__(self, *args):
            return None

    c = compose(f, NoSig())
    ann = c._combined_annotations()
    assert ann == {"return": int} or ann.get("return") == int


def test_compose_combined_annotations_nested_compose():
    def f(a: int) -> float:
        return 0.0

    def g(x: bool) -> int:
        return 1

    def h(y: str) -> bool:
        return True

    inner = compose(g, h)
    outer = compose(f, inner)
    ann = outer._combined_annotations()
    assert ann["return"] == float


def test_compose_wrapped_is_first():
    # __wrapped__ is self.first, which is the LAST func passed to compose
    # (applied first), i.e. g here.
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_signature_single_func_after_init():
    # Return annotation is taken from self.funcs[-1], which here is f (no
    # annotation -> empty).
    def f(x):
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(f, g)
    assert c.__signature__.return_annotation is inspect.Parameter.empty


def test_compose_attributes():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.first is g
    assert c.funcs == (f,)


# ---------------------------------------------------------------------------
# pipe / complement
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == "6"


def test_pipe_no_funcs():
    assert pipe(3) == 3


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
        return x - y

    assert juxt(f, g)(5, 3) == (8, 2)


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    new = juxt.__new__(juxt)
    new.__setstate__(state)
    assert new(1) == (2,)


def test_juxt_single_callable():
    f = lambda x: x + 1
    j = juxt(f)
    assert j.funcs == (f,)
    assert j(3) == (4,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do_returns_x():
    seen = []
    assert do(seen.append, 5) == 5
    assert seen == [5]


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
    data = [1, "a", "b", 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError("x")) is None


# ---------------------------------------------------------------------------
# excepts
# ---------------------------------------------------------------------------

def test_excepts_basic():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler_returns_none():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_name():
    excepting = excepts(ValueError, lambda a: a)
    assert excepting.__name__ == "<lambda>_excepting_ValueError"


def test_excepts_name_tuple():
    excepting = excepts((ValueError, KeyError), lambda a: a)
    assert excepting.__name__ == "<lambda>_excepting_ValueError_or_KeyError"


def test_excepts_name_fallback():
    class NotCallable:
        pass

    excepting = excepts(ValueError, NotCallable())
    assert excepting.__name__ == "excepting"


def test_excepts_docstring():
    def f(x):
        """func doc"""
        return x

    def handler(e):
        """handler doc"""
        return None

    excepting = excepts(ValueError, f, handler)
    assert "A wrapper around 'f'" in excepting.__doc__
    assert "ValueError" in excepting.__doc__
    assert "func doc" in excepting.__doc__
    assert "handler doc" in excepting.__doc__


def test_excepts_doc_tuple():
    def f(x):
        """func doc"""
        return x

    excepting = excepts((ValueError, KeyError), f)
    assert "(ValueError, KeyError)" in excepting.__doc__


def test_excepts_class_doc():
    assert excepts.__doc__ is not None


def test_excepts_doc_fallback():
    class NotCallable:
        pass

    excepting = excepts(ValueError, NotCallable())
    assert excepting.__doc__ == excepts.__doc__


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


def test_has_keywords_kwonly():
    def f(x, *, y):
        return x + y

    assert has_keywords(f) is True


def test_has_keywords_var_keyword():
    def f(x, **kwargs):
        return x

    assert has_keywords(f) is True


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
    assert is_arity(1, len) is True


def test_num_required_args_builtin():
    assert num_required_args(map) == 2


def test_is_valid_args_map():
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args_map():
    assert is_partial_args(map, (), {}) is True


def test_num_required_args_with_sigspec():
    def f(x, y, z=3):
        return x + y + z

    sigspec = inspect.signature(f)
    assert num_required_args(f, sigspec) == 2


def test_has_varargs_with_sigspec():
    def f(*args):
        return args

    sigspec = inspect.signature(f)
    assert has_varargs(f, sigspec) is True


def test_has_keywords_with_sigspec():
    def f(x, y=0):
        return x + y

    sigspec = inspect.signature(f)
    assert has_keywords(f, sigspec) is True


def test_is_valid_args_with_sigspec():
    def add(x, y):
        return x + y

    sigspec = inspect.signature(add)
    assert is_valid_args(add, (1,), {}, sigspec) is False


def test_is_partial_args_with_sigspec():
    def add(x, y):
        return x + y

    sigspec = inspect.signature(add)
    assert is_partial_args(add, (1,), {}, sigspec) is True


def test_is_arity_with_sigspec():
    def f(x):
        return x

    sigspec = inspect.signature(f)
    assert is_arity(1, f, sigspec) is True


def test_is_arity_builtin_non_matching():
    assert is_arity(2, sum) is False
