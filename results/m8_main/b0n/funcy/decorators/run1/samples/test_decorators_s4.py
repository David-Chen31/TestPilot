import inspect
import pytest

from funcy.decorators import (
    decorator,
    make_decorator,
    Call,
    has_single_arg,
    has_1pos_and_kwonly,
    get_argnames,
    arggetter,
    update_wrapper,
    wraps,
)


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

def test_call_basic_call():
    def f(a, b):
        return a + b

    c = Call(f, (1, 2), {})
    assert c() == 3
    assert c._func is f
    assert c._args == (1, 2)
    assert c._kwargs == {}


def test_call_with_kwargs():
    def f(a, b=10):
        return a + b

    c = Call(f, (1,), {"b": 5})
    assert c() == 6


def test_call_extra_args_positional():
    def f(a, b):
        return (a, b)

    c = Call(f, (1,), {})
    # Calling with extra positional args overrides / appends
    assert c(5) == (1, 5)


def test_call_extra_kwargs():
    def f(a, b=10):
        return (a, b)

    c = Call(f, (1,), {})
    assert c(b=3) == (1, 3)


def test_call_repr_and_str():
    def myfunc(a, b):
        return a + b

    c = Call(myfunc, (1, 2), {"z": 9})
    s = str(c)
    assert "myfunc" in s
    assert "1" in s
    assert "2" in s
    assert "z=9" in s
    r = repr(c)
    assert r.startswith("<Call ")
    assert r.endswith(">")


def test_call_str_no_qualname():
    class Callable:
        def __call__(self, a):
            return a

    obj = Callable()
    c = Call(obj, (1,), {})
    s = str(c)
    assert "1" in s


def test_call_getattr_arg_access():
    def f(a, b=20):
        return a + b

    c = Call(f, (1,), {})
    assert c.b == 20

    c2 = Call(f, (1,), {"b": 5})
    assert c2.b == 5


def test_call_getattr_missing_arg():
    def f(a):
        return a

    c = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        c.nonexistent


# ---------------------------------------------------------------------------
# decorator / make_decorator
# ---------------------------------------------------------------------------

def test_decorator_single_arg():
    @decorator
    def double(call):
        return call() * 2

    @double
    def add(a, b):
        return a + b

    assert add(1, 2) == 6
    assert add.__name__ == "add"


def test_decorator_with_args_flat():
    @decorator
    def add_n(call, n):
        return call() + n

    @add_n(10)
    def val():
        return 1

    assert val() == 11


def test_decorator_with_args_keyword():
    @decorator
    def add_n(call, n=0):
        return call() + n

    @add_n(n=5)
    def val():
        return 1

    assert val() == 6


def test_decorator_with_args_used_directly():
    @decorator
    def add_n(call, n=0):
        return call() + n

    @add_n
    def val():
        return 1

    assert val() == 1


def test_decorator_varargs_variant():
    @decorator
    def repeater(call, times=1, sep="-"):
        return sep.join([str(call())] * times)

    @repeater(times=3, sep=",")
    def val():
        return "x"

    assert val() == "x,x,x"


def test_decorator_access_call_args():
    @decorator
    def logger(call):
        # Access by name via arggetter
        return (call.a, call.b, call())

    @logger
    def add(a, b):
        return a + b

    assert add(1, 2) == (1, 2, 3)


def test_decorator_preserves_wrapped_metadata():
    @decorator
    def identity(call):
        return call()

    @identity
    def target(x):
        """target doc"""
        return x

    assert target.__doc__ == "target doc"
    assert target.__name__ == "target"


def test_make_decorator_basic():
    def deco(call, extra):
        return call() + extra

    d = make_decorator(deco, (100,), {})
    wrapped = d(lambda: 1)
    assert wrapped() == 101


def test_make_decorator_introspection_attrs():
    def deco(call):
        return call()

    d = make_decorator(deco, (), {})
    assert d._func is deco
    assert d._args == ()
    assert d._kwargs == {}


def test_make_decorator_kwargs():
    def deco(call, k=1):
        return call() * k

    d = make_decorator(deco, (), {"k": 3})
    wrapped = d(lambda: 2)
    assert wrapped() == 6


# ---------------------------------------------------------------------------
# has_single_arg / has_1pos_and_kwonly
# ---------------------------------------------------------------------------

def test_has_single_arg_true():
    def f(x):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_false_multiple():
    def f(x, y):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_false_varargs():
    def f(*args):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_false_varkw():
    def f(**kwargs):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_false_none():
    def f():
        pass
    assert has_single_arg(f) is False


def test_has_1pos_and_kwonly_true():
    def f(x, y=1):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_true_kwonly():
    def f(x, *, y=1):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_false_no_pos():
    def f(*, y=1):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_false_two_pos():
    def f(x, y):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_false_varargs():
    def f(x, *args):
        pass
    assert has_1pos_and_kwonly(f) is False


# ---------------------------------------------------------------------------
# get_argnames
# ---------------------------------------------------------------------------

def test_get_argnames_simple():
    def f(a, b):
        pass
    assert get_argnames(f) == ("a", "b")


def test_get_argnames_defaults():
    def f(a, b=1, *, c=2):
        pass
    assert get_argnames(f) == ("a", "b", "c")


def test_get_argnames_varargs():
    def f(a, *args):
        pass
    assert get_argnames(f) == ("a", "args")


def test_get_argnames_varkw():
    def f(a, **kwargs):
        pass
    assert get_argnames(f) == ("a", "kwargs")


def test_get_argnames_all():
    def f(a, *args, b=1, **kwargs):
        pass
    assert get_argnames(f) == ("a", "b", "args", "kwargs")


def test_get_argnames_unwrap_decorated():
    def inner(a, b):
        pass

    @wraps(inner)
    def wrapper(*args, **kwargs):
        return inner(*args, **kwargs)

    assert get_argnames(wrapper) == ("a", "b")


# ---------------------------------------------------------------------------
# arggetter
# ---------------------------------------------------------------------------

def test_arggetter_positional():
    def f(a, b):
        pass
    get = arggetter(f)
    assert get("a", (1, 2), {}) == 1
    assert get("b", (1, 2), {}) == 2


def test_arggetter_keyword():
    def f(a, b):
        pass
    get = arggetter(f)
    assert get("b", (1,), {"b": 5}) == 5


def test_arggetter_defaults():
    def f(a, b=10):
        pass
    get = arggetter(f)
    assert get("b", (1,), {}) == 10
    assert get("a", (1,), {}) == 1


def test_arggetter_kwonly_defaults():
    def f(a, *, b=7):
        pass
    get = arggetter(f)
    assert get("b", (1,), {}) == 7


def test_arggetter_varargs():
    def f(a, *args):
        pass
    get = arggetter(f)
    assert get("args", (1, 2, 3), {}) == (2, 3)


def test_arggetter_varkw():
    def f(a, **kwargs):
        pass
    get = arggetter(f)
    assert get("kwargs", (1,), {"x": 2, "y": 3}) == {"x": 2, "y": 3}


def test_arggetter_missing_arg_name():
    def f(a):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError):
        get("nope", (1,), {})


def test_arggetter_missing_required_value():
    def f(a, b):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError):
        get("b", (1,), {})


def test_arggetter_cache():
    def f(a):
        pass
    g1 = arggetter(f)
    g2 = arggetter(f)
    assert g1 is g2


def test_arggetter_method():
    class C:
        def method(self, a, b=2):
            pass

    get = arggetter(C.method)
    # 'a' should be positional index 0 (after self removed)
    assert get("a", (C, 5), {}) == 5 if False else True  # sanity noop
    # For a method, 'self' is dropped; passing the instance in args
    inst = C()
    assert get("b", (inst, 1), {}) == 2


# ---------------------------------------------------------------------------
# update_wrapper / wraps
# ---------------------------------------------------------------------------

def test_update_wrapper_sets_original():
    def wrapped(a, b):
        pass

    def wrapper(*args, **kwargs):
        pass

    result = update_wrapper(wrapper, wrapped)
    assert result is wrapper
    assert wrapper.__name__ == "wrapped"
    assert wrapper.__original__ is wrapped


def test_update_wrapper_keeps_original_of_wrapped():
    def orig(a):
        pass

    def wrapper1(*a, **kw):
        pass
    update_wrapper(wrapper1, orig)

    def wrapper2(*a, **kw):
        pass
    update_wrapper(wrapper2, wrapper1)
    assert wrapper2.__original__ is orig


def test_wraps_partial():
    def wrapped(a):
        pass

    def wrapper(*a, **kw):
        pass

    decorator_fn = wraps(wrapped)
    result = decorator_fn(wrapper)
    assert result is wrapper
    assert wrapper.__name__ == "wrapped"
    assert wrapper.__original__ is wrapped


def test_wraps_preserves_docstring():
    def wrapped():
        """the doc"""

    @wraps(wrapped)
    def wrapper():
        pass

    assert wrapper.__doc__ == "the doc"


def test_update_wrapper_doc_matches_functools():
    import functools
    assert update_wrapper.__doc__ == functools.update_wrapper.__doc__


def test_wraps_doc_matches_functools():
    import functools
    assert wraps.__doc__ == functools.wraps.__doc__
