import functools
import inspect
import sys

import pytest

from funcy.decorators import (
    Call,
    decorator,
    has_single_arg,
    has_1pos_and_kwonly,
    get_argnames,
    arggetter,
    make_decorator,
    update_wrapper,
    wraps,
)


# --------------------------- decorator ---------------------------

def test_decorator_single_arg_simple():
    @decorator
    def mydeco(call):
        return call() * 2

    @mydeco
    def add(a, b):
        return a + b

    assert add(2, 3) == 10
    assert add.__name__ == "add"


def test_decorator_single_arg_uses_call_attrs():
    seen = {}

    @decorator
    def mydeco(call):
        seen["args"] = call._args
        seen["kwargs"] = dict(call._kwargs)
        seen["str"] = str(call)
        seen["repr"] = repr(call)
        seen["value"] = call.a + call.b
        return call()

    @mydeco
    def add(a, b):
        return a + b

    assert add(1, b=5) == 6
    assert seen["args"] == (1,)
    assert seen["kwargs"] == {"b": 5}
    assert seen["value"] == 6
    assert "add(" in seen["str"]
    assert seen["repr"].startswith("<Call ")


def test_decorator_with_single_arg_custom_args():
    @decorator
    def mydeco(call, x):
        return call() + x

    @mydeco(10)
    def f():
        return 1

    assert f() == 11


def test_decorator_with_args_uses_factory():
    @decorator
    def mydeco(call, mult=1, add=0):
        return call() * mult + add

    @mydeco(mult=2, add=1)
    def f(x):
        return x

    assert f(3) == 7


def test_decorator_direct_usage_with_func():
    @decorator
    def mydeco(call, mult=1):
        return call() * mult

    def f(x):
        return x + 1

    decorated = mydeco(f, mult=3)
    assert decorated(2) == 9


# --------------------------- make_decorator ---------------------------

def test_make_decorator_basic():
    def deco(call):
        return call() + 1

    d = make_decorator(deco)

    @d
    def f(x):
        return x

    assert f(5) == 6


def test_make_decorator_with_dargs_dkwargs():
    def deco(call, a, b=0):
        return call() + a + b

    d = make_decorator(deco, (3,), {"b": 4})

    @d
    def f(x):
        return x

    assert f(1) == 8


def test_make_decorator_introspection_attrs():
    def deco(call):
        return call()

    d = make_decorator(deco, (1,), {"k": 2})
    assert d._func is deco
    assert d._args == (1,)
    assert d._kwargs == {"k": 2}


# --------------------------- Call ---------------------------

def test_call_basic():
    def f(a, b=2):
        return a + b

    c = Call(f, (1,), {})
    assert c() == 3


def test_call_extra_args():
    def f(a, b):
        return a + b

    c = Call(f, (1,), {})
    assert c(5) == 6


def test_call_extra_kwargs():
    def f(a, b=0):
        return a + b

    c = Call(f, (1,), {})
    assert c(b=10) == 11


def test_call_combines_and_overrides():
    def f(a, b):
        return (a, b)

    c = Call(f, (1,), {"b": 2})
    assert c(3, b=4) == (1, 3)


def test_call_attributes():
    def f(a, b):
        return a

    c = Call(f, (1,), {"b": 2})
    assert c._func is f
    assert c._args == (1,)
    assert c._kwargs == {"b": 2}


def test_call_getattr_arg():
    def f(a, b=5):
        return a + b

    c = Call(f, (1,), {})
    assert c.a == 1
    assert c.b == 5


def test_call_getattr_missing():
    def f(a):
        return a

    c = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        c.nonexistent


def test_call_getattr_missing_required_arg():
    def f(a, b):
        return a

    c = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        c.b


def test_call_str_and_repr():
    def f(a, b=2):
        return a

    c = Call(f, (1,), {"b": 3})
    s = str(c)
    assert "f(" in s
    assert "1" in s
    assert "b=3" in s
    assert repr(c) == "<Call %s>" % s


# --------------------------- has_single_arg ---------------------------

def test_has_single_arg_true():
    def f(call):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_false_two():
    def f(a, b):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_varargs():
    def f(*args):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_kwargs():
    def f(**kwargs):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_zero():
    def f():
        pass
    assert has_single_arg(f) is False


# --------------------------- has_1pos_and_kwonly ---------------------------

def test_has_1pos_and_kwonly_true_single_pos():
    def f(a, b=1):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_kwonly():
    def f(a, *, b=1):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_varargs_false():
    def f(a, *args):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_zero_false():
    def f(**kwargs):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_two_false():
    def f(a, b):
        pass
    assert has_1pos_and_kwonly(f) is False


# --------------------------- get_argnames ---------------------------

def test_get_argnames_simple():
    def f(a, b, c=1):
        pass
    assert get_argnames(f) == ("a", "b", "c")


def test_get_argnames_varargs():
    def f(a, *args):
        pass
    assert get_argnames(f) == ("a", "args")


def test_get_argnames_kwargs():
    def f(a, **kwargs):
        pass
    assert get_argnames(f) == ("a", "kwargs")


def test_get_argnames_both():
    def f(a, *args, **kwargs):
        pass
    assert get_argnames(f) == ("a", "args", "kwargs")


def test_get_argnames_kwonly():
    def f(a, *, b, c=1):
        pass
    assert get_argnames(f) == ("a", "b", "c")


def test_get_argnames_unwraps():
    def f(a, b):
        pass

    @functools.wraps(f)
    def g(*args, **kwargs):
        return f(*args, **kwargs)

    assert get_argnames(g) == ("a", "b")


# --------------------------- arggetter ---------------------------

def test_arggetter_positional():
    def f(a, b):
        pass
    get = arggetter(f)
    assert get("a", (1, 2), {}) == 1
    assert get("b", (1, 2), {}) == 2


def test_arggetter_kwargs():
    def f(a, b):
        pass
    get = arggetter(f)
    assert get("a", (), {"a": 5}) == 5


def test_arggetter_default_positional():
    def f(a, b=10):
        pass
    get = arggetter(f)
    assert get("b", (1,), {}) == 10


def test_arggetter_default_kwonly():
    def f(a, *, b=10):
        pass
    get = arggetter(f)
    assert get("b", (1,), {}) == 10


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


def test_arggetter_varkw_omits_named():
    def f(a, b=1, **kwargs):
        pass
    get = arggetter(f)
    assert get("kwargs", (1,), {"b": 2, "x": 3}) == {"x": 3}


def test_arggetter_missing_name():
    def f(a):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError):
        get("z", (1,), {})


def test_arggetter_missing_required():
    def f(a, b):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError):
        get("b", (1,), {})


def test_arggetter_positional_only():
    def f(a, /, b):
        pass
    get = arggetter(f)
    assert get("a", (1, 2), {}) == 1
    with pytest.raises(TypeError):
        get("a", (), {"a": 5})


def test_arggetter_method_skips_self():
    class C:
        def m(self, a, b=2):
            pass

    get = arggetter(C.m)
    assert get("a", (None, 1), {}) == 1


def test_arggetter_caches():
    def f(a):
        pass
    assert arggetter(f) is arggetter(f)


# --------------------------- update_wrapper ---------------------------

def test_update_wrapper_sets_metadata():
    def wrapped(a, b):
        "docstring"
        return a + b

    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    result = update_wrapper(wrapper, wrapped)
    assert result is wrapper
    assert wrapper.__name__ == "wrapped"
    assert wrapper.__doc__ == "docstring"
    assert wrapper.__wrapped__ is wrapped
    assert wrapper.__original__ is wrapped


def test_update_wrapper_unwraps_original():
    def base(a):
        pass

    @functools.wraps(base)
    def wrapped(*args, **kwargs):
        return base(*args, **kwargs)

    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    update_wrapper(wrapper, wrapped)
    assert wrapper.__original__ is base


# --------------------------- wraps ---------------------------

def test_wraps_decorates():
    def f(a):
        "f doc"
        return a

    @wraps(f)
    def g(*args, **kwargs):
        return f(*args, **kwargs)

    assert g.__name__ == "f"
    assert g.__doc__ == "f doc"
    assert g.__original__ is f
    assert g(3) == 3


def test_wraps_preserves_original_chain():
    def base(a):
        "base doc"
        return a

    @wraps(base)
    def middle(*args, **kwargs):
        return base(*args, **kwargs)

    @wraps(middle)
    def outer(*args, **kwargs):
        return middle(*args, **kwargs)

    assert outer.__original__ is base
    assert outer.__name__ == "base"
