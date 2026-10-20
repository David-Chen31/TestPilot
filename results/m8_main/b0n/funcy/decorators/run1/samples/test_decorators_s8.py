import functools
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
    ContextDecorator,
    contextmanager,
    unwrap,
)


# ---------------------------------------------------------------------------
# decorator
# ---------------------------------------------------------------------------

def test_decorator_single_arg_form():
    @decorator
    def double(call):
        return call() * 2

    @double
    def foo(x):
        return x + 1

    assert foo(3) == 8
    # introspection is preserved via wraps
    assert foo.__name__ == "foo"


def test_decorator_single_arg_receives_call_object():
    seen = {}

    @decorator
    def record(call):
        seen["func"] = call._func
        seen["args"] = call._args
        seen["kwargs"] = call._kwargs
        return call()

    @record
    def foo(a, b=2):
        return a + b

    assert foo(10, b=5) == 15
    assert seen["func"] is foo.__wrapped__
    assert seen["args"] == (10,)
    assert seen["kwargs"] == {"b": 5}


def test_decorator_with_kwonly_args():
    @decorator
    def add(call, n=1):
        return call() + n

    @add(n=10)
    def foo():
        return 1

    assert foo() == 11

    @add
    def bar():
        return 5

    # when used directly (no call) default applies
    assert bar() == 6


def test_decorator_with_kwonly_args_direct_use():
    @decorator
    def add(call, n=1):
        return call() + n

    @add
    def foo():
        return 1

    assert foo() == 2


def test_decorator_with_args_form():
    @decorator
    def mul(call, times):
        return call() * times

    @mul(3)
    def foo(x):
        return x + 1

    assert foo(2) == 9


def test_decorator_with_args_and_kwargs():
    @decorator
    def multi(call, *adds, **muls):
        result = call()
        for a in adds:
            result += a
        for m in muls.values():
            result *= m
        return result

    @multi(1, 2, x=3)
    def foo():
        return 1

    # (1 + 1 + 2) * 3 = 12
    assert foo() == 12


def test_decorator_preserves_metadata_for_fab():
    @decorator
    def deco(call, times, extra=None):
        """my deco doc"""
        return call() * times

    assert deco.__name__ == "deco"
    assert deco.__doc__ == "my deco doc"


def test_decorator_has_original():
    @decorator
    def deco(call):
        return call()

    assert hasattr(deco, "__original__")


# ---------------------------------------------------------------------------
# make_decorator
# ---------------------------------------------------------------------------

def test_make_decorator_basic():
    def deco(call):
        return call() + 1

    d = make_decorator(deco)

    @d
    def foo(x):
        return x

    assert foo(5) == 6
    assert d._func is deco
    assert d._args == ()
    assert d._kwargs == {}


def test_make_decorator_with_args():
    def deco(call, extra):
        return call() + extra

    d = make_decorator(deco, (100,), {"extra": 100})
    # note dargs and dkwargs both passed; extra in dkwargs will conflict if used; use just dargs first

    d2 = make_decorator(deco, (100,))

    @d2
    def foo(x):
        return x

    assert foo(1) == 101
    assert d2._args == (100,)
    assert d2._kwargs == {}


def test_make_decorator_with_kwargs():
    def deco(call, extra=0):
        return call() + extra

    d = make_decorator(deco, (), {"extra": 7})

    @d
    def foo(x):
        return x

    assert foo(1) == 8
    assert d._kwargs == {"extra": 7}


def test_make_decorator_preserves_name():
    def deco(call):
        return call()

    d = make_decorator(deco)

    @d
    def foo():
        return 1

    assert foo.__name__ == "foo"
    assert d.__name__ == "deco"


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

def test_call_call_no_args():
    def f(a, b):
        return a + b

    c = Call(f, (1,), {"b": 2})
    assert c() == 3


def test_call_call_with_args():
    def f(a, b, c=0):
        return a + b + c

    c = Call(f, (1,), {"b": 2})
    assert c(5) == 8  # a=1, b=2, c=5 -> wait: positional appended: f(1, 5, b=2)

def test_call_call_with_kwargs():
    def f(a, b):
        return a - b

    c = Call(f, (10,), {"b": 3})
    assert c() == 7
    # call with override kwarg
    assert c(b=1) == 9


def test_call_str():
    def my_func(a, b):
        pass

    c = Call(my_func, (1, 2), {"b": 3})
    s = str(c)
    assert s.startswith("my_func(")
    assert "1" in s and "2" in s and "b=3" in s


def test_call_str_no_qualname():
    class Callable:
        def __call__(self, a):
            pass

    c = Call(Callable(), (1,), {})
    s = str(c)
    assert s.startswith("(")
    assert s.endswith(")")
    assert "1" in s


def test_call_repr():
    def f(a):
        pass

    c = Call(f, (1,), {})
    assert repr(c) == "<Call %s>" % str(c)


def test_call_getattr_arg():
    def f(a, b):
        return a + b

    c = Call(f, (1,), {"b": 2})
    assert c.a == 1
    assert c.b == 2


def test_call_getattr_missing():
    def f(a):
        pass

    c = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        c.nonexistent


def test_call_getattr_default():
    def f(a, b=10):
        pass

    c = Call(f, (1,), {})
    assert c.b == 10


def test_call_getattr_varargs():
    def f(a, *args):
        pass

    c = Call(f, (1, 2, 3), {})
    assert c.args == (2, 3)


def test_call_getattr_varkw():
    def f(a, **kwargs):
        pass

    c = Call(f, (1,), {"x": 2, "y": 3})
    assert c.kwargs == {"x": 2, "y": 3}


def test_call_getattr_caches():
    def f(a):
        pass

    c = Call(f, (1,), {})
    val = c.a
    assert c.a == val
    # cached in __dict__
    assert "a" in c.__dict__


def test_call_getattr_missing_required():
    def f(a, b):
        pass

    c = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        c.b


# ---------------------------------------------------------------------------
# has_single_arg
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


def test_has_single_arg_false_zero():
    def f():
        pass

    assert has_single_arg(f) is False


# ---------------------------------------------------------------------------
# has_1pos_and_kwonly
# ---------------------------------------------------------------------------

def test_has_1pos_and_kwonly_true():
    def f(x, *, y):
        pass

    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_one_pos_no_kwonly():
    def f(x):
        pass

    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_two_pos():
    def f(x, y):
        pass

    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_varargs():
    def f(x, *args):
        pass

    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_only_kwonly():
    def f(*, x):
        pass

    assert has_1pos_and_kwonly(f) is False


# ---------------------------------------------------------------------------
# get_argnames
# ---------------------------------------------------------------------------

def test_get_argnames_basic():
    def f(a, b, c):
        pass

    assert get_argnames(f) == ("a", "b", "c")


def test_get_argnames_with_defaults():
    def f(a, b=1):
        pass

    assert get_argnames(f) == ("a", "b")


def test_get_argnames_kwonly():
    def f(a, *, b):
        pass

    assert get_argnames(f) == ("a", "b")


def test_get_argnames_varargs():
    def f(a, *args):
        pass

    assert get_argnames(f) == ("a", "args")


def test_get_argnames_varkw():
    def f(a, **kwargs):
        pass

    assert get_argnames(f) == ("a", "kwargs")


def test_get_argnames_full():
    def f(a, *args, b=1, **kwargs):
        pass

    assert get_argnames(f) == ("a", "args", "b", "kwargs")


def test_get_argnames_unwraps():
    def f(a, b):
        pass

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        pass

    assert get_argnames(wrapper) == ("a", "b")


def test_get_argnames_original_attr():
    def f(a, b):
        pass

    def wrapper(*args, **kwargs):
        pass

    wrapper.__original__ = f
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


def test_arggetter_kw():
    def f(a, b):
        pass

    get = arggetter(f)
    assert get("a", (), {"a": 10}) == 10
    assert get("b", (), {"b": 20}) == 20


def test_arggetter_defaults():
    def f(a, b=5):
        pass

    get = arggetter(f)
    assert get("b", (1,), {}) == 5


def test_arggetter_kwonly_default():
    def f(a, *, b=7):
        pass

    get = arggetter(f)
    assert get("b", (1,), {}) == 7


def test_arggetter_kwonly_given():
    def f(a, *, b):
        pass

    get = arggetter(f)
    assert get("b", (1,), {"b": 3}) == 3


def test_arggetter_varargs():
    def f(a, *rest):
        pass

    get = arggetter(f)
    assert get("rest", (1, 2, 3), {}) == (2, 3)


def test_arggetter_varkw():
    def f(a, **kw):
        pass

    get = arggetter(f)
    assert get("kw", (1,), {"x": 2}) == {"x": 2}


def test_arggetter_varkw_omits_named():
    def f(a, b, **kw):
        pass

    get = arggetter(f)
    assert get("kw", (1,), {"b": 2, "x": 3}) == {"x": 3}


def test_arggetter_missing_name():
    def f(a):
        pass

    get = arggetter(f)
    with pytest.raises(TypeError):
        get("nope", (1,), {})


def test_arggetter_missing_required():
    def f(a, b):
        pass

    get = arggetter(f)
    with pytest.raises(TypeError):
        get("b", (1,), {})


def test_arggetter_cache():
    def f(a):
        pass

    get1 = arggetter(f)
    get2 = arggetter(f)
    assert get1 is get2


def test_arggetter_positional_only():
    ns = {}
    exec("def f(a, /, b):\n    pass", ns)
    f = ns["f"]
    get = arggetter(f)
    assert get("a", (1, 2), {}) == 1
    # positional-only cannot be passed by keyword
    with pytest.raises(TypeError):
        get("a", (), {"a": 1})


def test_arggetter_method():
    class C:
        def method(self, a):
            pass

    c = C()
    get = arggetter(c.method)
    # self is skipped
    assert get("a", (5,), {}) == 5


# ---------------------------------------------------------------------------
# update_wrapper / wraps
# ---------------------------------------------------------------------------

def test_update_wrapper_copies_metadata():
    def original(a):
        """orig doc"""

    def wrapper(*args):
        pass

    result = update_wrapper(wrapper, original)
    assert result is wrapper
    assert wrapper.__name__ == "original"
    assert wrapper.__doc__ == "orig doc"
    assert wrapper.__wrapped__ is original


def test_update_wrapper_sets_original():
    def original(a):
        pass

    def wrapper(*args):
        pass

    update_wrapper(wrapper, original)
    assert wrapper.__original__ is original


def test_update_wrapper_chain_original():
    def original(a):
        pass

    def w1(*args):
        pass

    update_wrapper(w1, original)

    def w2(*args):
        pass

    update_wrapper(w2, w1)
    # __original__ should point to the deepest original
    assert w2.__original__ is original


def test_wraps_returns_partial_usable():
    def original(a, b=2):
        """orig doc"""

    wrapper = wraps(original)(lambda *args: None)
    assert wrapper.__name__ == "original"
    assert wrapper.__doc__ == "orig doc"


def test_wraps_preserves_original():
    def original():
        pass

    @wraps(original)
    def wrapper():
        pass

    assert wrapper.__original__ is original


# ---------------------------------------------------------------------------
# reexports
# ---------------------------------------------------------------------------

def test_reexports_exist():
    assert ContextDecorator is not None
    assert contextmanager is not None
    assert unwrap is not None


def test_unwrap_reexport_works():
    def f():
        pass

    @functools.wraps(f)
    def wrapper():
        pass

    assert unwrap(wrapper) is f
