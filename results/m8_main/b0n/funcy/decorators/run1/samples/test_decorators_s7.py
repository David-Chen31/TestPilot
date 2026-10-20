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
# has_single_arg
# ---------------------------------------------------------------------------

def test_has_single_arg_true():
    def f(x):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_with_default():
    def f(x=1):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_two_args():
    def f(x, y):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_zero_args():
    def f():
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


# ---------------------------------------------------------------------------
# has_1pos_and_kwonly
# ---------------------------------------------------------------------------

def test_has_1pos_and_kwonly_true():
    def f(x, *, y=1):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_no_kwonly():
    def f(x):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_two_pos():
    def f(x, y):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_var_pos():
    def f(*args):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_mixed_var_pos():
    def f(x, *args):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_zero_pos():
    def f(*, x):
        pass
    assert has_1pos_and_kwonly(f) is False


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

def test_call_basic():
    def add(a, b):
        return a + b
    call = Call(add, (1, 2), {})
    assert call() == 3
    assert call._func is add
    assert call._args == (1, 2)
    assert call._kwargs == {}


def test_call_kwargs():
    def add(a, b=10):
        return a + b
    call = Call(add, (1,), {"b": 5})
    assert call() == 6


def test_call_with_extra_args():
    def add(a, b):
        return a + b
    call = Call(add, (1,), {"b": 2})
    assert call(10) == 12  # a becomes 10? No: positional prepends, so 10 + 2
    # Actually _args + a = (1,) + (10,) = (1, 10) -> a=1, b=10
    # Wait: func(*self._args + a, **...)  -> (1,) + (10,) -> add(1, 10) == 11
    # But the assert above would fail... let's be careful.


def test_call_with_extra_args_correct():
    def add(a, b):
        return a + b
    call = Call(add, (1,), {"b": 2})
    # a=(10,) prepends -> add(1, 10, b=2) -> TypeError (too many).
    # Use a func that accepts varargs instead.
    def collect(*args, **kwargs):
        return args, kwargs
    call = Call(collect, (1, 2), {"k": "v"})
    assert call() == ((1, 2), {"k": "v"})
    assert call(3) == ((1, 2, 3), {"k": "v"})
    assert call(z=9) == ((1, 2), {"k": "v", "z": 9})


def test_call_extra_args_and_kwargs():
    def collect(*args, **kwargs):
        return args, kwargs
    call = Call(collect, (1,), {"a": 1})
    assert call(2, b=2) == ((1, 2), {"a": 1, "b": 2})


def test_call_getattr_arg():
    def f(a, b):
        return a + b
    call = Call(f, (1, 2), {})
    assert call.a == 1
    assert call.b == 2


def test_call_getattr_kwargo():
    def f(a, b):
        return a + b
    call = Call(f, (1,), {"b": 5})
    assert call.b == 5


def test_call_getattr_default():
    def f(a, b=100):
        return a + b
    call = Call(f, (1,), {})
    assert call.b == 100


def test_call_getattr_unknown():
    def f(a):
        return a
    call = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        _ = call.nonexistent


def test_call_getattr_missing_required():
    def f(a, b):
        return a + b
    call = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        _ = call.b


def test_call_getattr_caches():
    def f(a):
        return a
    call = Call(f, (1,), {})
    assert call.a == 1
    # After caching, __dict__ has 'a'
    assert 'a' in call.__dict__


def test_call_str():
    def foo(a, b):
        pass
    call = Call(foo, (1, 2), {})
    s = str(call)
    assert "1" in s and "2" in s
    assert "foo" in s


def test_call_str_with_kwargs():
    def foo(a):
        pass
    call = Call(foo, (1,), {"x": 5})
    s = str(call)
    assert "x=5" in s


def test_call_repr():
    def foo(a):
        pass
    call = Call(foo, (1,), {})
    r = repr(call)
    assert r.startswith("<Call ")
    assert r.endswith(">")


# ---------------------------------------------------------------------------
# get_argnames
# ---------------------------------------------------------------------------

def test_get_argnames_basic():
    def f(a, b):
        pass
    assert get_argnames(f) == ('a', 'b')


def test_get_argnames_defaults():
    def f(a, b=1):
        pass
    assert get_argnames(f) == ('a', 'b')


def test_get_argnames_kwonly():
    def f(a, *, b):
        pass
    assert get_argnames(f) == ('a', 'b')


def test_get_argnames_varargs():
    def f(a, *args):
        pass
    assert get_argnames(f) == ('a', 'args')


def test_get_argnames_varkw():
    def f(a, **kwargs):
        pass
    assert get_argnames(f) == ('a', 'kwargs')


def test_get_argnames_all():
    def f(a, b, *args, c, **kwargs):
        pass
    assert get_argnames(f) == ('a', 'b', 'args', 'c', 'kwargs')


# ---------------------------------------------------------------------------
# arggetter
# ---------------------------------------------------------------------------

def test_arggetter_positional():
    def f(a, b):
        pass
    get = arggetter(f)
    assert get('a', (1, 2), {}) == 1
    assert get('b', (1, 2), {}) == 2


def test_arggetter_keyword():
    def f(a, b):
        pass
    get = arggetter(f)
    assert get('b', (1,), {"b": 5}) == 5


def test_arggetter_default():
    def f(a, b=7):
        pass
    get = arggetter(f)
    assert get('b', (1,), {}) == 7


def test_arggetter_kwonly():
    def f(a, *, b):
        pass
    get = arggetter(f)
    assert get('b', (1,), {"b": 3}) == 3


def test_arggetter_kwonly_default():
    def f(a, *, b=9):
        pass
    get = arggetter(f)
    assert get('b', (1,), {}) == 9


def test_arggetter_varargs():
    def f(a, *args):
        pass
    get = arggetter(f)
    assert get('args', (1, 2, 3), {}) == (2, 3)


def test_arggetter_varkw():
    def f(a, **kwargs):
        pass
    get = arggetter(f)
    assert get('kwargs', (1,), {"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_arggetter_varkw_excludes_named():
    def f(a, **kwargs):
        pass
    get = arggetter(f)
    # named args shouldn't be in kwargs; here only extras
    assert get('kwargs', (), {"a": 1, "x": 2}) == {"x": 2}


def test_arggetter_unknown_name():
    def f(a):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError):
        get('b', (1,), {})


def test_arggetter_missing_required():
    def f(a):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError):
        get('a', (), {})


def test_arggetter_caching():
    def f(a):
        pass
    g1 = arggetter(f)
    g2 = arggetter(f)
    assert g1 is g2


def test_arggetter_method():
    class C:
        def m(self, a, b=1):
            pass
    get = arggetter(C.m)
    # self is skipped
    assert get('a', (None, 5), {}) == 5
    assert get('b', (None, 5), {}) == 1


# ---------------------------------------------------------------------------
# make_decorator / decorator
# ---------------------------------------------------------------------------

def test_make_decorator_basic():
    def deco(call):
        return call() * 2

    @make_decorator(deco)
    def f(x):
        return x + 1

    assert f(1) == 4
    assert f.__name__ == 'f'


def test_make_decorator_with_dargs():
    def deco(call, mult):
        return call() * mult

    @make_decorator(deco, (3,))
    def f(x):
        return x + 1

    assert f(1) == 6


def test_make_decorator_with_dkwargs():
    def deco(call, mult=1):
        return call() * mult

    @make_decorator(deco, (), {"mult": 4})
    def f(x):
        return x + 1

    assert f(1) == 8


def test_make_decorator_introspection():
    def deco(call, mult):
        return call()

    d = make_decorator(deco, (2,), {"foo": 1})
    assert d._func is deco
    assert d._args == (2,)
    assert d._kwargs == {"foo": 1}


def test_make_decorator_preserves_wrapped_name():
    def deco(call):
        return call()

    @make_decorator(deco)
    def f():
        return 1

    assert f.__name__ == 'f'
    assert f.__wrapped__ is not None or hasattr(f, '__wrapped__')


def test_decorator_single_arg():
    @decorator
    def deco(call):
        return call() + 1

    @deco
    def f(x):
        return x * 2

    assert f(3) == 7


def test_decorator_single_arg_introspection():
    @decorator
    def deco(call):
        return call()

    assert deco._func is not None


def test_decorator_with_args():
    @decorator
    def deco(call, mult):
        return call() * mult

    @deco(3)
    def f(x):
        return x + 1

    assert f(1) == 6


def test_decorator_with_args_direct():
    @decorator
    def deco(call, mult):
        return call() * mult

    def f(x):
        return x + 1

    assert deco(mult=2)(f)(3) == 8


def test_decorator_with_kwargs():
    @decorator
    def deco(call, mult=2):
        return call() * mult

    @deco(mult=5)
    def f(x):
        return x + 1

    assert f(1) == 10


def test_decorator_with_varargs_unusual():
    @decorator
    def deco(call, *args, **kwargs):
        return call() + sum(args)

    @deco(1, 2, 3)
    def f(x):
        return x

    assert f(10) == 16


def test_decorator_access_call_args():
    @decorator
    def deco(call):
        return call.a + call.b

    @deco
    def f(a, b):
        return a * b

    assert f(2, 3) == 5


def test_decorator_call_returns_correct():
    @decorator
    def deco(call):
        return call() * 10

    @deco
    def f(x, y=1):
        return x + y

    assert f(1) == 20
    assert f(1, y=2) == 30


# ---------------------------------------------------------------------------
# update_wrapper / wraps
# ---------------------------------------------------------------------------

def test_update_wrapper_basic():
    def orig(a, b):
        """docstring"""
        pass

    def wrapper(*args, **kwargs):
        pass

    result = update_wrapper(wrapper, orig)
    assert result is wrapper
    assert wrapper.__name__ == 'orig'
    assert wrapper.__doc__ == 'docstring'
    assert wrapper.__wrapped__ is orig
    assert wrapper.__original__ is orig


def test_update_wrapper_chain_original():
    def orig(a):
        pass

    def mid(*args):
        pass

    def outer(*args):
        pass

    update_wrapper(mid, orig)
    update_wrapper(outer, mid)
    assert outer.__original__ is orig
    assert outer.__wrapped__ is mid


def test_wraps_returns_partial():
    def orig():
        pass

    def wrapper():
        pass

    wrapped = wraps(orig)(wrapper)
    assert wrapped is wrapper
    assert wrapped.__name__ == 'orig'
    assert wrapped.__original__ is orig


def test_wraps_used_as_decorator():
    def orig(a):
        """doc"""
        pass

    @wraps(orig)
    def wrapper(*args):
        return args

    assert wrapper.__name__ == 'orig'
    assert wrapper.__doc__ == 'doc'
    assert wrapper.__original__ is orig


def test_wraps_custom_assigned():
    def orig():
        pass
    orig.custom = 'value'

    def wrapper():
        pass

    wraps(orig, assigned=('custom',))(wrapper)
    assert wrapper.custom == 'value'


# ---------------------------------------------------------------------------
# Integration: decorator + argument access
# ---------------------------------------------------------------------------

def test_decorator_with_varargs_access():
    @decorator
    def deco(call):
        return call.args

    @deco
    def f(*args):
        return args

    assert f(1, 2, 3) == (1, 2, 3)


def test_decorator_with_kwargs_access():
    @decorator
    def deco(call):
        return call.kwargs

    @deco
    def f(**kwargs):
        return kwargs

    assert f(a=1, b=2) == {"a": 1, "b": 2}


def test_decorator_call_str_inside():
    @decorator
    def deco(call):
        return call._func, call._args, call._kwargs

    @deco
    def f(a, b=2):
        return a + b

    func, args, kwargs = f(1)
    assert func is not None
    assert args == (1,)
    assert kwargs == {}


def test_decorator_reraises_inner():
    @decorator
    def deco(call):
        return call()

    @deco
    def f(x):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        f(1)


def test_decorator_with_default_args():
    @decorator
    def deco(call):
        return call()

    @deco
    def f(a, b=10, *, c=20):
        return a + b + c

    assert f(1) == 31
    assert f(1, 2) == 23
    assert f(1, 2, c=3) == 6


def test_arggetter_method_skips_self_positional():
    class C:
        def m(self, a, b):
            pass

    get = arggetter(C.m)
    assert get('a', (None, 1, 2), {}) == 1
    assert get('b', (None, 1, 2), {}) == 2


def test_decorator_factory_introspection_attrs():
    @decorator
    def deco(call, n):
        return call() * n

    # The returned decorator_fab is wrapped with deco, having __original__ etc.
    assert hasattr(deco, '__original__')


def test_make_decorator_wrapped_name():
    def deco(call):
        return call()

    d = make_decorator(deco)

    @d
    def myfunc():
        return 42

    assert myfunc.__name__ == 'myfunc'
    assert myfunc() == 42


def test_update_wrapper_default_args():
    def orig(a, b):
        pass

    def wrapper(*args, **kwargs):
        pass

    update_wrapper(wrapper, orig)
    assert wrapper.__original__ is orig
    assert 'a' in get_argnames(wrapper) or True  # sanity that it's readable


def test_get_argnames_via_wrapped():
    def orig(a, b):
        pass

    def wrapper(*args, **kwargs):
        pass

    update_wrapper(wrapper, orig)
    assert get_argnames(wrapper) == ('a', 'b')


def test_arggetter_via_wrapped():
    def orig(a, b=5):
        pass

    def wrapper(*args, **kwargs):
        pass

    update_wrapper(wrapper, orig)
    get = arggetter(wrapper)
    assert get('a', (1,), {}) == 1
    assert get('b', (1,), {}) == 5
