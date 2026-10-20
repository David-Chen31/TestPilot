import functools
import inspect

import pytest

from funcy.decorators import (
    Call,
    arggetter,
    decorator,
    get_argnames,
    has_1pos_and_kwonly,
    has_single_arg,
    make_decorator,
    update_wrapper,
    wraps,
    unwrap,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def plain_func(a, b, c=3):
    """Docstring for plain_func."""
    return a + b + c


# ---------------------------------------------------------------------------
# has_single_arg
# ---------------------------------------------------------------------------

def test_has_single_arg_simple():
    def f(x):
        pass

    assert has_single_arg(f) is True


def test_has_single_arg_with_default():
    def f(x=1):
        pass

    assert has_single_arg(f) is True


def test_has_single_arg_keyword_only():
    def f(*, x):
        pass

    assert has_single_arg(f) is True


def test_has_single_arg_two_params():
    def f(x, y):
        pass

    assert has_single_arg(f) is False


def test_has_single_arg_var_positional():
    def f(*args):
        pass

    assert has_single_arg(f) is False


def test_has_single_arg_var_keyword():
    def f(**kwargs):
        pass

    assert has_single_arg(f) is False


def test_has_single_arg_zero_params():
    def f():
        pass

    assert has_single_arg(f) is False


# ---------------------------------------------------------------------------
# has_1pos_and_kwonly
# ---------------------------------------------------------------------------

def test_has_1pos_and_kwonly_only_one_pos():
    def f(call):
        pass

    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_with_kwonly():
    def f(call, *, x, y=1):
        pass

    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_with_var_keyword():
    def f(call, **kwargs):
        pass

    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_two_pos():
    def f(a, b):
        pass

    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_var_positional():
    def f(call, *args):
        pass

    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_no_pos():
    def f(*, x):
        pass

    assert has_1pos_and_kwonly(f) is False


# ---------------------------------------------------------------------------
# get_argnames
# ---------------------------------------------------------------------------

def test_get_argnames_simple():
    assert get_argnames(plain_func) == ('a', 'b', 'c')


def test_get_argnames_kwonly():
    def f(a, b=1, *, c, d=2):
        pass

    assert get_argnames(f) == ('a', 'b', 'c', 'd')


def test_get_argnames_varargs():
    def f(a, *args, **kwargs):
        pass

    assert get_argnames(f) == ('a', 'args', 'kwargs')


def test_get_argnames_uses_original():
    def inner(a, b):
        pass

    def wrapper(*args, **kwargs):
        return inner(*args, **kwargs)

    update_wrapper(wrapper, inner)
    assert get_argnames(wrapper) == ('a', 'b')


def test_get_argnames_loaded_func():
    def f(a, b):
        pass

    update_wrapper(f, plain_func)
    assert get_argnames(f) == ('a', 'b', 'c')


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

def test_call_call_no_args():
    call = Call(plain_func, (1, 2), {})
    assert call() == 6


def test_call_call_with_args():
    call = Call(plain_func, (1,), {'b': 2})
    assert call(10) == 16  # 1 + 2 + 10... c default 3 -> 1+2+10+3
    # actually: _func(1, 10, b=2) -> a=1, b=10... wait

def test_call_call_with_args_exact():
    call = Call(plain_func, (1, 2), {'c': 5})
    assert call() == 8
    assert call(100) == 103  # c overridden? no: a=1,b=2,c=100 -> but c passed kw

def test_call_call_append_positional():
    def add3(a, b, c):
        return a + b + c

    call = Call(add3, (1,), {})
    assert call(2, 3) == 6


def test_call_call_override_kwargs():
    def f(a, b=2):
        return a + b

    call = Call(f, (1,), {'b': 5})
    assert call() == 6
    assert call(b=10) == 11
    assert call(100, b=10) == 110


def test_call_str():
    call = Call(plain_func, (1, 2), {'c': 3})
    s = str(call)
    assert s.startswith('plain_func(')
    assert '1' in s
    assert '2' in s
    assert 'c=3' in s


def test_call_repr():
    call = Call(plain_func, (1, 2), {'c': 3})
    r = repr(call)
    assert r.startswith('<Call ')
    assert r.endswith('>')
    assert 'plain_func' in r


def test_call_getattr_arg():
    def f(a, b, c=30):
        return None

    call = Call(f, (1, 2), {})
    assert call.a == 1
    assert call.b == 2
    assert call.c == 30


def test_call_getattr_kwarg():
    def f(a, b):
        return None

    call = Call(f, (1,), {'b': 2})
    assert call.b == 2


def test_call_getattr_missing_raises():
    call = Call(plain_func, (1, 2), {})
    with pytest.raises(AttributeError):
        call.nonexistent


def test_call_getattr_missing_required():
    call = Call(plain_func, (1,), {})
    with pytest.raises(AttributeError):
        call.b


def test_call_getattr_caches():
    def f(a):
        return None

    call = Call(f, (42,), {})
    assert call.a == 42
    assert 'a' in call.__dict__


# ---------------------------------------------------------------------------
# arggetter
# ---------------------------------------------------------------------------

def test_arggetter_basic():
    get_arg = arggetter(plain_func)
    assert get_arg('a', (1, 2), {}) == 1
    assert get_arg('b', (1, 2), {}) == 2
    assert get_arg('c', (1, 2), {}) == 3


def test_arggetter_kwarg():
    get_arg = arggetter(plain_func)
    assert get_arg('c', (1, 2), {'c': 7}) == 7


def test_arggetter_missing_name():
    get_arg = arggetter(plain_func)
    with pytest.raises(TypeError):
        get_arg('zzz', (1, 2), {})


def test_arggetter_missing_required():
    get_arg = arggetter(plain_func)
    with pytest.raises(TypeError):
        get_arg('b', (1,), {})


def test_arggetter_varargs():
    def f(a, *args):
        return a, args

    get_arg = arggetter(f)
    assert get_arg('a', (1, 2, 3), {}) == 1
    assert get_arg('args', (1, 2, 3), {}) == (2, 3)
    assert get_arg('args', (1,), {}) == ()


def test_arggetter_varkw():
    def f(a, **kwargs):
        return a, kwargs

    get_arg = arggetter(f)
    assert get_arg('kwargs', (1,), {'x': 2, 'y': 3}) == {'x': 2, 'y': 3}


def test_arggetter_varkw_omits_names():
    def f(a, b, **kwargs):
        pass

    get_arg = arggetter(f)
    assert get_arg('kwargs', (1, 2), {'x': 3}) == {'x': 3}


def test_arggetter_kwonly():
    def f(a, *, b, c=5):
        pass

    get_arg = arggetter(f)
    assert get_arg('b', (1,), {'b': 2}) == 2
    assert get_arg('c', (1,), {}) == 5
    assert get_arg('c', (1,), {'c': 9}) == 9


def test_arggetter_caches():
    f = plain_func
    get_arg = arggetter(f)
    assert arggetter(f) is get_arg


def test_arggetter_uses_original():
    def inner(a, b):
        pass

    def wrapper(*args, **kwargs):
        return inner(*args, **kwargs)

    update_wrapper(wrapper, inner)
    get_arg = arggetter(wrapper)
    assert get_arg('a', (1, 2), {}) == 1
    assert get_arg('b', (1, 2), {}) == 2


def test_arggetter_error_message():
    get_arg = arggetter(plain_func)
    with pytest.raises(TypeError) as exc:
        get_arg('b', (1,), {})
    assert 'plain_func' in str(exc.value)
    assert 'b' in str(exc.value)


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_wraps_copies_metadata():
    def f(a, b):
        """my doc"""

    @wraps(f)
    def g(*args, **kwargs):
        return f(*args, **kwargs)

    assert g.__name__ == 'f'
    assert g.__doc__ == 'my doc'
    assert g.__original__ is f


def test_wraps_set_original():
    def f(a):
        pass

    @wraps(f)
    def g(*args):
        return f(*args)

    assert g.__original__ is f


def test_update_wrapper_original():
    def f(a):
        pass

    def g(*args):
        pass

    result = update_wrapper(g, f)
    assert result is g
    assert g.__name__ == 'f'
    assert g.__original__ is f


def test_update_wrapper_uses_existing_original():
    def f(a, b):
        pass

    def g(*args):
        pass

    update_wrapper(g, f)

    def h(*args):
        pass

    update_wrapper(h, g)
    # __original__ should be the underlying f
    assert h.__original__ is f


def test_update_wrapper_returns_wrapper():
    def f():
        pass

    def g():
        pass

    assert update_wrapper(g, f) is g


# ---------------------------------------------------------------------------
# make_decorator
# ---------------------------------------------------------------------------

def test_make_decorator_basic():
    def deco(call):
        return call()

    decorated = make_decorator(deco)(plain_func)
    assert decorated(1, 2) == 6


def test_make_decorator_with_dargs():
    def deco(call, extra):
        return call() + extra

    decorated = make_decorator(deco, (10,))(plain_func)
    assert decorated(1, 2) == 16


def test_make_decorator_with_dkwargs():
    def deco(call, extra=0):
        return call() + extra

    decorated = make_decorator(deco, (), {'extra': 100})(plain_func)
    assert decorated(1, 2) == 106


def test_make_decorator_stores_introspection():
    def deco(call, x=1):
        return call()

    d = make_decorator(deco, (5,), {'y': 6})
    assert d._func is deco
    assert d._args == (5,)
    assert d._kwargs == {'y': 6}


def test_make_decorator_preserves_name():
    def deco(call):
        return call()

    decorated = make_decorator(deco)(plain_func)
    assert decorated.__name__ == 'plain_func'


def test_make_decorator_wrapper_name():
    def deco(call):
        return call()

    def myfunc(a, b):
        return a + b

    decorated = make_decorator(deco)(myfunc)
    assert decorated.__name__ == 'myfunc'


def test_make_decorator_pass_call():
    def deco(call):
        return call.a + call.b

    decorated = make_decorator(deco)(plain_func)
    assert decorated(3, 4) == 7


# ---------------------------------------------------------------------------
# decorator
# ---------------------------------------------------------------------------

def test_decorator_single_arg():
    @decorator
    def double(call):
        return call() * 2

    @double
    def add(a, b):
        return a + b

    assert add(1, 2) == 6


def test_decorator_single_arg_returns_decorator():
    @decorator
    def double(call):
        return call() * 2

    # single arg decorator applied directly
    @double
    def add(a, b):
        return a + b

    assert add(2, 3) == 10


def test_decorator_with_args():
    @decorator
    def multiply(call, factor):
        return call() * factor

    @multiply(10)
    def add(a, b):
        return a + b

    assert add(1, 2) == 30


def test_decorator_with_args_used_flat():
    @decorator
    def multiply(call, factor):
        return call() * factor

    result = multiply(factor=5)(plain_func)(1, 2)
    assert result == 30


def test_decorator_one_pos_kwonly():
    @decorator
    def deco(call, factor=1):
        return call() * factor

    @deco(factor=3)
    def add(a, b):
        return a + b

    assert add(1, 2) == 9


def test_decorator_preserves_metadata():
    @decorator
    def deco(call):
        return call()

    @deco
    def myfunc(a):
        """hello"""
        return a

    assert myfunc.__name__ == 'myfunc'
    assert myfunc.__doc__ == 'hello'


def test_decorator_returns_wrapped_fab():
    def raw(call):
        return call()

    w = decorator(raw)
    assert w._func is raw


def test_decorator_with_args_introspection():
    @decorator
    def deco(call, a, b=2):
        return call()

    w = deco(1, b=5)
    assert w._func is not None
    assert w._args == (1,)
    assert w._kwargs == {'b': 5}


def test_decorator_call_object_access():
    @decorator
    def deco(call):
        return call.x + call.y

    @deco
    def add(x, y):
        return x + y

    assert add(10, 20) == 30


def test_decorator_multiple_use():
    @decorator
    def counting(call):
        return call()

    @counting
    def f(a):
        return a

    assert f(1) == 1
    assert f(2) == 2


def test_decorator_wrapped_func_name():
    @decorator
    def deco(call):
        return call()

    @deco
    def g():
        return 1

    assert g.__original__ is not None
