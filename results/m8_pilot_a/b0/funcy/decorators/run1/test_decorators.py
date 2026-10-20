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
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sample(a, b=2, *args, c, d=4, **kwargs):
    return (a, b, args, c, d, kwargs)


def plain(x, y):
    return x + y


# ---------------------------------------------------------------------------
# has_single_arg
# ---------------------------------------------------------------------------

def test_has_single_arg_true():
    def f(call):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_false_multiple():
    def f(a, b):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_varargs_only():
    def f(*args):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_kwargs_only():
    def f(**kwargs):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_zero():
    def f():
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_keyword_only_single():
    def f(*, call):
        pass
    # single parameter, not var-positional / var-keyword -> True
    assert has_single_arg(f) is True


# ---------------------------------------------------------------------------
# has_1pos_and_kwonly
# ---------------------------------------------------------------------------

def test_has_1pos_and_kwonly_true():
    def f(call, methods="GET"):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_only_one_pos():
    def f(call):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_two_pos():
    def f(a, b):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_with_varargs():
    def f(call, *args):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_zero_pos():
    def f(**kwargs):
        pass
    assert has_1pos_and_kwonly(f) is False


# ---------------------------------------------------------------------------
# get_argnames
# ---------------------------------------------------------------------------

def test_get_argnames_simple():
    def f(a, b, c):
        pass
    assert get_argnames(f) == ('a', 'b', 'c')


def test_get_argnames_with_kwonly():
    assert get_argnames(sample) == ('a', 'b', 'args', 'c', 'd', 'kwargs')


def test_get_argnames_method():
    class A:
        def m(self, x, y):
            pass
    assert get_argnames(A().m) == ('self', 'x', 'y')


def test_get_argnames_uses_original():
    def f(a, b):
        pass

    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    wrapper.__original__ = f
    assert get_argnames(wrapper) == ('a', 'b')


def test_get_argnames_unwraps():
    def f(a, b):
        pass

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    assert get_argnames(wrapper) == ('a', 'b')


# ---------------------------------------------------------------------------
# arggetter
# ---------------------------------------------------------------------------

def test_arggetter_positional():
    get = arggetter(plain)
    assert get('x', (1, 2), {}) == 1
    assert get('y', (1, 2), {}) == 2


def test_arggetter_default():
    def f(a, b=7):
        pass
    get = arggetter(f)
    assert get('b', (1,), {}) == 7


def test_arggetter_kwonly_default():
    def f(a, *, c=9):
        pass
    get = arggetter(f)
    assert get('c', (1,), {}) == 9


def test_arggetter_kwonly_from_kwargs():
    def f(a, *, c=9):
        pass
    get = arggetter(f)
    assert get('c', (1,), {'c': 3}) == 3


def test_arggetter_varargs():
    get = arggetter(sample)
    assert get('args', (1, 2, 3, 4), {'c': 5}) == (3, 4)


def test_arggetter_varkw():
    get = arggetter(sample)
    assert get('kwargs', (1, 2), {'c': 5, 'extra': 6}) == {'extra': 6}


def test_arggetter_unknown_name():
    get = arggetter(plain)
    with pytest.raises(TypeError):
        get('nope', (1, 2), {})


def test_arggetter_missing_required():
    get = arggetter(plain)
    with pytest.raises(TypeError):
        get('y', (1,), {})


def test_arggetter_cache():
    get1 = arggetter(plain)
    get2 = arggetter(plain)
    assert get1 is get2


def test_arggetter_method_skips_self():
    class A:
        def m(self, x, y=2):
            pass
    inst = A()
    get = arggetter(inst.m)
    assert get('x', (10,), {}) == 10
    assert get('y', (10,), {}) == 2


def test_arggetter_positional_only():
    def f(a, b, /, c=3):
        pass
    get = arggetter(f)
    assert get('a', (1, 2), {}) == 1
    assert get('b', (1, 2), {}) == 2
    assert get('c', (1, 2), {}) == 3


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

def test_call_basic():
    call = Call(plain, (1, 2), {})
    assert call() == 3
    assert call._func is plain
    assert call._args == (1, 2)
    assert call._kwargs == {}


def test_call_extra_args_append():
    def f(*args):
        return args
    call = Call(f, (1,), {})
    assert call(2, 3) == (1, 2, 3)


def test_call_extra_kwargs_merge():
    def f(a, b=0):
        return (a, b)
    call = Call(f, (1,), {})
    assert call(b=5) == (1, 5)
    assert call(b=5).__class__ is tuple


def test_call_attribute_lookup():
    def f(x, y=2):
        return x + y
    call = Call(f, (1,), {})
    assert call.x == 1
    assert call.y == 2


def test_call_attribute_missing():
    call = Call(plain, (1, 2), {})
    with pytest.raises(AttributeError):
        call.nonexistent


def test_call_str():
    call = Call(plain, (1,), {'y': 2})
    assert str(call) == "plain(1, y=2)"
    assert repr(call) == "<Call plain(1, y=2)>"


def test_call_str_fallback():
    obj = object()
    call = Call(obj, (), {})
    assert str(call) == "%s()" % str(obj)


# ---------------------------------------------------------------------------
# make_decorator
# ---------------------------------------------------------------------------

def test_make_decorator_plain():
    def deco(call):
        return call() * 2

    d = make_decorator(deco)
    assert d._func is deco
    assert d._args == ()
    assert d._kwargs == {}

    @d
    def f(x):
        return x + 1

    assert f(3) == 8
    assert f.__name__ == 'f'


def test_make_decorator_with_args():
    def deco(call, mult):
        return call() * mult

    d = make_decorator(deco, (3,))
    assert d._args == (3,)

    @d
    def f(x):
        return x

    assert f(2) == 6


def test_make_decorator_with_kwargs():
    def deco(call, mult=1):
        return call() * mult

    d = make_decorator(deco, (), {'mult': 4})

    @d
    def f(x):
        return x

    assert f(2) == 8


# ---------------------------------------------------------------------------
# decorator
# ---------------------------------------------------------------------------

def test_decorator_single_arg():
    @decorator
    def double(call):
        return call() * 2

    @double
    def f(x):
        return x + 1

    assert f(5) == 12
    assert f.__name__ == 'f'


def test_decorator_single_arg_wrapped_meta():
    @decorator
    def deco(call):
        return call()

    assert hasattr(deco, '__original__')
    assert deco.__original__.__name__ == 'deco'


def test_decorator_with_args_positional():
    @decorator
    def multiply(call, factor, add=0):
        return call() * factor + add

    @multiply(2, add=10)
    def f(x):
        return x

    assert f(3) == 16


def test_decorator_with_args_used_bare():
    @decorator
    def multiply(call, factor=1):
        return call() * factor

    @multiply
    def f(x):
        return x

    assert f(3) == 3


def test_decorator_with_args_kwonly():
    @decorator
    def deco(call, factor=1):
        return call() * factor

    @deco(factor=5)
    def f(x):
        return x
    assert f(2) == 10


def test_decorator_with_args_regular_fab():
    @decorator
    def deco(call, a, b=2):
        return call() + a + b

    @deco(1, b=10)
    def f(x):
        return x
    assert f(5) == 16


def test_decorator_preserves_wrapper_call_meta():
    @decorator
    def deco(call):
        return call()

    @deco
    def f(x, y=2):
        return x + y

    assert f.__name__ == 'f'
    assert f.__original__ is f.__original__


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_wraps_copies_metadata():
    def f(x):
        """docstring"""
        return x

    def w(*args, **kwargs):
        return f(*args, **kwargs)

    wrapped = wraps(f)(w)
    assert wrapped.__name__ == 'f'
    assert wrapped.__doc__ == 'docstring'
    assert wrapped.__original__ is f


def test_wraps_copies_original():
    def f(x):
        return x

    def w(*args, **kwargs):
        return f(*args, **kwargs)

    wraps(f)(w)
    assert w.__original__ is f


def test_update_wrapper_sets_original():
    def f(a):
        """doc"""
        return a

    def w(*args, **kwargs):
        return f(*args, **kwargs)

    result = update_wrapper(w, f)
    assert result is w
    assert w.__original__ is f
    assert w.__name__ == 'f'
    assert w.__doc__ == 'doc'


def test_update_wrapper_uses_existing_original():
    def real(a):
        return a

    def f(*args, **kwargs):
        return real(*args, **kwargs)
    f.__original__ = real

    def w():
        pass

    update_wrapper(w, f)
    assert w.__original__ is real


def test_wraps_passes_assigned_updated():
    def f(a):
        """doc"""
        return a

    def w():
        pass
    w.custom = 'x'

    wraps(f, assigned=('__name__',), updated=())(w)
    assert w.__name__ == 'f'
    assert w.__original__ is f
