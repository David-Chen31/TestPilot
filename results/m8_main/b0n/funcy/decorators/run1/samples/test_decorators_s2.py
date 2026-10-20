import inspect
import functools
import pytest

from funcy.decorators import (
    decorator, make_decorator, Call, has_single_arg, has_1pos_and_kwonly,
    get_argnames, arggetter, update_wrapper, wraps, unwrap,
    ContextDecorator, contextmanager,
)


# ---------------------------------------------------------------- decorator

def test_decorator_single_arg():
    @decorator
    def simple(call):
        return call() * 2

    @simple
    def f(x, y=3):
        return x + y

    assert f(1) == 8
    assert f(1, 2) == 6
    # wraps applied
    assert f.__name__ == 'f'


def test_decorator_single_arg_variadic_is_not_single_arg():
    # *args single param is not treated as single arg -> decorator_fab
    @decorator
    def d(*args, **kwargs):
        return make_decorator(d2)

    def d2(call):
        return call()

    result = d()
    assert inspect.isfunction(result)


def test_decorator_with_params_pos_and_kwonly():
    @decorator
    def mult(call, factor=2):
        return call() * factor

    @mult
    def f(x):
        return x

    assert f(3) == 6

    @mult(factor=5)
    def g(x):
        return x

    assert g(3) == 15

    # direct call with func
    @mult(3)
    def h(x):
        return x

    assert h(3) == 9


def test_decorator_with_positional_args():
    @decorator
    def add(call, extra):
        return call() + extra

    @add(10)
    def f(x):
        return x

    assert f(5) == 15


# ------------------------------------------------------------ make_decorator

def test_make_decorator():
    def deco(call, a, b=1):
        return call() + a + b

    d = make_decorator(deco, (2,), {'b': 3})

    def func(x):
        return x

    wrapped = d(func)
    assert wrapped(10) == 15
    assert d._func is deco
    assert d._args == (2,)
    assert d._kwargs == {'b': 3}


def test_make_decorator_preserves_metadata():
    def deco(call):
        return call()

    def func(x):
        "docstring"
        return x

    d = make_decorator(deco)
    wrapped = d(func)
    assert wrapped.__name__ == 'func'
    assert wrapped.__doc__ == 'docstring'


# --------------------------------------------------------------- Call

def test_call_basic():
    def f(a, b=2):
        return a + b

    c = Call(f, (1,), {})
    assert c() == 3
    assert c(10) == 13
    assert c(b=100) == 101


def test_call_str_repr():
    def myfunc(a, b=2):
        return a

    c = Call(myfunc, (1,), {'b': 5})
    assert str(c) == 'myfunc(1, b=5)'
    assert repr(c) == '<Call myfunc(1, b=5)>'


def test_call_str_no_qualname():
    class F:
        def __call__(self, *a):
            return 0

    inst = F()
    c = Call(inst, (1,), {})
    s = str(c)
    assert s.endswith('(1)')


def test_call_getattr_argument():
    def f(a, b=2, *args, c=3, **kwargs):
        return a

    c = Call(f, (1, 2, 3, 4), {'c': 10, 'd': 20})
    assert c.a == 1
    assert c.b == 2
    assert c.args == (3, 4)
    assert c.c == 10
    # kwargs capture d
    assert c.kwargs == {'d': 20}


def test_call_getattr_missing_raises_attributeerror():
    def f(a):
        return a

    c = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        c.nonexistent


def test_call_getattr_missing_required_raises_attributeerror():
    def f(a, b):
        return a

    c = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        c.b


def test_call_getattr_caches():
    def f(a):
        return a

    c = Call(f, (1,), {})
    assert c.a == 1
    # cached in __dict__
    assert 'a' in c.__dict__


# ------------------------------------------------------------- has_single_arg

def test_has_single_arg():
    assert has_single_arg(lambda x: x)
    assert has_single_arg(lambda x, y: x) is False
    assert has_single_arg(lambda *a: a) is False
    assert has_single_arg(lambda **k: k) is False

    def f(x):
        pass

    assert has_single_arg(f)


def test_has_1pos_and_kwonly():
    def f(a, *, b=1):
        pass

    assert has_1pos_and_kwonly(f)

    def g(a, b, *, c=1):
        pass

    assert has_1pos_and_kwonly(g) is False

    def h(a, *args):
        pass

    assert has_1pos_and_kwonly(h) is False

    def i(*, a=1):
        pass

    assert has_1pos_and_kwonly(i) is False


# ------------------------------------------------------------- get_argnames

def test_get_argnames():
    def f(a, b, c=1, *args, d=2, **kwargs):
        pass

    assert get_argnames(f) == ('a', 'b', 'c', 'args', 'd', 'kwargs')


def test_get_argnames_no_var():
    def f(a, b):
        pass

    assert get_argnames(f) == ('a', 'b')


def test_get_argnames_uses_original():
    def f(a):
        pass

    def wrapper(*args):
        pass

    wrapper.__original__ = f
    assert get_argnames(wrapper) == ('a',)


# --------------------------------------------------------------- arggetter

def test_arggetter_positional():
    def f(a, b=2):
        return a, b

    get = arggetter(f)
    assert get('a', (1,), {}) == 1
    assert get('b', (1,), {}) == 2
    assert get('b', (1, 5), {}) == 5


def test_arggetter_keyword():
    def f(a, b=2):
        return a

    get = arggetter(f)
    assert get('b', (), {'b': 9}) == 9


def test_arggetter_default():
    def f(a, b=7):
        return a

    get = arggetter(f)
    assert get('b', (1,), {}) == 7


def test_arggetter_varargs():
    def f(a, *args):
        return a

    get = arggetter(f)
    assert get('args', (1, 2, 3), {}) == (2, 3)


def test_arggetter_varkwargs():
    def f(a, **kwargs):
        return a

    get = arggetter(f)
    assert get('kwargs', (1,), {'b': 2, 'c': 3}) == {'b': 2, 'c': 3}


def test_arggetter_missing_name():
    def f(a):
        return a

    get = arggetter(f)
    with pytest.raises(TypeError):
        get('nope', (1,), {})


def test_arggetter_missing_required():
    def f(a, b):
        return a

    get = arggetter(f)
    with pytest.raises(TypeError):
        get('b', (1,), {})


def test_arggetter_kwonly_default():
    def f(a, *, b=5):
        return a

    get = arggetter(f)
    assert get('b', (1,), {}) == 5
    assert get('b', (1,), {'b': 11}) == 11


def test_arggetter_caching():
    def f(a):
        return a

    g1 = arggetter(f)
    g2 = arggetter(f)
    assert g1 is g2


def test_arggetter_method():
    class C:
        def m(self, a, b=2):
            return a

    get = arggetter(C.m)
    assert get('a', (1,), {}) == 1
    assert get('b', (1,), {}) == 2


# ------------------------------------------------------------ update_wrapper

def test_update_wrapper():
    def wrapped(a, b):
        "doc"
        pass

    def wrapper(*args, **kwargs):
        pass

    result = update_wrapper(wrapper, wrapped)
    assert result is wrapper
    assert wrapper.__name__ == 'wrapped'
    assert wrapper.__doc__ == 'doc'
    assert wrapper.__wrapped__ is wrapped
    assert wrapper.__original__ is wrapped


def test_update_wrapper_keeps_original():
    def orig(a):
        pass

    def mid(*args):
        pass

    mid.__original__ = orig

    def wrapper(*args):
        pass

    update_wrapper(wrapper, mid)
    assert wrapper.__original__ is orig


# --------------------------------------------------------------- wraps

def test_wraps():
    def wrapped(a):
        "d"
        pass

    @wraps(wrapped)
    def wrapper(*args):
        pass

    assert wrapper.__name__ == 'wrapped'
    assert wrapper.__doc__ == 'd'
    assert wrapper.__original__ is wrapped


# ------------------------------------------------------ reexports / misc

def test_reexports():
    assert unwrap is inspect.unwrap
    assert ContextDecorator is not None
    assert contextmanager is not None
