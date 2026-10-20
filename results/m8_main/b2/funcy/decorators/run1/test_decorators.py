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
# decorator
# ---------------------------------------------------------------------------

def test_decorator_single_arg_basic():
    @decorator
    def deco(call):
        call.called = True
        return call()

    @deco
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert add(10, 20) == 30


def test_decorator_single_arg_uses_call_attrs():
    @decorator
    def deco(call):
        call.x *= 2
        assert call.x == 10
        return call()

    @deco
    def double(x):
        return x * 10

    assert double(5) == 50


def test_decorator_single_arg_preserves_metadata():
    @decorator
    def deco(call):
        return call()

    @deco
    def named(a, b):
        """my doc"""
        return a + b

    assert named.__name__ == "named"
    assert named.__doc__ == "my doc"
    assert named(1, 2) == 3


def test_decorator_kwonly_args_produce_factory():
    @decorator
    def deco(call, *, mult=1):
        return call() * mult

    times3 = deco(mult=3)

    @times3
    def f(x):
        return x

    assert f(4) == 12

    @deco
    def g(x):
        return x + 1

    assert g(5) == 6


def test_decorator_kwonly_factory_passes_kwargs():
    seen = {}

    @decorator
    def deco(call, *, mult=1, add=0):
        seen["mult"] = mult
        seen["add"] = add
        return call() * mult + add

    @deco(mult=2, add=5)
    def f(x):
        return x

    assert f(10) == 25
    assert seen == {"mult": 2, "add": 5}


def test_decorator_kwonly_callable_directly():
    @decorator
    def deco(call, *, mult=1):
        return call() * mult

    def base(x):
        return x

    wrapped = deco(_func=base, mult=4)
    assert wrapped(3) == 12


def test_decorator_multiple_args():
    @decorator
    def deco(call, a, b):
        return call(a, b)

    @deco(1, 2)
    def add2(x, y, a, b):
        return (x, y, a, b)

    assert add2(100, 200) == (100, 200, 1, 2)


def test_decorator_multiple_args_dargs_passed():
    @decorator
    def deco(call, a, b, c=0):
        return (a, b, c, call())

    @deco(1, 2, c=3)
    def base():
        return "x"

    assert base() == (1, 2, 3, "x")


# ---------------------------------------------------------------------------
# make_decorator
# ---------------------------------------------------------------------------

def test_make_decorator_basic():
    def deco(call):
        return call() + 1

    wrapped = make_decorator(deco)

    def base(x):
        return x * 2

    f = wrapped(base)
    assert f(5) == 11


def test_make_decorator_with_args():
    def deco(call, n):
        return call() + n

    wrapped = make_decorator(deco, (10,))

    def base():
        return 1

    f = wrapped(base)
    assert f() == 11


def test_make_decorator_with_kwargs():
    def deco(call, n=0):
        return call() + n

    wrapped = make_decorator(deco, (), {"n": 7})

    def base():
        return 2

    f = wrapped(base)
    assert f() == 9


def test_make_decorator_metadata_attrs():
    def deco(call):
        return call()

    wrapped = make_decorator(deco)
    assert wrapped._func is deco
    assert wrapped._args == ()
    assert wrapped._kwargs == {}

    wrapped2 = make_decorator(deco, (1, 2), {"a": 3})
    assert wrapped2._func is deco
    assert wrapped2._args == (1, 2)
    assert wrapped2._kwargs == {"a": 3}


def test_make_decorator_dkwargs_actually_used():
    def deco(call, **kw):
        return (call(), kw)

    wrapped = make_decorator(deco, (), {"x": 1, "y": 2})

    @wrapped
    def base():
        return "r"

    assert base() == ("r", {"x": 1, "y": 2})


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

def test_call_basic():
    def func(a, b):
        return a + b

    call = Call(func, (1, 2), {})
    assert call() == 3


def test_call_extra_args():
    def func(a, b):
        return (a, b)

    call = Call(func, (1,), {})
    assert call(2) == (1, 2)


def test_call_extra_kwargs():
    def func(a, b):
        return (a, b)

    call = Call(func, (), {"a": 1})
    assert call(b=2) == (1, 2)


def test_call_extra_kwargs_override():
    def func(a, b):
        return (a, b)

    call = Call(func, (), {"a": 1, "b": 2})
    assert call(b=3) == (1, 3)


def test_call_no_extra_args_uses_saved_kwargs():
    seen = {}

    def func(a, b=0):
        seen["a"] = a
        seen["b"] = b

    call = Call(func, (5,), {"b": 9})
    call()
    assert seen == {"a": 5, "b": 9}


def test_call_getattr_positional():
    def func(a, b):
        return a + b

    call = Call(func, (1, 2), {})
    assert call.a == 1
    assert call.b == 2


def test_call_getattr_kwarg():
    def func(a, b):
        return a + b

    call = Call(func, (1,), {"b": 5})
    assert call.a == 1
    assert call.b == 5


def test_call_getattr_default():
    def func(a, b=10):
        return a + b

    call = Call(func, (1,), {})
    assert call.b == 10


def test_call_getattr_missing_raises_attribute_error():
    def func(a, b):
        return a + b

    call = Call(func, (1,), {})
    with pytest.raises(AttributeError):
        call.b


def test_call_getattr_unknown_name():
    def func(a):
        return a

    call = Call(func, (1,), {})
    with pytest.raises(AttributeError):
        call.nonexistent


def test_call_getattr_error_message():
    def somefunc(a):
        return a

    call = Call(somefunc, (1,), {})
    with pytest.raises(AttributeError) as exc:
        call.missing
    assert "somefunc() doesn't have argument named missing" in str(exc.value)


def test_call_str():
    def func(a, b):
        return a + b

    call = Call(func, (1,), {"b": 2})
    s = str(call)
    assert "func" in s
    assert "1" in s
    assert "b=2" in s


def test_call_str_exact():
    def func(a, b):
        return a + b

    call = Call(func, (1,), {"b": 2})
    assert str(call) == "%s(1, b=2)" % func.__qualname__


def test_call_str_no_args():
    def func():
        return 1

    call = Call(func, (), {})
    assert str(call) == "%s()" % func.__qualname__


def test_call_str_multiple_kwargs():
    def func():
        return 1

    call = Call(func, (1, 2), {"x": 3, "y": 4})
    assert str(call) == "%s(1, 2, x=3, y=4)" % func.__qualname__


def test_call_repr():
    def func(a):
        return a

    call = Call(func, (1,), {})
    r = repr(call)
    assert r.startswith("<Call ")
    assert r.endswith(">")


def test_call_str_qualname():
    def func(a):
        return a

    call = Call(func, (5,), {})
    assert str(call) == "%s(5)" % func.__qualname__


def test_call_str_functools_partial_no_qualname():
    class Callable:
        def __call__(self, a):
            return a

    obj = Callable()
    call = Call(obj, (1,), {})
    assert str(call) == "%s(1)" % str(obj)


def test_call_str_functools_partial():
    import functools

    def base(a, b):
        return a + b

    partial = functools.partial(base, 1)
    call = Call(partial, (2,), {"b": 3})
    assert str(call) == "%s(2, b=3)" % str(partial)


# ---------------------------------------------------------------------------
# has_single_arg
# ---------------------------------------------------------------------------

def test_has_single_arg_true():
    def f(x):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_true_kwonly_single():
    def f(*, x):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_true_with_default():
    def f(x=1):
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


def test_has_single_arg_false_kwargs_only():
    def f(*args, **kwargs):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_zero():
    def f():
        pass
    assert has_single_arg(f) is False


# ---------------------------------------------------------------------------
# has_1pos_and_kwonly
# ---------------------------------------------------------------------------

def test_has_1pos_and_kwonly_true():
    def f(x, *, a=1):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_true_no_kwonly():
    def f(x):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_true_positional_only():
    def f(x, /, *, a=1):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_false_two_pos():
    def f(x, y, *, a=1):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_false_varargs():
    def f(x, *args, a=1):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_false_zero_pos():
    def f(*, a=1):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_false_no_pos_no_kw():
    def f():
        pass
    assert has_1pos_and_kwonly(f) is False


# ---------------------------------------------------------------------------
# get_argnames
# ---------------------------------------------------------------------------

def test_get_argnames_basic():
    def f(a, b, c):
        pass
    assert get_argnames(f) == ("a", "b", "c")


def test_get_argnames_defaults():
    def f(a, b=1, c=2):
        pass
    assert get_argnames(f) == ("a", "b", "c")


def test_get_argnames_kwonly():
    def f(a, *, b, c=1):
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


def test_get_argnames_varargs_and_varkw():
    def f(a, *args, **kwargs):
        pass
    assert get_argnames(f) == ("a", "args", "kwargs")


def test_get_argnames_kwonly_and_varargs_no_varkw():
    def f(a, *args, b):
        pass
    assert get_argnames(f) == ("a", "b", "args")


def test_get_argnames_no_varargs_kwonly_only():
    def f(a, *, b):
        pass
    assert get_argnames(f) == ("a", "b")


def test_get_argnames_all():
    def f(a, b=1, *args, c, **kwargs):
        pass
    assert get_argnames(f) == ("a", "b", "c", "args", "kwargs")


def test_get_argnames_uses_original_attr():
    def original(a, b, c):
        pass

    def wrapper(*args, **kwargs):
        pass

    wrapper.__original__ = original
    assert get_argnames(wrapper) == ("a", "b", "c")


def test_get_argnames_through_wraps():
    def original(a, b, c):
        pass

    @wraps(original)
    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    assert get_argnames(wrapper) == ("a", "b", "c")


# ---------------------------------------------------------------------------
# arggetter
# ---------------------------------------------------------------------------

def test_arggetter_positional():
    def f(a, b):
        pass
    getter = arggetter(f)
    assert getter("a", (1, 2), {}) == 1
    assert getter("b", (1, 2), {}) == 2


def test_arggetter_keyword():
    def f(a, b):
        pass
    getter = arggetter(f)
    assert getter("a", (), {"a": 5}) == 5
    assert getter("b", (), {"b": 7}) == 7


def test_arggetter_positional_priority_over_kwarg():
    def f(a):
        pass
    getter = arggetter(f)
    assert getter("a", (1,), {"a": 2}) == 1


def test_arggetter_defaults():
    def f(a, b=10):
        pass
    getter = arggetter(f)
    assert getter("b", (1,), {}) == 10


def test_arggetter_kwonly_default():
    def f(a, *, b=20):
        pass
    getter = arggetter(f)
    assert getter("b", (1,), {}) == 20


def test_arggetter_kwonly_value():
    def f(a, *, b=20):
        pass
    getter = arggetter(f)
    assert getter("b", (1,), {"b": 99}) == 99


def test_arggetter_varargs():
    def f(a, *args):
        pass
    getter = arggetter(f)
    assert getter("args", (1, 2, 3), {}) == (2, 3)


def test_arggetter_varargs_empty():
    def f(a, *args):
        pass
    getter = arggetter(f)
    assert getter("args", (1,), {}) == ()


def test_arggetter_varkw():
    def f(a, **kwargs):
        pass
    getter = arggetter(f)
    assert getter("kwargs", (1,), {"x": 2, "y": 3}) == {"x": 2, "y": 3}


def test_arggetter_varkw_excludes_named():
    def f(a, b=0, **kwargs):
        pass
    getter = arggetter(f)
    assert getter("kwargs", (1,), {"b": 2, "x": 3}) == {"x": 3}


def test_arggetter_varkw_excludes_kwonly():
    def f(a, *, c=0, **kwargs):
        pass
    getter = arggetter(f)
    assert getter("kwargs", (1,), {"c": 2, "x": 3}) == {"x": 3}


def test_arggetter_missing_required_raises():
    def f(a, b):
        pass
    getter = arggetter(f)
    with pytest.raises(TypeError):
        getter("b", (1,), {})


def test_arggetter_missing_required_message():
    def somefn(a, b):
        pass
    getter = arggetter(somefn)
    with pytest.raises(TypeError) as exc:
        getter("b", (1,), {})
    assert "somefn() missing required argument: 'b'" in str(exc.value)


def test_arggetter_unknown_name_raises():
    def f(a):
        pass
    getter = arggetter(f)
    with pytest.raises(TypeError):
        getter("zzz", (1,), {})


def test_arggetter_unknown_name_message():
    def somefn(a):
        pass
    getter = arggetter(somefn)
    with pytest.raises(TypeError) as exc:
        getter("zzz", (1,), {})
    assert "somefn() doesn't have argument named zzz" in str(exc.value)


def test_arggetter_cache():
    def f(a):
        pass
    g1 = arggetter(f)
    g2 = arggetter(f)
    assert g1 is g2


def test_arggetter_uses_original_attr():
    def original(a, b):
        pass

    def wrapper(*args, **kwargs):
        pass

    wrapper.__original__ = original
    getter = arggetter(wrapper)
    assert getter("a", (1, 2), {}) == 1
    assert getter("b", (1, 2), {}) == 2


def test_arggetter_method_skips_self():
    class C:
        def m(self, a, b=2):
            pass

    c = C()
    getter = arggetter(c.m)
    assert getter("a", (1,), {}) == 1
    assert getter("b", (1,), {}) == 2


def test_arggetter_bound_method_no_self_name():
    class C:
        def m(self, a):
            pass

    c = C()
    getter = arggetter(c.m)
    with pytest.raises(TypeError):
        getter("self", (1,), {})


def test_arggetter_positional_only():
    def f(a, b, /, c):
        pass
    getter = arggetter(f)
    with pytest.raises(TypeError):
        getter("a", (), {"a": 5})
    assert getter("c", (), {"c": 7}) == 7


def test_arggetter_kwonly_through_kwargs():
    def f(a, *, b):
        pass
    getter = arggetter(f)
    assert getter("b", (1,), {"b": 42}) == 42


def test_arggetter_through_wraps_uses_original_code():
    def original(a, b):
        pass

    @wraps(original)
    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    getter = arggetter(wrapper)
    assert getter("a", (1, 2), {}) == 1
    assert getter("b", (1, 2), {}) == 2


# ---------------------------------------------------------------------------
# Additional arggetter tests to kill surviving mutants
# ---------------------------------------------------------------------------

def _fresh_arggetter(func):
    # arggetter caches by func; each func object defined inline is fresh.
    return arggetter(func)


def test_arggetter_positional_only_via_positional_index():
    # Kills posonly default mutations on line 121 / 124: with a positional-only
    # arg passed positionally, index lookup should still return it.
    def f(a, /, b):
        pass

    getter = _fresh_arggetter(f)
    assert getter("a", (1, 2), {}) == 1
    assert getter("b", (1, 2), {}) == 2
    assert getter("b", (), {"b": 9}) == 9


def test_arggetter_positional_only_not_in_kwnames():
    # The posonly name must not be recognized as a keyword name.
    def f(a, /, b):
        pass

    getter = _fresh_arggetter(f)
    # 'a' is positional-only: kwargs 'a' must not be picked up.
    with pytest.raises(TypeError):
        getter("a", (), {"a": 5})
    # 'b' is a normal positional-or-keyword arg.
    assert getter("b", (), {"b": 9}) == 9


def test_arggetter_method_with_positional_only():
    # Method: skip self, handle positional-only count adjustment correctly.
    class C:
        def m(self, a, /, b=2):
            pass

    c = C()
    getter = _fresh_arggetter(c.m)
    assert getter("a", (1,), {}) == 1
    assert getter("b", (1,), {}) == 2
    # 'a' is positional-only, so kwargs 'a' must not be returned.
    with pytest.raises(TypeError):
        getter("a", (), {"a": 5})
    # 'b' is positional-or-keyword.
    assert getter("b", (), {"b": 9}) == 9


def test_arggetter_method_with_positional_only_varkw():
    class C:
        def m(self, a, /, *args, **kwargs):
            pass

    c = C()
    getter = _fresh_arggetter(c.m)
    assert getter("a", (1,), {}) == 1
    assert getter("args", (1, 2, 3), {}) == (2, 3)
    assert getter("kwargs", (1,), {"x": 5, "y": 6}) == {"x": 5, "y": 6}


def test_arggetter_plain_function_is_method_check_false():
    # Plain functions: inspect.ismethod(func) is False and
    # inspect.ismethod(original) is False. No skip.
    def f(a, b=2):
        pass

    getter = _fresh_arggetter(f)
    assert getter("a", (1,), {}) == 1
    assert getter("b", (1,), {}) == 2


def test_arggetter_bound_method_no_skip_from_original_alone():
    # A bound method passed with __original__ not being a method? Ensure the
    # method-skip works even if only the bound func is a method (original
    # from unwrap is the plain function, so inspect.ismethod(original) is
    # False). This kills the "and" mutation, which would skip the self-strip.
    class C:
        def m(self, a, b=2):
            pass

    c = C()
    getter = _fresh_arggetter(c.m)
    # Without the self-strip, 'a' positionally at index 0 would be 'self'.
    assert getter("a", (10,), {}) == 10
    assert getter("b", (10,), {"b": 30}) == 30


def test_arggetter_method_varargs_and_varkw():
    class C:
        def m(self, a, *args, **kwargs):
            pass

    c = C()
    getter = _fresh_arggetter(c.m)
    assert getter("a", (1,), {}) == 1
    # After self skip, posnames = [a]; args beyond index 0.
    assert getter("args", (1, 2, 3), {}) == (2, 3)
    assert getter("kwargs", (1,), {"x": 5, "y": 6}) == {"x": 5, "y": 6}


def test_arggetter_varargs_with_defaults_and_kwonly():
    def f(a, b=1, *args, c=2, **kwargs):
        pass

    getter = _fresh_arggetter(f)
    assert getter("a", (10,), {}) == 10
    assert getter("b", (10,), {}) == 1
    assert getter("args", (10, 20, 30), {}) == (30,)
    assert getter("c", (10,), {}) == 2
    assert getter("kwargs", (10,), {"x": 1}) == {"x": 1}
    # varargs and varkw should be empty when no extras
    assert getter("args", (10, 20), {}) == ()
    assert getter("kwargs", (10,), {}) == {}


def test_arggetter_varargs_and_kwonlys_together():
    # Kills line 133 n-offset mutants: varargs and varkw live after all
    # pos and kwonly names. If n is wrong, they'd map to wrong names.
    def f(a, *args, b=1, **kwargs):
        pass

    getter = _fresh_arggetter(f)
    assert getter("args", (1, 2, 3), {}) == (2, 3)
    assert getter("b", (1, 2, 3), {}) == 1
    assert getter("b", (1, 2, 3), {"b": 9}) == 9
    assert getter("kwargs", (1, 2), {"b": 9, "x": 5}) == {"x": 5}


def test_arggetter_none_varposname_no_false_match():
    # varposname/varkwname default None; name lookup for an arg named "None"
    # is impossible, but ensure a real arg named like the sentinels doesn't
    # trigger weirdness. More directly: no varargs means "args" isn't valid.
    def f(a, b):
        pass

    getter = _fresh_arggetter(f)
    with pytest.raises(TypeError):
        getter("args", (1, 2), {})


def test_arggetter_defaults_from_end_of_posnames():
    # Kills mutants around defaults zip boundary on line 141.
    # __defaults__ apply to the LAST len(__defaults__) positional names.
    def f(a, b=10, c=20):
        pass

    getter = _fresh_arggetter(f)
    assert getter("a", (1,), {}) == 1
    assert getter("b", (1,), {}) == 10
    assert getter("c", (1,), {}) == 20
    # Provide both, ensure only remaining picked default.
    assert getter("b", (1,), {}) == 10
    assert getter("c", (1,), {}) == 20


def test_arggetter_defaults_partial_supplied():
    def f(a, b=10, c=20):
        pass

    getter = _fresh_arggetter(f)
    assert getter("b", (1, 2), {}) == 2
    assert getter("c", (1, 2), {}) == 20


def test_arggetter_defaults_only_b():
    # Single default: applies to last positional name (b).
    def f(a, b=99):
        pass

    getter = _fresh_arggetter(f)
    assert getter("a", (1,), {}) == 1
    assert getter("b", (1,), {}) == 99


def test_arggetter_kwonly_after_defaults():
    # kwdefaults don't overlap with pos defaults in identity.
    def f(a, b=5, *, c=7, d):
        pass

    getter = _fresh_arggetter(f)
    assert getter("b", (1,), {}) == 5
    assert getter("c", (1,), {}) == 7
    with pytest.raises(TypeError):
        getter("d", (1,), {})


def test_arggetter_positional_index_for_kwonly_not_confused():
    # Ensure kwonly names don't appear in indexes (positional index map).
    def f(a, *, b=1):
        pass

    getter = _fresh_arggetter(f)
    # b has no positional index; keyword route should apply.
    assert getter("b", (1, 2, 3), {}) == 1
    assert getter("b", (1, 2, 3), {"b": 9}) == 9


def test_arggetter_positional_only_multi():
    # Multiple positional-only args: posonly_n must be exact.
    def f(a, b, /, c, d=4):
        pass

    getter = _fresh_arggetter(f)
    assert getter("a", (1, 2, 3), {}) == 1
    assert getter("b", (1, 2, 3), {}) == 2
    assert getter("c", (1, 2, 3), {}) == 3
    assert getter("d", (1, 2, 3), {}) == 4
    # 'a' and 'b' are positional-only -> not keyword-retrievable.
    with pytest.raises(TypeError):
        getter("a", (), {"a": 9})
    with pytest.raises(TypeError):
        getter("b", (), {"b": 9})
    assert getter("c", (), {"c": 9}) == 9


def test_arggetter_varkw_returns_omitted_plain_dict():
    def f(a, **kwargs):
        pass

    getter = _fresh_arggetter(f)
    result = getter("kwargs", (1,), {"x": 2, "y": 3})
    assert result == {"x": 2, "y": 3}
    assert isinstance(result, dict)


def test_arggetter_varkw_with_posonly_and_kwonly():
    def f(a, /, b=1, *, c=2, **kwargs):
        pass

    getter = _fresh_arggetter(f)
    result = getter("kwargs", (10, 20), {"c": 5, "x": 1, "y": 2})
    assert result == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# update_wrapper / wraps
# ---------------------------------------------------------------------------

def test_update_wrapper_copies_metadata():
    def wrapped(a, b):
        """docstring here"""
        return a + b

    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    result = update_wrapper(wrapper, wrapped)
    assert result is wrapper
    assert wrapper.__name__ == "wrapped"
    assert wrapper.__doc__ == "docstring here"


def test_update_wrapper_sets_original():
    def wrapped(a, b):
        return a + b

    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    update_wrapper(wrapper, wrapped)
    assert wrapper.__original__ is wrapped


def test_update_wrapper_preserves_original_chain():
    def original(a):
        return a

    def wrapped(*args, **kwargs):
        return original(*args, **kwargs)

    update_wrapper(wrapped, original)
    assert wrapped.__original__ is original

    def wrapper2(*args, **kwargs):
        return wrapped(*args, **kwargs)

    update_wrapper(wrapper2, wrapped)
    assert wrapper2.__original__ is original


def test_update_wrapper_uses_unwrap_for_wrapped():
    def original(a):
        return a

    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    update_wrapper(wrapper, original)
    assert wrapper.__original__ is original
    assert wrapper.__wrapped__ is original


def test_wraps_decorator_usage():
    def wrapped(a):
        """the doc"""
        return a

    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    assert wrapper.__name__ == "wrapped"
    assert wrapper.__doc__ == "the doc"
    assert wrapper.__original__ is wrapped
    assert wrapper(5) == 5


def test_wraps_with_custom_updated():
    def wrapped(a):
        return a

    @wraps(wrapped, updated=())
    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    assert wrapper.__name__ == "wrapped"
    assert wrapper.__original__ is wrapped
    assert wrapper.__wrapped__ is wrapped


def test_wraps_with_custom_assigned():
    def wrapped(a):
        """doc"""
        return a

    @wraps(wrapped, assigned=())
    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    assert wrapper.__name__ == "wrapper"
    assert wrapper.__original__ is wrapped


def test_wraps_returns_partial_setup():
    def wrapped(a):
        return a

    factory = wraps(wrapped)
    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    ret = factory(wrapper)
    assert ret is wrapper
    assert wrapper.__original__ is wrapped


def test_update_wrapper_preserves_original_via_wrapped():
    def deepest(a):
        return a

    def mid(*args, **kwargs):
        return deepest(*args, **kwargs)

    update_wrapper(mid, deepest)
    assert mid.__original__ is deepest

    def outer(*args, **kwargs):
        return mid(*args, **kwargs)

    update_wrapper(outer, mid)
    assert outer.__original__ is deepest
