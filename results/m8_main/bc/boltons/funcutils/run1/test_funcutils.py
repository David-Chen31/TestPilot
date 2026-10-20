import inspect
import sys
import threading
import types
import functools
import asyncio

import pytest

from boltons.funcutils import (
    inspect_formatargspec,
    get_module_callables,
    mro_items,
    dir_dict,
    copy_function,
    partial_ordering,
    InstancePartial,
    CachedInstancePartial,
    format_invocation,
    format_exp_repr,
    format_nonexp_repr,
    wraps,
    update_wrapper,
    FunctionBuilder,
    MissingArgument,
    ExistingArgument,
    noop,
    once,
    NO_DEFAULT,
)


# ---------------------------------------------------------------------------
# inspect_formatargspec
# ---------------------------------------------------------------------------

class TestInspectFormatargspec:
    def test_basic(self):
        assert inspect_formatargspec(['a', 'b']) == '(a, b)'

    def test_empty(self):
        assert inspect_formatargspec([]) == '()'

    def test_defaults(self):
        assert inspect_formatargspec(['a', 'b'], defaults=(1,)) == '(a, b=1)'
        assert inspect_formatargspec(['a', 'b'], defaults=(1, 2)) == '(a=1, b=2)'

    def test_varargs(self):
        assert inspect_formatargspec(['a'], varargs='args') == '(a, *args)'

    def test_varkw(self):
        assert inspect_formatargspec(['a'], varkw='kwargs') == '(a, **kwargs)'

    def test_kwonly_no_varargs_adds_star(self):
        result = inspect_formatargspec([], kwonlyargs=['k'])
        assert result == '(*, k)'

    def test_kwonly_defaults(self):
        result = inspect_formatargspec([], kwonlyargs=['k'], kwonlydefaults={'k': 5})
        assert result == '(*, k=5)'

    def test_kwonly_with_varargs(self):
        result = inspect_formatargspec([], varargs='a', kwonlyargs=['k'])
        assert result == '(*a, k)'

    def test_annotations(self):
        result = inspect_formatargspec(['a'], annotations={'a': int})
        assert result == '(a: int)'

    def test_return_annotation(self):
        result = inspect_formatargspec([], annotations={'return': int})
        assert result == '() -> int'

    def test_custom_formatters(self):
        result = inspect_formatargspec(
            ['a'], defaults=(1,),
            formatvalue=lambda v: '<%r>' % v)
        assert result == '(a<1>)'

    def test_full(self):
        result = inspect_formatargspec(
            ['a', 'b'], varargs='args', varkw='kw',
            defaults=(2,), kwonlyargs=['c'], kwonlydefaults={'c': 3},
            annotations={'a': int, 'return': str})
        assert result == '(a: int, b=2, *args, c=3, **kw) -> str'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

class TestGetModuleCallables:
    def test_by_instance(self):
        mod = types.ModuleType('my_test_mod')

        def foo():
            pass

        class Bar:
            pass

        def not_mine():
            pass

        not_mine.__module__ = 'some_other_mod'

        mod.foo = foo
        foo.__module__ = 'my_test_mod'
        Bar.__module__ = 'my_test_mod'
        mod.Bar = Bar
        mod.not_mine = not_mine

        types_map, funcs_map = get_module_callables(mod)
        assert 'Bar' in types_map
        assert types_map['Bar'] is Bar
        assert 'foo' in funcs_map
        assert funcs_map['foo'] is foo
        assert 'not_mine' not in funcs_map

    def test_by_string_name(self):
        mod = types.ModuleType('str_named_mod')

        def func():
            pass

        func.__module__ = 'str_named_mod'
        mod.func = func
        sys.modules['str_named_mod'] = mod
        try:
            types_map, funcs_map = get_module_callables('str_named_mod')
            assert funcs_map == {'func': func}
            assert types_map == {}
        finally:
            del sys.modules['str_named_mod']

    def test_ignore(self):
        mod = types.ModuleType('ignored_mod')

        def foo():
            pass

        def bar():
            pass

        foo.__module__ = 'ignored_mod'
        bar.__module__ = 'ignored_mod'
        mod.foo = foo
        mod.bar = bar

        types_map, funcs_map = get_module_callables(
            mod, ignore=lambda name: name == 'foo')
        assert 'foo' not in funcs_map
        assert 'bar' in funcs_map

    def test_attr_without_module_skipped(self):
        mod = types.ModuleType('no_module_mod')
        mod.x = 5
        mod.y = object()
        types_map, funcs_map = get_module_callables(mod)
        assert 'y' not in funcs_map
        assert 'y' not in types_map


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

class TestMroItems:
    def test_basic(self):
        class A:
            x = 1

        class B(A):
            y = 2

        items = dict(mro_items(B))
        assert items['x'] == 1
        assert items['y'] == 2

    def test_includes_object(self):
        items = list(mro_items(int))
        keys = [k for k, v in items]
        assert '__init__' in keys


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

class TestDirDict:
    def test_basic(self):
        class A:
            x = 1

            def method(self):
                return 2

        a = A()
        a.y = 3
        d = dir_dict(a)
        assert d['x'] == 1
        assert d['y'] == 3
        assert d['method'] == a.method

    def test_inheritance(self):
        class A:
            ax = 10

        class B(A):
            bx = 20

        d = dir_dict(B())
        assert d['ax'] == 10
        assert d['bx'] == 20

    def test_raise_exc(self):
        class BadProp:
            @property
            def boom(self):
                raise RuntimeError('boom')

        obj = BadProp()
        d = dir_dict(obj)
        assert 'boom' not in d

        with pytest.raises(RuntimeError):
            dir_dict(obj, raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

class TestCopyFunction:
    def test_basic(self):
        def func():
            return func

        cp = copy_function(func)
        assert cp() is func
        assert cp is not func
        assert cp.__name__ == func.__name__

    def test_defaults_and_kwdefaults(self):
        def func(a, b=2, *, c=3):
            return a, b, c

        cp = copy_function(func)
        assert cp(1) == (1, 2, 3)
        assert cp.__defaults__ == (2,)
        assert cp.__kwdefaults__ == {'c': 3}

    def test_closure(self):
        def outer():
            value = 42

            def inner():
                return value

            return inner

        inner = outer()
        cp = copy_function(inner)
        assert cp() == 42

    def test_copy_dict(self):
        def func():
            pass

        func.custom = 'hello'
        cp = copy_function(func)
        assert cp.custom == 'hello'

        cp2 = copy_function(func, copy_dict=False)
        assert not hasattr(cp2, 'custom')


# ---------------------------------------------------------------------------
# partial_ordering
# ---------------------------------------------------------------------------

class TestPartialOrdering:
    def test_set_example(self):
        @partial_ordering
        class MySet(set):
            def __le__(self, other):
                return self.issubset(other)

            def __ge__(self, other):
                return self.issuperset(other)

        a = MySet([1, 2, 3])
        b = MySet([1, 2])
        c = MySet([1, 2, 4])
        assert (b < a) is True
        assert (b > a) is False
        assert (b < c) is True
        assert (a < c) is False
        assert (c > a) is False
        assert (b == MySet([1, 2])) is True
        assert (b == a) is False

    def test_does_not_override_existing(self):
        class MySet(set):
            def __le__(self, other):
                return True

            def __ge__(self, other):
                return True

            def __lt__(self, other):
                return 'custom'

        partial_ordering(MySet)
        assert MySet([1]).__lt__(MySet([2])) == 'custom'


# ---------------------------------------------------------------------------
# InstancePartial / CachedInstancePartial
# ---------------------------------------------------------------------------
# Both bind via MethodType(self, obj), so on call, the instance is passed
# as the *first* argument to the partial's __call__, which prepends the
# stored args. So the resulting call is func(*stored_args, instance).

class TestInstancePartial:
    def test_get_binds_instance(self):
        def greet(*args):
            return args

        class A:
            hello = InstancePartial(greet, 'hello')

        a = A()
        # Stored args come first, then the bound instance.
        assert a.hello() == ('hello', a)

    def test_as_method_binds_instance_first(self):
        def greet(*args):
            return args

        class A:
            hello = InstancePartial(greet, 'hello')

        a = A()
        result = a.hello()
        assert result == ('hello', a)

    def test_partialmethod_property(self):
        def base(a, b):
            return a + b

        p = InstancePartial(base, 1)
        assert isinstance(p._partialmethod, functools.partialmethod)
        assert isinstance(p.__partialmethod__, functools.partialmethod)

    def test_is_partial_subclass(self):
        def base(a):
            return a

        p = InstancePartial(base, 1)
        assert isinstance(p, functools.partial)
        assert p() == 1


class TestCachedInstancePartial:
    def test_as_method_binds_instance_first(self):
        def greet(*args):
            return args

        class A:
            hello = CachedInstancePartial(greet, 'hello')

        a = A()
        assert a.hello() == ('hello', a)
        # cached: second access returns the same bound method
        assert a.hello is a.hello
        assert 'hello' in a.__dict__

    def test_instance_access_sets_metadata(self):
        class A:
            def greet(self, x):
                """docs here"""
                return x

            g = CachedInstancePartial(greet, 1)

        a = A()
        a.g
        # Metadata is set on the partial during instance access
        assert A.__dict__['g'].__doc__ == A.greet.__doc__
        assert A.__dict__['g'].__module__ == A.greet.__module__

    def test_set_name(self):
        class A:
            def f(self, x):
                return x

            g = CachedInstancePartial(f, 1)

        # __set_name__ assigns __name__ at class creation time.
        assert A.__dict__['g'].__name__ == 'g'

    def test_class_access_raises(self):
        class A:
            def f(self, x):
                return x

            g = CachedInstancePartial(f, 1)

        with pytest.raises(TypeError):
            A.g

    def test_instance_access_returns_methodtype(self):
        class A:
            def f(self, x):
                return x

            g = CachedInstancePartial(f, 1)

        a = A()
        assert isinstance(a.g, types.MethodType)

    def test_partialmethod_property(self):
        def base(a, b):
            return a + b

        p = CachedInstancePartial(base, 1)
        assert isinstance(p._partialmethod, functools.partialmethod)
        assert isinstance(p.__partialmethod__, functools.partialmethod)

    def test_alias_partial(self):
        from boltons.funcutils import partial
        assert partial is CachedInstancePartial


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

class TestFormatInvocation:
    def test_basic(self):
        assert format_invocation('func', args=(1, 2), kwargs={'c': 3}) == 'func(1, 2, c=3)'

    def test_args_only(self):
        assert format_invocation('a_func', args=(1,)) == 'a_func(1)'

    def test_kwargs_list(self):
        assert format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]) == 'kw_func(a=1, b=2)'

    def test_empty(self):
        assert format_invocation('f') == 'f()'

    def test_dict_sorted(self):
        result = format_invocation('f', kwargs={'b': 2, 'a': 1})
        assert result == 'f(a=1, b=2)'

    def test_custom_repr(self):
        result = format_invocation('f', kwargs={'a': 'x'}, repr=lambda v: '"%s"' % v)
        assert result == 'f(a="x")'

    def test_unexpected_kw(self):
        with pytest.raises(TypeError):
            format_invocation('f', bad=True)


# ---------------------------------------------------------------------------
# format_exp_repr / format_nonexp_repr
# ---------------------------------------------------------------------------

class Flag:
    def __init__(self, length, width, depth=None):
        self.length = length
        self.width = width
        self.depth = depth


class TestFormatExpRepr:
    def test_basic(self):
        flag = Flag(5, 10)
        assert format_exp_repr(flag, ['length', 'width'], [], ['depth']) == 'Flag(5, 10)'

    def test_keyword_expansion(self):
        flag = Flag(5, 15, 2)
        assert format_exp_repr(flag, ['length'], ['width', 'depth']) == 'Flag(5, width=15, depth=2)'

    def test_dedup(self):
        flag = Flag(5, 15)
        result = format_exp_repr(flag, ['length'], ['width'], ['width'])
        assert result == 'Flag(5, width=15)'

    def test_custom_opt_key(self):
        flag = Flag(5, 15, 0)
        result = format_exp_repr(flag, ['length'], [], ['depth'],
                                 opt_key=lambda v: False)
        assert result == 'Flag(5, depth=0)'


class TestFormatNonexpRepr:
    def test_basic(self):
        flag = Flag(5, 10)
        assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == '<Flag length=5 width=10>'

    def test_id_fallback(self):
        flag = Flag(5, 10)
        result = format_nonexp_repr(flag)
        assert result.startswith('<Flag id=')
        assert result.endswith('>')

    def test_opt_excluded(self):
        flag = Flag(5, 10)
        result = format_nonexp_repr(flag, ['length'], ['depth'])
        assert result == '<Flag length=5>'


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

class TestUpdateWrapper:
    def test_basic(self):
        def print_return(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return update_wrapper(wrapper, func)

        @print_return
        def example():
            """docstring"""
            return 'value'

        assert example() == 'value'
        assert example.__name__ == 'example'
        assert example.__doc__ == 'docstring'

    def test_signature_preserved(self):
        def func(a, b, c=3):
            return a + b + c

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapped = update_wrapper(wrapper, func)
        sig = inspect.signature(wrapped)
        assert list(sig.parameters) == ['a', 'b', 'c']

    def test_injected_forwards_through(self):
        def func(a, b, c):
            return a, b, c

        def wrapper(*args, **kwargs):
            return func('injected', *args, **kwargs)

        wrapped = update_wrapper(wrapper, func, injected=['a'])
        assert wrapped(2, 3) == ('injected', 2, 3)

    def test_injected_missing_raises(self):
        def func(a):
            return a

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        with pytest.raises(MissingArgument):
            update_wrapper(wrapper, func, injected=['z'])

    def test_injected_with_varkw(self):
        def func(a, **kwargs):
            return a, kwargs

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Injected arg missing from params but varkw is present -> no raise.
        wrapped = update_wrapper(wrapper, func, injected=['missing'])
        assert callable(wrapped)

    # `expected` args are added to the builder's forward-list, so the
    # resulting wrapper's generated invocation forwards them as names.
    # When the underlying `func` doesn't accept those names, calling the
    # wrapper will raise TypeError.
    def test_expected_forwarded_to_func(self):
        def func(a, newarg):
            return a, newarg

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapped = update_wrapper(wrapper, func, expected='newarg')
        assert wrapped(5, 7) == (5, 7)

    def test_expected_missing_in_func_raises(self):
        def func(a):
            return a

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapped = update_wrapper(wrapper, func, expected='newarg')
        with pytest.raises(TypeError):
            wrapped(5)

    def test_expected_pair_with_func_default(self):
        def func(a, newarg=10):
            return a, newarg

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapped = update_wrapper(wrapper, func, expected=[('newarg', 5)])
        assert wrapped(1) == (1, 10)
        assert wrapped(1, 2) == (1, 2)

    def test_expected_invalid_raises(self):
        def func(a):
            return a

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        with pytest.raises(ValueError):
            update_wrapper(wrapper, func, expected=123)

    def test_expected_bad_pair(self):
        def func(a):
            return a

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        with pytest.raises(ValueError):
            update_wrapper(wrapper, func, expected=[(1, 2, 3)])

    def test_expected_nonstring_name(self):
        def func(a):
            return a

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        with pytest.raises(ValueError):
            update_wrapper(wrapper, func, expected=[(5, 6)])

    def test_classmethod_raises(self):
        class A:
            @classmethod
            def cm(cls):
                pass

        def wrapper(*args, **kwargs):
            pass

        with pytest.raises(TypeError):
            update_wrapper(wrapper, A.__dict__['cm'])

    def test_unexpected_kw(self):
        def func(a):
            return a

        def wrapper(*args, **kwargs):
            pass

        with pytest.raises(TypeError):
            update_wrapper(wrapper, func, bad_kw=True)

    def test_hide_wrapped(self):
        def func(a):
            return a

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapped = update_wrapper(wrapper, func, hide_wrapped=True)
        assert not hasattr(wrapped, '__wrapped__')

        wrapped2 = update_wrapper(wrapper, func)
        assert wrapped2.__wrapped__ is func

    def test_update_dict_false(self):
        def func(a):
            return a

        func.custom = 1

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapped = update_wrapper(wrapper, func, update_dict=False)
        assert not hasattr(wrapped, 'custom')

    def test_async_wrapper(self):
        async def a_func(a):
            return a

        async def wrapper(*args, **kwargs):
            return await a_func(*args, **kwargs)

        wrapped = update_wrapper(wrapper, a_func)
        assert asyncio.run(wrapped(3)) == 3
        assert inspect.iscoroutinefunction(wrapped)


class TestWraps:
    def test_basic(self):
        def deco(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        @deco
        def example():
            """doc"""
            return 'val'

        assert example() == 'val'
        assert example.__name__ == 'example'
        assert example.__doc__ == 'doc'


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

class TestFunctionBuilder:
    def test_from_scratch(self):
        fb = FunctionBuilder('return_five', doc='returns 5', body='return 5')
        f = fb.get_func()
        assert f() == 5
        assert f.__name__ == 'return_five'
        assert f.__doc__ == 'returns 5'

    def test_varkw(self):
        fb = FunctionBuilder('return_five', body='return 5')
        fb.varkw = 'kw'
        f = fb.get_func()
        assert f(ignored='val') == 5

    def test_args_and_defaults(self):
        fb = FunctionBuilder('add', args=['a', 'b'], defaults=(2,), body='return a + b')
        f = fb.get_func()
        assert f(1) == 3
        assert f(1, 5) == 6

    def test_kwonly(self):
        fb = FunctionBuilder('konly', args=[], kwonlyargs=['x'],
                             kwonlydefaults={'x': 7}, body='return x')
        f = fb.get_func()
        assert f() == 7
        assert f(x=3) == 3

    def test_annotations(self):
        fb = FunctionBuilder('ann', args=['a'], annotations={'a': int, 'return': int},
                             body='return a')
        f = fb.get_func()
        assert f.__annotations__ == {'a': int, 'return': int}

    def test_unexpected_kw(self):
        with pytest.raises(TypeError):
            FunctionBuilder('f', bad=True)

    def test_get_sig_str(self):
        # NOTE: get_sig_str passes [] as defaults to inspect_formatargspec,
        # so defaults do not appear in the returned signature string.
        fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,),
                             kwonlyargs=['c'], kwonlydefaults={'c': 3})
        assert fb.get_sig_str() == '(a, b, *, c)'

    def test_get_sig_str_no_annotations(self):
        fb = FunctionBuilder('f', args=['a'], annotations={'a': int})
        assert fb.get_sig_str(with_annotations=False) == '(a)'
        assert fb.get_sig_str(with_annotations=True) == '(a: int)'

    def test_invocation_str_kwonly_default(self):
        fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
        assert fb.get_invocation_str() == 'a, b=b'

    def test_invocation_str_varargs(self):
        fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,), varargs='args')
        assert fb.get_invocation_str() == 'a, b, *args'

    def test_invocation_str_kwonly(self):
        fb = FunctionBuilder('f', args=['a'], kwonlyargs=['k'])
        assert fb.get_invocation_str() == 'a, k=k'

    def test_invocation_str_varkw(self):
        fb = FunctionBuilder('f', args=['a'], varkw='kw')
        assert fb.get_invocation_str() == 'a, **kw'

    def test_from_func(self):
        def func(a, b=2, *args, c=3, **kw):
            return a

        fb = FunctionBuilder.from_func(func)
        assert fb.name == 'func'
        assert fb.args == ['a', 'b']
        assert fb.defaults == (2,)
        assert fb.varargs == 'args'
        assert fb.kwonlyargs == ['c']
        assert fb.kwonlydefaults == {'c': 3}
        assert fb.varkw == 'kw'

    def test_from_func_not_callable(self):
        with pytest.raises(TypeError):
            FunctionBuilder.from_func(5)

    def test_from_func_partial(self):
        def base(a, b=2):
            return a + b

        p = functools.partial(base, 1)
        fb = FunctionBuilder.from_func(p)
        assert fb.name == 'base'

    def test_from_func_async(self):
        # NOTE: get_func for async builders sets body to
        # "return await _call(...)" only via update_wrapper; the default
        # body for an async builder is 'pass', so running the generated
        # coroutine returns None.
        async def afunc(a):
            return a

        fb = FunctionBuilder.from_func(afunc)
        assert fb.is_async is True
        f = fb.get_func({'afunc': afunc})
        assert inspect.iscoroutinefunction(f)
        assert asyncio.run(f(1)) is None

    def test_from_func_async_with_body(self):
        async def afunc(a):
            return a

        fb = FunctionBuilder.from_func(afunc)
        fb.body = 'return a'
        f = fb.get_func()
        assert asyncio.run(f(5)) == 5

    def test_posonly_args(self):
        def func(a, b, /, c):
            return a, b, c

        fb = FunctionBuilder.from_func(func)
        assert fb.posonlyargs == ['a', 'b']
        f = fb.get_func()
        sig = inspect.signature(f)
        assert list(sig.parameters) == ['a', 'b', 'c']

    def test_get_func_with_dict(self):
        fb = FunctionBuilder('f', body='return 1', dict={'custom': 'value'})
        f = fb.get_func()
        assert f.custom == 'value'

    def test_get_func_without_dict(self):
        fb = FunctionBuilder('f', body='return 1', dict={'custom': 'value'})
        f = fb.get_func(with_dict=False)
        assert not hasattr(f, 'custom')

    def test_get_func_no_source(self):
        fb = FunctionBuilder('f', body='return 1')
        f = fb.get_func(add_source=False)
        assert not hasattr(f, '__source__')

    def test_get_func_has_source(self):
        fb = FunctionBuilder('f', body='return 1')
        f = fb.get_func()
        assert 'def f(' in f.__source__

    def test_lambda_name(self):
        fb = FunctionBuilder('<lambda>', body='return 1')
        f = fb.get_func()
        assert f.__name__ == '<lambda>'

    def test_get_defaults_dict(self):
        fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3),
                             kwonlyargs=['k'], kwonlydefaults={'k': 4})
        d = fb.get_defaults_dict()
        assert d == {'b': 2, 'c': 3, 'k': 4}

    def test_get_arg_names(self):
        fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,),
                             kwonlyargs=['k'], kwonlydefaults={'k': 3})
        assert fb.get_arg_names() == ('a', 'b', 'k')
        assert fb.get_arg_names(only_required=True) == ('a',)

    def test_add_arg(self):
        fb = FunctionBuilder('f', args=['a'])
        fb.add_arg('b')
        assert fb.args == ['a', 'b']

    def test_add_arg_with_default(self):
        fb = FunctionBuilder('f', args=['a'])
        fb.add_arg('b', 5)
        assert fb.defaults == (5,)

    def test_add_arg_kwonly(self):
        fb = FunctionBuilder('f')
        fb.add_arg('k', 10, kwonly=True)
        assert fb.kwonlyargs == ['k']
        assert fb.kwonlydefaults == {'k': 10}

    def test_add_existing_arg(self):
        fb = FunctionBuilder('f', args=['a'])
        with pytest.raises(ExistingArgument):
            fb.add_arg('a')

    def test_add_existing_kwonly(self):
        fb = FunctionBuilder('f', kwonlyargs=['k'])
        with pytest.raises(ExistingArgument):
            fb.add_arg('k')

    def test_remove_arg(self):
        fb = FunctionBuilder('f', args=['a', 'b'])
        fb.remove_arg('a')
        assert fb.args == ['b']

    def test_remove_arg_adjusts_defaults(self):
        fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3))
        fb.remove_arg('b')
        assert fb.args == ['a', 'c']
        assert fb.defaults == (3,)

    def test_remove_kwonly(self):
        fb = FunctionBuilder('f', kwonlyargs=['k'], kwonlydefaults={'k': 3})
        fb.remove_arg('k')
        assert fb.kwonlyargs == []
        assert fb.kwonlydefaults == {}

    def test_remove_missing_raises(self):
        fb = FunctionBuilder('f', args=['a'])
        with pytest.raises(MissingArgument) as exc_info:
            fb.remove_arg('z')
        assert exc_info.value.arg_name == 'z'

    def test_resulting_func_signature(self):
        fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,), body='return a + b')
        f = fb.get_func()
        assert f(2) == 3
        assert f(2, 3) == 5


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------

class TestNoop:
    def test_returns_none(self):
        assert noop() is None

    def test_accepts_args(self):
        assert noop(1, 2, 3, x=4) is None


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

class TestOnce:
    def test_basic(self):
        calls = []

        @once
        def setup():
            calls.append(1)
            return 'value'

        assert setup() == 'value'
        assert setup() == 'value'
        assert len(calls) == 1

    def test_preserves_wraps(self):
        @once
        def my_func():
            """docs"""
            return 1

        assert my_func.__name__ == 'my_func'
        assert my_func.__doc__ == 'docs'

    def test_thread_safety(self):
        calls = []

        @once
        def setup():
            calls.append(1)
            return 'result'

        results = []
        threads = []

        def worker():
            results.append(setup())

        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert all(r == 'result' for r in results)
        assert len(calls) == 1
