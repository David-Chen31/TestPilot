"""Unit tests for boltons.funcutils."""
import inspect
import functools
import sys
import types
import threading

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

def test_formatargspec_simple():
    assert inspect_formatargspec(['a', 'b']) == '(a, b)'


def test_formatargspec_defaults():
    assert inspect_formatargspec(['a', 'b'], defaults=(1,)) == '(a, b=1)'


def test_formatargspec_varargs_varkw():
    assert inspect_formatargspec(['a'], varargs='args', varkw='kwargs') == '(a, *args, **kwargs)'


def test_formatargspec_kwonly():
    assert inspect_formatargspec([], kwonlyargs=['x', 'y']) == '(*, x, y)'


def test_formatargspec_kwonly_defaults():
    res = inspect_formatargspec([], kwonlyargs=['x'], kwonlydefaults={'x': 5})
    assert res == '(*, x=5)'


def test_formatargspec_kwonly_with_varargs():
    res = inspect_formatargspec(['a'], varargs='args', kwonlyargs=['x'])
    assert res == '(a, *args, x)'


def test_formatargspec_annotations_and_return():
    res = inspect_formatargspec(
        ['a'], annotations={'a': int, 'return': str})
    assert res == '(a: int) -> str'


def test_formatargspec_custom_formatters():
    res = inspect_formatargspec(
        ['a'],  # noqa
        formatarg=lambda n: n.upper())
    assert res == '(A)'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def test_get_module_callables_by_module_obj():
    mod = types.ModuleType('test_mod_callables')

    class Foo:
        pass

    def bar():
        pass

    Foo.__module__ = 'test_mod_callables'
    bar.__module__ = 'test_mod_callables'
    mod.Foo = Foo
    mod.bar = bar
    mod.other = 42

    types_map, funcs_map = get_module_callables(mod)
    assert types_map == {'Foo': Foo}
    assert funcs_map == {'bar': bar}


def test_get_module_callables_by_name():
    mod = types.ModuleType('test_mod_callables_by_name')

    def foo():
        pass

    foo.__module__ = 'test_mod_callables_by_name'
    mod.foo = foo
    sys.modules['test_mod_callables_by_name'] = mod
    try:
        types_map, funcs_map = get_module_callables('test_mod_callables_by_name')
        assert funcs_map == {'foo': foo}
    finally:
        del sys.modules['test_mod_callables_by_name']


def test_get_module_callables_ignore():
    mod = types.ModuleType('test_mod_ignore')

    def foo():
        pass

    def bar():
        pass

    foo.__module__ = 'test_mod_ignore'
    bar.__module__ = 'test_mod_ignore'
    mod.foo = foo
    mod.bar = bar

    _, funcs_map = get_module_callables(mod, ignore=lambda n: n == 'foo')
    assert 'foo' not in funcs_map
    assert 'bar' in funcs_map


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items_contains_int_attrs():
    items = dict(mro_items(int))
    # int has these non-callable class vars
    for k in ('denominator', 'numerator', 'imag', 'real'):
        assert k in items


def test_mro_items_inherits():
    class A:
        x = 1

    class B(A):
        y = 2

    items = dict(mro_items(B))
    assert items['x'] == 1
    assert items['y'] == 2


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

def test_dir_dict_includes_inherited():
    class A:
        a = 1

    class B(A):
        b = 2

    d = dir_dict(B())
    assert d['a'] == 1
    assert d['b'] == 2


def test_dir_dict_swallow_exception():
    class Broken:
        @property
        def bad(self):
            raise RuntimeError('boom')

        ok = 5

    d = dir_dict(Broken())
    assert d['ok'] == 5
    assert 'bad' not in d


def test_dir_dict_raise_exc():
    class Broken:
        @property
        def bad(self):
            raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        dir_dict(Broken(), raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_closure_and_globals():
    def func():
        return func

    copied = copy_function(func)
    assert copied is not func
    assert copied() is func


def test_copy_function_dict():
    def func():
        pass

    func.custom = 'value'
    copied = copy_function(func)
    assert copied.custom == 'value'
    copied_nodict = copy_function(func, copy_dict=False)
    assert not hasattr(copied_nodict, 'custom')


def test_copy_function_kwdefaults():
    def func(a, *, b=10):
        return a + b

    copied = copy_function(func)
    assert copied(1) == 11
    assert copied.__kwdefaults__ == {'b': 10}


# ---------------------------------------------------------------------------
# partial_ordering
# ---------------------------------------------------------------------------

def test_partial_ordering_basic():
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


def test_partial_ordering_does_not_override():
    class Base:
        def __lt__(self, other):
            return 'custom'

    Base.__le__ = lambda self, other: False
    Base.__ge__ = lambda self, other: False

    decorated = partial_ordering(Base)
    assert decorated.__lt__ is Base.__lt__


def test_partial_ordering_eq():
    @partial_ordering
    class MySet(set):
        def __le__(self, other):
            return self.issubset(other)

        def __ge__(self, other):
            return self.issuperset(other)

    a = MySet([1, 2])
    b = MySet([1, 2])
    assert (a == b) is True


# ---------------------------------------------------------------------------
# InstancePartial
# ---------------------------------------------------------------------------

def test_instance_partial_as_method():
    def add(self, x, y):
        return self + x + y

    class Thing:
        add = InstancePartial(add, 10, 20)

    t = Thing()
    assert t.add() == 0 + 10 + 20


def test_instance_partial_get_descriptor_direct():
    def f(a, b):
        return a + b

    class C:
        m = InstancePartial(f, 1)

    assert isinstance(C.__dict__['m'].__get__(None, C), types.MethodType)


def test_instance_partial_partialmethod_attr():
    def f(a, b):
        return a + b

    p = InstancePartial(f, 1)
    assert isinstance(p._partialmethod, functools.partialmethod)
    assert isinstance(p.__partialmethod__, functools.partialmethod)


# ---------------------------------------------------------------------------
# CachedInstancePartial
# ---------------------------------------------------------------------------

def test_cached_instance_partial_caches():
    def mul(self, x, y):
        return self * x * y

    class N:
        calc = CachedInstancePartial(mul, 2, 3)

    n = N()
    n = 4
    # create real instance
    class Real:
        __mul__ = lambda self, other: 1

    class M:
        calc = CachedInstancePartial(mul, 2, 3)

    class IntLike:
        def __init__(self):
            self.val = 4

        def __mul__(self, other):
            return self.val * other

    m = M()
    m.val = 4
    first = m.calc()
    second = m.calc()
    assert first == second
    # Cached on instance dict
    assert 'calc' in m.__dict__


def test_cached_instance_partial_set_name():
    def f(self):
        return 1

    class C:
        method = CachedInstancePartial(f)

    c = C()
    c.method()
    assert C.__dict__['method'].__name__ == 'method'
    assert C.__dict__['method'].__doc__ == f.__doc__
    assert C.__dict__['method'].__module__ == f.__module__


def test_cached_instance_partial_class_access():
    def f(self):
        return 1

    class C:
        method = CachedInstancePartial(f)

    bound = C.method
    assert isinstance(bound, types.MethodType)


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_basic():
    assert format_invocation('func', args=(1, 2), kwargs={'c': 3}) == 'func(1, 2, c=3)'


def test_format_invocation_no_kwargs():
    assert format_invocation('a_func', args=(1,)) == 'a_func(1)'


def test_format_invocation_list_kwargs():
    assert format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]) == 'kw_func(a=1, b=2)'


def test_format_invocation_sorted_dict_kwargs():
    assert format_invocation('f', kwargs={'b': 2, 'a': 1}) == 'f(a=1, b=2)'


def test_format_invocation_empty():
    assert format_invocation('f') == 'f()'


def test_format_invocation_unexpected_kw():
    with pytest.raises(TypeError):
        format_invocation('f', unexpected=1)


def test_format_invocation_custom_repr():
    assert format_invocation('f', args=('a',), repr=str) == "f(a)"


# ---------------------------------------------------------------------------
# format_exp_repr
# ---------------------------------------------------------------------------

def test_format_exp_repr_pos_and_opt():
    class Flag:
        def __init__(self, length, width, depth=None):
            self.length = length
            self.width = width
            self.depth = depth

    flag = Flag(5, 10)
    assert format_exp_repr(flag, ['length', 'width'], [], ['depth']) == 'Flag(5, 10)'

    flag2 = Flag(5, 15, 2)
    assert format_exp_repr(flag2, ['length'], ['width', 'depth']) == 'Flag(5, width=15, depth=2)'


def test_format_exp_repr_dedupes():
    class Obj:
        def __init__(self):
            self.a = 1

    o = Obj()
    assert format_exp_repr(o, [], ['a'], ['a']) == 'Obj(a=1)'


def test_format_exp_repr_custom_opt_key():
    class Obj:
        def __init__(self):
            self.x = 0

    o = Obj()
    res = format_exp_repr(o, [], [], ['x'], opt_key=lambda v: v == 0)
    assert res == 'Obj()'


# ---------------------------------------------------------------------------
# format_nonexp_repr
# ---------------------------------------------------------------------------

def test_format_nonexp_repr_basic():
    class Flag:
        def __init__(self, length, width, depth=None):
            self.length = length
            self.width = width
            self.depth = depth

    flag = Flag(5, 10)
    assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == '<Flag length=5 width=10>'


def test_format_nonexp_repr_fallback_id():
    class Flag:
        pass

    flag = Flag()
    res = format_nonexp_repr(flag)
    assert res == '<Flag id=%s>' % id(flag)


def test_format_nonexp_repr_dedupes():
    class Obj:
        def __init__(self):
            self.a = 1

    o = Obj()
    assert format_nonexp_repr(o, ['a'], ['a']) == '<Obj a=1>'


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_wraps_basic():
    def print_return(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            print(ret)
            return ret

        return wrapper

    @print_return
    def example():
        """docstring"""
        return 'example return value'

    assert example() == 'example return value'
    assert example.__name__ == 'example'
    assert example.__doc__ == 'docstring'


def test_update_wrapper_copies_signature():
    def func(a, b=2):
        """doc"""
        return a + b

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    w = update_wrapper(wrapper, func)
    sig = inspect.signature(w)
    assert list(sig.parameters) == ['a', 'b']
    assert w(1, 3) == 4
    assert w.__name__ == 'func'
    assert w.__doc__ == 'doc'


def test_update_wrapper_injected():
    def func(a, b):
        return a + b

    def wrapper(b):
        return func(1, b)

    w = update_wrapper(wrapper, func, injected=['a'])
    sig = inspect.signature(w)
    assert list(sig.parameters) == ['b']
    assert w(5) == 6


def test_update_wrapper_expected():
    def func(a):
        return a

    def wrapper(a, b=10):
        return func(a) + b

    w = update_wrapper(wrapper, func, expected=['b'])
    sig = inspect.signature(w)
    assert 'b' in sig.parameters


def test_update_wrapper_classmethod_raises():
    class Foo:
        @classmethod
        def cm(cls):
            pass

    def wrapper(*args, **kwargs):
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, Foo.cm)


def test_update_wrapper_unexpected_kwargs():
    def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    with pytest.raises(TypeError):
        update_wrapper(wrapper, func, bogus=1)


def test_update_wrapper_inject_to_varkw():
    def func(a, **kwargs):
        return (a, kwargs)

    def wrapper(**kwargs):
        return func(1, **kwargs)

    w = update_wrapper(wrapper, func, injected=['a'])
    assert w(x=2) == (1, {'x': 2})


def test_update_wrapper_inject_to_varkw_false():
    def func(a):
        return a

    def wrapper(**kwargs):
        return func(**kwargs)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, func, injected=['missing'],
                       inject_to_varkw=False)


def test_update_wrapper_hide_wrapped():
    def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    w = update_wrapper(wrapper, func, hide_wrapped=True)
    assert not hasattr(w, '__wrapped__')

    w2 = update_wrapper(wrapper, func)
    assert w2.__wrapped__ is func


def test_update_wrapper_update_dict_false():
    def func(a):
        return a

    func.custom = 'x'

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    w = update_wrapper(wrapper, func, update_dict=False)
    assert 'custom' not in w.__dict__


def test_parse_expected_bad_values():
    def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    with pytest.raises(ValueError):
        update_wrapper(wrapper, func, expected=42)

    with pytest.raises(ValueError):
        update_wrapper(wrapper, func, expected=[(1, 2, 3)])

    with pytest.raises(ValueError):
        update_wrapper(wrapper, func, expected=[(1, 2)])


def test_parse_expected_string():
    def func(a):
        return a

    def wrapper(a, b=0):
        return func(a) + b

    w = update_wrapper(wrapper, func, expected='b')
    assert 'b' in inspect.signature(w).parameters


def test_parse_expected_mapping():
    def func(a):
        return a

    def wrapper(a, b=0, c=0):
        return func(a) + b + c

    w = update_wrapper(wrapper, func, expected={'b': 5, 'c': 6})
    sig = inspect.signature(w)
    assert sig.parameters['b'].default == 5
    assert sig.parameters['c'].default == 6


def test_update_wrapper_async():
    async def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    w = update_wrapper(wrapper, func)
    assert inspect.iscoroutinefunction(w)


def test_update_wrapper_partial_wrapper():
    def func(a, b=2):
        return a + b

    p = functools.partial(func, 1)

    def wrapper(*args, **kwargs):
        return p(*args, **kwargs)

    # build_from defaults to wrapper if partial built from func
    w = update_wrapper(wrapper, func)
    assert w() == 3


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_function_builder_basic():
    fb = FunctionBuilder('return_five', doc='returns the integer 5',
                         body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__name__ == 'return_five'
    assert f.__doc__ == 'returns the integer 5'


def test_function_builder_varkw():
    fb = FunctionBuilder('return_five', doc='returns the integer 5',
                         body='return 5')
    fb.varkw = 'kw'
    f = fb.get_func()
    assert f(ignored_arg='ignored_val') == 5


def test_function_builder_unexpected_kwargs():
    with pytest.raises(TypeError):
        FunctionBuilder('name', bogus=1)


def test_function_builder_from_func():
    def orig(a, b=2, *args, c=3, **kwargs):
        return a, b, args, c, kwargs

    fb = FunctionBuilder.from_func(orig)
    assert fb.name == 'orig'
    assert fb.args == ['a', 'b']
    assert fb.varargs == 'args'
    assert fb.varkw == 'kwargs'
    assert fb.defaults == (2,)
    assert fb.kwonlyargs == ['c']
    assert fb.kwonlydefaults == {'c': 3}


def test_function_builder_from_func_not_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(42)


def test_function_builder_from_func_partial():
    def func(a, b=2):
        """doc"""
        return a + b

    p = functools.partial(func, 1)
    fb = FunctionBuilder.from_func(p)
    assert fb.name == 'func'


def test_function_builder_from_async():
    async def afunc(a):
        return a

    fb = FunctionBuilder.from_func(afunc)
    assert fb.is_async is True
    f = fb.get_func()
    assert inspect.iscoroutinefunction(f)


def test_function_builder_get_sig_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
    assert fb.get_sig_str() == '(a, b=2)'
    fb.annotations = {'a': int}
    assert fb.get_sig_str() == '(a: int, b=2)'
    assert fb.get_sig_str(with_annotations=False) == '(a, b=2)'


def test_function_builder_sig_uses_default_value_repr():
    # Inspect returned by get_sig_str uses default value from fb.defaults
    fb = FunctionBuilder('f', args=['a'], defaults=(10,))
    assert 'a' in fb.get_sig_str()


def test_function_builder_get_invocation_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_invocation_varargs_positional():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,), varargs='args')
    # with varargs, all args positional
    assert fb.get_invocation_str() == 'a, b, *args'


def test_function_builder_invocation_kwonly():
    fb = FunctionBuilder('f', kwonlyargs=['x'], varkw='kwargs')
    assert fb.get_invocation_str() == 'x=x, **kwargs'


def test_function_builder_invocation_target_kwonly():
    def target(a, *, b):
        return a, b

    fb = FunctionBuilder('f', args=['a', 'b'])
    res = fb.get_invocation_str(target=target)
    assert res == 'a, b=b'


def test_function_builder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,),
                         kwonlyargs=['c'], kwonlydefaults={'c': 5})
    assert fb.get_defaults_dict() == {'b': 1, 'c': 5}


def test_function_builder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,),
                         kwonlyargs=['c'], kwonlydefaults={'c': 5})
    assert fb.get_arg_names() == ('a', 'b', 'c')
    assert fb.get_arg_names(only_required=True) == ('a',)


def test_function_builder_add_arg():
    fb = FunctionBuilder('f')
    fb.add_arg('a')
    assert fb.args == ['a']
    assert fb.defaults in (None, ())

    fb.add_arg('b', 5)
    assert fb.args == ['a', 'b']
    assert fb.defaults == (5,)

    fb.add_arg('c', 7, kwonly=True)
    assert fb.kwonlyargs == ['c']
    assert fb.kwonlydefaults == {'c': 7}


def test_function_builder_add_arg_existing():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')

    fb2 = FunctionBuilder('f', kwonlyargs=['a'])
    with pytest.raises(ExistingArgument):
        fb2.add_arg('a', kwonly=True)


def test_function_builder_remove_arg():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3))
    fb.remove_arg('b')
    assert fb.args == ['a', 'c']
    assert fb.defaults == (3,)


def test_function_builder_remove_arg_kwonly():
    fb = FunctionBuilder('f', kwonlyargs=['a', 'b'], kwonlydefaults={'b': 5})
    fb.remove_arg('b')
    assert fb.kwonlyargs == ['a']
    assert 'b' not in fb.kwonlydefaults


def test_function_builder_remove_missing():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(MissingArgument) as exc_info:
        fb.remove_arg('missing')
    assert exc_info.value.arg_name == 'missing'


def test_function_builder_get_func_source():
    fb = FunctionBuilder('f', body='return 1')
    f = fb.get_func()
    assert '__source__' in f.__dict__
    assert 'return 1' in f.__source__


def test_function_builder_get_func_with_dict():
    fb = FunctionBuilder('f', body='return 1', dict={'x': 5})
    f = fb.get_func()
    assert f.x == 5
    f2 = fb.get_func(with_dict=False)
    assert not hasattr(f2, 'x')


def test_function_builder_get_func_execdict():
    fb = FunctionBuilder('f', body='return _val()')
    f = fb.get_func(execdict={'_val': lambda: 42})
    assert f() == 42


def test_function_builder_lambda_name():
    fb = FunctionBuilder('<lambda>', body='return 1')
    f = fb.get_func()
    assert f() == 1
    assert f.__name__ == '<lambda>'


def test_function_builder_annotation():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int})
    f = fb.get_func()
    assert f.__annotations__ == {'a': int}


def test_function_builder_module():
    fb = FunctionBuilder('f', module='mymodule')
    f = fb.get_func()
    assert f.__module__ == 'mymodule'


def test_function_builder_indent():
    fb = FunctionBuilder('f', body='if True:\n    return 1', indent=4)
    f = fb.get_func()
    assert f() == 1


def test_function_builder_posonly_from_func():
    def func(a, b, /, c, d=4):
        return a, b, c, d

    fb = FunctionBuilder.from_func(func)
    assert fb.posonlyargs == ['a', 'b']
    # posonlyargs never keyword-forwarded
    inv = fb.get_invocation_str()
    assert inv == 'a, b, c, d=d'


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------

def test_noop_returns_none():
    assert noop() is None
    assert noop(1, 2, a=3) is None


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

def test_once_caches_result():
    calls = []

    @once
    def setup():
        calls.append(1)
        return 'initialized'

    assert setup() == 'initialized'
    assert setup() == 'initialized'
    assert len(calls) == 1


def test_once_preserves_name():
    @once
    def my_setup():
        return 1

    assert my_setup.__name__ == 'my_setup'


def test_once_thread_safe():
    calls = []
    started = threading.Event()

    @once
    def setup():
        started.set()
        calls.append(1)
        return 'value'

    results = []

    def worker():
        results.append(setup())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r == 'value' for r in results)
    assert len(calls) == 1
