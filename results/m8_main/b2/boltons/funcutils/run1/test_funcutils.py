import sys
import types
import inspect
import functools
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

def test_formatargspec_basic():
    assert inspect_formatargspec(['a', 'b']) == '(a, b)'


def test_formatargspec_defaults():
    assert inspect_formatargspec(['a', 'b'], defaults=(1,)) == '(a, b=1)'


def test_formatargspec_varargs_varkw():
    out = inspect_formatargspec(['a'], varargs='args', varkw='kwargs')
    assert out == '(a, *args, **kwargs)'


def test_formatargspec_kwonly_without_varargs():
    out = inspect_formatargspec(['a'], kwonlyargs=['b'])
    assert out == '(a, *, b)'


def test_formatargspec_kwonly_with_defaults():
    out = inspect_formatargspec(['a'], kwonlyargs=['b'], kwonlydefaults={'b': 5})
    assert out == '(a, *, b=5)'


def test_formatargspec_annotations():
    out = inspect_formatargspec(['a'], annotations={'a': int})
    assert out == '(a: int)'


def test_formatargspec_return_annotation():
    out = inspect_formatargspec(['a'], annotations={'return': int})
    assert out == '(a) -> int'


def test_formatargspec_empty():
    assert inspect_formatargspec([]) == '()'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def test_get_module_callables_by_instance():
    mod = types.ModuleType('test_fake_module_xyz')

    class Foo:
        pass

    def bar():
        pass

    Foo.__module__ = 'test_fake_module_xyz'
    bar.__module__ = 'test_fake_module_xyz'
    mod.Foo = Foo
    mod.bar = bar
    mod.other = 42

    types_map, funcs_map = get_module_callables(mod)
    assert types_map == {'Foo': Foo}
    assert funcs_map == {'bar': bar}


def test_get_module_callables_by_name():
    mod = types.ModuleType('test_fake_module_abc')
    sys.modules['test_fake_module_abc'] = mod
    try:
        def baz():
            pass
        baz.__module__ = 'test_fake_module_abc'
        mod.baz = baz
        types_map, funcs_map = get_module_callables('test_fake_module_abc')
        assert types_map == {}
        assert funcs_map == {'baz': baz}
    finally:
        del sys.modules['test_fake_module_abc']


def test_get_module_callables_ignore():
    mod = types.ModuleType('test_fake_module_ignore')

    def bar():
        pass
    bar.__module__ = 'test_fake_module_ignore'
    mod.bar = bar

    types_map, funcs_map = get_module_callables(
        mod, ignore=lambda name: name == 'bar')
    assert funcs_map == {}


def test_get_module_callables_skips_foreign_module():
    mod = types.ModuleType('test_fake_module_foreign')

    def external():
        pass
    external.__module__ = 'some_other_module'
    mod.external = external

    types_map, funcs_map = get_module_callables(mod)
    assert funcs_map == {}


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items_includes_base_attrs():
    class Base:
        base_attr = 1

    class Child(Base):
        child_attr = 2

    items = dict(mro_items(Child))
    assert items['base_attr'] == 1
    assert items['child_attr'] == 2


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

def test_dir_dict_includes_inherited():
    class Base:
        a = 1

    class Child(Base):
        b = 2

    d = dir_dict(Child())
    assert d['a'] == 1
    assert d['b'] == 2


def test_dir_dict_raise_exc_false_skips():
    class Obj:
        def __dir__(self):
            return ['bad', 'good']

        @property
        def bad(self):
            raise RuntimeError('boom')

        @property
        def good(self):
            return 42

    d = dir_dict(Obj(), raise_exc=False)
    assert d['good'] == 42
    assert 'bad' not in d


def test_dir_dict_raise_exc_true():
    class Obj:
        def __dir__(self):
            return ['bad']

        @property
        def bad(self):
            raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        dir_dict(Obj(), raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_identity_closure():
    func = lambda: func
    cp = copy_function(func)
    assert cp is not func
    assert cp() is func


def test_copy_function_copies_dict():
    def orig():
        return 1
    orig.custom = 'yes'
    cp = copy_function(orig)
    assert cp.custom == 'yes'
    cp2 = copy_function(orig, copy_dict=False)
    assert 'custom' not in cp2.__dict__


def test_copy_function_defaults_kwonly():
    def orig(a, b=2, *, c=3):
        return a, b, c
    cp = copy_function(orig)
    assert cp.__defaults__ == (2,)
    assert cp.__kwdefaults__ == {'c': 3}
    assert cp(1) == (1, 2, 3)


# ---------------------------------------------------------------------------
# partial_ordering
# ---------------------------------------------------------------------------

def test_partial_ordering():
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
    assert (a > b) is True
    assert (a == b) is False
    assert (a == MySet([1, 2, 3])) is True


def test_partial_ordering_does_not_override():
    class Foo:
        def __lt__(self, other):
            return 'custom'
        def __le__(self, other):
            return True
        def __ge__(self, other):
            return True

    original_lt = Foo.__lt__

    decorated = partial_ordering(Foo)
    assert isinstance(decorated, type)
    # __lt__ was already present, so it is not overridden
    assert Foo.__lt__ is original_lt
    # calling the method directly returns the custom value
    assert Foo.__lt__(Foo(), Foo()) == 'custom'
    foo = Foo()
    assert foo == foo


# ---------------------------------------------------------------------------
# InstancePartial / CachedInstancePartial
# ---------------------------------------------------------------------------
#
# functools.partial prepends its stored args to the call arguments. When
# InstancePartial(f, 5) is bound to an instance ``o`` via __get__, the
# resulting MethodType calls partial.__call__(o) -> f(5, o). So the
# instance gets passed LAST, not first.

def _make_partial_target():
    def func(x, self):
        return self.val + x
    return func


def test_instance_partial_method_binding():
    func = _make_partial_target()

    class Obj:
        val = 10
        add = InstancePartial(func, 5)

    o = Obj()
    # partial stored arg (5) comes first, then the bound instance -> 15
    assert o.add() == 15


def test_cached_instance_partial_method_binding():
    func = _make_partial_target()

    class Obj:
        val = 20
        add = CachedInstancePartial(func, 3)

    o = Obj()
    assert o.add() == 23
    # cached on instance __dict__; same bound method object returned
    assert o.__dict__['add'] is o.add


def test_partial_alias_and_partialmethod_property():
    func = _make_partial_target()
    p = InstancePartial(func, 1)
    pm = p._partialmethod
    assert isinstance(pm, functools.partialmethod)
    assert isinstance(p.__partialmethod__, functools.partialmethod)


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_positional():
    assert format_invocation('f', args=(1, 2)) == 'f(1, 2)'


def test_format_invocation_kwargs_sorted():
    assert format_invocation('f', kwargs={'b': 2, 'a': 1}) == 'f(a=1, b=2)'


def test_format_invocation_kwargs_list():
    assert format_invocation('f', kwargs=[('a', 1), ('b', 2)]) == 'f(a=1, b=2)'


def test_format_invocation_mixed():
    assert format_invocation('f', args=(1,), kwargs={'c': 3}) == 'f(1, c=3)'


def test_format_invocation_custom_repr():
    assert format_invocation('f', args=('x',), repr=str) == 'f(x)'


def test_format_invocation_unexpected_kw():
    with pytest.raises(TypeError):
        format_invocation('f', unexpected=True)


# ---------------------------------------------------------------------------
# format_exp_repr
# ---------------------------------------------------------------------------

class _Flag:
    def __init__(self, length, width, depth=None):
        self.length = length
        self.width = width
        self.depth = depth


def test_format_exp_repr_positional():
    flag = _Flag(5, 10)
    assert format_exp_repr(flag, ['length', 'width'], [], ['depth']) == '_Flag(5, 10)'


def test_format_exp_repr_keyword():
    flag = _Flag(5, 15, 2)
    assert format_exp_repr(
        flag, ['length'], ['width', 'depth']) == '_Flag(5, width=15, depth=2)'


def test_format_exp_repr_dedup():
    flag = _Flag(5, 10)
    out = format_exp_repr(flag, [], ['length', 'length'])
    assert out == '_Flag(length=5)'


def test_format_exp_repr_custom_opt_key():
    # opt_key returns True -> excluded (opt_key is a "should skip" predicate
    # for opt_names: item included only when NOT opt_key(val))
    flag = _Flag(5, 10, 0)
    # opt_key(val) = (val != 0) -> for depth=0 => False => included
    out = format_exp_repr(flag, [], [], ['depth'], opt_key=lambda v: v != 0)
    assert out == '_Flag(depth=0)'

    # opt_key(val) = (val == 0) -> for depth=0 => True => excluded
    out2 = format_exp_repr(flag, [], [], ['depth'], opt_key=lambda v: v == 0)
    assert out2 == '_Flag()'


# ---------------------------------------------------------------------------
# format_nonexp_repr
# ---------------------------------------------------------------------------

def test_format_nonexp_repr():
    flag = _Flag(5, 10)
    assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == \
        '<_Flag length=5 width=10>'


def test_format_nonexp_repr_id_fallback():
    flag = _Flag(5, 10)
    out = format_nonexp_repr(flag)
    assert out.startswith('<_Flag id=')
    assert out.endswith('>')


def test_format_nonexp_repr_with_value():
    flag = _Flag(5, 10, 3)
    out = format_nonexp_repr(flag, [], ['depth'])
    assert out == '<_Flag depth=3>'


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_wraps_basic():
    def func(a, b=2):
        """docstring"""
        return a + b

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    assert wrapper(1) == 3
    assert wrapper.__name__ == 'func'
    assert wrapper.__doc__ == 'docstring'
    assert wrapper.__wrapped__ is func


def test_wraps_injected():
    def func(a, b):
        return a + b

    @wraps(func, injected='a')
    def wrapper(*args, **kwargs):
        return func(10, **kwargs)

    params = list(inspect.signature(wrapper).parameters)
    assert params == ['b']


def test_wraps_expected():
    def func(a):
        return a

    @wraps(func, expected={'b': 5})
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    params = inspect.signature(wrapper).parameters
    assert 'a' in params


def test_update_wrapper_returns_copy():
    def func():
        return 1

    def wrapper():
        return func()

    result = update_wrapper(wrapper, func)
    assert result is not wrapper
    assert result() == 1


def test_update_wrapper_hide_wrapped():
    def func():
        return 1

    def wrapper():
        return func()

    result = update_wrapper(wrapper, func, hide_wrapped=True)
    assert not hasattr(result, '__wrapped__')


def test_update_wrapper_unexpected_kwarg():
    def func():
        pass

    def wrapper():
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, func, bogus=1)


def test_update_wrapper_rejects_classmethod():
    class C:
        @classmethod
        def cm(cls):
            pass

    def wrapper():
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, C.__dict__['cm'])


def test_update_wrapper_inject_missing_no_varkw():
    def func(a):
        pass

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, func, injected='missing')


def test_update_wrapper_inject_missing_with_varkw():
    def func(a, **kwargs):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    result = update_wrapper(wrapper, func, injected='missing')
    assert result(a=3) == 3


def test_update_wrapper_async():
    async def func(a, b=2):
        return a + b

    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    result = update_wrapper(wrapper, func)
    assert inspect.iscoroutinefunction(result)


# ---------------------------------------------------------------------------
# _parse_wraps_expected error paths (via update_wrapper)
# ---------------------------------------------------------------------------

def test_expected_bad_type():
    def func(a):
        pass

    def wrapper(*args, **kwargs):
        pass

    with pytest.raises(ValueError):
        update_wrapper(wrapper, func, expected=12345)


def test_expected_bad_pair():
    def func(a):
        pass

    def wrapper(*args, **kwargs):
        pass

    with pytest.raises(ValueError):
        update_wrapper(wrapper, func, expected=[('a', 'b', 'c')])


def test_expected_non_string_name():
    def func(a):
        pass

    def wrapper(*args, **kwargs):
        pass

    with pytest.raises(ValueError):
        update_wrapper(wrapper, func, expected=[(1, 2)])


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_function_builder_basic():
    fb = FunctionBuilder('return_five', doc='returns 5', body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__name__ == 'return_five'
    assert f.__doc__ == 'returns 5'
    assert '__source__' in f.__dict__


def test_function_builder_varkw():
    fb = FunctionBuilder('return_five', body='return 5')
    fb.varkw = 'kw'
    f = fb.get_func()
    assert f(ignored_arg='ignored_val') == 5


def test_function_builder_defaults():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(10,), body='return a + b')
    f = fb.get_func()
    assert f(1) == 11
    assert f(1, 2) == 3


def test_function_builder_kwonly():
    fb = FunctionBuilder('f', args=['a'], kwonlyargs=['b'],
                         kwonlydefaults={'b': 7}, body='return a + b')
    f = fb.get_func()
    assert f(1) == 8
    assert f(1, b=3) == 4


def test_function_builder_annotations():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int, 'return': int},
                         body='return a')
    f = fb.get_func()
    assert f.__annotations__ == {'a': int, 'return': int}
    assert f(3) == 3


def test_function_builder_unexpected_kwarg():
    with pytest.raises(TypeError):
        FunctionBuilder('f', bogus=1)


def test_function_builder_from_func():
    def orig(a, b=2):
        return a + b

    fb = FunctionBuilder.from_func(orig)
    assert fb.name == 'orig'
    assert fb.args == ['a', 'b']
    assert fb.defaults == (2,)
    f = fb.get_func()
    assert f(3) is None
    assert f(3, 4) is None


def test_function_builder_from_func_not_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(42)


def test_function_builder_from_partial():
    def orig(a, b):
        return a + b
    p = functools.partial(orig, 1)
    fb = FunctionBuilder.from_func(p)
    assert fb.name == 'orig'


def test_function_builder_from_async():
    async def orig(a):
        return a
    fb = FunctionBuilder.from_func(orig)
    assert fb.is_async is True
    f = fb.get_func()
    assert inspect.iscoroutinefunction(f)


def test_function_builder_posonly():
    def orig(a, b, /, c):
        return a + b + c
    fb = FunctionBuilder.from_func(orig)
    assert fb.posonlyargs == ['a', 'b']


def test_function_builder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,),
                         kwonlyargs=['c'])
    assert fb.get_arg_names() == ('a', 'b', 'c')
    assert fb.get_arg_names(only_required=True) == ('a', 'c')


def test_function_builder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,),
                         kwonlyargs=['c'], kwonlydefaults={'c': 5})
    assert fb.get_defaults_dict() == {'b': 1, 'c': 5}


def test_function_builder_add_arg():
    fb = FunctionBuilder('f', body='return a')
    fb.add_arg('a')
    assert fb.args == ['a']
    f = fb.get_func()
    assert f(9) == 9


def test_function_builder_add_arg_default():
    fb = FunctionBuilder('f', body='return a')
    fb.add_arg('a', 3)
    assert fb.defaults == (3,)


def test_function_builder_add_kwonly():
    fb = FunctionBuilder('f', body='return a')
    fb.add_arg('a', kwonly=True)
    assert fb.kwonlyargs == ['a']


def test_function_builder_add_existing_arg():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_function_builder_add_existing_kwonly():
    fb = FunctionBuilder('f', kwonlyargs=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_function_builder_remove_arg():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(5,), body='return a')
    fb.remove_arg('b')
    assert fb.args == ['a']
    assert fb.defaults == ()
    f = fb.get_func()
    assert f(1) == 1


def test_function_builder_remove_kwonly():
    fb = FunctionBuilder('f', kwonlyargs=['a'], kwonlydefaults={'a': 1},
                         body='return 1')
    fb.remove_arg('a')
    assert fb.kwonlyargs == []
    assert fb.kwonlydefaults == {}


def test_function_builder_remove_missing():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(MissingArgument):
        fb.remove_arg('z')


def test_function_builder_get_invocation_str_defaults_kw():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    # b has default and no varargs -> forwarded as keyword
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_get_invocation_str_with_varargs():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,), varargs='args')
    assert fb.get_invocation_str() == 'a, b, *args'


def test_function_builder_get_invocation_str_kwonly():
    fb = FunctionBuilder('f', args=['a'], kwonlyargs=['b'])
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_get_invocation_str_posonly():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,), posonlyargs=['a'])
    # b has a default and is not posonly -> forwarded as keyword
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_get_sig_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    assert fb.get_sig_str() == '(a, b)'


def test_function_builder_lambda_name():
    fb = FunctionBuilder('<lambda>', body='return 1')
    f = fb.get_func()
    assert f.__name__ == '<lambda>'
    assert f() == 1


def test_function_builder_with_dict():
    fb = FunctionBuilder('f', dict={'custom': 'val'}, body='return 1')
    f = fb.get_func()
    assert f.custom == 'val'
    f2 = fb.get_func(with_dict=False)
    assert not hasattr(f2, 'custom')


def test_function_builder_no_source():
    fb = FunctionBuilder('f', body='return 1')
    f = fb.get_func(add_source=False)
    assert not hasattr(f, '__source__')


# ---------------------------------------------------------------------------
# MissingArgument / ExistingArgument
# ---------------------------------------------------------------------------

def test_exception_types():
    assert issubclass(MissingArgument, ValueError)
    assert issubclass(ExistingArgument, ValueError)


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------

def test_noop():
    assert noop() is None
    assert noop(1, 2, a=3) is None


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

def test_once_caches():
    calls = []

    @once
    def expensive():
        calls.append(1)
        return 'initialized'

    assert expensive() == 'initialized'
    assert expensive() == 'initialized'
    assert len(calls) == 1


def test_once_preserves_metadata():
    @once
    def func():
        """docs"""
        return 1

    assert func.__name__ == 'func'
    assert func.__doc__ == 'docs'


def test_once_thread_safe():
    calls = []
    barrier = threading.Barrier(5)

    @once
    def expensive():
        calls.append(1)
        return 42

    results = []

    def worker():
        barrier.wait()
        results.append(expensive())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert results == [42, 42, 42, 42, 42]


def test_no_default_sentinel():
    fb = FunctionBuilder('f', args=['a'])
    fb.add_arg('b')
    assert fb.defaults is None
