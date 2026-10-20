import inspect
import sys
import types
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
    total_ordering,
)


# ---------------------------------------------------------------------------
# inspect_formatargspec
# ---------------------------------------------------------------------------

def test_inspect_formatargspec_basic():
    result = inspect_formatargspec(['a', 'b'])
    assert result == '(a, b)'


def test_inspect_formatargspec_with_defaults():
    result = inspect_formatargspec(['a', 'b'], defaults=(1,))
    assert result == '(a, b=1)'


def test_inspect_formatargspec_varargs_varkw():
    result = inspect_formatargspec(['a'], varargs='args', varkw='kwargs')
    assert result == '(a, *args, **kwargs)'


def test_inspect_formatargspec_kwonly_only():
    # no varargs but kwonly -> introduces bare '*'
    result = inspect_formatargspec(['a'], kwonlyargs=['b'])
    assert result == '(a, *, b)'


def test_inspect_formatargspec_kwonly_defaults():
    result = inspect_formatargspec(
        ['a'], kwonlyargs=['b', 'c'], kwonlydefaults={'b': 1})
    assert result == '(a, *, b=1, c)'


def test_inspect_formatargspec_varargs_with_kwonly():
    result = inspect_formatargspec(
        ['a'], varargs='args', kwonlyargs=['b'], kwonlydefaults={'b': 2})
    assert result == '(a, *args, b=2)'


def test_inspect_formatargspec_annotations():
    result = inspect_formatargspec(
        ['a'], annotations={'a': int, 'return': str})
    assert result == '(a: int) -> str'


def test_inspect_formatargspec_annotations_varargs():
    result = inspect_formatargspec(
        [], varargs='args', annotations={'args': int})
    assert result == '(*args: int)'


def test_inspect_formatargspec_custom_formatarg():
    result = inspect_formatargspec(
        ['a'], formatarg=lambda n: n.upper())
    assert result == '(A)'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def test_get_module_callables_current_module():
    types_map, funcs_map = get_module_callables(sys.modules[__name__])
    # this test function itself should be among funcs
    assert 'test_get_module_callables_current_module' in funcs_map


def test_get_module_callables_by_string():
    mod = types.ModuleType('fake_mod_for_callables')

    def inner():
        pass

    inner.__module__ = 'fake_mod_for_callables'
    mod.inner = inner
    mod.some_int = 5

    class Klass:
        pass

    Klass.__module__ = 'fake_mod_for_callables'
    mod.Klass = Klass

    mod.__dict__['inner'] = inner
    mod.__dict__['Klass'] = Klass

    key = 'fake_mod_for_callables'
    sys.modules[key] = mod
    try:
        types_map, funcs_map = get_module_callables(key)
    finally:
        del sys.modules[key]

    assert types_map == {'Klass': Klass}
    assert funcs_map == {'inner': inner}


def test_get_module_callables_ignore():
    mod = types.ModuleType('fake_mod_ignore')

    def keep():
        pass

    def skip():
        pass

    keep.__module__ = 'fake_mod_ignore'
    skip.__module__ = 'fake_mod_ignore'
    mod.__dict__['keep'] = keep
    mod.__dict__['skip'] = skip

    key = 'fake_mod_ignore'
    sys.modules[key] = mod
    try:
        types_map, funcs_map = get_module_callables(
            key, ignore=lambda name: name == 'skip')
    finally:
        del sys.modules[key]

    assert 'skip' not in funcs_map
    assert 'keep' in funcs_map


def test_get_module_callables_skips_foreign_module():
    mod = types.ModuleType('fake_mod_foreign')

    def foreign():
        pass

    foreign.__module__ = 'some_other_module'
    mod.__dict__['foreign'] = foreign

    key = 'fake_mod_foreign'
    sys.modules[key] = mod
    try:
        types_map, funcs_map = get_module_callables(key)
    finally:
        del sys.modules[key]

    assert types_map == {}
    assert funcs_map == {}


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items_int():
    names = [k for k, v in mro_items(int)]
    assert 'denominator' in names
    assert 'numerator' in names


def test_mro_items_respects_mro_order():
    class A:
        x = 1

    class B(A):
        y = 2

    keys = [k for k, v in mro_items(B)]
    assert 'y' in keys
    assert 'x' in keys
    # B's own dict should come before A's
    assert keys.index('y') < keys.index('x')


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

def test_dir_dict_includes_parent_attrs():
    class Base:
        base_attr = 'b'

    class Child(Base):
        child_attr = 'c'

    obj = Child()
    result = dir_dict(obj)
    assert result['base_attr'] == 'b'
    assert result['child_attr'] == 'c'


def test_dir_dict_raise_exc():
    class Bad:
        @property
        def boom(self):
            raise RuntimeError('boom')

    obj = Bad()
    # default does not raise
    result = dir_dict(obj)
    assert 'boom' not in result
    with pytest.raises(RuntimeError):
        dir_dict(obj, raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_returns_new_function():
    def orig():
        return 42

    orig.attr = 'hello'
    cp = copy_function(orig)
    assert cp is not orig
    assert cp() == 42
    assert cp.attr == 'hello'
    assert cp.__name__ == 'orig'


def test_copy_function_no_copy_dict():
    def orig():
        return 7

    orig.attr = 'hello'
    cp = copy_function(orig, copy_dict=False)
    assert not hasattr(cp, 'attr')
    assert cp() == 7


def test_copy_function_self_reference():
    def func():
        return func

    cp = copy_function(func)
    assert cp() is func


def test_copy_function_with_closure():
    x = 10

    def orig():
        return x

    cp = copy_function(orig)
    assert cp() == 10


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
    assert b < a
    assert not (b > a)
    assert b < c
    assert not (a < c)
    assert not (c > a)


def test_partial_ordering_does_not_override():
    def my_lt(self, other):
        return 'custom'

    @partial_ordering
    class Foo:
        __lt__ = my_lt

        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

    assert Foo.__lt__ is my_lt
    assert Foo.__eq__ is not None


def test_partial_ordering_eq():
    @partial_ordering
    class S(set):
        def __le__(self, other):
            return self.issubset(other)

        def __ge__(self, other):
            return self.issuperset(other)

    assert S([1, 2]) == S([1, 2])
    assert not (S([1, 2]) == S([1, 2, 3]))


# ---------------------------------------------------------------------------
# InstancePartial & CachedInstancePartial
# ---------------------------------------------------------------------------

def test_instance_partial_as_method():
    def mul(self, x):
        return self.factor * x

    class Obj:
        factor = 3
        triple = InstancePartial(mul)

    obj = Obj()
    assert obj.triple(4) == 12


def test_instance_partial_get_descriptor():
    def f(x):
        return x

    p = InstancePartial(f)
    bound = p.__get__(None, object)
    assert bound == p


def test_instance_partial_partialmethod_property():
    def f(a, b):
        return a + b

    p = InstancePartial(f, 1)
    pm = p.__partialmethod__
    assert isinstance(pm, functools.partialmethod)
    assert p._partialmethod is pm


def test_cached_instance_partial_as_method():
    def mul(self, x):
        return self.factor * x

    class Obj:
        factor = 5
        times = CachedInstancePartial(mul)

    obj = Obj()
    assert obj.times(2) == 10
    # second access should return the cached bound method
    first = obj.times
    second = obj.times
    assert first == second


def test_cached_instance_partial_caches_on_instance():
    def f(self):
        return 1

    class Obj:
        m = CachedInstancePartial(f)

    obj = Obj()
    bound = obj.m
    assert 'm' in obj.__dict__
    assert obj.__dict__['m'] is bound


def test_cached_instance_partial_set_name():
    def f(self):
        return 1

    class Obj:
        meth = CachedInstancePartial(f)

    assert Obj.__dict__['meth'].__name__ == 'meth'


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_basic():
    assert format_invocation('func', args=(1, 2), kwargs={'c': 3}) == \
        'func(1, 2, c=3)'


def test_format_invocation_only_args():
    assert format_invocation('a_func', args=(1,)) == 'a_func(1)'


def test_format_invocation_only_kwargs_list():
    assert format_invocation(
        'kw_func', kwargs=[('a', 1), ('b', 2)]) == 'kw_func(a=1, b=2)'


def test_format_invocation_no_args():
    assert format_invocation('empty') == 'empty()'


def test_format_invocation_sorted_dict():
    assert format_invocation(
        'f', kwargs={'b': 2, 'a': 1}) == 'f(a=1, b=2)'


def test_format_invocation_custom_repr():
    assert format_invocation(
        'f', args=('x',), repr=str) == 'f(x)'


def test_format_invocation_unexpected_kwarg():
    with pytest.raises(TypeError):
        format_invocation('f', bogus=1)


# ---------------------------------------------------------------------------
# format_exp_repr
# ---------------------------------------------------------------------------

class _Flag:
    def __init__(self, length, width, depth=None):
        self.length = length
        self.width = width
        self.depth = depth


def test_format_exp_repr_positional_only():
    flag = _Flag(5, 10)
    out = format_exp_repr(flag, ['length', 'width'], [], ['depth'])
    assert out == '_Flag(5, 10)'


def test_format_exp_repr_with_required():
    flag = _Flag(5, 15, 2)
    out = format_exp_repr(
        flag, ['length'], ['width', 'depth'])
    assert out == '_Flag(5, width=15, depth=2)'


def test_format_exp_repr_opt_excludes_none():
    flag = _Flag(5, 10)
    out = format_exp_repr(
        flag, ['length', 'width'], [], ['depth'])
    assert 'depth' not in out


def test_format_exp_repr_custom_opt_key():
    flag = _Flag(5, 10, 0)
    out = format_exp_repr(
        flag, ['length'], [], ['width', 'depth'],
        opt_key=lambda v: False)
    assert 'width=10' in out
    assert 'depth=0' in out


def test_format_exp_repr_dedupe_names():
    flag = _Flag(1, 2)
    out = format_exp_repr(flag, ['length'], ['width'], ['width'])
    assert out == '_Flag(1, width=2)'


def test_format_exp_repr_opt_key_not_callable():
    flag = _Flag(1, 2)
    with pytest.raises(AssertionError):
        format_exp_repr(flag, ['length'], opt_key='not callable')


# ---------------------------------------------------------------------------
# format_nonexp_repr
# ---------------------------------------------------------------------------

def test_format_nonexp_repr_basic():
    flag = _Flag(5, 10)
    out = format_nonexp_repr(flag, ['length', 'width'], ['depth'])
    assert out == '<_Flag length=5 width=10>'


def test_format_nonexp_repr_id_fallback():
    flag = _Flag(5, 10)
    out = format_nonexp_repr(flag)
    assert out.startswith('<_Flag id=')
    assert out.endswith('>')
    assert str(id(flag)) in out


def test_format_nonexp_repr_opt_excluded():
    flag = _Flag(5, 10)
    out = format_nonexp_repr(flag, ['length'], ['width', 'depth'])
    assert 'width=10' in out
    assert 'depth' not in out


def test_format_nonexp_repr_opt_key_not_callable():
    flag = _Flag(5, 10)
    with pytest.raises(AssertionError):
        format_nonexp_repr(flag, ['length'], opt_key=5)


# ---------------------------------------------------------------------------
# wraps & update_wrapper
# ---------------------------------------------------------------------------

def test_wraps_copies_name_and_doc():
    def orig(a, b=2):
        """orig doc"""
        return a + b

    @wraps(orig)
    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    assert wrapper.__name__ == 'orig'
    assert wrapper.__doc__ == 'orig doc'
    assert wrapper(1) == 3
    assert wrapper(1, b=5) == 6
    assert wrapper.__wrapped__ is orig


def test_wraps_signature_matches():
    def orig(a, b=2):
        return a + b

    @wraps(orig)
    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    sig = inspect.signature(wrapper)
    assert list(sig.parameters) == ['a', 'b']
    assert sig.parameters['b'].default == 2


def test_update_wrapper_injected():
    def orig(a, b):
        return a + b

    def wrapper(x, a, b):
        return orig(a, b)

    new = update_wrapper(wrapper, orig, injected=['x'])
    # x should no longer appear in the new signature
    assert 'x' not in inspect.signature(new).parameters
    assert new(1, 2) == 3


def test_update_wrapper_expected():
    def orig(a):
        return a

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    new = update_wrapper(wrapper, orig, expected=[('extra', 99)])
    params = inspect.signature(new).parameters
    assert 'extra' in params
    assert params['extra'].default == 99


def test_update_wrapper_expected_str():
    def orig(a):
        return a

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    new = update_wrapper(wrapper, orig, expected='extra')
    params = inspect.signature(new).parameters
    assert 'extra' in params
    assert params['extra'].default is NO_DEFAULT


def test_update_wrapper_expected_dict():
    def orig(a):
        return a

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    new = update_wrapper(wrapper, orig, expected={'extra': 7})
    params = inspect.signature(new).parameters
    assert params['extra'].default == 7


def test_update_wrapper_hide_wrapped():
    def orig():
        return 1

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    new = update_wrapper(wrapper, orig, hide_wrapped=True)
    assert not hasattr(new, '__wrapped__')


def test_update_wrapper_unexpected_kwarg():
    def orig():
        return 1

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    with pytest.raises(TypeError):
        update_wrapper(wrapper, orig, bogus_kwarg=1)


def test_update_wrapper_classmethod_error():
    def orig():
        return 1

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    with pytest.raises(TypeError):
        update_wrapper(wrapper, classmethod(orig))


def test_update_wrapper_injected_varkw():
    def orig(a):
        return a

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    new = update_wrapper(wrapper, orig, injected='missing')
    # missing arg is caught by varkw, so no error
    assert 'missing' not in inspect.signature(new).parameters


def test_update_wrapper_injected_missing_no_varkw():
    def orig(a):
        return a

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, orig, injected='missing',
                      inject_to_varkw=False)


def test_update_wrapper_does_not_mutate_original():
    def orig(a):
        return a

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    orig_sig_before = inspect.signature(orig)
    update_wrapper(wrapper, orig, expected='extra')
    assert inspect.signature(orig) == orig_sig_before


def test_update_wrapper_async():
    async def orig(a, b=2):
        return a + b

    async def wrapper(*args, **kwargs):
        return await orig(*args, **kwargs)

    new = update_wrapper(wrapper, orig)
    assert inspect.iscoroutinefunction(new)
    # run it
    import asyncio
    assert asyncio.run(new(1, 3)) == 4


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_function_builder_basic():
    fb = FunctionBuilder('return_five', doc='returns 5', body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__name__ == 'return_five'
    assert f.__doc__ == 'returns 5'


def test_function_builder_varkw():
    fb = FunctionBuilder('return_five', body='return 5')
    fb.varkw = 'kw'
    f = fb.get_func()
    assert f(ignored_arg='ignored_val') == 5


def test_function_builder_default_body():
    fb = FunctionBuilder('noop_func')
    f = fb.get_func()
    assert f() is None


def test_function_builder_with_args():
    fb = FunctionBuilder('add', args=['a', 'b'], body='return a + b')
    f = fb.get_func()
    assert f(1, 2) == 3


def test_function_builder_with_defaults():
    fb = FunctionBuilder('add', args=['a', 'b'], defaults=(2,),
                         body='return a + b')
    f = fb.get_func()
    assert f(1) == 3
    assert f(1, 5) == 6


def test_function_builder_kwonly():
    fb = FunctionBuilder('f', args=['a'], kwonlyargs=['b'],
                         kwonlydefaults={'b': 10},
                         body='return a + b')
    f = fb.get_func()
    assert f(1) == 11
    assert f(1, b=2) == 3


def test_function_builder_annotations():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int},
                         body='return a')
    f = fb.get_func()
    assert f.__annotations__ == {'a': int}


def test_function_builder_source_attr():
    fb = FunctionBuilder('f', body='return 1')
    f = fb.get_func()
    assert hasattr(f, '__source__')
    assert 'def f' in f.__source__


def test_function_builder_no_source():
    fb = FunctionBuilder('f', body='return 1')
    f = fb.get_func(add_source=False)
    assert not hasattr(f, '__source__')


def test_function_builder_from_func():
    def orig(a, b=2, *args, c=3, **kwargs):
        return a

    fb = FunctionBuilder.from_func(orig)
    assert fb.name == 'orig'
    assert fb.args == ['a', 'b']
    assert fb.varargs == 'args'
    assert fb.varkw == 'kwargs'
    assert fb.kwonlyargs == ['c']
    assert fb.kwonlydefaults == {'c': 3}
    assert fb.defaults == (2,)


def test_function_builder_from_func_not_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(5)


def test_function_builder_get_sig_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    assert fb.get_sig_str() == '(a, b=1)'


def test_function_builder_get_sig_str_no_annotations():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int})
    assert fb.get_sig_str(with_annotations=False) == '(a)'


def test_function_builder_get_invocation_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    # b has a default, gets keyword forwarded (no varargs)
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,),
                         kwonlyargs=['c'], kwonlydefaults={'c': 5})
    dd = fb.get_defaults_dict()
    assert dd == {'b': 1, 'c': 5}


def test_function_builder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,),
                         kwonlyargs=['c'])
    assert fb.get_arg_names() == ('a', 'b', 'c')
    assert fb.get_arg_names(only_required=True) == ('a', 'c')


def test_function_builder_add_arg():
    fb = FunctionBuilder('f')
    fb.add_arg('a')
    assert fb.args == ['a']
    fb.add_arg('b', 5)
    assert fb.args == ['a', 'b']
    assert fb.defaults == (5,)


def test_function_builder_add_arg_kwonly():
    fb = FunctionBuilder('f')
    fb.add_arg('k', 1, kwonly=True)
    assert fb.kwonlyargs == ['k']
    assert fb.kwonlydefaults == {'k': 1}


def test_function_builder_add_existing_arg():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_function_builder_add_existing_kwonly_arg():
    fb = FunctionBuilder('f', kwonlyargs=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a', kwonly=True)


def test_function_builder_remove_arg():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    fb.remove_arg('b')
    assert fb.args == ['a']
    assert fb.defaults == ()


def test_function_builder_remove_kwonly_arg():
    fb = FunctionBuilder('f', kwonlyargs=['k'], kwonlydefaults={'k': 1})
    fb.remove_arg('k')
    assert fb.kwonlyargs == []
    assert fb.kwonlydefaults == {}


def test_function_builder_remove_missing_arg():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(MissingArgument):
        fb.remove_arg('missing')


def test_function_builder_unexpected_kwarg():
    with pytest.raises(TypeError):
        FunctionBuilder('f', not_a_real_arg=1)


def test_function_builder_posonly_forwarding():
    def orig(a, /, b):
        return a + b

    fb = FunctionBuilder.from_func(orig)
    assert fb.posonlyargs == ['a']
    inv = fb.get_invocation_str()
    # positional-only must be forwarded positionally
    assert 'a' in inv.split(',')
    f = fb.get_func({'orig': orig}, with_dict=False)
    assert f(1, 2) == 3


def test_function_builder_indent():
    fb = FunctionBuilder('f', indent=2,
                         body='x = 1\nreturn x')
    f = fb.get_func()
    assert f() == 1


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
    assert noop(1, 2, a=3, b=4) is None


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


def test_once_returns_cached_value():
    @once
    def f():
        return object()

    first = f()
    second = f()
    assert first is second


def test_once_preserves_name():
    @once
    def my_func():
        return 1

    assert my_func.__name__ == 'my_func'


def test_once_thread_safe():
    calls = []

    @once
    def setup():
        calls.append(1)
        return 42

    results = []
    barrier = threading.Barrier(5)

    def worker():
        barrier.wait()
        results.append(setup())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [42] * 5
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# total_ordering re-export sanity
# ---------------------------------------------------------------------------

def test_total_ordering_reexport():
    assert total_ordering is functools.total_ordering
