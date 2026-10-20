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

def test_inspect_formatargspec_empty():
    assert inspect_formatargspec([]) == '()'


def test_inspect_formatargspec_simple_args():
    assert inspect_formatargspec(['a', 'b']) == '(a, b)'


def test_inspect_formatargspec_defaults():
    assert inspect_formatargspec(['a', 'b'], defaults=(1,)) == '(a, b=1)'


def test_inspect_formatargspec_varargs():
    assert inspect_formatargspec(['a'], varargs='args') == '(a, *args)'


def test_inspect_formatargspec_varkw():
    assert inspect_formatargspec(['a'], varkw='kwargs') == '(a, **kwargs)'


def test_inspect_formatargspec_kwonly_no_varargs_uses_star():
    assert inspect_formatargspec([], kwonlyargs=['x']) == '(*, x)'


def test_inspect_formatargspec_kwonly_defaults():
    res = inspect_formatargspec([], kwonlyargs=['x'], kwonlydefaults={'x': 3})
    assert res == '(*, x=3)'


def test_inspect_formatargspec_annotations():
    res = inspect_formatargspec(
        ['a'], annotations={'a': int, 'return': str}
    )
    assert res == '(a: int) -> str'


def test_inspect_formatargspec_custom_formatters():
    res = inspect_formatargspec(
        ['a', 'b'],
        defaults=(5,),
        formatarg=lambda n: n.upper(),
        formatvalue=lambda v: '<%s>' % v,
    )
    assert res == '(A, B<5>)'


def test_inspect_formatargspec_varargs_with_annotation():
    res = inspect_formatargspec(['a'], varargs='args',
                                annotations={'args': int})
    assert res == '(a, *args: int)'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def _build_module():
    mod = types.ModuleType('_test_funcutils_mod')

    def _local_func():
        return 1

    _local_func.__module__ = '_test_funcutils_mod'
    mod.local_func = _local_func

    class LocalClass:
        pass

    LocalClass.__module__ = '_test_funcutils_mod'
    mod.LocalClass = LocalClass

    def _other_func():
        return 2

    _other_func.__module__ = 'some_other_module'
    mod.other_func = _other_func

    mod.not_callable = 42
    return mod


def test_get_module_callables_by_instance():
    mod = _build_module()
    types_map, funcs_map = get_module_callables(mod)
    assert 'LocalClass' in types_map
    assert types_map['LocalClass'] is mod.LocalClass
    assert 'local_func' in funcs_map
    assert funcs_map['local_func'] is mod.local_func
    assert 'other_func' not in funcs_map
    assert 'not_callable' not in funcs_map


def test_get_module_callables_by_name(monkeypatch):
    mod = _build_module()
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    types_map, funcs_map = get_module_callables(mod.__name__)
    assert 'LocalClass' in types_map
    assert 'local_func' in funcs_map


def test_get_module_callables_ignore():
    mod = _build_module()
    types_map, funcs_map = get_module_callables(
        mod, ignore=lambda n: n in ('LocalClass', 'local_func'))
    assert 'LocalClass' not in types_map
    assert 'local_func' not in funcs_map


def test_get_module_callables_skips_errors(monkeypatch):
    mod = types.ModuleType('_test_funcutils_mod_err')

    class Boom:
        def __getattr__(self, name):
            raise RuntimeError('nope')

    mod.Boom = Boom
    # it's a class so has __module__; also add a plain attribute that raises
    types_map, funcs_map = get_module_callables(mod)
    assert 'Boom' in types_map


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items_includes_inherited():
    class A:
        x = 1

    class B(A):
        y = 2

    items = dict(mro_items(B))
    assert items['x'] == 1
    assert items['y'] == 2


def test_mro_items_order():
    class A:
        pass

    class B(A):
        pass

    d = dict(mro_items(B))
    # object's dict entries present
    assert '__class__' in d


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

def test_dir_dict_includes_parent_attrs():
    class A:
        a = 1

    class B(A):
        b = 2

    b = B()
    d = dir_dict(b)
    assert d['a'] == 1
    assert d['b'] == 2


def test_dir_dict_raise_exc():
    class Bad:
        @property
        def boom(self):
            raise RuntimeError('bad')

    bad = Bad()
    # by default, errors are skipped
    d = dir_dict(bad)
    assert 'boom' not in d
    with pytest.raises(RuntimeError):
        dir_dict(bad, raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_basic():
    def orig(a, b=2):
        return a + b

    orig.attr = 'x'
    copy = copy_function(orig)
    assert copy is not orig
    assert copy(1) == 3
    assert copy.__name__ == orig.__name__
    assert copy.attr == 'x'


def test_copy_function_no_dict():
    def orig():
        return 1

    orig.attr = 'x'
    copy = copy_function(orig, copy_dict=False)
    assert not hasattr(copy, 'attr')


def test_copy_function_closure():
    def outer():
        val = 10

        def inner():
            return val

        return inner

    f = outer()
    c = copy_function(f)
    assert c() == 10


def test_copy_function_kwdefaults():
    def orig(a, *, b=3):
        return a + b

    c = copy_function(orig)
    assert c(1) == 4
    assert c(1, b=10) == 11


# ---------------------------------------------------------------------------
# partial_ordering
# ---------------------------------------------------------------------------

@partial_ordering
class _MySet(set):
    def __le__(self, other):
        return self.issubset(other)

    def __ge__(self, other):
        return self.issuperset(other)


def test_partial_ordering_lt():
    a = _MySet([1, 2, 3])
    b = _MySet([1, 2])
    assert (b < a) is True
    assert (b > a) is False


def test_partial_ordering_incomparable():
    a = _MySet([1, 2, 3])
    c = _MySet([1, 2, 4])
    assert (a < c) is False
    assert (c > a) is False
    assert (a == c) is False


def test_partial_ordering_eq():
    a = _MySet([1, 2])
    b = _MySet([2, 1])
    assert (a == b) is True


def test_partial_ordering_does_not_override():
    @partial_ordering
    class HasLt:
        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

        def __lt__(self, other):
            return 'custom'

    assert HasLt().__lt__(None) == 'custom'


# ---------------------------------------------------------------------------
# InstancePartial / CachedInstancePartial
# ---------------------------------------------------------------------------

def _make_add(a, b):
    return a + b


class _InstPartialHolder:
    add_five = InstancePartial(_make_add, 5)


def test_instance_partial_as_method():
    obj = _InstPartialHolder()
    assert obj.add_five(10) == 15


def test_instance_partial_descriptor_access_on_class():
    obj = _InstPartialHolder()
    bound = obj.add_five
    assert isinstance(bound, types.MethodType)


class _CachedPartialHolder:
    add_five = CachedInstancePartial(_make_add, 5)


def test_cached_instance_partial():
    obj = _CachedPartialHolder()
    assert obj.add_five(10) == 15
    # cached
    first = obj.add_five
    second = obj.add_five
    assert first == second


def test_cached_instance_partial_set_name():
    assert _CachedPartialHolder.__dict__['add_five'].__name__ == 'add_five'
    obj = _CachedPartialHolder()
    assert obj.add_five(1) == 6
    assert obj.__dict__['add_five'] is not None


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_basic():
    assert format_invocation('func', args=(1, 2), kwargs={'c': 3}) == 'func(1, 2, c=3)'


def test_format_invocation_no_kwargs():
    assert format_invocation('a_func', args=(1,)) == 'a_func(1)'


def test_format_invocation_kwarg_list():
    assert format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]) == 'kw_func(a=1, b=2)'


def test_format_invocation_empty():
    assert format_invocation() == '()'


def test_format_invocation_kwargs_sorted():
    assert format_invocation('f', kwargs={'b': 2, 'a': 1}) == 'f(a=1, b=2)'


def test_format_invocation_custom_repr():
    assert format_invocation('f', args=(1,), repr=lambda v: 'X') == 'f(X)'


def test_format_invocation_unexpected_kw():
    with pytest.raises(TypeError):
        format_invocation('f', bad_kw=1)


# ---------------------------------------------------------------------------
# format_exp_repr
# ---------------------------------------------------------------------------

class _Flag:
    def __init__(self, length, width, depth=None):
        self.length = length
        self.width = width
        self.depth = depth


def test_format_exp_repr_basic():
    flag = _Flag(5, 10)
    assert format_exp_repr(flag, ['length', 'width'], [], ['depth']) == 'Flag(5, 10)'


def test_format_exp_repr_with_kwargs():
    flag = _Flag(5, 15, 2)
    res = format_exp_repr(flag, ['length'], ['width', 'depth'])
    assert res == 'Flag(5, width=15, depth=2)'


def test_format_exp_repr_dedupes():
    flag = _Flag(1, 2)
    res = format_exp_repr(flag, [], ['length', 'length'])
    assert res == 'Flag(length=1)'


def test_format_exp_repr_custom_opt_key():
    flag = _Flag(1, 2, 3)
    res = format_exp_repr(flag, [], [], ['depth'], opt_key=lambda v: v == 3)
    assert res == 'Flag()'


# ---------------------------------------------------------------------------
# format_nonexp_repr
# ---------------------------------------------------------------------------

def test_format_nonexp_repr_basic():
    flag = _Flag(5, 10)
    assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == '<Flag length=5 width=10>'


def test_format_nonexp_repr_fallback_id():
    flag = _Flag(5, 10)
    res = format_nonexp_repr(flag)
    assert res == '<Flag id=%s>' % id(flag)


def test_format_nonexp_repr_custom_opt_key():
    flag = _Flag(1, 2, 3)
    res = format_nonexp_repr(flag, [], ['depth'], opt_key=lambda v: v == 3)
    assert res == '<Flag id=%s>' % id(flag)


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_update_wrapper_basic():
    def orig():
        "original doc"
        return 'val'

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    wrapped = update_wrapper(wrapper, orig)
    assert wrapped() == 'val'
    assert wrapped.__name__ == 'orig'
    assert wrapped.__doc__ == 'original doc'
    assert wrapped.__wrapped__ is orig


def test_update_wrapper_injected():
    def orig(a, b):
        return a + b

    def wrapper(b):
        return orig(10, b)

    wrapped = update_wrapper(wrapper, orig, injected=['a'])
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['b']
    assert wrapped(5) == 15


def test_update_wrapper_injected_missing_raises():
    def orig(a):
        return a

    def wrapper():
        return orig(1)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, orig, injected=['nonexistent'])


def test_update_wrapper_injected_varkw():
    def orig(**kwargs):
        return kwargs

    def wrapper(**kwargs):
        return orig(**kwargs)

    wrapped = update_wrapper(wrapper, orig, injected=['whatever'])
    assert wrapped() == {}


def test_update_wrapper_expected():
    def orig(a):
        return a

    def wrapper(a, extra):
        return orig(a) + extra

    wrapped = update_wrapper(wrapper, orig, expected='extra',
                             )
    sig = inspect.signature(wrapped)
    assert 'extra' in sig.parameters
    assert wrapped(1, 2) == 3


def test_update_wrapper_hide_wrapped():
    def orig():
        return 1

    def wrapper():
        return orig()

    wrapped = update_wrapper(wrapper, orig, hide_wrapped=True)
    assert not hasattr(wrapped, '__wrapped__')


def test_update_wrapper_unexpected_kw():
    def orig():
        return 1

    def wrapper():
        return 1

    with pytest.raises(TypeError):
        update_wrapper(wrapper, orig, bad_kw=True)


def test_update_wrapper_classmethod_raises():
    class C:
        @classmethod
        def cm(cls):
            pass

    def wrapper():
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, C.__dict__['cm'])


def test_update_wrapper_update_dict_false():
    def orig():
        return 1

    orig.custom = 'x'

    def wrapper():
        return orig()

    wrapped = update_wrapper(wrapper, orig, update_dict=False)
    assert not hasattr(wrapped, 'custom')


def test_wraps_decorator():
    @wraps(_make_add)
    def wrapper(*args, **kwargs):
        return _make_add(*args, **kwargs)

    assert wrapper.__name__ == '_make_add'
    assert wrapper(1, 2) == 3
    assert wrapper.__wrapped__ is _make_add


def test_update_wrapper_async():
    async def orig(a):
        return a

    async def wrapper(*args, **kwargs):
        return await orig(*args, **kwargs)

    wrapped = update_wrapper(wrapper, orig)
    assert inspect.iscoroutinefunction(wrapped)
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['a']


# ---------------------------------------------------------------------------
# _parse_wraps_expected via update_wrapper
# ---------------------------------------------------------------------------

def test_expected_mapping():
    def orig():
        return 'o'

    def wrapper(x=None, y=None):
        return (orig(), x, y)

    wrapped = update_wrapper(wrapper, orig, expected={'x': 5, 'y': 6})
    sig = inspect.signature(wrapped)
    assert sig.parameters['x'].default == 5
    assert sig.parameters['y'].default == 6


def test_expected_list_of_pairs():
    def orig():
        return 'o'

    def wrapper(z=None):
        return (orig(), z)

    wrapped = update_wrapper(wrapper, orig, expected=[('z', 9)])
    sig = inspect.signature(wrapped)
    assert sig.parameters['z'].default == 9


def test_expected_invalid():
    def orig():
        return 1

    def wrapper():
        return 1

    with pytest.raises(ValueError):
        update_wrapper(wrapper, orig, expected=123)


def test_expected_bad_argname():
    def orig():
        return 1

    def wrapper():
        return 1

    with pytest.raises(ValueError):
        update_wrapper(wrapper, orig, expected=[(123, 4)])


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_function_builder_basic():
    fb = FunctionBuilder('return_five', doc='returns the integer 5',
                         body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__doc__ == 'returns the integer 5'
    assert f.__name__ == 'return_five'


def test_function_builder_varkw():
    fb = FunctionBuilder('return_five', body='return 5')
    fb.varkw = 'kw'
    f = fb.get_func()
    assert f(ignored_arg='ignored_val') == 5


def test_function_builder_args_and_defaults():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,),
                         body='return a + b')
    f = fb.get_func()
    assert f(1) == 3
    assert f(1, 10) == 11


def test_function_builder_get_func_with_dict():
    fb = FunctionBuilder('f', body='return 1', dict={'custom': 'val'})
    f = fb.get_func()
    assert f.custom == 'val'


def test_function_builder_get_func_without_dict():
    fb = FunctionBuilder('f', body='return 1', dict={'custom': 'val'})
    f = fb.get_func(with_dict=False)
    assert not hasattr(f, 'custom')


def test_function_builder_source_attr():
    fb = FunctionBuilder('f', body='return 1')
    f = fb.get_func()
    assert '__source__' in f.__dict__
    assert 'def f' in f.__source__

    f2 = fb.get_func(add_source=False)
    assert '__source__' not in f2.__dict__


def test_function_builder_from_func():
    def orig(a, b=3, *args, c=4, **kwargs):
        return a, b, args, c, kwargs

    fb = FunctionBuilder.from_func(orig)
    assert fb.name == 'orig'
    assert list(fb.args) == ['a', 'b']
    assert fb.varargs == 'args'
    assert fb.varkw == 'kwargs'
    assert list(fb.kwonlyargs) == ['c']
    assert fb.defaults == (3,)
    assert fb.kwonlydefaults == {'c': 4}


def test_function_builder_from_func_not_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(42)


def test_function_builder_from_func_partial():
    def orig(a, b):
        return a + b

    p = functools.partial(orig, 1)
    fb = FunctionBuilder.from_func(p)
    assert fb.name == 'orig'


def test_function_builder_from_func_async():
    async def orig():
        return 1

    fb = FunctionBuilder.from_func(orig)
    assert fb.is_async


def test_function_builder_posonly():
    def orig(a, b, /, c):
        return a + b + c

    fb = FunctionBuilder.from_func(orig)
    assert fb.posonlyargs == ['a', 'b']
    f = fb.get_func({'orig': orig})
    # invocation string should keep a and b positional
    inv = fb.get_invocation_str()
    assert inv.startswith('a, b')


def test_function_builder_unexpected_kwargs():
    with pytest.raises(TypeError):
        FunctionBuilder('f', not_a_real_arg=1)


def test_function_builder_add_arg():
    fb = FunctionBuilder('f', body='return a, b')
    fb.add_arg('a')
    fb.add_arg('b', default=5)
    f = fb.get_func()
    assert f(1) == (1, 5)


def test_function_builder_add_arg_existing():
    fb = FunctionBuilder('f', args=['a'], body='return a')
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_function_builder_add_kwonly_arg():
    fb = FunctionBuilder('f', body='return a')
    fb.add_arg('a', default=7, kwonly=True)
    assert 'a' in fb.kwonlyargs
    assert fb.kwonlydefaults == {'a': 7}


def test_function_builder_add_arg_kwonly_existing():
    fb = FunctionBuilder('f', kwonlyargs=['x'], body='return x')
    with pytest.raises(ExistingArgument):
        fb.add_arg('x', kwonly=True)


def test_function_builder_remove_arg():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3),
                         body='return a')
    fb.remove_arg('b')
    assert list(fb.args) == ['a', 'c']
    assert fb.defaults == (3,)


def test_function_builder_remove_kwonly_arg():
    fb = FunctionBuilder('f', kwonlyargs=['x'], kwonlydefaults={'x': 1},
                         body='return 1')
    fb.remove_arg('x')
    assert 'x' not in fb.kwonlyargs
    assert fb.kwonlydefaults == {}


def test_function_builder_remove_missing_raises():
    fb = FunctionBuilder('f', args=['a'], body='return a')
    with pytest.raises(MissingArgument) as excinfo:
        fb.remove_arg('zzz')
    assert excinfo.value.arg_name == 'zzz'


def test_function_builder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,),
                         kwonlyargs=['c'], kwonlydefaults={'c': 2},
                         body='return 1')
    d = fb.get_defaults_dict()
    assert d == {'b': 1, 'c': 2}


def test_function_builder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,),
                         kwonlyargs=['c'], body='return 1')
    assert fb.get_arg_names() == ('a', 'b', 'c')
    assert fb.get_arg_names(only_required=True) == ('a', 'c')


def test_function_builder_get_sig_str():
    fb = FunctionBuilder('f', args=['a'], defaults=(1,), body='return a')
    assert fb.get_sig_str() == '(a)'


def test_function_builder_lambda_name():
    fb = FunctionBuilder('<lambda>', body='return 1')
    f = fb.get_func()
    assert f.__name__ == '<lambda>'


def test_function_builder_get_invocation_str_with_varargs():
    fb = FunctionBuilder('f', args=['a', 'b'], varargs='args', body='return 1')
    assert fb.get_invocation_str() == 'a, b, *args'


def test_function_builder_get_invocation_str_keyword_forwarding():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,), body='return 1')
    # b has default and no varargs -> forwarded as keyword
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_get_invocation_str_varkw():
    fb = FunctionBuilder('f', args=['a'], varkw='kwargs', body='return 1')
    assert fb.get_invocation_str() == 'a, **kwargs'


# ---------------------------------------------------------------------------
# MissingArgument / ExistingArgument
# ---------------------------------------------------------------------------

def test_exception_classes():
    assert issubclass(MissingArgument, ValueError)
    assert issubclass(ExistingArgument, ValueError)


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------

def test_noop_returns_none():
    assert noop() is None
    assert noop(1, 2, x=3) is None


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

def test_once_caches():
    calls = []

    @once
    def setup():
        calls.append(1)
        return 'val'

    assert setup() == 'val'
    assert setup() == 'val'
    assert calls == [1]


def test_once_preserves_name():
    @once
    def my_setup():
        return 1

    assert my_setup.__name__ == 'my_setup'


def test_once_thread_safety():
    counter = []
    barrier = threading.Barrier(5)

    @once
    def setup():
        counter.append(1)
        return 'result'

    results = []

    def worker():
        barrier.wait()
        results.append(setup())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter == [1]
    assert results == ['result'] * 5
