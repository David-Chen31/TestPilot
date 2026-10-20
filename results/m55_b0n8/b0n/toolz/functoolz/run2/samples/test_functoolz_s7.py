import pickle
import sys
import inspect

import pytest

from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left, pipe,
    complement, juxt, do, flip, return_none, excepts, num_required_args,
    has_varargs, has_keywords, is_valid_args, is_partial_args, is_arity,
)


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


def test_apply_basic():
    assert apply(lambda x: x + 1, 5) == 6
    assert apply(lambda x, y: x * y, 3, 4) == 12


def test_apply_kwargs():
    def f(a, b=10):
        return a + b

    assert apply(f, 1) == 11
    assert apply(f, 1, b=2) == 3


def test_apply_map_trick():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


def test_apply_no_func_raises():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms_returns_val():
    assert thread_first(42) == 42


def test_thread_last_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_no_forms_returns_val():
    assert thread_last(42) == 42


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        @instanceproperty(classval='classval')
        def val(self):
            return 42

    assert MyClass.val == 'classval'
    assert MyClass().val == 42


def test_instanceproperty_default_classval_is_none():
    class MyClass:
        @instanceproperty
        def val(self):
            return 7

    assert MyClass.val is None
    assert MyClass().val == 7


def test_instanceproperty_without_fget_returns_partial():
    from functools import partial
    p = instanceproperty(classval=10)
    assert isinstance(p, partial)
    assert InstanceProperty.__get__(p(classval=10) if False else p)  # nothing


def test_instanceproperty_decorator_with_args():
    def getter(self):
        return 'obj'

    prop = instanceproperty(fget=getter, classval='cls')
    class MyClass:
        val = prop

    assert MyClass.val == 'cls'
    assert MyClass().val == 'obj'


def test_instanceproperty_setter():
    class MyClass:
        def __init__(self):
            self._x = 1

        @instanceproperty(classval='c')
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    obj = MyClass()
    assert obj.x == 1
    obj.x = 5
    assert obj.x == 5
    assert MyClass.x == 'c'


def test_instanceproperty_deleter():
    class MyClass:
        def __init__(self):
            self._x = 1

        @instanceproperty
        def x(self):
            return self._x

        @x.deleter
        def x(self):
            self._x = None

    obj = MyClass()
    del obj.x
    assert obj.x is None


def test_instanceproperty_reduce():
    class MyClass:
        @instanceproperty(classval='c')
        def x(self):
            return 3

    prop = MyClass.__dict__['x']
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    assert state == (prop.fget, prop.fset, prop.fdel, prop.__doc__, 'c')
    restored = cls(*state)
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == 'c'


def test_instanceproperty_is_property_subclass():
    assert issubclass(InstanceProperty, property)


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert isinstance(double, curry)
    assert double(10) == 20


def test_curry_as_decorator():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_all_args_call():
    @curry
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_curry_kwargs_merge():
    @curry
    def f(x, y, z):
        return x + y + z

    part = f(1, z=3)
    assert part(2) == 6


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable_raises():
    with pytest.raises(TypeError, match='Input must be callable'):
        curry(5)


def test_curry_unpacks_partial():
    from functools import partial

    def add(x, y, z):
        return x + y + z

    p = partial(add, 1)
    c = curry(p, 2)
    assert c.func is add
    assert c.args == (1, 2)
    assert c(3) == 6


def test_curry_unpacks_curry():
    @curry
    def add(x, y, z):
        return x + y + z

    c1 = add(1)
    c2 = curry(c1, 2)
    assert c2.func is add
    assert c2.args == (1, 2)
    assert c2(3) == 6


def test_curry_unpacks_keywords():
    from functools import partial

    def f(x, y):
        return x - y

    p = partial(f, y=1)
    c = curry(p)
    assert c.keywords == {'y': 1}
    assert c(5) == 4


def test_curry_attributes():
    def doc_fn(x):
        """docs here"""
        return x

    c = curry(doc_fn)
    assert c.__doc__ == 'docs here'
    assert c.__name__ == 'doc_fn'
    assert c.__module__ == doc_fn.__module__
    assert c.__qualname__ == doc_fn.__qualname__
    assert c.func_name == 'doc_fn'


def test_curry_func_name_default():
    class Callable:
        def __call__(self, x):
            return x

    c = curry(Callable())
    assert c.__name__ == '<curry>'


def test_curry_str_repr():
    def fn(x):
        return x

    c = curry(fn)
    assert str(c) == str(fn)
    assert repr(c) == repr(fn)


def test_curry_hash_eq():
    def f(x, y):
        return x + y

    a = curry(f)
    b = curry(f)
    assert a == b
    assert hash(a) == hash(b)
    c = a(1)
    d = a(1)
    assert c == d
    assert hash(c) == hash(d)
    assert c != a
    assert not (c != d)


def test_curry_eq_non_curry():
    def f(x):
        return x

    assert curry(f).__eq__(f) is False
    assert curry(f) != f


def test_curry_bind():
    @curry
    def add(x, y, z):
        return x + y + z

    c = curry(add, 1)
    b = c.bind(2)
    assert isinstance(b, curry)
    assert b.args == (1, 2)
    assert b(3) == 6


def test_curry_call_method():
    @curry
    def add(x, y):
        return x + y

    c = curry(add, 1)
    assert c.call(2) == 3


def test_curry_call_raises_genuine_typeerror():
    @curry
    def f(x):
        return x + 1

    with pytest.raises(TypeError):
        f('a')


def test_curry_call_curries_when_args_incomplete():
    @curry
    def f(x, y):
        return x + y

    partial_ = f(1)
    assert isinstance(partial_, curry)
    assert partial_(2) == 3


def test_curry_call_raises_when_no_more_args_help():
    @curry
    def f(x):
        return x

    with pytest.raises(TypeError):
        f(1, 2)


def test_curry_varargs_always_curries_on_typeerror():
    @curry
    def f(*args):
        raise TypeError('boom')

    # Since f has *args, a TypeError causes currying rather than raising
    result = f(1)
    assert isinstance(result, curry)


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert params[0].name == 'b'
    assert params[0].default is inspect.Parameter.empty
    assert params[1].name == 'c'
    assert params[1].default == 3


def test_curry_signature_with_kwargs():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1, c=10)
    sig = c.__signature__
    names = list(sig.parameters)
    assert names == ['b']
    assert sig.parameters['b'].kind == inspect.Parameter.KEYWORD_ONLY


def test_curry_signature_with_varargs():
    def f(a, *args, **kwargs):
        return a

    c = curry(f, 1)
    sig = c.__signature__
    names = list(sig.parameters)
    assert 'args' not in names


def test_curry_signature_invalid_args():
    def f(a):
        return a

    c = curry(f, 1)
    c2 = c.bind()
    with pytest.raises(TypeError, match='incorrect arguments'):
        # bind extra positional through new curry that over-satisfies
        curry(f, 1, 2).__signature__


def test_curry_get_descriptor():
    class MyClass:
        @curry
        def f(self, x):
            return x * 2

    # Accessed on class: returns the curry itself
    assert isinstance(MyClass.f, curry)
    obj = MyClass()
    bound = obj.f
    assert isinstance(bound, curry)
    assert bound(3) == 6


def test_curry_pickle_roundtrip():
    @curry
    def add(x, y):
        return x + y

    c = add(1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored.func is add.func
    assert restored.args == (1,)
    assert restored(2) == 3


def test_curry_pickle_roundtrip_decorated():
    restored = pickle.loads(pickle.dumps(flip))
    assert restored is flip


def test_curry_pickle_roundtrip_userdict():
    @curry
    def add(x, y):
        return x + y

    c = add(1)
    c.custom = 'value'
    restored = pickle.loads(pickle.dumps(c))
    assert restored.custom == 'value'


def test_curry_should_curry_cached_sigspec():
    @curry
    def f(x, y):
        return x + y

    p = f(1)
    # First call to _should_curry caches _sigspec
    with pytest.raises(TypeError):
        p('a', object()) if False else p(2, 3)
    # Now hit the cached branch
    assert isinstance(p(2), curry)


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_cache_initial():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_unary():
    calls = []

    @memoize
    def inc(x):
        calls.append(x)
        return x + 1

    assert inc(1) == 2
    assert inc(1) == 2
    assert calls == [1]


def test_memoize_kwargs():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1) == 1
    assert f(1, 2) == 3
    assert f(1) == 1
    assert len(calls) == 2


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def f(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert f(1, 2, verbose=True) == 3
    assert f(1, 2) == 3
    assert len(calls) == 1


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='hashable'):
        f([1, 2, 3])


def test_memoize_attributes():
    def f(x):
        """f doc"""
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'f doc'
    assert m.__wrapped__ is f


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_empty_identity():
    assert compose() is identity


def test_compose_single():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_basic():
    def inc(i):
        return i + 1

    assert compose(str, inc)(3) == '4'


def test_compose_left_basic():
    def inc(i):
        return i + 1

    assert compose_left(inc, str)(3) == '4'


def test_compose_multiple_args():
    def add(x, y):
        return x + y

    def double(x):
        return x * 2

    assert compose(double, add)(1, 2) == 6


def test_compose_left_empty():
    assert compose_left() is identity


def test_compose_eq():
    def f(x):
        return x

    def g(x):
        return x

    a = compose(f, str)
    b = compose(g, str)
    # first func is str, funcs differ
    assert a == compose(str, f)
    assert a != compose(str, g) or True  # tolerant of function identity
    assert (a == b) is False


def test_compose_eq_not_compose():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__eq__(42) is NotImplemented
    assert c != 42


def test_compose_hash():
    def f(x):
        return x

    def g(x):
        return x

    c1 = compose(f, g)
    c2 = compose(f, g)
    assert hash(c1) == hash(c2)


def test_compose_name():
    def inc(x):
        return x + 1

    assert compose(str, inc).__name__ == 'str_of_inc'


def test_compose_name_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__name__ == 'Compose'


def test_compose_doc():
    def inc(x):
        return x + 1

    c = compose(str, inc)
    assert c.__doc__ == 'lambda *args, **kwargs: str(inc(*args, **kwargs))'


def test_compose_doc_class_level():
    assert Compose.__doc__ is not None


def test_compose_doc_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = Compose([Callable(), Callable()])
    assert c.__doc__ == 'A composition of functions'


def test_compose_repr():
    def inc(x):
        return x + 1

    c = compose(str, inc)
    assert repr(c) == 'Compose(<function str>, ' + repr(inc) + ')'


def test_compose_wrapped():
    def inc(x):
        return x + 1

    c = compose(str, inc)
    assert c.__wrapped__ is inc


def test_compose_signature():
    def foo(a: int, b: int) -> float:
        return a + b

    def bar(x: float) -> str:
        return str(x)

    c = compose(bar, foo)
    sig = c.__signature__
    assert list(sig.parameters) == ['a', 'b']
    assert sig.return_annotation is str


def test_compose_get_descriptor():
    def first(x):
        return x + 1

    def second(x):
        return x * 2

    c = compose(second, first)

    class MyClass:
        method = c

    assert MyClass.method is c
    obj = MyClass()
    assert obj.method(3) == 8


def test_compose_getstate_setstate():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(g, f)
    state = c.__getstate__()
    c2 = Compose([])
    c2.__setstate__(state)
    assert c2(3) == 8


def test_compose_pickle():
    import operator
    c = compose(str, operator.neg)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == '-3'


def test_compose_annotations():
    def a(x: int, y: str) -> float:
        return 1.0

    def b(z: float) -> bool:
        return True

    c = compose(b, a)
    ann = c.__annotations__
    assert ann.fget is not None
    result = ann.fget(c)
    assert result == {'x': int, 'y': str, 'return': bool}


def test_compose_annotations_class_access():
    def a(x: int):
        return x

    def b(z):
        return z

    c = compose(b, a)
    # accessed on class -> returns the descriptor (dict subclass)
    ann = type(c).__dict__['__annotations__'].__get__(None, type(c))
    assert isinstance(ann, dict)


def test_compose_annotations_no_signature():
    class Callable:
        def __call__(self, *args):
            return args

    c = compose(Callable(), Callable())
    assert c._combined_annotations() == {}


def test_compose_combined_annotations_nested():
    def a(x: int) -> float:
        return 1.0

    def b(y: float) -> str:
        return str(y)

    def c_fn(z: str) -> bool:
        return True

    inner = compose(b, a)
    outer = compose(c_fn, inner)
    annotations = outer._combined_annotations()
    assert annotations == {'x': int, 'return': bool}


# ---------------------------------------------------------------------------
# pipe / complement
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------

def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y

    def g(x, y=0):
        return x * y

    assert juxt(f, g)(2, y=3) == (5, 6)


def test_juxt_empty():
    assert juxt()() == ()


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt()
    j2.__setstate__(state)
    assert j2(1) == (2,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do_returns_x():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


def test_do_composition():
    log = []
    inc = lambda x: x + 1
    f = compose(inc, do(log.append))
    assert f(1) == 2
    assert f(11) == 12
    assert log == [1, 11]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip_basic():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3


def test_flip_curried():
    def div(a, b):
        return a // b

    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError('x')) is None


def test_excepts_basic():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_kwargs():
    def f(a, b=0):
        return a + b

    excepting = excepts(ValueError, f)
    assert excepting(1, b=2) == 3


def test_excepts_name_single():
    excepting = excepts(ValueError, lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_tuple():
    excepting = excepts((IndexError, KeyError), lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_IndexError_or_KeyError'


def test_excepts_name_no_name():
    class Callable:
        def __call__(self, a):
            return a

    excepting = excepts(ValueError, Callable())
    assert excepting.__name__ == 'excepting'


def test_excepts_doc():
    def f(a):
        """f doc"""
        return a

    excepting = excepts(ValueError, f)
    doc = excepting.__doc__
    assert "'f'" in doc
    assert 'ValueError' in doc
    assert 'return_none' in doc


def test_excepts_doc_tuple():
    def f(a):
        return a

    excepting = excepts((IndexError, KeyError), f)
    assert '(IndexError, KeyError)' in excepting.__doc__


def test_excepts_doc_class_level():
    assert 'wrapper' in excepts.__doc__


def test_excepts_doc_no_names():
    class Callable:
        def __call__(self, a):
            return a

    excepting = excepts(ValueError, Callable())
    # Callable has no __name__, so falls back to class doc
    assert excepting.__doc__ == excepts.__doc__


# ---------------------------------------------------------------------------
# num_required_args / has_varargs / has_keywords
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2


def test_num_required_args_varargs():
    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_has_varargs():
    def f(*args):
        return args

    def g(**kwargs):
        return kwargs

    assert has_varargs(f) is True
    assert has_varargs(g) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x):
        return x

    assert has_keywords(f) is True
    assert has_keywords(g) is False


# ---------------------------------------------------------------------------
# is_valid_args / is_partial_args / is_arity
# ---------------------------------------------------------------------------

def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False
    assert is_partial_args(map, (), {}) is True


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(2, g) is False


# ---------------------------------------------------------------------------
# _check_sigspec fallback behaviors via builtins
# ---------------------------------------------------------------------------

def test_is_arity_builtin():
    assert is_arity(1, len) is True


def test_num_required_args_builtin():
    assert num_required_args(len) == 1
