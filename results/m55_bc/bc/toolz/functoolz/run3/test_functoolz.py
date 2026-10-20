import sys
import pickle
import inspect
import functools

import pytest

from toolz.functoolz import (
    identity,
    apply,
    thread_first,
    thread_last,
    instanceproperty,
    InstanceProperty,
    curry,
    memoize,
    Compose,
    compose,
    compose_left,
    pipe,
    complement,
    juxt,
    do,
    flip,
    return_none,
    excepts,
    num_required_args,
    has_varargs,
    has_keywords,
    is_valid_args,
    is_partial_args,
    is_arity,
)
from toolz.utils import no_default


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


def test_apply_basic():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert apply(double, 5) == 10
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000
    )


def test_apply_kwargs():
    def f(a, b=10):
        return a + b

    assert apply(f, 1, b=2) == 3
    assert apply(f, 1) == 11


def test_apply_no_func():
    with pytest.raises(TypeError) as excinfo:
        apply()
    assert 'func argument is required' in str(excinfo.value)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_first(1, inc, double) == 4
    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_last():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_last(1, inc, double) == 4
    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_no_forms():
    assert thread_last(5) == 5


def test_thread_first_non_callable_non_tuple_returns_none():
    # A non-callable, non-tuple form falls through both branches -> None.
    assert thread_first(1, 5) is None


def test_thread_last_non_callable_non_tuple_returns_none():
    assert thread_last(1, 5) is None


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'An object docstring'

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == 'The class docstring'
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == 'An object docstring'
    assert obj.val == 42


def test_instanceproperty_is_property():
    class C:
        @instanceproperty
        def v(self):
            return 1

    assert isinstance(C.__dict__['v'], InstanceProperty)
    assert isinstance(C.__dict__['v'], property)


def test_instanceproperty_without_fget_returns_partial():
    result = instanceproperty(fset=lambda self, v: None, classval=99)
    assert isinstance(result, functools.partial)
    # The partial itself, when accessed as a plain class attribute without
    # being a proper descriptor, is returned unchanged.
    class C:
        v = result
    assert C.__dict__['v'] is result
    assert C.v is result


def test_instanceproperty_setter():
    class C:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    c = C()
    c.x = 10
    assert c.x == 10


def test_instanceproperty_reduce():
    @instanceproperty(classval='cls')
    def val(self):
        return 'inst'

    fget, fset, fdel, doc, classval = val.__reduce__()[1]
    assert classval == 'cls'
    assert fget is not None
    assert fset is None
    assert fdel is None


def test_instance_property_pickle_roundtrip():
    # Pickling an InstanceProperty requires its fget to be picklable, so the
    # getter is a module-level function (the property is created inside a
    # top-level-class body whose getter is a global).  Use a builtin-like
    # picklable getter instead: define a module-level helper.
    prop = InstanceProperty(fget=len, classval='classval')
    restored = pickle.loads(pickle.dumps(prop))
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == 'classval'
    assert restored.fget is len


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20
    assert isinstance(double, curry)


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable():
    with pytest.raises(TypeError) as excinfo:
        curry(1)
    assert 'callable' in str(excinfo.value)


def test_curry_unpacks_partial():
    def add(x, y, z):
        return x + y + z

    p = functools.partial(add, 1)
    c = curry(p)
    assert c.func is add
    assert c.args == (1,)


def test_curry_unpacks_partial_with_keywords():
    def add(x, y, z=0, w=0):
        return x + y + z + w

    p = functools.partial(add, 1, z=5)
    c = curry(p, w=2)
    assert c.func is add
    assert c.args == (1,)
    assert c.keywords == {'z': 5, 'w': 2}
    assert c(2) == 10


def test_curry_unpacks_curry():
    def add(x, y, z):
        return x + y + z

    c1 = curry(add, 1)
    c2 = curry(c1, 2)
    assert c2.func is add
    assert c2.args == (1, 2)
    assert c2(3) == 6


def test_curry_state():
    def foo(x, y):
        """doc"""
        return x + y

    c = curry(foo)
    assert c.func is foo
    assert c.args == ()
    assert c.keywords == {}
    assert c.__name__ == 'foo'
    assert c.__doc__ == 'doc'
    assert c.func_name == 'foo'


def test_curry_str_repr():
    def foo(x):
        return x

    c = curry(foo)
    assert str(c) == str(foo)
    assert repr(c) == repr(foo)


def test_curry_hash_eq():
    def foo(x, y):
        return x + y

    a = curry(foo, 1)
    b = curry(foo, 1)
    c = curry(foo, 2)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert not (a == 5)
    assert a != 5


def test_curry_kwargs_hash():
    def foo(x, y=0):
        return x + y

    a = curry(foo, y=1)
    b = curry(foo, y=1)
    assert hash(a) == hash(b)
    assert a == b


def test_curry_call_raises_genuine_type_error():
    def f(x):
        return x + 1

    c = curry(f)
    with pytest.raises(TypeError):
        c('a', 'b', 'c')


def test_curry_call_success():
    def add(x, y):
        return x + y

    assert curry(add)(1, 2) == 3


def test_curry_varargs():
    def deco(*args, **kwargs):
        def wrapper(f):
            def inner(x):
                return f(x) + len(args) + len(kwargs)
            return inner
        return wrapper

    @curry
    def f(x):
        return x

    partial_applied = curry(deco)
    # Calling with plausible arguments succeeds; the result is the wrapper.
    result = partial_applied(1, a=2)
    assert callable(result)
    assert result(f)(10) == 10 + 1 + 1


def test_curry_bind_and_call():
    def add(x, y):
        return x + y

    c = curry(add)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert c.call(1, 2) == 3


def test_curry_get():
    def f(self, x):
        return x + self.v

    class C:
        v = 10

    c = curry(f)
    assert c.__get__(None, C) is c
    inst = C()
    bound = c.__get__(inst, C)
    assert isinstance(bound, curry)
    assert bound(5) == 15


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert params[0].name == 'y'
    assert params[1].name == 'z'
    assert params[1].default == 3


def test_curry_signature_wrong_args():
    def f(x, y):
        return x + y

    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError) as excinfo:
        c.__signature__
    assert 'incorrect arguments' in str(excinfo.value)


def test_curry_signature_kwonly_from_keywords():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1, z=5)
    sig = c.__signature__
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names == ['y', 'z']
    z_param = params[1]
    assert z_param.default == 5
    assert z_param.kind == inspect.Parameter.KEYWORD_ONLY


def test_curry_signature_varargs_break():
    # When *args consumes the bound positional arguments, the loop breaks.
    def f(*args, **kwargs):
        return args, kwargs

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    # The VAR_POSITIONAL param remains and should be preserved.
    kinds = [p.kind for p in params]
    assert inspect.Parameter.VAR_POSITIONAL in kinds


def test_curry_signature_var_keyword_preserved():
    # VAR_KEYWORD parameters are iterated over but kept in the signature
    # (the code passes on them without skipping).
    def f(x, **kwargs):
        return x, kwargs

    c = curry(f)
    sig = c.__signature__
    names = [p.name for p in sig.parameters.values()]
    assert names == ['x', 'kwargs']
    assert sig.parameters['kwargs'].kind == inspect.Parameter.VAR_KEYWORD


def test_curry_signature_kwonly_following_varargs():
    # After a keyword-only param appears via keywords, a later VAR_POSITIONAL
    # is skipped.
    def f(x, *args, y=0):
        return x, args, y

    c = curry(f, 1, y=5)
    sig = c.__signature__
    names = [p.name for p in sig.parameters.values()]
    assert names == ['args', 'y']
    y_param = sig.parameters['y']
    assert y_param.kind == inspect.Parameter.KEYWORD_ONLY
    assert y_param.default == 5


def test_curry_signature_required_becomes_no_default():
    def f(x, y, z):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    assert sig.parameters['y'].default is no_default
    assert sig.parameters['z'].default is no_default


def test_curry_should_curry_merges_keywords():
    def f(x, y, *, a=0):
        return x + y + a

    c = curry(f, a=5)
    # Missing y; TypeError, should curry and merge keywords.
    c2 = c(1)
    assert isinstance(c2, curry)
    assert c2.keywords == {'a': 5}
    assert c2(2) == 8


def test_curry_pickle():
    from operator import add as add_op
    c = curry(add_op, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


def test_curry_pickle_preserves_userdict():
    from operator import add as add_op

    c = curry(add_op, 1)
    c.custom_attr = 'hello'
    restored = pickle.loads(pickle.dumps(c))
    assert restored.custom_attr == 'hello'
    assert restored(2) == 3


def test_curry_pickle_decorated():
    c = pickle.loads(pickle.dumps(memoize))
    assert isinstance(c, curry)


def test_curry_pickle_toplevel_func():
    # Use an importable module-level callable so pickle can round-trip.
    from operator import add as add_op

    c = curry(add_op, 2)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == 5
    assert restored.args == (2,)


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    mem = memoize(add)
    assert mem(1, 2) == 3
    assert mem(1, 2) == 3
    assert calls == [(1, 2)]
    assert mem.__wrapped__ is add
    assert mem.__name__ == 'add'


def test_memoize_decorator():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_with_cache_kwarg():
    calls = []

    @memoize(cache={(1, 2): 100})
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 100
    assert calls == []


def test_memoize_unary():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]


def test_memoize_key_function():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError) as excinfo:
        f([1, 2, 3])
    assert 'hashable' in str(excinfo.value)


def test_memoize_docstring_copied():
    def f(x):
        """my docstring"""
        return x

    mem = memoize(f)
    assert mem.__doc__ == 'my docstring'


def test_memoize_no_kwargs_key():
    # Function without keywords -> uses args tuple as key.
    calls = []

    def f(x, y):
        calls.append((x, y))
        return x + y

    mem = memoize(f)
    assert mem(1, 2) == 3
    assert mem(1, 2) == 3
    assert calls == [(1, 2)]


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    def f(x):
        return x
    assert compose(f) is f


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity


def test_compose_multiple_args():
    c = compose(str, lambda a, b: a + b)
    assert c(1, 2) == '3'


def test_compose_kwargs():
    c = compose(str, lambda a, b=0: a + b)
    assert c(1, b=2) == '3'


def test_compose_name():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert c.__name__ == 'foo_of_bar'


def test_compose_doc():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert c.__doc__ == 'lambda *args, **kwargs: foo(bar(*args, **kwargs))'


def test_compose_doc_nameless():
    c = compose(len, lambda x: x)
    assert isinstance(c, Compose)
    assert c.__doc__ == 'lambda *args, **kwargs: len(<lambda>(*args, **kwargs))'


def test_compose_doc_no_name_attribute():
    class Callable:
        def __call__(self, x):
            return x

    def foo(x):
        return x

    c = compose(foo, Callable())
    assert c.__doc__ == 'A composition of functions'


def test_compose_name_nameless():
    c = compose(len, lambda x: x)
    assert c.__name__ == 'len_of_<lambda>'


def test_compose_name_no_name_attribute():
    class Callable:
        def __call__(self, x):
            return x

    def foo(x):
        return x

    c = compose(foo, Callable())
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert repr(c).startswith('Compose(')


def test_compose_eq_hash():
    def foo(x):
        return x

    def bar(x):
        return x

    a = compose(foo, bar)
    b = compose(foo, bar)
    c = compose(bar, foo)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert not (a == 5)


def test_compose_ne_notimplemented():
    c = compose(len, str)
    # Comparison with unrelated type returns True for != via NotImplemented.
    assert c.__ne__(5) is NotImplemented


def test_compose_eq_notimplemented():
    c = compose(len, str)
    assert c.__eq__(5) is NotImplemented


def test_compose_get_descriptor():
    def f(self):
        return self.v

    def g(x):
        return x * 2

    class C:
        v = 5
        m = compose(g, f)

    inst = C()
    assert inst.m() == 10
    assert C.m is not None


def test_compose_get_returns_self_on_class():
    c = compose(len, str)
    assert c.__get__(None) is c


def test_compose_signature():
    def f(x: int) -> int:
        return x

    def g(y) -> str:
        return str(y)

    c = compose(g, f)
    sig = c.__signature__
    assert sig.parameters['x'].annotation is int
    assert sig.return_annotation is str


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    # __wrapped__ is the first function applied (rightmost in compose).
    assert c.__wrapped__ is g


def test_compose_combined_annotations():
    def f(x: int, y: str) -> float:
        return x

    def g(a: float) -> str:
        return str(a)

    c = compose(g, f)
    annotations = c._combined_annotations()
    assert annotations['x'] is int
    assert annotations['y'] is str
    assert annotations['return'] is str


def test_compose_combined_annotations_nested():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    inner = compose(g, f)
    outer = compose(len, inner)
    ann = outer._combined_annotations()
    assert ann['x'] is int
    # ``len`` has no return annotation, so 'return' is absent.
    assert 'return' not in ann


def test_compose_combined_annotations_no_return():
    def f(x: int):
        return x

    def g(y):
        return y

    c = compose(g, f)
    ann = c._combined_annotations()
    assert ann == {'x': int}


def test_compose_combined_annotations_unsignatureable():
    class NoSig:
        __signature__ = None

    def f(x: int) -> int:
        return x

    c = compose(f, NoSig())
    # Should not raise; NoSig contributes nothing.
    ann = c._combined_annotations()
    assert ann['return'] is int


def test_compose_annotations_descriptor():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    ann = c.__annotations__
    assert isinstance(ann, dict)
    assert ann['x'] is int
    assert ann['return'] is str
    assert isinstance(Compose.__annotations__, dict)


def test_compose_getstate_setstate():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2 == c


def test_compose_single_elem_doc():
    def foo(x):
        return x

    c = Compose([foo])
    assert c.__doc__ == 'lambda *args, **kwargs: foo(*args, **kwargs)'


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------

def test_juxt():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y

    def g(x, y=0):
        return x - y

    j = juxt(f, g)
    assert j(5, y=2) == (7, 3)


def test_juxt_pickle():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2.funcs == j.funcs


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    result = do(log.append, 1)
    assert result == 1
    assert log == [1]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
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
    assert return_none(ValueError()) is None


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


def test_excepts_name():
    excepting = excepts(ValueError, lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_multiple():
    excepting = excepts((IndexError, KeyError), lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_IndexError_or_KeyError'


def test_excepts_doc():
    def handler(e):
        return None

    excepting = excepts(ValueError, len, handler)
    doc = excepting.__doc__
    assert 'ValueError' in doc
    assert 'handler' in doc


def test_excepts_doc_classval():
    # Accessing the property on the class yields the class docstring.
    assert excepts.__doc__ is excepts.__doc__
    assert 'A wrapper around' in excepts.__doc__


def test_excepts_name_attribute_error():
    class Callable:
        def __call__(self, x):
            return x

    excepting = excepts(ValueError, Callable())
    assert excepting.__name__ == 'excepting'


def test_excepts_doc_attribute_error():
    class Callable:
        def __call__(self, x):
            return x

    class Handler:
        def __call__(self, e):
            return None

    excepting = excepts(ValueError, Callable(), Handler())
    # func/handler have no __name__: doc falls back to class docstring.
    assert 'A wrapper around' in excepting.__doc__


def test_excepts_class_doc():
    assert isinstance(excepts.__doc__, str)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

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


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


def test_is_arity_kwargs():
    def f(x, **kwargs):
        return x

    assert is_arity(1, f) is False


def test_is_arity_builtin():
    assert is_arity(1, len) is True


def test_num_required_args_builtin():
    assert num_required_args(len) == 1


def test_has_varargs_builtin():
    assert has_varargs(print) in (True, None)


def test_signature_helpers_with_sigspec():
    def add(x, y, z=0):
        return x + y + z

    sig = inspect.signature(add)
    assert num_required_args(add, sig) == 2
    assert has_varargs(add, sig) is False
    assert has_keywords(add, sig) is True
    assert is_valid_args(add, (1, 2), {}, sig) is True
    assert is_partial_args(add, (1,), {}, sig) is True
    assert is_arity(3, add, sig) is False


def test_num_required_args_signature_value_error():
    rv = num_required_args(len, ValueError('x'))
    # builtin fallback used
    assert rv == 1


def test_is_valid_args_with_sigspec_valueerror():
    # Passing a ValueError sigspec triggers the builtin fallback path.
    assert is_valid_args(len, ([1, 2],), {}, ValueError('x')) is True


def test_is_arity_unknown_returns_none_branch():
    class NoSig:
        __signature__ = 5  # not a Signature -> _check_sigspec non-Signature path

        def __call__(self):
            pass

    # Falls to _sigs lookup; behavior is whatever the registry says.
    # Just ensure calling doesn't raise and returns a bool/None.
    res = is_arity(0, NoSig())
    assert res in (True, False, None)
