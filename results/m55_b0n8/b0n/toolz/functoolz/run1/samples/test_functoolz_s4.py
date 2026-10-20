import inspect
import pickle
import sys

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


# ---------------------------------------------------------------- helpers

def inc(x):
    return x + 1


def double(x):
    return 2 * x


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def variadic(*args, **kwargs):
    return (args, kwargs)


# ---------------------------------------------------------------- identity

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


# ---------------------------------------------------------------- apply

def test_apply():
    assert apply(double, 5) == 10
    assert apply(add, 2, 3) == 5
    assert apply(add, x=2, y=3) == 5


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


def test_apply_map():
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


# ---------------------------------------------------------------- thread_first

def test_thread_first_simple():
    assert thread_first(1, inc, double) == 4
    assert thread_first(1) == 1


def test_thread_first_tuple_forms():
    def power(x, y):
        return x ** y
    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_first_appends_value_first():
    def three(x, y, z):
        return (x, y, z)
    assert thread_first(1, (three, 2, 3)) == (1, 2, 3)


# ---------------------------------------------------------------- thread_last

def test_thread_last_simple():
    assert thread_last(1, inc, double) == 4
    assert thread_last(1) == 1


def test_thread_last_tuple_forms():
    def power(x, y):
        return x ** y
    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_appends_value_last():
    def three(x, y, z):
        return (x, y, z)
    assert thread_last(1, (three, 2, 3)) == (2, 3, 1)


def test_thread_last_map_filter():
    def iseven(x):
        return x % 2 == 0
    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


# ---------------------------------------------------------------- instanceproperty

def test_instanceproperty_classval():
    class C:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'obj-doc'

        @instanceproperty
        def val(self):
            return 42

    assert C.__doc__ == 'The class docstring'
    assert C.val is None
    obj = C()
    assert obj.__doc__ == 'obj-doc'
    assert obj.val == 42


def test_instanceproperty_lazy_without_fget():
    class C:
        @instanceproperty(classval='classval')
        def val(self):
            return 'instanceval'

    assert C.val == 'classval'
    assert C().val == 'instanceval'


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
    assert c.x == 0
    c.x = 5
    assert c.x == 5


def test_instance_property_is_property():
    assert issubclass(InstanceProperty, property)


# ---------------------------------------------------------------- curry

def test_curry_basic():
    mul = curry(lambda x, y: x * y)
    double_ = mul(2)
    assert isinstance(double_, curry)
    assert double_(10) == 20
    assert mul(2, 3) == 6
    assert mul(x=2, y=3) == 6


def test_curry_keywords():
    @curry
    def f(x, y, a=10):
        return a * (x + y)
    add_ = f(a=1)
    assert add_(2, 3) == 5


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_attributes():
    def myfunc(x):
        """my doc"""
        return x

    c = curry(myfunc)
    assert c.func is myfunc
    assert c.__name__ == 'myfunc'
    assert c.__doc__ == 'my doc'
    assert str(c) == str(myfunc)
    assert repr(c) == repr(myfunc)
    assert c.func_name == 'myfunc'


def test_curry_args_keywords_properties():
    @curry
    def f(x, y, z=3):
        return x + y + z

    g = f(1)
    assert g.args == (1,)
    assert g.keywords == {}
    h = f(1, z=5)
    assert h.args == (1,)
    assert h.keywords == {'z': 5}


def test_curry_call_vs_curry():
    c = curry(add)
    # Call with too few args curries
    result = c(1)
    assert isinstance(result, curry)
    assert result(2) == 3


def test_curry_genuine_typeerror():
    def bad(x):
        return x + 'no'
    c = curry(bad)
    with pytest.raises(TypeError):
        c(1)


def test_curry_bind_and_call():
    c = curry(add)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound(2) == 3
    c2 = curry(add)
    assert c2.call(3, 4) == 7


def test_curry_unpack_partial():
    from functools import partial
    p = partial(add, 1)
    c = curry(p)
    assert c.func is add
    assert c.args == (1,)
    assert c(2) == 3


def test_curry_unpack_curry():
    c1 = curry(add)(1)
    c2 = curry(c1)
    assert c2.func is add
    assert c2.args == (1,)
    assert c2(2) == 3


def test_curry_eq_hash():
    c1 = curry(add)(1)
    c2 = curry(add)(1)
    c3 = curry(add)(2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert not (c1 == 5)


def test_curry_signature():
    @curry
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f(1))
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['y', 'z']


def test_curry_signature_keyword():
    @curry
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f(z=5))
    assert 'z' in sig.parameters
    assert sig.parameters['z'].kind == inspect.Parameter.KEYWORD_ONLY


def test_curry_invalid_signature_args():
    @curry
    def f(x, y):
        return x + y

    c = curry(f)
    # Cannot create a curry with too many positional args
    with pytest.raises(TypeError):
        curry(curry(f)(1), 2, 3).__signature__


def test_curry_descriptor():
    class C:
        @curry
        def meth(self, x):
            return self.value + x

    c = C()
    c.value = 10
    assert C.meth is not None
    bound = c.meth
    assert isinstance(bound, curry)
    assert bound(5) == 15


def test_curry_get_returns_self_on_class():
    @curry
    def f(x):
        return x
    assert curry.__get__(f, None, None) is f


def test_curry_pickle():
    def f(x, y):
        return x + y
    f.__module__ = 'test_module_pickle'
    f.__qualname__ = 'f'

    c = curry(f)(1)
    # not importable; just exercise reduce of a plain curry
    c2 = curry(add)(1)
    restored = pickle.loads(pickle.dumps(c2))
    assert restored.func is add
    assert restored.args == (1,)
    assert restored(2) == 3


def test_curry_varargs_curried_anyway():
    @curry
    def f(*args, **kwargs):
        return (args, kwargs)

    # *args functions curry even when TypeError raised; but here no TypeError
    assert f(1) == ((1,), {})
    assert f() == ((), {})


def test_curry_should_curry_unknown_args():
    # a function with *args that raises TypeError
    @curry
    def f(*args):
        if len(args) < 2:
            raise TypeError('need two')
        return args

    result = f(1)
    assert isinstance(result, curry)


# ---------------------------------------------------------------- memoize

def test_memoize_basic():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_memoize_with_cache_kwarg():
    @memoize(cache={(1, 2): 3})
    def add2(x, y):
        return x + y

    assert add2(1, 2) == 3
    assert add2(2, 3) == 5


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def f(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert f(1, 2, verbose=True) == 3
    assert f(1, 2, verbose=False) == 3
    assert calls == [(1, 2)]


def test_memoize_kwargs():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert len(calls) == 1
    assert f(1, y=3) == 4
    assert len(calls) == 2


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_wrapped():
    def f(x):
        """doc"""
        return x

    g = memoize(f)
    assert g.__wrapped__ is f
    assert g.__name__ == 'f'
    assert g.__doc__ == 'doc'


def test_memoize_unary_varargs():
    calls = []

    @memoize
    def f(*args):
        calls.append(args)
        return sum(args)

    assert f(1, 2) == 3
    assert f(1, 2) == 3
    assert len(calls) == 1


# ---------------------------------------------------------------- compose

def test_compose_basic():
    assert compose(str, inc)(3) == '4'
    assert compose(double, inc)(1) == 4


def test_compose_no_args_identity():
    f = compose()
    assert f is identity


def test_compose_one_arg():
    assert compose(inc) is inc


def test_compose_left():
    assert compose_left(inc, str)(3) == '4'
    assert compose_left(inc)(1) == 2


def test_compose_class():
    c = Compose((str, inc))
    assert isinstance(c, Compose)
    assert c(3) == '4'


def test_compose_doc():
    def f(x):
        return x
    def g(x):
        return x
    c = compose(f, g)
    assert 'f(g(' in c.__doc__


def test_compose_doc_no_names():
    class Callable:
        def __call__(self, x):
            return x
    c = compose(Callable(), Callable())
    assert c.__doc__ == 'A composition of functions'


def test_compose_name():
    def f(x):
        return x
    def g(x):
        return x
    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_name_fallback():
    class Callable:
        def __call__(self, x):
            return x
    c = compose(Callable(), Callable())
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def f(x):
        return x
    def g(x):
        return x
    c = compose(f, g)
    assert repr(c).startswith('Compose(')
    assert 'f' in repr(c) and 'g' in repr(c)


def test_compose_eq():
    def f(x):
        return x
    def g(x):
        return x
    assert compose(f, g) == compose(f, g)
    assert compose(f, g) != compose(g, f)
    assert compose(f, g) != 5


def test_compose_hash():
    def f(x):
        return x
    def g(x):
        return x
    assert hash(compose(f, g)) == hash(compose(f, g))


def test_compose_pickle():
    c = compose(str, double)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == '4'


def test_compose_wrapped():
    def f(x):
        return x
    def g(x):
        return x
    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_signature():
    def f(x, y=1):
        return x + y
    def g(a):
        return a
    c = compose(f, g)
    sig = inspect.signature(c)
    assert 'a' in sig.parameters


def test_compose_descriptor():
    c = compose(str, double)

    class C:
        method = c

    obj = C()
    assert obj.method(2) == '4'


def test_compose_annotations_combined():
    def f(x: int) -> float:
        return float(x)
    def g(y: str) -> int:
        return int(y)

    c = compose(f, g)
    annotations = c._combined_annotations()
    assert annotations.get('y') == str
    assert annotations.get('return') == float


def test_compose_annotations_nested():
    def f(x: int) -> float:
        return float(x)
    def g(y: str) -> int:
        return int(y)
    def h(z: bytes) -> str:
        return z.decode()

    c = compose(f, g, h)
    annotations = c._combined_annotations()
    assert annotations.get('z') == bytes
    assert annotations.get('return') == float


def test_compose_annotations_bad_signature_ignored():
    def f(x):
        return x

    class Bad:
        __signature__ = 'not a signature'

        def __call__(self, x):
            return x

    c = compose(f, Bad())
    # Should not raise
    annotations = c._combined_annotations()
    assert annotations.get('x') is inspect.Parameter.empty or 'x' not in annotations


# ---------------------------------------------------------------- pipe

def test_pipe():
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3
    assert pipe(1, inc, inc, inc) == 4


# ---------------------------------------------------------------- complement

def test_complement():
    def iseven(n):
        return n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------- juxt

def test_juxt_basic():
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_single_callable():
    j = juxt(inc)
    assert j(1) == (2,)


def test_juxt_pickle():
    j = juxt(inc, double)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(3) == (4, 6)


# ---------------------------------------------------------------- do

def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


# ---------------------------------------------------------------- flip

def test_flip():
    assert flip(sub, 2, 10) == 8  # sub(10, 2)
    assert flip(sub, 2, 10) == sub(10, 2)


def test_flip_curried():
    div_by_two = flip(sub, 2)
    assert div_by_two(10) == 8


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------- return_none

def test_return_none():
    assert return_none(ValueError('x')) is None


# ---------------------------------------------------------------- excepts

def test_excepts_basic():
    excepting = excepts(ValueError, lambda a: [1, 2].index(a), lambda _: -1)
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_passes_exception_to_handler():
    def handler(e):
        return type(e).__name__
    e = excepts(ValueError, lambda: int('x'), handler)
    assert e() == 'ValueError'


def test_excepts_doc_single():
    def f(x):
        """docs for f"""
        return x

    def handler(e):
        """docs for handler"""
        return None

    e = excepts(ValueError, f, handler)
    doc = e.__doc__
    assert "f" in doc
    assert "ValueError" in doc
    assert "docs for f" in doc
    assert "handler" in doc


def test_excepts_doc_tuple():
    e = excepts((ValueError, KeyError), lambda x: x)
    assert '(ValueError, KeyError)' in e.__doc__


def test_excepts_class_doc_without_attr():
    e = excepts(ValueError, lambda x: x)
    # On the class, __doc__ is the class docstring
    assert excepts.__doc__ is not None


def test_excepts_name():
    def f(x):
        return x
    e = excepts(ValueError, f)
    assert e.__name__ == 'f_excepting_ValueError'


def test_excepts_name_tuple():
    def f(x):
        return x
    e = excepts((ValueError, KeyError), f)
    assert e.__name__ == 'f_excepting_ValueError_or_KeyError'


def test_excepts_name_fallback():
    e = excepts(ValueError, lambda x: x)
    # lambda has __name__
    assert e.__name__ == '<lambda>_excepting_ValueError'


# ---------------------------------------------------------------- introspection

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z
    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass
    assert num_required_args(g) == 0

    def h(a, b, c):
        pass
    assert num_required_args(h) == 3


def test_has_varargs():
    def f(*args):
        return args
    assert has_varargs(f) is True

    def g(**kwargs):
        return kwargs
    assert has_varargs(g) is False

    def h(x):
        return x
    assert has_varargs(h) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y
    assert has_keywords(f) is True

    def g(x):
        return x
    assert has_keywords(g) is False

    def h(**kwargs):
        return kwargs
    assert has_keywords(h) is True


def test_is_valid_args_examples():
    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(add, (1, 2, 3), {}) is False
    assert is_valid_args(add, (), {}) is False


def test_is_partial_args_examples():
    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False
    assert is_partial_args(add, (), {}) is True


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    def h(*args):
        return args

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False
    assert is_arity(2, add) is True
    assert is_arity(1, add) is False


def test_is_arity_builtin():
    # len is a builtin with known signature via registry
    result = is_arity(1, len)
    assert result in (True, None)


def test_signature_helpers_with_sigspec():
    sigspec = inspect.signature(add)
    assert num_required_args(add, sigspec) == 2
    assert has_varargs(add, sigspec) is False
    assert has_keywords(add, sigspec) is False
    assert is_valid_args(add, (1, 2), {}, sigspec) is True
    assert is_partial_args(add, (1,), {}, sigspec) is True
    assert is_arity(2, add, sigspec) is True


def test_curry_arg_introspection():
    @curry
    def f(x, y):
        return x + y

    sigspec = inspect.signature(f)
    assert num_required_args(f) == 2
    assert has_varargs(f) is False
    assert has_keywords(f) is False


# ---------------------------------------------------------------- misc

def test_all_public_names_callable():
    for name in ('identity', 'apply', 'thread_first', 'thread_last',
                 'instanceproperty', 'InstanceProperty', 'curry', 'memoize',
                 'Compose', 'compose', 'compose_left', 'pipe', 'complement',
                 'juxt', 'do', 'flip', 'return_none', 'excepts',
                 'num_required_args', 'has_varargs', 'has_keywords',
                 'is_valid_args', 'is_partial_args', 'is_arity'):
        from toolz import functoolz
        assert hasattr(functoolz, name)


def test_compose_callable_through_pipe():
    assert pipe([1, 2, 3], compose(sum, identity)) == 6
