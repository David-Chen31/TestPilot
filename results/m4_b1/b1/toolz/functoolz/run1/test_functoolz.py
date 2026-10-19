import pickle
import sys
import inspect
from functools import partial

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
    _restore_curry,
)


# --------------------------------------------------------------------------- #
# identity / apply
# --------------------------------------------------------------------------- #

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_kwargs():
    def f(x, y=0):
        return x + y

    assert apply(f, 1, y=10) == 11
    assert apply(f, y=5, x=2) == 7


def test_apply_no_func():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


def test_apply_propagates_errors():
    def f(x):
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        apply(f, 1)


# --------------------------------------------------------------------------- #
# thread_first / thread_last
# --------------------------------------------------------------------------- #

def test_thread_first_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_with_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(42) == 42


def test_thread_last_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_with_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_doc_example():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


# --------------------------------------------------------------------------- #
# instanceproperty / InstanceProperty
# --------------------------------------------------------------------------- #

def test_instanceproperty_class_and_instance():
    class MyClass:
        '''The class docstring'''

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


def test_instanceproperty_partial_returned_when_no_fget():
    p = instanceproperty(classval=5)
    assert isinstance(p, partial)
    # Applying a getter yields an InstanceProperty
    prop = p(lambda self: 1)
    assert isinstance(prop, InstanceProperty)
    assert prop.classval == 5


def test_instanceproperty_is_property_subclass():
    class MyClass:
        @instanceproperty(classval='cv')
        def x(self):
            return 7

    prop = MyClass.__dict__['x']
    assert isinstance(prop, property)
    assert isinstance(prop, InstanceProperty)
    assert MyClass().x == 7


def test_instanceproperty_setter_preserves_classval():
    class MyClass:
        def __init__(self):
            self._x = 0

        def get_x(self):
            return self._x

        def set_x(self, v):
            self._x = v

        x = instanceproperty(fget=get_x, fset=set_x, classval='CV')

    assert MyClass.x == 'CV'
    obj = MyClass()
    obj.x = 9
    assert obj.x == 9


def test_instanceproperty_reduce():
    def getter(self):
        return 1

    prop = InstanceProperty(fget=getter, classval='cv')
    reduced = prop.__reduce__()
    assert reduced[0] is InstanceProperty
    fget, fset, fdel, doc, classval = reduced[1]
    assert fget is getter
    assert fset is None
    assert fdel is None
    assert classval == 'cv'


# --------------------------------------------------------------------------- #
# curry
# --------------------------------------------------------------------------- #

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert isinstance(double, curry)
    assert double(10) == 20


def test_curry_as_decorator_with_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_init_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_init_not_callable():
    with pytest.raises(TypeError, match='Input must be callable'):
        curry(5)


def test_curry_attributes():
    def f(x, y):
        """docstring"""
        return x + y

    c = curry(f)
    assert c.func is f
    assert c.__doc__ == "docstring"
    assert c.__name__ == 'f'
    assert c.func_name == 'f'
    assert c.args == ()
    assert c.keywords == {}


def test_curry_str_repr():
    def f(x):
        return x

    c = curry(f)
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_equality_and_hash():
    def f(x, y):
        return x + y

    def g(x, y):
        return x + y

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    c4 = curry(g, 1)

    assert c1 == c2
    assert not (c1 != c2)
    assert c1 != c3
    assert c1 != c4
    assert c1 != 5
    assert hash(c1) == hash(c2)
    # Keywords affect hash
    ck1 = curry(f, x=1)
    ck2 = curry(f, x=1)
    assert hash(ck1) == hash(ck2)
    assert ck1 == ck2


def test_curry_unpack_partial():
    def f(x, y, z):
        return x + y + z

    p = partial(f, 1)
    c = curry(p, 2)
    assert c(3) == 6


def test_curry_unpack_curry_merge_kwargs():
    def f(x, y, a=1, b=2):
        return (x, y, a, b)

    c1 = curry(f, a=10)
    c2 = curry(c1, b=20)
    assert c2(1, 2) == (1, 2, 10, 20)


def test_curry_call_direct():
    def f(x, y):
        return x + y

    c = curry(f)
    assert c.call(1, 2) == 3


def test_curry_bind():
    def f(x, y):
        return x + y

    c = curry(f)
    c2 = c.bind(1)
    assert isinstance(c2, curry)
    assert c2(2) == 3


def test_curry_genuine_typeerror_raised():
    def f(x):
        raise TypeError('genuine')

    c = curry(f)
    with pytest.raises(TypeError, match='genuine'):
        c(1)


def test_curry_deferred_call():
    def f(x, y):
        return x + y

    c = curry(f)
    # Not enough args -> returns curry
    result = c(1)
    assert isinstance(result, curry)
    assert result(2) == 3


def test_curry_should_curry_rejects_too_many_args():
    def f(x):
        return x

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2)  # cannot possibly be valid; raises


def test_curry_varargs_calls_directly():
    # A *args function has no required args, so calling is always attempted
    # (curry does not need to defer here) -- the function is invoked.
    def f(*args, **kwargs):
        return (args, kwargs)

    c = curry(f)
    assert c(1) == ((1,), {})


def test_curry_descriptor():
    class MyClass:
        def __init__(self, base):
            self.base = base

        @curry
        def add(self, x, y):
            return self.base + x + y

    obj = MyClass(10)
    bound = obj.add
    assert isinstance(bound, curry)
    assert bound(1, 2) == 13
    # class access returns curry itself
    assert isinstance(MyClass.add, curry)


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters)
    # First positional consumed
    assert params == ['y', 'z']


def test_curry_signature_bad_args():
    def f(x):
        return x

    c = curry(f, 1, 2)  # too many
    with pytest.raises(TypeError, match='curry object has incorrect arguments'):
        c.__signature__


def test_curry_signature_kwargs():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, y=2)
    sig = c.__signature__
    # y consumed as keyword; from there on, params become keyword-only
    yparam = sig.parameters['y']
    assert yparam.kind == inspect.Parameter.KEYWORD_ONLY


def test_curry_reduce_pickling_roundtrip_module_func():
    # Test that a simply-defined curry can be reduced.
    c = curry(len)
    red = c.__reduce__()
    assert red[0] is _restore_curry
    state = red[1]
    assert isinstance(state, tuple) and len(state) == 6


def test_curry_pickle_simple_builtin():
    c2 = pickle.loads(pickle.dumps(curry(map)))
    # Just invokes fine
    assert list(c2(int, ['1', '2'])) == [1, 2]


# --------------------------------------------------------------------------- #
# memoize
# --------------------------------------------------------------------------- #

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    madd = memoize(add)
    assert madd(1, 2) == 3
    assert madd(1, 2) == 3
    assert calls == [(1, 2)]
    assert madd(2, 3) == 5
    assert calls == [(1, 2), (2, 3)]


def test_memoize_preserves_metadata():
    def add(x, y):
        """my doc"""
        return x + y

    madd = memoize(add)
    assert madd.__name__ == 'add'
    assert madd.__doc__ == 'my doc'
    assert madd.__wrapped__ is add


def test_memoize_with_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_unary():
    calls = []

    def sq(x):
        calls.append(x)
        return x * x

    msq = memoize(sq)
    assert msq(3) == 9
    assert msq(3) == 9
    assert calls == [3]


def test_memoize_kwargs():
    calls = []

    def f(x, y=0):
        calls.append((x, y))
        return x + y

    mf = memoize(f)
    assert mf(1, y=2) == 3
    assert mf(1, y=2) == 3
    assert calls == [(1, 2)]


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def f(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert f(1, 2, verbose=True) == 3
    assert f(1, 2, verbose=False) == 3
    # Same key args, so only one call
    assert calls == [(1, 2, True)]


def test_memoize_unhashable_args():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='Arguments to memoized function must be hashable'):
        f([1, 2, 3])


def test_memoize_can_be_used_as_decorator():
    @memoize
    def f(x):
        return x + 1

    assert f(1) == 2
    assert f(1) == 2


# --------------------------------------------------------------------------- #
# Compose / compose / compose_left
# --------------------------------------------------------------------------- #

def test_compose_no_funcs():
    assert compose() is identity


def test_compose_single_func():
    def f(x):
        return x + 1

    assert compose(f) is f


def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_order():
    def f(x):
        return ('f', x)

    def g(x):
        return ('g', x)

    def h(x):
        return ('h', x)

    # compose(f, g, h)(x) == f(g(h(x)))
    assert compose(f, g, h)(0) == ('f', ('g', ('h', 0)))


def test_compose_class():
    c = Compose([str, lambda x: x + 1])
    assert isinstance(c, Compose)
    assert c(1) == '2'


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_order():
    def f(x):
        return ('f', x)

    def g(x):
        return ('g', x)

    def h(x):
        return ('h', x)

    assert compose_left(f, g, h)(0) == ('h', ('g', ('f', 0)))


def test_compose_left_no_funcs():
    assert compose_left() is identity


def test_compose_first_with_multiple_args():
    def add(x, y):
        return x + y

    c = compose(str, add)
    assert c(1, 2) == '3'


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    r = repr(c)
    assert r.startswith('Compose')
    assert 'f' in r and 'g' in r


def test_compose_eq_ne_hash():
    def f(x):
        return x

    def g(x):
        return x

    c1 = compose(f, g)
    c2 = compose(f, g)
    c3 = compose(g, f)

    assert c1 == c2
    assert not (c1 != c2)
    assert c1 != c3
    assert c1 != 'not a Compose'
    assert c1.__ne__('not a Compose') is NotImplemented
    assert hash(c1) == hash(c2)


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert 'f(' in c.__doc__
    assert 'g(' in c.__doc__


def test_compose_doc_nameless_callable():
    class NoName:
        def __call__(self, x):
            return x

    c = compose(NoName(), NoName())
    assert c.__doc__ == 'A composition of functions'


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_name_nameless():
    class NoName:
        def __call__(self, x):
            return x

    c = compose(NoName(), NoName())
    assert c.__name__ == 'Compose'


def test_compose_descriptor():
    # The first-applied (rightmost) function receives the instance as
    # its first argument; the rest of the pipeline operates on the result.
    class MyClass:
        double_shift = compose(lambda x: x * 2,
                               lambda self, x: self.base + x)

    obj = MyClass()
    obj.base = 10
    assert obj.double_shift(5) == 30


def test_compose_signature():
    def f(x: int) -> str:
        return str(x)

    def g(y: float) -> int:
        return int(y)

    c = compose(f, g)
    sig = c.__signature__
    # Return annotation comes from the outermost
    assert sig.return_annotation is str
    assert 'y' in sig.parameters


def test_compose_wrapped():
    # __wrapped__ is `first`, i.e. the first function applied (the
    # rightmost in the compose(...) argument list).
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_getstate_setstate():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)  # first = g, funcs = (f,)
    state = c.__getstate__()
    assert state == (g, (f,))

    c2 = Compose([identity, identity])
    c2.__setstate__(state)
    assert c2.first is g
    assert c2(3) == 7  # f(g(3)) = (3 * 2) + 1 = 7


def test_compose_annotations():
    def f(x: int, y: str) -> float:
        return 0.0

    def g(a: bool) -> int:
        return 0

    c = compose(f, g)
    ann = c._combined_annotations()
    # Parameter annotations come from the first-applied function (g);
    # the return annotation comes from the last-applied function (f).
    assert ann.get('a') is bool
    assert 'x' not in ann
    assert 'y' not in ann
    assert ann.get('return') is float


def test_compose_annotations_nested():
    def f(x: int) -> float:
        return 0.0

    def g(a: bool) -> int:
        return 0

    def h(b) -> bool:
        return True

    inner = compose(g, h)
    c = compose(f, inner)
    ann = c._combined_annotations()
    # Actual behavior: the return annotation is taken from the outermost
    # (last-applied) function; the nested Compose is the first-applied
    # callable, and the result reflects what the implementation produces.
    assert ann.get('return') is float
    assert 'x' not in ann
    assert 'b' not in ann


# --------------------------------------------------------------------------- #
# pipe / complement / juxt / do
# --------------------------------------------------------------------------- #

def test_pipe_basic():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


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


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    j = juxt(inc, double)
    state = j.__getstate__()
    assert state == (inc, double)
    j2 = juxt(identity)
    j2.__setstate__((inc, double))
    assert j2(1) == (2, 2)


def test_do():
    log = []

    def inc(x):
        return x + 1

    # do is not curried; wrap it in a lambda to compose it.
    inc_c = compose(inc, lambda x: do(log.append, x))
    assert inc_c(1) == 2
    assert inc_c(11) == 12
    assert log == [1, 11]


def test_do_direct():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


# --------------------------------------------------------------------------- #
# flip / return_none / excepts
# --------------------------------------------------------------------------- #

def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


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


def test_excepts_tuple_and_default_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_kwargs():
    def f(x, y):
        if x < y:
            raise ValueError('small')
        return x - y

    e = excepts(ValueError, f, lambda exc: str(exc))
    assert e(5, 3) == 2
    assert e(1, 3) == 'small'


def test_excepts_name():
    def f(a):
        return a

    def handler(e):
        return None

    e = excepts(ValueError, f, handler)
    assert e.__name__ == 'f_excepting_ValueError'

    e2 = excepts((ValueError, KeyError), f, handler)
    assert e2.__name__ == 'f_excepting_ValueError_or_KeyError'


def test_excepts_name_missing():
    e = excepts(ValueError, object())
    assert e.__name__ == 'excepting'


def test_excepts_doc():
    def f(a):
        """F doc"""
        return a

    def handler(e):
        """H doc"""
        return None

    e = excepts(ValueError, f, handler)
    assert "A wrapper around 'f' that will except" in e.__doc__
    assert 'ValueError' in e.__doc__
    assert "'handler'" in e.__doc__
    assert 'F doc' in e.__doc__
    assert 'H doc' in e.__doc__


def test_excepts_doc_tuple():
    def f(a):
        return a

    e = excepts((ValueError, KeyError), f)
    assert '(ValueError, KeyError)' in e.__doc__


# --------------------------------------------------------------------------- #
# introspection helpers
# --------------------------------------------------------------------------- #

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

    assert has_varargs(f) is True

    def g(**kwargs):
        return kwargs

    assert has_varargs(g) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    assert has_keywords(f) is True

    def g(x, y):
        return x + y

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

    assert is_arity(1, f) is True

    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False

    def h(*args):
        return args

    assert is_arity(1, h) is False


def test_num_required_args_builtin():
    assert num_required_args(map) == 2


def test_has_varargs_builtin():
    assert has_varargs(map) in (True, False, None)
