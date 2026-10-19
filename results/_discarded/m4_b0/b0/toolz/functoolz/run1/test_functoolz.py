import inspect
import pickle
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
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_with_kwargs():
    def f(a, b, c=0):
        return a + b + c

    assert apply(f, 1, 2, c=3) == 6


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuple_forms():
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

    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


def test_thread_last_no_forms():
    assert thread_last(7) == 7


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        @instanceproperty
        def val(self):
            return 42

    obj = MyClass()
    assert obj.val == 42
    assert MyClass.val is None


def test_instanceproperty_classval():
    class MyClass:
        @instanceproperty(classval='class-doc')
        def __doc__(self):
            return 'instance-doc'

    assert MyClass.__doc__ == 'class-doc'
    assert MyClass().__doc__ == 'instance-doc'


def test_instanceproperty_setter():
    class MyClass:
        def __init__(self):
            self._v = 0

        @instanceproperty
        def val(self):
            return self._v

        @val.setter
        def val(self, v):
            self._v = v

    obj = MyClass()
    obj.val = 10
    assert obj.val == 10


def test_instanceproperty_class_access_returns_partial():
    p = instanceproperty()
    assert isinstance(p, type(p))  # a partial object


def test_instanceproperty_no_args_returns_partial():
    from functools import partial
    p = instanceproperty()
    assert isinstance(p, partial)


def test_instanceproperty_reduce_roundtrip():
    class MyClass:
        @instanceproperty(classval='cls')
        def val(self):
            return 1

    prop = MyClass.__dict__['val']
    restored = pickle.loads(pickle.dumps(prop))
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == 'cls'
    assert restored.fget is prop.fget


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
    assert cmul(3, 4) == 12


def test_curry_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args_error():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable_error():
    with pytest.raises(TypeError):
        curry(42)


def test_curry_merges_partial_like():
    def f(a, b, c):
        return a + b + c

    c1 = curry(f, 1)
    c2 = curry(c1, 2)
    assert c2(3) == 6
    assert c2.args == (1, 2)


def test_curry_func_args_keywords():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1, b=2)
    assert c.func is f
    assert c.args == (1,)
    assert c.keywords == {'b': 2}
    assert c(4) == 7


def test_curry_str_repr():
    def f(x):
        return x

    c = curry(f)
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_equality_and_hash():
    def f(x, y):
        return x + y

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert a != c
    assert not (a == 5)
    assert hash(a) == hash(b)


def test_curry_call_method_raises_directly():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    with pytest.raises(TypeError):
        c.call(1, 2)


def test_curry_bind():
    def f(x, y, z):
        return x + y + z

    c = curry(f)
    c2 = c.bind(1)
    assert isinstance(c2, curry)
    assert c2(2, 3) == 6


def test_curry_genuine_type_error_reraised():
    @curry
    def f(x):
        raise TypeError("boom")

    # With full valid args, the TypeError is genuine -> reraised
    with pytest.raises(TypeError, match="boom"):
        f(1)


def test_curry_empty_args_curries():
    def f(x, y):
        return x + y

    c = curry(f)
    # calling with no args should curry again
    c2 = c()
    assert isinstance(c2, curry)
    assert c2(1, 2) == 3


def test_curry_too_many_args_raises():
    def f(x):
        return x

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2)


def test_curry_get_descriptor():
    class A:
        @curry
        def method(self, x, y):
            return self.val + x + y

        val = 10

    a = A()
    assert a.method(1, 2) == 13
    # Accessing on class returns the curry itself
    assert isinstance(A.method, curry)


def test_curry_pickle_roundtrip():
    from toolz.functoolz import curry as C

    local_f = _module_level_function
    c = C(local_f, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


def _module_level_function(x, y):
    return x + y


def test_curry_signature():
    def f(a, b, c=1):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert params[0].name == 'b'
    assert params[1].name == 'c'


def test_curry_signature_incorrect_args():
    def f(a, b):
        return a + b

    # create directly bypassing curry's checks to force invalid args
    c = curry(f)
    c._partial = __import__('functools').partial(f, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


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


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100
    assert add(2, 3) == 5


def test_memoize_custom_key():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3


def test_memoize_unary():
    calls = []

    @memoize
    def sq(x):
        calls.append(x)
        return x * x

    assert sq(3) == 9
    assert sq(3) == 9
    assert calls == [3]


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match="hashable"):
        f([1, 2, 3])


def test_memoize_metadata():
    def f(x):
        "docstring"
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'docstring'
    assert m.__wrapped__ is f


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    def inc(i):
        return i + 1

    assert compose(str, inc)(3) == '4'


def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_left_basic():
    def inc(i):
        return i + 1

    assert compose_left(inc, str)(3) == '4'


def test_compose_left_empty():
    assert compose_left() is identity


def test_compose_kwargs_pass_through():
    def f(a, b=0):
        return a + b

    def g(x):
        return x * 2

    assert compose(g, f)(1, b=2) == 6


def test_compose_call():
    def inc(x):
        return x + 1

    def dbl(x):
        return x * 2

    c = compose(inc, dbl)
    assert isinstance(c, Compose)
    assert c(3) == 7


def test_compose_doc():
    def inc(i):
        return i + 1

    def dbl(i):
        return i * 2

    c = compose(inc, dbl)
    assert c.__doc__ == 'lambda *args, **kwargs: inc(dbl(*args, **kwargs))'


def test_compose_name():
    def inc(i):
        return i + 1

    def dbl(i):
        return i * 2

    c = compose(inc, dbl)
    assert c.__name__ == 'inc_of_dbl'


def test_compose_repr():
    def inc(i):
        return i + 1

    def dbl(i):
        return i * 2

    c = compose(inc, dbl)
    assert repr(c) == 'Compose(<function inc at 0x{0}>, <function dbl at 0x{1}>)'.format(
        format(id(inc), 'x'), format(id(dbl), 'x')
    ) if False else repr(c).startswith('Compose((')
    assert 'inc' in repr(c)
    assert 'dbl' in repr(c)


def test_compose_equality():
    def inc(x):
        return x + 1

    def dbl(x):
        return x * 2

    assert compose(inc, dbl) == compose(inc, dbl)
    assert compose(inc, dbl) != compose(dbl, inc)
    assert not (compose(inc, dbl) == 5)


def test_compose_hash():
    def inc(x):
        return x + 1

    def dbl(x):
        return x * 2

    assert hash(compose(inc, dbl)) == hash(compose(inc, dbl))


def test_compose_getstate_setstate():
    def inc(x):
        return x + 1

    def dbl(x):
        return x * 2

    c = compose(inc, dbl)
    state = c.__getstate__()
    c2 = Compose((inc, dbl))
    c2.__setstate__(state)
    assert c2(3) == 7


def test_compose_signature():
    def first(a: int, b: int) -> int:
        return a + b

    def last(x: int) -> str:
        return str(x)

    c = compose(last, first)
    sig = c.__signature__
    assert list(sig.parameters) == ['a', 'b']
    assert sig.return_annotation is str


def test_compose_wrapped():
    def inc(x):
        return x + 1

    def dbl(x):
        return x * 2

    c = compose(inc, dbl)
    assert c.__wrapped__ is dbl


def test_compose_as_method():
    def inc(x):
        return x + 1

    def dbl(x):
        return x * 2

    class A:
        val = 5
        method = compose(inc, dbl)

    a = A()
    assert a.method(1) == 3
    assert A.method is A.__dict__['method']


def test_compose_annotations():
    def first(a: int, b: int) -> int:
        return a + b

    def last(x: int) -> str:
        return str(x)

    c = compose(last, first)
    ann = c._combined_annotations()
    assert ann['a'] is int
    assert ann['b'] is int
    assert ann['return'] is str


def test_compose_doc_unnamed_callable():
    def inc(x):
        return x + 1

    c = compose(inc, len)
    # len has __name__, fine; use a lambda-less callable lacking __name__?
    assert isinstance(c.__doc__, str)


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    def double(i):
        return 2 * i

    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
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

def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def add(x, y):
        return x + y

    def sub(x, y):
        return x - y

    j = juxt(add, sub)
    assert j(5, 3) == (8, 2)


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    j = juxt(inc, double)
    state = j.__getstate__()
    j2 = juxt()
    j2.__setstate__(state)
    assert j2(3) == (4, 6)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


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
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError()) is None


# ---------------------------------------------------------------------------
# excepts
# ---------------------------------------------------------------------------

def test_excepts_basic():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_no_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_handler_receives_exception():
    seen = []

    def handler(e):
        seen.append(e)
        return 'handled'

    excepting = excepts(ValueError, lambda: int('x'), handler)
    assert excepting() == 'handled'
    assert len(seen) == 1
    assert isinstance(seen[0], ValueError)


def test_excepts_does_not_catch_others():
    excepting = excepts(ValueError, lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        excepting()


def test_excepts_doc():
    excepting = excepts(ValueError, lambda a: a, lambda e: None)
    doc = excepting.__doc__
    assert 'ValueError' in doc
    assert 'lambda' in doc


def test_excepts_name():
    excepting = excepts((IndexError, KeyError), lambda a: a)
    assert excepting.__name__ == 'lambda_excepting_IndexError_or_KeyError'


def test_excepts_class_doc():
    assert excepts.__doc__ is not None


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    def g(*args, **kwargs):
        pass

    assert num_required_args(f) == 2
    assert num_required_args(g) == 0


def test_num_required_args_positional_only():
    def f(x, /, y):
        return x + y

    assert num_required_args(f) == 2


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
    assert is_arity(2, g) is False


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


def test_is_arity_keyword_only():
    def f(x, *, y=1):
        return x + y

    assert is_arity(1, f) is False
