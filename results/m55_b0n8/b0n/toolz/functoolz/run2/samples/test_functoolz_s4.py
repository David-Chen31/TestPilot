import inspect
import pickle
import sys
import pytest

from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left,
    pipe, complement, juxt, do, flip, return_none, excepts,
    num_required_args, has_varargs, has_keywords, is_valid_args,
    is_partial_args, is_arity,
)


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply_basic():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert apply(double, 5) == 10
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


def test_apply_kwargs():
    def f(a, b=10):
        return a + b

    assert apply(f, 1) == 11
    assert apply(f, 1, b=5) == 6


def test_apply_no_args_raises():
    with pytest.raises(TypeError):
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
    assert thread_first(5) == 5
    assert thread_first(5, inc) == 6


def test_thread_first_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_last_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4
    assert thread_last(5) == 5


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

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

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


def test_instanceproperty_setter_deleter():
    class Holder:
        def __init__(self):
            self._x = 1

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

        @x.deleter
        def x(self):
            self._x = None

    h = Holder()
    assert h.x == 1
    h.x = 5
    assert h.x == 5
    del h.x
    assert h.x is None


def test_instance_property_reduce():
    class MyClass:
        @instanceproperty(classval='clsval')
        def val(self):
            return 42

    prop = MyClass.__dict__['val']
    assert isinstance(prop, InstanceProperty)
    restored_cls, state = prop.__reduce__()
    assert restored_cls is InstanceProperty
    fget, fset, fdel, doc, classval = state
    assert fget is prop.fget
    assert classval == 'clsval'
    new_prop = restored_cls(*state)
    assert new_prop.classval == 'clsval'


def test_instance_property_class_val():
    class MyClass:
        @instanceproperty(classval='class-level')
        def val(self):
            return 'instance-level'

    assert MyClass.val == 'class-level'
    assert MyClass().val == 'instance-level'


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20
    assert cmul(3, 4) == 12


def test_curry_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5
    assert f(2, 3) == 50


def test_curry_init_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_init_not_callable_raises():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_genuine_type_error_is_raised():
    @curry
    def f(x):
        return 1 / x if x != 0 else x.undefined

    # wrong type leads TypeError from int + ...
    @curry
    def g(x):
        return x + 1

    with pytest.raises(TypeError):
        g('a', 'b')


def test_curry_genuine_type_error_inside_body():
    @curry
    def f(x):
        return None + x

    # All required arguments present, so a TypeError inside must propagate.
    with pytest.raises(TypeError):
        f(1)


def test_curry_attributes():
    def f(x, y):
        "docstring"
        return x + y

    cf = curry(f)
    assert cf.__doc__ == 'docstring'
    assert cf.__name__ == 'f'
    assert cf.func is f
    assert cf.func_name == 'f'
    assert cf.args == ()
    assert cf.keywords == {}

    cf2 = cf(1)
    assert cf2.args == (1,)

    cf3 = cf(x=1)
    assert cf3.keywords == {'x': 1}


def test_curry_str_repr():
    def f(x, y):
        return x + y

    cf = curry(f)
    assert str(cf) == str(f)
    assert repr(cf) == repr(f)


def test_curry_equality_and_hash():
    def f(x, y):
        return x + y

    a = curry(f)
    b = curry(f)
    assert a == b
    assert hash(a) == hash(b)
    assert (a != b) is False

    c = curry(f, 1)
    d = curry(f, 1)
    assert c == d
    assert hash(c) == hash(d)
    assert a != c
    assert (a == 5) is False
    assert (a != 5) is True


def test_curry_flatten_partial():
    def f(x, y, z):
        return x + y + z

    c1 = curry(f, 1)
    c2 = c1(2)
    assert c2(3) == 6
    assert c2.args == (1, 2)
    assert c2.func is f


def test_curry_bind_and_call():
    def f(x, y):
        return x + y

    c = curry(f)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound.args == (1,)
    assert bound.call(2) == 3


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x, y):
            return x + y

    obj = C()
    assert obj.method(1, 2) == 3
    assert obj.method(1)(2) == 3
    assert C.method.func.__name__ == 'method'


def test_curry_signature():
    def f(x, y, a=10, **kwargs):
        return x + y + a

    cf = curry(f, 1)
    sig = cf.__signature__
    assert list(sig.parameters) == ['y', 'a', 'kwargs']


def test_curry_signature_incorrect_arguments():
    def f(x):
        return x

    cf = curry(f, 1, 2)
    with pytest.raises(TypeError):
        cf.__signature__


def test_curry_pickle():
    import toolz.functoolz as ft

    c = curry(ft.identity)
    restored = pickle.loads(pickle.dumps(c))
    assert isinstance(restored, curry)
    assert restored.func is ft.identity


def test_curry_pickle_with_args():
    from toolz.functoolz import apply as f_apply

    def _add(x, y):
        return x + y

    # module-level functions pickle; _add is not module level, but use builtin
    c = curry(int, base=2)
    restored = pickle.loads(pickle.dumps(c))
    assert restored('10') == 2


def test_curry_varargs_func():
    @curry
    def f(*args):
        return sum(args)

    assert f(1, 2, 3) == 6
    # With varargs, a TypeError inside is curried anyway -> empty call returns 0
    assert f() == 0


def test_curry_no_signature_falls_back():
    # A callable without an inspectable signature still curries
    try:
        sig = inspect.signature(map)
    except (TypeError, ValueError):
        pass

    class Callable:
        def __call__(self, *args, **kwargs):
            return (args, kwargs)

    c = curry(Callable())
    assert c() == ((), {})


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_counts_calls():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    madd = memoize(add)
    assert madd(1, 2) == 3
    assert madd(1, 2) == 3
    assert len(calls) == 1
    assert madd(2, 3) == 5
    assert len(calls) == 2


def test_memoize_unary():
    calls = []

    def square(x):
        calls.append(x)
        return x * x

    msq = memoize(square)
    assert msq(3) == 9
    assert msq(3) == 9
    assert msq(4) == 16
    assert calls == [3, 4]


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100
    assert add(2, 3) == 5


def test_memoize_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_keyword_function():
    calls = []

    @memoize
    def f(x, y=1):
        calls.append((x, y))
        return x + y

    assert f(1) == 2
    assert f(1) == 2
    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert len(calls) == 2


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_attributes():
    def f(x):
        "docstring"
        return x

    mf = memoize(f)
    assert mf.__name__ == 'f'
    assert mf.__doc__ == 'docstring'
    assert mf.__wrapped__ is f


# ---------------------------------------------------------------------------
# Compose / compose / compose_left / pipe
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_empty_returns_identity():
    assert compose() is identity


def test_compose_single_returns_func():
    inc = lambda i: i + 1
    assert compose(inc) is inc


def test_compose_returns_compose_obj():
    f = lambda x: x
    g = lambda x: x
    c = compose(f, g)
    assert isinstance(c, Compose)


def test_compose_kwargs_forwarded():
    c = compose(lambda x, y: x + y, lambda x: x * 2)
    assert c(3) == 6

    def first(x, y):
        return x + y

    c2 = compose(str, first)
    assert c2(1, 2) == '3'


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert 'f(' in c.__doc__
    assert 'g(' in c.__doc__


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_name_fallback():
    c = Compose((lambda x: x, len))
    assert c.__name__ == '_of_'.join(['lambda', 'len'])


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert repr(c).startswith('Compose(')
    assert 'f' in repr(c)
    assert 'g' in repr(c)


def test_compose_equality():
    def f(x):
        return x

    def g(x):
        return x

    a = compose(f, g)
    b = compose(f, g)
    assert a == b
    assert (a != b) is False
    assert a != compose(g, f)
    assert (a == 5) is False
    assert (a != 5) is True


def test_compose_hash():
    def f(x):
        return x

    def g(x):
        return x

    a = compose(f, g)
    b = compose(f, g)
    assert hash(a) == hash(b)


def test_compose_getstate_setstate():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2(5) == c(5)
    assert c2.first is c.first
    assert c2.funcs == c.funcs


def test_compose_descriptor():
    class C:
        def __init__(self):
            self.mult = 2

        compose_method = compose(lambda x: x + 1, lambda x: x)

    # Descriptor behaves like a method when bound
    obj = C()
    assert obj.compose_method(3) == 3
    assert C.compose_method(3) == 3


def test_compose_signature():
    def f(x: int, y: int) -> str:
        return str(x + y)

    def g(a) -> int:
        return a + 1

    c = compose(f, g)
    sig = c.__signature__
    assert list(sig.parameters) == ['a']
    assert sig.return_annotation is str


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(a: float) -> int:
        return int(a)

    c = compose(f, g)
    ann = c.__annotations__
    assert ann.get('a') is float
    assert ann.get('return') is str


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity
    f = lambda x: x
    assert compose_left(f) is f


def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3
    assert pipe(3, double) == 6


# ---------------------------------------------------------------------------
# complement / juxt / do / flip
# ---------------------------------------------------------------------------

def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


def test_juxt():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=1):
        return x + y

    def g(x, y=1):
        return x * y

    assert juxt(f, g)(2, y=3) == (5, 6)


def test_juxt_state():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(1) == j(1)


def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


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


def test_excepts_non_matching_propagates():
    excepting = excepts(ValueError, lambda a: a[0])
    with pytest.raises(TypeError):
        excepting(None)


def test_excepts_doc():
    def func(a):
        "Docs for func"
        return a[0]

    def handler(e):
        "Docs for handler"
        return None

    excepting = excepts(IndexError, func, handler)
    doc = excepting.__doc__
    assert 'func' in doc
    assert 'handler' in doc
    assert 'IndexError' in doc


def test_excepts_doc_tuple():
    def f(a):
        return a[0]

    excepting = excepts((IndexError, KeyError), f)
    assert '(IndexError, KeyError)' in excepting.__doc__


def test_excepts_doc_default():
    # Class-level docstring when attributes missing
    e = excepts(ValueError, lambda a: a)
    assert 'A wrapper around' in e.__doc__


def test_excepts_name():
    def f(a):
        return a[0]

    def handler(e):
        return None

    excepting = excepts(IndexError, f, handler)
    assert excepting.__name__ == 'f_excepting_IndexError'

    excepting2 = excepts((IndexError, KeyError), f)
    assert excepting2.__name__ == 'f_excepting_IndexError_or_KeyError'


def test_excepts_name_fallback():
    class Callable:
        exc = ValueError

        def __call__(self, x):
            return x

    e = excepts(ValueError, Callable())
    # Callable has no __name__: fallback
    assert e.__name__ == 'excepting'


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


def test_num_required_args_positional_only():
    def f(a, /, b, c=3):
        return a + b + c

    assert num_required_args(f) == 2


def test_has_varargs():
    def f(*args):
        return args

    def g(**kwargs):
        return kwargs

    def h(x):
        return x

    assert has_varargs(f) is True
    assert has_varargs(g) is False
    assert has_varargs(h) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x, y):
        return x + y

    def h(x, *, y):
        return x + y

    def i(*args, **kwargs):
        return kwargs

    assert has_keywords(f) is True
    assert has_keywords(g) is False
    assert has_keywords(h) is True
    assert has_keywords(i) is True


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(add, (1, 2, 3), {}) is False


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    def h(*args):
        return args

    def i(**kwargs):
        return kwargs

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False
    assert is_arity(1, i) is False


def test_builtin_signature_fallbacks():
    # Builtins may not have inspectable signatures; helpers should fall back
    # to the signature registry and not raise.
    res = is_partial_args(map, (), {})
    assert res in (True, False, None)

    res = is_valid_args(map, (), {})
    assert res in (True, False, None)


def test_introspection_with_sigspec():
    def add(x, y):
        return x + y

    sig = inspect.signature(add)
    assert is_valid_args(add, (1, 2), {}, sig) is True
    assert is_partial_args(add, (1,), {}, sig) is True
    assert num_required_args(add, sig) == 2
    assert has_varargs(add, sig) is False
    assert has_keywords(add, sig) is False
    assert is_arity(2, add, sig) is True


def test_curry_signature_after_partial_keyword():
    def f(a, b, c):
        return a + b + c

    cf = curry(f, a=1)
    sig = cf.__signature__
    names = list(sig.parameters)
    assert 'b' in names and 'c' in names
