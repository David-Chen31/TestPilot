import pickle
import sys
import types

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

    assert apply(double, 5) == 10


def test_apply_kwargs():
    def f(a, b, c=0):
        return a + b + c

    assert apply(f, 1, 2, c=10) == 13


def test_apply_map_usage():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


def test_apply_no_args_raises():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_callables():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_callables():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_no_forms():
    assert thread_last(5) == 5


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        @instanceproperty
        def val(self):
            return 42

    assert MyClass.val is None
    assert MyClass().val == 42


def test_instanceproperty_classval():
    class MyClass:
        @instanceproperty(classval='class-doc')
        def doc(self):
            return 'instance-doc'

    assert MyClass.doc == 'class-doc'
    assert MyClass().doc == 'instance-doc'


def test_instanceproperty_partial_usage():
    # calling with no fget returns a partial usable as decorator
    class MyClass:
        @instanceproperty(classval='cv')
        def val(self):
            return 7

    assert MyClass.val == 'cv'
    assert MyClass().val == 7


def test_instanceproperty_setter():
    class MyClass:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    obj = MyClass()
    obj.x = 5
    assert obj.x == 5


def test_instanceproperty_pickle():
    class MyClass:
        @instanceproperty(classval='cv')
        def val(self):
            return 3

    obj = MyClass()
    restored = pickle.loads(pickle.dumps(obj.val))
    assert restored == 3


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    double = curry(mul)(2)
    assert double(10) == 20


def test_curry_decorator():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable_raises():
    with pytest.raises(TypeError, match='Input must be callable'):
        curry(5)


def test_curry_genuine_typeerror_propagates():
    @curry
    def f(x):
        return x + 1

    with pytest.raises(TypeError):
        f('a')


def test_curry_unpack_partial_like():
    from functools import partial
    def add3(a, b, c):
        return a + b + c

    p = partial(add3, 1)
    c = curry(p)
    assert c.args == (1,)
    assert c(2, 3) == 6


def test_curry_merge_kwargs():
    from functools import partial
    def f(a, b, c=0):
        return a + b + c

    p = partial(f, 1, c=5)
    c = curry(p, b=2)
    assert c() == 8


def test_curry_func_attribute():
    def mul(x, y):
        return x * y

    c = curry(mul, 2)
    assert c.func is mul
    assert c.args == (2,)
    assert c.keywords == {}


def test_curry_str_repr():
    def myfunc(a, b):
        return a + b

    c = curry(myfunc)
    assert str(c) == str(myfunc)
    assert repr(c) == repr(myfunc)


def test_curry_equality_hash():
    def f(a, b):
        return a + b

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_curry_call_method():
    def f(a, b):
        return a + b

    c = curry(f)
    assert c.call(1, 2) == 3


def test_curry_bind():
    def f(a, b, c):
        return a + b + c

    c = curry(f)
    c2 = c.bind(1)
    assert isinstance(c2, curry)
    assert c2(2, 3) == 6


def test_curry_get_descriptor():
    class MyClass:
        def method(self, x):
            return x + 1

        cmethod = curry(method)

    obj = MyClass()
    assert obj.cmethod(1) == 2
    # access on class returns curry
    assert isinstance(MyClass.cmethod, curry)


def test_curry_signature():
    import inspect

    @curry
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f(1))
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[0].name == 'y'
    assert params[1].name == 'z'
    assert params[1].default == 3


def test_curry_signature_bad_args():
    def f(x, y):
        return x + y

    c = curry(object.__new__(curry))
    # build via constructor to trigger property; but we want invalid args
    # Use curry(f) and force bad partial args manually is hard; instead
    # use bind which usually raises via signature property
    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_should_curry_varargs():
    @curry
    def f(*args, **kwargs):
        raise TypeError("boom")

    # should curry because *args means unknown
    res = f(1)
    assert isinstance(res, curry)


def test_curry_doc_name_attributes():
    def f(a):
        """my doc"""
        return a

    c = curry(f)
    assert c.__doc__ == 'my doc'
    assert c.__name__ == 'f'
    assert c.__module__ == f.__module__
    assert c.__qualname__ == f.__qualname__


def test_curry_func_name():
    def f(a):
        return a

    c = curry(f)
    assert c.func_name == 'f'


def test_curry_pickle_basic():
    import toolz.functoolz as mod

    c = curry(mod.identity)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(5) == 5


def test_curry_pickle_decorated_function():
    # decorated functions (module-level) should round-trip
    memoized = memoize
    restored = pickle.loads(pickle.dumps(memoized))
    assert isinstance(restored, curry)


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


def test_memoize_preserves_wrapped_and_doc():
    def add(x, y):
        """doc for add"""
        return x + y

    m = memoize(add)
    assert m.__wrapped__ is add
    assert m.__name__ == 'add'
    assert m.__doc__ == 'doc for add'


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100
    assert add(2, 3) == 5


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, verbose=True) == 3
    assert len(calls) == 1


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='hashable'):
        f([1, 2, 3])


def test_memoize_kwargs_key():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert len(calls) == 1


def test_memoize_unary():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1]


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    c = compose(str, inc)
    assert c(3) == '4'


def test_compose_no_args():
    assert compose() is identity


def test_compose_single_func():
    def f(x):
        return x + 1

    assert compose(f) is f


def test_compose_left_basic():
    inc = lambda i: i + 1
    c = compose_left(inc, str)
    assert c(3) == '4'


def test_compose_left_no_args():
    assert compose_left() is identity


def test_compose_multiple_args():
    def add(x, y):
        return x + y

    c = compose(str, add)
    assert c(1, 2) == '3'


def test_compose_attributes():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = compose(double, inc)
    assert isinstance(c, Compose)
    assert c.first is inc
    assert c.funcs == (double,)
    assert c.__wrapped__ is inc
    assert c.__name__ == 'double_of_inc'


def test_compose_eq_hash():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c1 = compose(double, inc)
    c2 = compose(double, inc)
    c3 = compose(inc, double)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_compose_ne_notimplemented():
    c = compose(str, int)
    assert c.__ne__(5) is NotImplemented
    assert c.__eq__(5) is NotImplemented


def test_compose_repr():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = compose(double, inc)
    assert repr(c).startswith('Compose(')


def test_compose_signature():
    import inspect

    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    sig = inspect.signature(c)
    assert sig.parameters['x'].annotation is int
    assert sig.return_annotation is str


def test_compose_doc():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = compose(double, inc)
    assert 'lambda *args, **kwargs:' in c.__doc__
    assert 'double(inc(' in c.__doc__


def test_compose_doc_no_names():
    c = compose(5)  # not callable, single returns 5 though
    # build Compose directly with non-named callables
    class Callable:
        def __call__(self, x):
            return x

    c = Compose([Callable(), Callable()])
    assert c.__doc__ == 'A composition of functions'


def test_compose_name_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = Compose([Callable(), Callable()])
    assert c.__name__ == 'Compose'


def test_compose_descriptor_binding():
    class MyClass:
        @compose(str, lambda self: 1)
        def method(self):
            pass

    obj = MyClass()
    # accessing through instance returns MethodType
    result = obj.method()
    assert callable(obj.method)
    assert result is not None


def test_compose_pickle():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(g, f)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == 8


def test_compose_annotations():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    ann = c.__annotations__
    assert ann('x') if callable(ann) else True  # callable only via instance
    # access via instance
    inst_ann = c.__annotations__
    # On class-level __get__, obj is not None so fget returns dict
    assert inst_ann == {'x': int, 'return': str}


def test_compose_nested_annotations():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    inner = compose(f)
    # build nested explicitly
    c = Compose([g, f])
    ann = c._combined_annotations()
    assert ann == {'x': int, 'return': str}


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


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


def test_juxt_list_arg():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_callable_single():
    f = juxt(lambda x: x + 1)
    assert f(1) == (2,)


def test_juxt_kwargs():
    def f(x, y):
        return x + y

    def g(x, y):
        return x * y

    assert juxt(f, g)(2, y=3) == (5, 6)


def test_juxt_pickle():
    inc = lambda x: x + 1
    j = juxt(inc)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(1) == (2,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_compose():
    log = []
    inc = lambda x: x + 1
    inc = compose(inc, do(log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


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
    data = [1, 'a', 2, 1.5, 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError()) is None


def test_excepts_catches():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_tuple():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_doc():
    def myfunc(a):
        """myfunc doc"""
        return a

    def myhandler(e):
        """handler doc"""
        return None

    e = excepts(ValueError, myfunc, myhandler)
    doc = e.__doc__
    assert "'myfunc'" in doc
    assert 'ValueError' in doc
    assert "'myhandler'" in doc
    assert 'myfunc doc' in doc
    assert 'handler doc' in doc


def test_excepts_doc_tuple():
    def myfunc(a):
        return a

    e = excepts((IndexError, KeyError), myfunc)
    assert '(IndexError, KeyError)' in e.__doc__


def test_excepts_doc_attribute_error():
    e = excepts(ValueError, lambda a: a)
    doc = e.__doc__
    assert 'A wrapper' in doc  # falls back to class doc or partial


def test_excepts_name():
    def myfunc(a):
        return a

    e = excepts(ValueError, myfunc)
    assert e.__name__ == 'myfunc_excepting_ValueError'


def test_excepts_name_tuple():
    def myfunc(a):
        return a

    e = excepts((IndexError, KeyError), myfunc)
    assert e.__name__ == 'myfunc_excepting_IndexError_or_KeyError'


def test_excepts_name_no_attribute():
    class NotCallable:
        __name__ = 'weird'

    class Bad:
        pass

    e = excepts(ValueError, Bad())
    # Bad has no __name__, catches AttributeError
    assert e.__name__ == 'excepting'


# ---------------------------------------------------------------------------
# introspection helpers
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

    def h(x):
        return x

    assert has_varargs(f) is True
    assert has_varargs(g) is False
    assert has_varargs(h) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x):
        return x

    def h(**kwargs):
        return kwargs

    def k(*, a):
        return a

    assert has_keywords(f) is True
    assert has_keywords(g) is False
    assert has_keywords(h) is True
    assert has_keywords(k) is True


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True


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

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False


def test_is_valid_args_builtin():
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args_builtin():
    assert is_partial_args(map, (), {}) is True


def test_num_required_args_builtin():
    # map requires at least one arg in a way signatures can determine
    result = num_required_args(map)
    assert result is not None


def test_has_keywords_builtin():
    result = has_keywords(dict)
    assert result is not None
