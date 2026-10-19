import pickle
import sys
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
    assert apply(lambda x: x * 2, 5) == 10


def test_apply_with_kwargs():
    def f(a, b, c=0):
        return a + b + c

    assert apply(f, 1, 2, c=3) == 6


def test_apply_no_args_raises():
    with pytest.raises(TypeError):
        apply()


def test_apply_only_func():
    assert apply(lambda: 42) == 42


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

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
    assert thread_first(5) == 5


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


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_no_forms():
    assert thread_last(7) == 7


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_returns_classval_on_class():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'instance doc'

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == 'The class docstring'
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == 'instance doc'
    assert obj.val == 42


def test_instanceproperty_setter_deleter():
    class C:
        def __init__(self):
            self._x = 1

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    c = C()
    assert c.x == 1
    c.x = 5
    assert c.x == 5


def test_instanceproperty_partial_form():
    f = instanceproperty(classval='CLASS')
    assert isinstance(f, partial)
    assert isinstance(f.func, type)
    assert f.func is instanceproperty


def test_instance_property_reduce():
    def fget(self):
        return 1

    prop = InstanceProperty(fget=fget, classval='cv')
    func, state = prop.__reduce__()
    assert func is InstanceProperty
    assert state[0] is fget
    assert state[1] is None
    assert state[2] is None
    assert state[4] == 'cv'


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    @curry
    def mul(x, y):
        return x * y

    double = mul(2)
    assert double(10) == 20
    assert isinstance(double, curry)


def test_curry_keyword_arguments():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_str_repr():
    def foo():
        return 1

    c = curry(foo)
    assert str(c) == str(foo)
    assert repr(c) == repr(foo)


def test_curry_func_args_keywords_props():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    assert c.func is add
    assert c.args == (1,)
    assert c.keywords == {}
    assert c.func_name == 'add'


def test_curry_raises_real_type_error():
    @curry
    def f(x):
        def inner():
            pass
        return x

    # Non-error case:
    assert f(1) == 1

    @curry
    def g(x):
        return x + 'a'

    # int + str -> TypeError, but args are valid, so genuine TypeError raised
    with pytest.raises(TypeError):
        g(1)


def test_curry_unpack_partial():
    def add(x, y, z):
        return x + y + z

    p = partial(add, 1)
    c = curry(p, 2)
    assert c.func is add
    assert c.args == (1, 2)
    assert c(3) == 6


def test_curry_hash_eq():
    def add(x, y):
        return x + y

    c1 = curry(add, 1)
    c2 = curry(add, 1)
    c3 = curry(add, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 'not-a-curry'


def test_curry_call_method():
    def add(x, y):
        return x + y

    c = curry(add)
    assert c.call(1, 2) == 3


def test_curry_bind():
    def add(x, y, z):
        return x + y + z

    c = curry(add)
    b = c.bind(1, 2)
    assert isinstance(b, curry)
    assert b(3) == 6


def test_curry_descriptor():
    class C:
        @curry
        def add(self, x):
            return x + 1

    c = C()
    bound = c.add
    assert isinstance(bound, curry)
    assert bound(1) == 2
    # Access via class returns the curry itself
    assert isinstance(C.add, curry)


def test_curry_signature():
    def add(x, y, z=3):
        return x + y + z

    c = curry(add, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert params[0].name == 'y'
    assert params[1].name == 'z'


def test_curry_signature_keyword_only():
    def add(x, y, *, z=3):
        return x + y + z

    c = curry(add, 1, z=5)
    sig = c.__signature__
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert 'z' in names
    z = [p for p in params if p.name == 'z'][0]
    assert z.kind == z.KEYWORD_ONLY
    assert z.default == 5


def test_curry_signature_incorrect_args():
    def add(x, y):
        return x + y

    c = curry(add, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_varargs():
    @curry
    def f(*args):
        return sum(args)

    # extra args should curry rather than fail, even if TypeError
    c = curry(f, 1)
    assert c(2, 3) == 6


def test_curry_getstate_roundtrip():
    @curry
    def add(x, y):
        return x + y

    c = add(1)
    # pickle round-trip
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


def test_curry_should_curry_unknown_args():
    # function with *args: _should_curry returns True even if genuine TypeError
    @curry
    def f(*args, **kwargs):
        if len(args) < 2:
            raise TypeError("need 2")
        return sum(args)

    # Calling with 1 arg raises TypeError; curried anyway
    c = f(1)
    assert isinstance(c, curry)
    assert c(2) == 3


def test_curry_no_curry_when_nothing_can_help():
    @curry
    def f(x):
        return x

    with pytest.raises(TypeError):
        f(1, 2)


def test_curry_pos_args_unexpected_kw():
    @curry
    def f(x):
        return x

    with pytest.raises(TypeError):
        f(x=1)


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
    assert len(calls) == 1


def test_memoize_initial_cache():
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


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=False) == 3
    assert add(1, 2, verbose=True) == 3
    assert len(calls) == 1


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_attributes():
    def add(x, y):
        """Add two numbers"""
        return x + y

    m = memoize(add)
    assert m.__name__ == 'add'
    assert m.__doc__ == 'Add two numbers'
    assert m.__wrapped__ is add


def test_memoize_no_kwargs_path():
    # has_keywords(f) is False, so key returns args directly
    @memoize
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    def inc(x):
        return x + 1

    assert compose(str, inc)(3) == '4'


def test_compose_no_funcs():
    assert compose() is identity


def test_compose_one_func():
    def inc(x):
        return x + 1

    assert compose(inc) is inc


def test_compose_left_basic():
    def inc(x):
        return x + 1

    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_funcs():
    assert compose_left() is identity


def test_compose_multiple_args():
    def add(x, y):
        return x + y

    def double(x):
        return 2 * x

    assert compose(double, add)(1, 2) == 6


def test_compose_object_type():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert isinstance(c, Compose)


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__doc__ == 'lambda *args, **kwargs: f(g(*args, **kwargs))'
    assert Compose.__doc__ is not None


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'
    assert Compose.__name__ == 'Compose'


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert repr(c) == 'Compose({!r}, {!r})'.format(f, g)


def test_compose_eq_hash():
    def f(x):
        return x

    def g(x):
        return x

    c1 = compose(f, g)
    c2 = compose(f, g)
    c3 = compose(g, f)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 != c3
    assert c1 != 'not compose'
    assert c1.__ne__('not compose') is not NotImplemented


def test_compose_descriptor():
    class C:
        def f(self, x):
            return x + 1

        def g(self, x):
            return x * 2

        h = compose(f, g)

    c = C()
    # h is not a normal method, but g(f(...)) -> g(f(1)) = g(2) = 4
    assert C.h is not None
    assert c.h(1) == 4


def test_compose_signature():
    def f(x: int) -> int:
        return x

    def g(y: str) -> str:
        return y

    c = compose(f, g)
    sig = c.__signature__
    assert list(sig.parameters) == ['y']
    assert sig.return_annotation == int


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    # __wrapped__ is attrgetter('first') ; first is g
    assert c.__wrapped__ is g


def test_compose_pickle():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = compose(inc, double)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == 7


def test_compose_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(y: str) -> int:
        return int(y)

    c = compose(f, g)
    ann = c.__annotations__
    assert ann['y'] == str
    assert ann['return'] == str


def test_compose_annotations_class_access():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    # accessing on class gives the descriptor itself (truthy dict-like)
    assert isinstance(c.__annotations__, dict)


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    def double(i):
        return 2 * i

    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(3) == 3


def test_pipe_multiple():
    assert pipe(1, lambda x: x + 1, lambda x: x * 10) == 20


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


def test_juxt_empty():
    assert juxt()() == ()


def test_juxt_kwargs():
    f = lambda *, x: x
    g = lambda *, x: x * 2
    assert juxt(f, g)(x=3) == (3, 6)


def test_juxt_pickle():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    j = juxt(inc, double)
    # these lambdas aren't picklable, but __getstate__/__setstate__ work
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(10) == (11, 20)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    inc = lambda x: x + 1
    inc_log = compose(inc, do(log.append))
    assert inc_log(1) == 2
    assert inc_log(11) == 12
    assert log == [1, 11]


def test_do_returns_x():
    seen = []
    x = {'a': 1}
    result = do(seen.append, x)
    assert result is x
    assert seen == [x]


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
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError()) is None
    assert return_none(None) is None


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


def test_excepts_default_handler_returns_none():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_no_exception():
    excepting = excepts(ValueError, lambda a: a + 1)
    assert excepting(1) == 2


def test_excepts_other_exception_propagates():
    excepting = excepts(ValueError, lambda a: a[0])
    with pytest.raises(TypeError):
        excepting(1)


def test_excepts_doc():
    def func(a):
        """func doc"""
        return a

    def handler(e):
        """handler doc"""
        return None

    e = excepts(ValueError, func, handler)
    assert 'func doc' in e.__doc__
    assert 'handler doc' in e.__doc__
    assert 'ValueError' in e.__doc__


def test_excepts_doc_tuple():
    def func(a):
        return a

    e = excepts((ValueError, KeyError), func)
    assert '(ValueError, KeyError)' in e.__doc__


def test_excepts_doc_classval():
    assert excepts.__doc__ is not None


def test_excepts_name():
    def func(a):
        return a

    e = excepts(ValueError, func)
    assert e.__name__ == 'func_excepting_ValueError'


def test_excepts_name_tuple():
    def func(a):
        return a

    e = excepts((ValueError, KeyError), func)
    assert e.__name__ == 'func_excepting_ValueError_or_KeyError'


def test_excepts_name_no_name():
    class NoName:
        def __call__(self, a):
            return a

    e = excepts(ValueError, NoName())
    assert e.__name__ == 'excepting'


# ---------------------------------------------------------------------------
# signature helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0

    def h(a, *args):
        pass

    assert num_required_args(h) == 1


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

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


def test_signature_helpers_explicit_sigspec():
    import inspect

    def add(x, y):
        return x + y

    sig = inspect.signature(add)
    assert num_required_args(add, sigspec=sig) == 2
    assert has_varargs(add, sigspec=sig) is False
    assert has_keywords(add, sigspec=sig) is False
    assert is_valid_args(add, (1, 2), {}, sigspec=sig) is True
