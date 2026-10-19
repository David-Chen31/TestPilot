import inspect
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


def test_apply_basic():
    assert apply(lambda x: x * 2, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3


def test_apply_kwargs():
    def f(x, y=0):
        return x + y
    assert apply(f, 1, y=2) == 3


def test_apply_no_function():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


def test_apply_map():
    double = lambda x: 2 * x
    inc = lambda x: x + 1
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------
def test_thread_first():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
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
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_last(1, inc, double) == 4


def test_thread_last_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    inc = lambda x: x + 1
    iseven = lambda x: x % 2 == 0
    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_no_forms():
    assert thread_last(5) == 5


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------
def test_instanceproperty_classval():
    class MyClass:
        @instanceproperty(classval='class-doc')
        def val(self):
            return 42

    assert MyClass.val == 'class-doc'
    assert MyClass().val == 42


def test_instanceproperty_default_none_classval():
    class MyClass:
        @instanceproperty
        def val(self):
            return 42

    assert MyClass.val is None
    assert MyClass().val == 42


def test_instanceproperty_is_property_subclass():
    @instanceproperty
    def f(self):
        return 1

    assert isinstance(f, InstanceProperty)
    assert isinstance(f, property)


def test_instanceproperty_doc():
    class MyClass:
        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'instance doc'

    assert MyClass.__doc__ == __doc__
    assert MyClass().__doc__ == 'instance doc'


def test_instanceproperty_decorator_with_args_not_direct_call():
    # Calling instanceproperty() without fget returns a partial.
    p = instanceproperty(classval=5)
    assert isinstance(p, partial)

    def fget(self):
        return 1

    prop = p(fget)
    assert isinstance(prop, InstanceProperty)
    assert prop.classval == 5


def test_instanceproperty_reduce_roundtrip():
    class MyClass:
        @instanceproperty(classval='cv')
        def val(self):
            return 10

    restored = pickle.loads(pickle.dumps(MyClass.val))
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == 'cv'
    assert restored.fget.__name__ == 'val'


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------
def test_curry_partial_application():
    def mul(x, y):
        return x * y
    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20


def test_curry_decorator_with_kwargs():
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


def test_curry_unpack_partial():
    def add(x, y, z):
        return x + y + z
    p = partial(add, 1)
    c = curry(p, 2)
    assert c(3) == 6
    assert c.func is add
    assert c.args == (1, 2)


def test_curry_unpack_curry_merges_kwargs():
    def add(x=0, y=0):
        return x + y
    c1 = curry(add, x=1)
    c2 = curry(c1, y=2)
    assert c2() == 3


def test_curry_func_args_keywords_props():
    def f(x, y):
        """my doc"""
        return x + y
    c = curry(f, 1)
    assert c.func is f
    assert c.args == (1,)
    assert c.keywords == {}
    assert c.__doc__ == 'my doc'
    assert c.__name__ == 'f'
    assert c.func_name == 'f'


def test_curry_str_repr_hash_eq():
    def f(x, y):
        return x + y
    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert str(c1) == str(f)
    assert repr(c1) == repr(f)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 != c3
    assert not (c1 == 'notacurry')


def test_curry_call_raises_genuine_typeerror():
    def f(x):
        if x < 0:
            raise TypeError('bad value')
        return x
    c = curry(f)
    with pytest.raises(TypeError, match='bad value'):
        c(-1)


def test_curry_bind():
    def add(x, y):
        return x + y
    c = curry(add)
    b = c.bind(1)
    assert isinstance(b, curry)
    assert b(2) == 3


def test_curry_call_method():
    def add(x, y):
        return x + y
    c = curry(add, 1)
    assert c.call(2) == 3


def test_curry_should_curry_with_varargs():
    @curry
    def wrapper(*args, **kwargs):
        if len(args) < 2:
            raise TypeError('need more')
        return args
    # First call raises TypeError; because of varargs it curries.
    partial_obj = wrapper(1)
    assert isinstance(partial_obj, curry)
    assert partial_obj(2) == (1, 2)


def test_curry_should_curry_invalid_args_false():
    def add(x, y):
        return x + y
    c = curry(add)
    with pytest.raises(TypeError):
        c(1, 2, 3)


def test_curry_check_valid_args():
    def add(x, y):
        return x + y
    c = curry(add, 1)
    with pytest.raises(TypeError):
        c(2, 3)  # too many args, cannot be made valid


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z
    c = curry(f, 1)
    sig = c.__signature__
    names = list(sig.parameters)
    assert 'x' not in names
    assert names[0] == 'y'


def test_curry_signature_keyword_as_kwonly():
    def f(x, y, z=3):
        return x + y + z
    c = curry(f, z=10)
    sig = c.__signature__
    params = sig.parameters
    assert params['z'].kind == inspect.Parameter.KEYWORD_ONLY
    assert params['z'].default == 10


def test_curry_signature_incorrect_args():
    def f(x):
        return x
    c = curry(f, 1)
    # Reassign internal partial to have invalid args
    c._partial = partial(f, 1, 2)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_get_descriptor():
    def f(self, x):
        return x * 2

    class C:
        method = curry(f)

    c = C()
    bound = C.__dict__['method'].__get__(c, C)
    assert isinstance(bound, curry)
    assert bound(5) == 10
    # class access returns the curry itself
    assert C.__dict__['method'].__get__(None, C) is C.__dict__['method']


def test_curry_pickling_module_level():
    def add(x, y):
        return x + y
    add.__module__ = __name__
    add.__qualname__ = 'test_curry_pickling_module_level.<locals>.add'
    c = curry(add, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored.func is add
    assert restored.args == (1,)


def test_curry_pickling_decorated_function():
    # Pickling a plain curry wrapping a function whose qualname points
    # back to it works through the module lookup.
    restored = pickle.loads(pickle.dumps(memoize))
    assert isinstance(restored, curry)


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------
def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(add)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_as_decorator():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert len(calls) == 1


def test_memoize_custom_cache():
    @memoize(cache={(1, 2): 99})
    def add(x, y):
        return x + y

    assert add(1, 2) == 99


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_unary():
    calls = []

    @memoize
    def sq(x):
        calls.append(x)
        return x * x

    assert sq(3) == 9
    assert sq(3) == 9
    assert calls == [3]


def test_memoize_kwargs_default_key():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert len(calls) == 1


def test_memoize_unhashable_args_raises():
    @memoize
    def f(x, extra=None):
        return x

    with pytest.raises(TypeError, match='hashable'):
        f([1, 2, 3])


def test_memoize_name_doc_wrapped():
    def original(x):
        """orig doc"""
        return x

    m = memoize(original)
    assert m.__name__ == 'original'
    assert m.__doc__ == 'orig doc'
    assert m.__wrapped__ is original


def test_memoize_name_attribute_error():
    class Callable:
        def __call__(self, x):
            return x
        __doc__ = 'callable doc'

    c = Callable()
    m = memoize(c)
    # __name__ not set on the object, so name assignment is skipped
    assert m.__doc__ == 'callable doc'


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------
def test_compose_basic():
    def inc(x):
        return x + 1
    assert compose(str, inc)(3) == '4'


def test_compose_zero_funcs():
    assert compose() is identity


def test_compose_one_func():
    f = lambda x: x + 1
    assert compose(f) is f


def test_compose_left_basic():
    def inc(x):
        return x + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_zero():
    assert compose_left() is identity


def test_compose_kwargs_first_func():
    def add(x, y):
        return x + y
    c = compose(str, add)
    assert c(1, 2) == '3'


def test_compose_slots():
    f = compose(str, lambda x: x + 1)
    assert isinstance(f, Compose)
    assert f.first.__name__ == '<lambda>'
    assert len(f.funcs) == 1


def test_compose_getstate_setstate():
    f = compose(str, lambda x: x + 1)
    state = f.__getstate__()
    g = Compose.__new__(Compose)
    g.__setstate__(state)
    assert g(2) == '3'


def test_compose_doc():
    def inc(x):
        return x + 1
    c = compose(str, inc)
    assert c.__doc__ == 'lambda *args, **kwargs: str(inc(*args, **kwargs))'


def test_compose_doc_no_names():
    class C:
        def __call__(self, x):
            return x

    c = compose(C(), C())
    assert c.__doc__ == 'A composition of functions'


def test_compose_name():
    def inc(x):
        return x + 1
    c = compose(str, inc)
    assert c.__name__ == 'str_of_inc'


def test_compose_name_no_names():
    class C:
        def __call__(self, x):
            return x

    c = compose(C(), C())
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def inc(x):
        return x + 1
    c = compose(str, inc)
    assert repr(c) == "Compose(<function str>, <function test_compose_repr.<locals>.inc at " + repr(inc)[repr(inc).index('0x'):]


def test_compose_eq_ne_hash():
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
    assert (c1 == 'x') is False
    assert (c1 != 'x') is True
    assert hash(c3) == (hash(f) ^ hash((g,)))


def test_compose_descriptor():
    def add(x, y):
        return x + y

    class C:
        method = compose(str, add)

    c = C()
    bound = c.method
    assert bound(1, 2) == '3'
    # class access returns the Compose itself
    assert C.__dict__['method'].__get__(None, C) is C.__dict__['method']


def test_compose_signature():
    def f(x: int) -> int:
        return x

    def g(y) -> str:
        return str(y)

    c = compose(g, f)
    sig = c.__signature__
    assert sig.return_annotation is str
    assert list(sig.parameters) == ['x']


def test_compose_wrapped():
    def f(x):
        return x + 1
    c = compose(str, f)
    assert c.__wrapped__ is f


def test_compose_annotations():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    ann = c.__annotations__
    assert ann['return'] is str
    # parameter annotation comes from first applied function (f)
    assert ann['x'] is int


def test_compose_annotations_no_return_on_first():
    def f(x: int):
        return x

    def g(y) -> str:
        return str(y)

    c = compose(g, f)
    assert c.__annotations__['return'] is str


def test_compose_annotations_first_no_annotation():
    def f(x):
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    ann = c.__annotations__
    assert 'x' not in ann
    assert ann['return'] is str


def test_compose_annotations_nested_compose():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    def h(z: str) -> str:
        return z + '!'

    inner = compose(g, f)
    outer = compose(h, inner)
    ann = outer.__annotations__
    assert ann['return'] is str
    assert ann['x'] is int


def test_compose_annotations_signature_error_ignored():
    def f(x):
        return x

    # functions without signature contribute nothing
    c = compose(len, f)
    ann = c.__annotations__
    # len has no meaningful signature args matching
    assert isinstance(ann, dict)


def test_instance_annotations_class_access():
    def f(x):
        return x

    c = compose(str, f)
    # Accessing the annotations descriptor on the class returns the dict
    from toolz.functoolz import _InstanceAnnotations
    raw = _InstanceAnnotations.__get__(c.__class__.__dict__.get('__annotations__') or
                                        _InstanceAnnotations(lambda s: {}),
                                        None, c.__class__)
    assert isinstance(raw, _InstanceAnnotations)


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------
def test_pipe():
    double = lambda x: 2 * x
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------
def test_complement():
    iseven = lambda n: n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------
def test_juxt_variadic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_single_sequence():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_single_callable():
    inc = lambda x: x + 1
    assert juxt(inc)(10) == (11,)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y
    assert juxt(f)(1, y=2) == (3,)


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(1) == (2,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------
def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


def test_do_side_effect_in_compose():
    log = []
    inc = compose(lambda x: x + 1, do(log.append))
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


def test_flip_is_curried():
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


def test_excepts_name():
    e = excepts(ValueError, lambda a: [1, 2].index(a))

    def handler(e):
        return -1

    e2 = excepts((ValueError, KeyError), lambda a: a, handler)
    assert 'ValueError' in e.__name__
    assert 'ValueError_or_KeyError' in e2.__name__


def test_excepts_name_no_name():
    class Callable:
        def __call__(self, x):
            return x

    class Exc(Exception):
        pass

    e = excepts(Callable, Callable())
    assert e.__name__ == 'excepting'


def test_excepts_doc():
    def f(a):
        """f doc"""
        return a

    def handler(e):
        """handler doc"""
        return None

    e = excepts(ValueError, f, handler)
    doc = e.__doc__
    assert "'f'" in doc
    assert 'ValueError' in doc
    assert "'handler'" in doc
    assert 'f doc' in doc
    assert 'handler doc' in doc


def test_excepts_doc_class_access():
    assert excepts.__doc__ is not None
    assert 'wrapper' in excepts.__doc__


def test_excepts_doc_no_names():
    class Callable:
        def __call__(self, x):
            return x

    class Exc(Exception):
        pass

    e = excepts(Callable, Callable())
    assert e.__doc__ is not None


def test_excepts_handler_receives_exception():
    seen = []

    def handler(exc):
        seen.append(exc)
        return 'handled'

    e = excepts(ValueError, lambda x: int(x), handler)
    assert e('nope') == 'handled'
    assert isinstance(seen[0], ValueError)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------
def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2


def test_num_required_args_varargs():
    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_num_required_args_positional_only():
    def h(x, y, /, z=1):
        return x + y + z

    assert num_required_args(h) == 2


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


def test_has_keywords_var_keyword():
    def f(**kwargs):
        return kwargs

    assert has_keywords(f) is True


def test_has_keywords_kwonly():
    def f(*, x):
        return x

    assert has_keywords(f) is True


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True


def test_is_valid_args_builtin_map():
    # map has no inspectable signature; use the registry
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False


def test_is_partial_args_builtin_map():
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


def test_num_required_args_with_sigspec():
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2


def test_has_varargs_with_sigspec():
    def f(*args):
        return args

    sig = inspect.signature(f)
    assert has_varargs(f, sig) is True


def test_has_keywords_with_sigspec():
    def f(x, y=0):
        return x + y

    sig = inspect.signature(f)
    assert has_keywords(f, sig) is True


def test_is_valid_args_with_sigspec():
    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert is_valid_args(f, (1, 2), {}, sig) is True
    assert is_valid_args(f, (1,), {}, sig) is False


def test_is_partial_args_with_sigspec():
    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert is_partial_args(f, (1,), {}, sig) is True


def test_is_arity_with_sigspec():
    def f(x):
        return x

    sig = inspect.signature(f)
    assert is_arity(1, f, sig) is True


def test_introspection_builtin_registry():
    # Tests that the _signatures registry path is used for builtins.
    assert num_required_args(map) == 2
    assert has_varargs(map) is False
