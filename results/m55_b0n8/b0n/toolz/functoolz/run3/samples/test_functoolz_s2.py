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


# ---------- identity / apply ----------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply_basic():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3


def test_apply_kwargs():
    assert apply(lambda x, y: x - y, 10, y=3) == 7


def test_apply_requires_func():
    with pytest.raises(TypeError):
        apply()


def test_apply_map_usage():
    double = lambda x: 2 * x
    inc = lambda x: x + 1
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


# ---------- thread_first / thread_last ----------

def test_thread_first_basic():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_first(1, inc, double) == 4


def test_thread_first_tuple_forms():
    add = lambda x, y: x + y
    pow_ = lambda x, y: x ** y
    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_basic():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_last(1, inc, double) == 4


def test_thread_last_tuple_forms():
    add = lambda x, y: x + y
    pow_ = lambda x, y: x ** y
    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    inc = lambda x: x + 1
    iseven = lambda x: x % 2 == 0
    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


# ---------- instanceproperty / InstanceProperty ----------

def test_instanceproperty_classval_and_instance():
    class MyClass:
        @instanceproperty(classval="CV")
        def val(self):
            return 42

    assert MyClass.val == "CV"
    assert MyClass().val == 42


def test_instanceproperty_partial_form():
    # Call without fget returns a partial usable as decorator carrying metadata
    deco = instanceproperty(classval="X")
    class C:
        @deco
        def prop(self):
            return 7
    assert C.prop == "X"
    assert C().prop == 7


def test_instanceproperty_doc_classval():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return "An object docstring"

    obj = MyClass()
    # Note: assigning a property named __doc__ is tricky; check instance access
    assert obj.__doc__ == "An object docstring"


def test_instanceproperty_setter_deleter():
    class C:
        def __init__(self):
            self._x = 1

        @instanceproperty(classval="CLS")
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

        @x.deleter
        def x(self):
            del self._x

    c = C()
    assert c.x == 1
    c.x = 5
    assert c.x == 5
    del c.x
    assert not hasattr(c, "_x")


def test_instanceproperty_reduce():
    def fget(self):
        return 1

    ip = InstanceProperty(fget=fget, classval="cv")
    restored_cls, state = ip.__reduce__()
    assert restored_cls is InstanceProperty
    assert state[0] is fget
    assert state[4] == "cv"


def test_instanceproperty_pickle_roundtrip():
    class C:
        @instanceproperty(classval="CV")
        def val(self):
            return 3

    # InstanceProperty should be picklable via __reduce__
    ip = C.__dict__["val"]
    restored = pickle.loads(pickle.dumps(ip))
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == "CV"


# ---------- curry ----------

def test_curry_basic():
    def mul(x, y):
        return x * y

    mul = curry(mul)
    double = mul(2)
    assert double(10) == 20
    assert mul(3, 4) == 12


def test_curry_as_decorator():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_bind_and_call():
    @curry
    def add(x, y):
        return x + y

    bound = add.bind(1)
    assert bound(2) == 3
    assert add.call(1, 2) == 3


def test_curry_genuine_typeerror_propagates():
    @curry
    def f(x):
        return x + 1

    with pytest.raises(TypeError):
        f("a", "b")  # too many args, cannot be cured


def test_curry_typeerror_from_body():
    @curry
    def f(x):
        raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        f(1)


def test_curry_varargs_always_curries_on_typeerror():
    @curry
    def f(*args, **kwargs):
        raise TypeError("x")

    # Even with valid-looking args, a TypeError causes currying
    result = f(1)
    assert isinstance(result, curry)


def test_curry_equality_hash():
    def add(x, y):
        return x + y

    a = curry(add)
    b = curry(add)
    assert a == b
    assert hash(a) == hash(b)

    c = a(1)
    d = a(1)
    assert c == d
    assert c != a


def test_curry_repr_str():
    def f(x):
        return x

    c = curry(f)
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_properties():
    def f(x, y=1):
        return x + y

    c = curry(f)
    assert c.func is f
    assert c.args == ()
    assert c.keywords == {}
    assert c.func_name == "f"
    assert c.__name__ == "f"
    assert c.__doc__ == f.__doc__


def test_curry_merges_partial():
    def add(x, y, z):
        return x + y + z

    p = curry(add, 1)
    q = p(2)
    assert q(3) == 6
    # Merging existing curry
    r = curry(q, 2)
    assert r.args == (1, 2)


def test_curry_signature_simple():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f)
    sig = c.__signature__
    assert list(sig.parameters) == ["x", "y", "z"]


def test_curry_signature_after_partial():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ["y", "z"]


def test_curry_get_method_binding():
    class C:
        @curry
        def method(self, x):
            return x + 1

    obj = C()
    assert obj.method(5) == 6
    assert C.method(5)  # bound curry when accessed on class


def test_curry_pickle_roundtrip_module_level():
    # Use a module-level function whitelisted for pickle
    c = curry(num_required_args, has_keywords)
    restored = pickle.loads(pickle.dumps(c))
    assert isinstance(restored, curry)
    assert restored.args == (has_keywords,)


def test_curry_docstring_copied():
    def f(x):
        """my doc"""
        return x

    c = curry(f)
    assert c.__doc__ == "my doc"


# ---------- memoize ----------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(add)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_unary():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]


def test_memoize_kwargs():
    calls = []

    @memoize
    def f(x, y=1):
        calls.append(1)
        return x + y

    assert f(1) == 2
    assert f(1) == 2
    assert len(calls) == 1
    assert f(1, y=2) == 3


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append(1)
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2])


def test_memoize_wrapped_and_name():
    def f(x):
        """doc"""
        return x

    m = memoize(f)
    assert m.__wrapped__ is f
    assert m.__name__ == "f"
    assert m.__doc__ == "doc"


def test_memoize_named_cache():
    cache = {}
    def f(x):
        return x * 2

    m = memoize(f, cache=cache)
    assert m(2) == 4
    assert 2 in cache


# ---------- compose / Compose ----------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == "4"


def test_compose_no_funcs():
    assert compose()(5) == 5


def test_compose_single_func():
    f = lambda x: x * 2
    assert compose(f) is f


def test_compose_order():
    f = lambda x: x + "f"
    g = lambda x: x + "g"
    h = lambda x: x + "h"
    assert compose(f, g, h)("") == "hgf"


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == "4"


def test_compose_left_order():
    f = lambda x: x + "f"
    g = lambda x: x + "g"
    h = lambda x: x + "h"
    assert compose_left(f, g, h)("") == "fgh"


def test_compose_left_no_funcs():
    assert compose_left()(5) == 5


def test_compose_equality():
    f = lambda x: x
    g = lambda x: x
    assert compose(f, g) == compose(f, g)
    assert compose(f, g) != compose(g, f)
    assert compose(f, g) != 1


def test_compose_hash():
    f = lambda x: x
    g = lambda x: x
    assert hash(compose(f, g)) == hash(compose(f, g))


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    assert compose(f, g).__name__ == "f_of_g"


def test_compose_name_no_names():
    class NoName:
        def __call__(self, x):
            return x

    c = compose(NoName(), NoName())
    assert c.__name__ == "Compose"


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    r = repr(compose(f, g))
    assert r.startswith("Compose(")


def test_compose_docstring():
    def f(x):
        return x

    def g(x):
        return x

    doc = compose(f, g).__doc__
    assert doc.startswith("lambda *args, **kwargs: f(g(")


def test_compose_class_doc():
    assert Compose.__doc__ == Compose.__doc__  # class-level attr returns string
    assert isinstance(Compose.__doc__, str)


def test_compose_signature():
    def f(x: int, y: int = 0) -> float:
        return float(x + y)

    def g(a: str) -> int:
        return len(a)

    c = compose(f, g)
    sig = c.__signature__
    assert list(sig.parameters) == ["a"]
    assert sig.return_annotation is float


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    assert compose(f, g).__wrapped__ is g


def test_compose_pickle():
    c = compose(abs, int)
    restored = pickle.loads(pickle.dumps(c))
    assert restored("-3") == 3


def test_compose_get_descriptor():
    def add_one(x):
        return x + 1

    class C:
        method = compose(add_one)

    obj = C()
    # Calling via the class needs the instance
    assert obj.method(1) == 2


def test_compose_annotations():
    def f(x: int) -> float:
        return float(x)

    def g(y: str) -> int:
        return int(y)

    c = compose(f, g)
    ann = c.__annotations__
    assert ann == {"return": float}


def test_compose_annotations_empty_for_no_annotation():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__annotations__ == {}


# ---------- pipe ----------

def test_pipe_basic():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == "6"


def test_pipe_no_funcs():
    assert pipe(5) == 5


# ---------- complement ----------

def test_complement():
    iseven = lambda n: n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------- juxt ----------

def test_juxt_args():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_pickles():
    j = juxt(abs, int)
    restored = pickle.loads(pickle.dumps(j))
    assert restored.funcs == (abs, int)


# ---------- do ----------

def test_do_returns_input():
    seen = []
    assert do(seen.append, 5) == 5
    assert seen == [5]


# ---------- flip ----------

def test_flip_basic():
    div = lambda a, b: a // b
    assert flip(div, 2, 6) == 3


def test_flip_curried():
    div = lambda a, b: a // b
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, "a", 2, 1.5, 3]
    assert list(filter(flip(isinstance, int), data)) == [1, 2, 3]


# ---------- return_none / excepts ----------

def test_return_none():
    assert return_none(ValueError()) is None


def test_excepts_basic():
    excepting = excepts(ValueError, lambda a: [1, 2].index(a), lambda _: -1)
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_multiple_and_default():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_name():
    excepting = excepts(ValueError, lambda x: x)
    assert excepting.__name__ == "<lambda>_excepting_ValueError"


def test_excepts_name_tuple():
    def f(x):
        return x
    e = excepts((ValueError, KeyError), f)
    assert e.__name__ == "f_excepting_ValueError_or_KeyError"


def test_excepts_doc_classval():
    assert isinstance(excepts.__doc__, str)


def test_excepts_instance_doc():
    def f(x):
        """func doc"""
        return x

    def handler(e):
        """handler doc"""
        return None

    e = excepts(ValueError, f, handler)
    doc = e.__doc__
    assert "f" in doc
    assert "ValueError" in doc
    assert "handler" in doc


def test_excepts_does_not_catch_other():
    e = excepts(ValueError, lambda x: int(x))
    with pytest.raises(TypeError):
        e(None)


# ---------- introspection helpers ----------

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


def test_has_keywords_var_keyword():
    def f(**kwargs):
        return kwargs

    assert has_keywords(f) is True


def test_has_keywords_kwonly():
    def f(x, *, y):
        return x + y

    assert has_keywords(f) is True


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

    assert is_arity(1, f) is True

    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


def test_is_arity_builtin_known():
    # abs has signature known to inspect
    assert is_arity(1, abs) is True


# ---------- introspection on builtins / unknowns ----------

def test_num_required_args_builtin():
    assert num_required_args(abs) == 1


def test_is_valid_args_builtin():
    assert is_valid_args(abs, (1,), {}) is True
    assert is_valid_args(abs, (), {}) is False


def test_is_partial_args_builtin():
    assert is_partial_args(abs, (), {}) is True


def test_curry_signature_property_for_builtin():
    # curry(abs) should give a signature for abs
    c = curry(abs)
    sig = c.__signature__
    assert list(sig.parameters) == ["x"]
