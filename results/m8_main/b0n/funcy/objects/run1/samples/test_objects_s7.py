import pytest

from funcy.objects import cached_property, cached_readonly, wrap_prop, monkey, LazyObject


# ------------------------- cached_property -------------------------

def test_cached_property_computes_and_caches():
    calls = []

    class A:
        @cached_property
        def value(self):
            """docstring here"""
            calls.append(1)
            return 42

    a = A()
    assert a.value == 42
    assert a.value == 42
    assert len(calls) == 1
    # cached in instance dict
    assert a.__dict__['value'] == 42


def test_cached_property_class_access_returns_descriptor():
    class A:
        @cached_property
        def value(self):
            return 1

    assert isinstance(A.value, cached_property)
    assert A.value.fget is not None


def test_cached_property_doc():
    def f(self):
        """my docs"""
        return 1

    cp = cached_property(f)
    assert cp.__doc__ == "my docs"
    # mimic property attributes
    assert cp.fset is None
    assert cp.fdel is None


def test_cached_property_settable():
    class A:
        @cached_property
        def value(self):
            return 1

    a = A()
    assert a.value == 1
    # Can overwrite since it's just a dict entry (cached_property is non-data descriptor)
    a.value = 5
    assert a.value == 5
    assert a.__dict__['value'] == 5


# ------------------------- cached_readonly -------------------------

def test_cached_readonly_caches():
    calls = []

    class A:
        @cached_readonly
        def value(self):
            calls.append(1)
            return 7

    a = A()
    assert a.value == 7
    assert a.value == 7
    assert len(calls) == 1


def test_cached_readonly_set_raises():
    class A:
        @cached_readonly
        def value(self):
            return 1

    a = A()
    # Access first
    assert a.value == 1
    with pytest.raises(AttributeError):
        a.value = 2


def test_cached_readonly_set_before_access_raises():
    class A:
        @cached_readonly
        def value(self):
            return 1

    a = A()
    with pytest.raises(AttributeError):
        a.value = 2


def test_cached_readonly_class_access():
    class A:
        @cached_readonly
        def value(self):
            return 1

    assert isinstance(A.value, cached_readonly)


# ------------------------- wrap_prop -------------------------

def test_wrap_prop_get():
    events = []
    import contextlib

    @contextlib.contextmanager
    def ctx():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    class A:
        def __init__(self):
            self._x = 10

        @property
        def x(self):
            events.append("get")
            return self._x

        x = wrap_prop(ctx())(x)

    a = A()
    assert a.x == 10
    assert events == ["enter", "get", "exit"]


def test_wrap_prop_set():
    events = []
    import contextlib

    @contextlib.contextmanager
    def ctx():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    class A:
        def __init__(self):
            self._x = 0

        @property
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            events.append("set")
            self._x = value

        x = wrap_prop(ctx())(x)

    a = A()
    a.x = 5
    assert a.x == 5
    assert events == ["enter", "set", "exit", "enter", "exit"]


def test_wrap_prop_delete():
    events = []
    import contextlib

    @contextlib.contextmanager
    def ctx():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    class A:
        def __init__(self):
            self._x = 1

        @property
        def x(self):
            return self._x

        @x.deleter
        def x(self):
            events.append("delete")
            del self._x

        x = wrap_prop(ctx())(x)

    a = A()
    del a.x
    assert events == ["enter", "delete", "exit"]
    assert "_x" not in a.__dict__


def test_wrap_prop_class_access():
    import contextlib

    @contextlib.contextmanager
    def ctx():
        yield

    class A:
        @property
        def x(self):
            return 1

        x = wrap_prop(ctx())(x)

    # Accessing from class returns wrapper, repr reflects wrapped prop
    assert repr(A.x) == repr(A.__dict__['x'])


def test_wrap_prop_class_access_no_instance():
    import contextlib

    entered = []

    @contextlib.contextmanager
    def ctx():
        entered.append(True)
        yield

    class A:
        @property
        def x(self):
            return 1

        x = wrap_prop(ctx())(x)

    # Class access should NOT enter context
    assert A.x is not None
    assert entered == []


# ------------------------- monkey -------------------------

def test_monkey_class():
    class A:
        def foo(self):
            return "original"

    @monkey(A)
    def foo(self):
        return "patched"

    assert A().foo() == "patched"
    assert A.foo.original is not None
    original = A.foo.original
    assert original(A()) == "original"


def test_monkey_prefix_strip():
    class B:
        def __init__(self):
            self.x = 1

    @monkey(B)
    def B__run(self):
        return "ran"

    b = B()
    assert b.run() == "ran"
    assert hasattr(B, "run")
    assert not hasattr(B, "B__run")


def test_monkey_explicit_name():
    class A:
        pass

    @monkey(A, name="custom")
    def other(self):
        return 5

    a = A()
    # function's name is set to custom
    assert A.custom.original is None
    assert A().custom() == 5


def test_monkey_overwrite_original_attr():
    class A:
        def foo(self):
            return 1

    old = A.foo

    @monkey(A)
    def foo(self):
        return 2

    assert A().foo() == 2
    assert A.foo.original is old


def test_monkey_module():
    import types

    mod = types.ModuleType("test_mod")
    mod.value = 1

    @monkey(mod)
    def mod__get_value():
        return 99

    assert mod.get_value() == 99


def test_monkey_non_class_raises():
    with pytest.raises(AssertionError):
        monkey(123)
    with pytest.raises(AssertionError):
        monkey("string")


def test_monkey_property():
    class A:
        pass

    @monkey(A)
    @property
    def val(self):
        return 42

    a = A()
    assert a.val == 42
    # original stored on fget
    assert A.val.fget.original is None


def test_monkey_returns_value():
    class A:
        pass

    @monkey(A)
    def f(self):
        return 0

    assert callable(f)
    # name was rewritten
    assert f.__name__ == "f"


# ------------------------- LazyObject -------------------------

def test_lazy_object_setup_on_access():
    class Real:
        def __init__(self):
            self.value = 10

        def method(self):
            return "real"

    calls = []

    def init():
        calls.append(1)
        return Real()

    obj = LazyObject(init)
    assert len(calls) == 0
    assert obj.value == 10
    assert len(calls) == 1
    assert obj.method() == "real"
    # Only initialized once
    assert len(calls) == 1


def test_lazy_object_class_swap():
    class Real:
        pass

    lo = LazyObject(lambda: Real())
    assert type(lo) is LazyObject
    lo._setup()
    assert type(lo) is Real


def test_lazy_object_setattr_triggers_setup():
    class Real:
        def __init__(self):
            self.x = 1

    obj = LazyObject(lambda: Real())
    obj.y = 5
    assert obj.x == 1
    assert obj.y == 5


def test_lazy_object_getattr_before_setup():
    class Real:
        def __init__(self):
            self.a = "a"

    obj = LazyObject(lambda: Real())
    assert obj.a == "a"


def test_lazy_object_init_attr_allowed_via_dict():
    obj = LazyObject(lambda: object())
    assert "_init" in obj.__dict__
