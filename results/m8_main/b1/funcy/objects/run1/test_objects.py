import types
import pytest

from funcy.objects import (
    cached_property,
    cached_readonly,
    wrap_prop,
    monkey,
    LazyObject,
)


# ----------------------------------------------------------------------------
# cached_property
# ----------------------------------------------------------------------------

class TestCachedProperty:
    def test_basic_caching(self):
        calls = []

        class A:
            @cached_property
            def value(self):
                calls.append(1)
                return 42

        a = A()
        assert a.value == 42
        assert a.value == 42
        assert calls == [1]
        assert a.__dict__['value'] == 42

    def test_class_access_returns_descriptor(self):
        class A:
            @cached_property
            def value(self):
                return 1

        assert isinstance(A.value, cached_property)
        assert A.value.fget is not None

    def test_fset_fdel_none(self):
        class A:
            @cached_property
            def value(self):
                return 1

        assert A.value.fset is None
        assert A.value.fdel is None

    def test_doc(self):
        class A:
            @cached_property
            def value(self):
                """my doc"""
                return 1

        assert A.value.__doc__ == "my doc"

    def test_caching_different_instances(self):
        class A:
            @cached_property
            def value(self):
                return []

        a1 = A()
        a2 = A()
        v1 = a1.value
        v2 = a2.value
        assert v1 is not v2
        assert v1 is a1.value
        assert v2 is a2.value


# ----------------------------------------------------------------------------
# cached_readonly
# ----------------------------------------------------------------------------

class TestCachedReadonly:
    def test_basic_caching(self):
        calls = []

        class A:
            @cached_readonly
            def value(self):
                calls.append(1)
                return 7

        a = A()
        assert a.value == 7
        assert a.value == 7
        assert calls == [1]

    def test_readonly_raises(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        a = A()
        assert a.value == 1
        with pytest.raises(AttributeError, match="read-only"):
            a.value = 5

    def test_readonly_raises_before_compute(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        a = A()
        with pytest.raises(AttributeError, match="read-only"):
            a.value = 5

    def test_class_access(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        assert isinstance(A.value, cached_readonly)

    def test_cached_short_circuit(self):
        calls = []

        class A:
            @cached_readonly
            def value(self):
                calls.append(1)
                return 3

        a = A()
        assert a.value == 3
        assert a.__dict__['value'] == 3
        # Manually change cached value; readonly should return the dict value.
        a.__dict__['value'] = 99
        assert a.value == 99
        assert calls == [1]


# ----------------------------------------------------------------------------
# wrap_prop
# ----------------------------------------------------------------------------

class TestWrapProp:
    def test_wraps_property_get(self):
        events = []

        class Ctx:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *args):
                events.append("exit")

        prop = property(lambda self: 123)
        wrapped = wrap_prop(Ctx())(prop)

        class A:
            value = wrapped

        assert A.value is not None  # class access returns wrapper descriptor
        assert isinstance(A.value, type(wrapped))
        assert repr(A.value) == repr(prop)

        a = A()
        assert a.value == 123
        assert events == ["enter", "exit"]

    def test_wrap_set(self):
        events = []
        store = {}

        class Ctx:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *args):
                events.append("exit")

        def fget(self):
            return store.get("v")

        def fset(self, value):
            store["v"] = value

        prop = property(fget, fset)
        wrapped = wrap_prop(Ctx())(prop)

        class A:
            value = wrapped

        a = A()
        a.value = 55
        assert store["v"] == 55
        # wrap_prop's __set__ is only defined when prop has __set__; the
        # wrapper's __set__ passes through but does not re-run the ctx here
        # because the ctx has already been entered/exited during the get.
        # Actual observed behavior: only one enter/exit pair.
        assert events == ["enter", "exit"]

    def test_wrap_delete(self):
        events = []
        deleted = []

        class Ctx:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *args):
                events.append("exit")

        def fget(self):
            return 1

        def fdel(self):
            deleted.append(True)

        prop = property(fget, fdel=fdel)
        wrapped = wrap_prop(Ctx())(prop)

        class A:
            value = wrapped

        a = A()
        del a.value
        assert deleted == [True]
        assert events == ["enter", "exit"]

    def test_wrap_get_only_no_set(self):
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        prop = property(lambda self: 1)
        wrapped = wrap_prop(Ctx())(prop)

        class A:
            value = wrapped

        a = A()
        with pytest.raises(AttributeError):
            a.value = 2


# ----------------------------------------------------------------------------
# monkey
# ----------------------------------------------------------------------------

class TestMonkey:
    def test_rejects_non_class_non_module(self):
        with pytest.raises(AssertionError, match="non-class and non-module"):
            monkey(lambda: None)

        with pytest.raises(AssertionError):
            monkey("string")

        with pytest.raises(AssertionError):
            monkey(123)

    def test_monkey_class(self):
        class A:
            pass

        @monkey(A)
        def A__foo(self):
            return "bar"

        a = A()
        assert a.foo() == "bar"
        assert A.foo.__name__ == "foo"
        assert A.foo.original is None

    def test_monkey_class_preserves_original(self):
        class A:
            def foo(self):
                return "orig"

        @monkey(A)
        def A__foo(self):
            return "new"

        a = A()
        assert a.foo() == "new"
        assert A.foo.original(a) == "orig"

    def test_monkey_with_explicit_name(self):
        class A:
            pass

        @monkey(A, name="custom")
        def whatever(self):
            return "x"

        a = A()
        assert a.custom() == "x"
        assert A.custom.__name__ == "custom"

    def test_monkey_returns_value(self):
        class A:
            pass

        def A__foo(self):
            return 1

        result = monkey(A)(A__foo)
        assert result is A__foo
        assert A.foo is A__foo

    def test_monkey_module(self):
        mod = types.ModuleType("test_mod")

        @monkey(mod)
        def test_mod__bar():
            return "module-bar"

        assert mod.bar() == "module-bar"
        assert mod.bar.__name__ == "bar"

    def test_monkey_property(self):
        class A:
            pass

        @monkey(A)
        @property
        def A__prop(self):
            return 5

        a = A()
        assert a.prop == 5
        # func_name derived from fget name
        assert A.prop.fget.__name__ == "prop"
        assert A.prop.fget.original is None

    def test_monkey_explicit_name_for_property(self):
        class A:
            pass

        @monkey(A, name="renamed")
        @property
        def A__prop(self):
            return 9

        a = A()
        assert a.renamed == 9
        assert A.renamed.fget.__name__ == "renamed"


# ----------------------------------------------------------------------------
# LazyObject
# ----------------------------------------------------------------------------

class TestLazyObject:
    def test_init_stores_init(self):
        lo = LazyObject(lambda: object())
        assert lo.__dict__ == {"_init": lo.__dict__["_init"]}
        assert callable(lo.__dict__["_init"])

    def test_getattr_triggers_setup(self):
        class Target:
            def __init__(self):
                self.x = 10

            def method(self):
                return self.x * 2

        lo = LazyObject(Target)
        assert lo.x == 10
        assert isinstance(lo, Target)
        assert lo.method() == 20
        assert not hasattr(lo, "_init") or lo._init is not None  # replaced __dict__

    def test_setattr_triggers_setup(self):
        class Target:
            def __init__(self):
                self.y = 1

        lo = LazyObject(Target)
        lo.y = 5
        assert lo.y == 5
        assert isinstance(lo, Target)

    def test_setup_swaps_dict(self):
        class Target:
            def __init__(self):
                self.attr = "abc"

        lo = LazyObject(Target)
        lo.attr  # trigger
        assert lo.__dict__ == {"attr": "abc"}
        assert isinstance(lo, Target)

    def test_lazy_not_initialized_until_access(self):
        calls = []

        def init():
            calls.append(1)
            return Target()

        class Target:
            z = 0

        lo = LazyObject(init)
        assert calls == []
        assert lo.z == 0
        assert calls == [1]

    def test_setattr_before_getattr_triggers_setup(self):
        calls = []

        class Target:
            def __init__(self):
                calls.append("init")

        lo = LazyObject(Target)
        # Setting an attribute calls _setup(), which has __setattr__ overridden
        # on the original LazyObject instance, so setting __class__ triggers
        # the setup path.
        lo.__class__ = Target
        # The LazyObject __setattr__ calls _setup, which invokes Target.
        assert calls == ["init"]
