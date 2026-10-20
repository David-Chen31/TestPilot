import pytest

from funcy.objects import cached_property, cached_readonly, wrap_prop, monkey, LazyObject


# ----------------------------- cached_property -----------------------------

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
        assert calls == [1]  # computed once

    def test_stored_in_instance_dict(self):
        class A:
            @cached_property
            def value(self):
                return "x"

        a = A()
        a.value
        assert a.__dict__["value"] == "x"

    def test_class_access_returns_descriptor(self):
        class A:
            @cached_property
            def value(self):
                return 1

        assert isinstance(A.value, cached_property)
        assert A.value.fget.__name__ == "value"

    def test_doc_preserved(self):
        class A:
            @cached_property
            def value(self):
                """my doc"""
                return 1

        assert A.value.__doc__ == "my doc"

    def test_fset_fdel_are_none(self):
        class A:
            @cached_property
            def value(self):
                return 1

        assert cached_property.fset is None
        assert cached_property.fdel is None

    def test_separate_instances(self):
        class A:
            @cached_property
            def value(self):
                return object()

        a1, a2 = A(), A()
        assert a1.value is a1.value
        assert a2.value is a2.value
        assert a1.value is not a2.value


# ----------------------------- cached_readonly -----------------------------

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

    def test_class_access_returns_descriptor(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        assert isinstance(A.value, cached_readonly)

    def test_set_raises(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        a = A()
        with pytest.raises(AttributeError):
            a.value = 5

    def test_writes_to_instance_dict_directly_are_respected(self):
        calls = []

        class A:
            @cached_readonly
            def value(self):
                calls.append(1)
                return 1

        a = A()
        a.__dict__["value"] = 99
        assert a.value == 99
        assert calls == []

    def test_is_cached_property_subclass(self):
        assert issubclass(cached_readonly, cached_property)


# ------------------------------- wrap_prop ---------------------------------

class TestWrapProp:
    def test_get_wraps_with_context(self):
        events = []

        class Ctx:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *exc):
                events.append("exit")

        class A:
            @wrap_prop(Ctx())
            def value(self):
                events.append("get")
                return 5

        a = A()
        assert a.value == 5
        assert events == ["enter", "get", "exit"]

    def test_class_access_returns_wrapper(self):
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

        class A:
            @wrap_prop(Ctx())
            def value(self):
                return 1

        assert isinstance(A.value, type(A.value))
        assert A.value.__class__.__name__ == "WrapperProp"

    def test_repr_delegates_to_prop(self):
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

        class A:
            @wrap_prop(Ctx())
            def value(self):
                return 1

        prop = A.__dict__["value"]
        # The underlying property repr should appear in wrapper repr
        assert "property" in repr(prop)

    def test_set_wraps_with_context(self):
        events = []

        class Ctx:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *exc):
                events.append("exit")

        class A:
            def _set(self, value):
                events.append("set")
                self._v = value

            def _get(self):
                return getattr(self, "_v", None)

            value = property(_get, _set)

        wrapped = wrap_prop(Ctx())(A.__dict__["value"])
        # Build a class using wrapped property
        class B:
            pass

        B.value = wrapped
        b = B()
        b.value = 10
        assert b._v == 10
        assert events == ["enter", "set", "exit"]

    def test_delete_wraps_with_context(self):
        events = []

        class Ctx:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *exc):
                events.append("exit")

        class A:
            def _get(self):
                return getattr(self, "_v", None)

            def _set(self, value):
                self._v = value

            def _del(self):
                events.append("del")
                del self._v

            value = property(_get, _set, _del)

        wrapped = wrap_prop(Ctx())(A.__dict__["value"])

        class B:
            pass

        B.value = wrapped
        b = B()
        b.value = 3
        del b.value
        assert events[-2:] == ["enter", "exit"]
        assert "del" in events


# -------------------------------- monkey -----------------------------------

class TestMonkey:
    def test_monkey_patch_class(self):
        class Target:
            def greet(self):
                return "orig"

        @monkey(Target)
        def Target__greet(self):
            return "patched"

        assert Target().greet() == "patched"
        # original accessible
        assert Target.greet.original is not None
        assert Target.greet.original(Target()) == "orig"

    def test_monkey_patch_module(self):
        import types
        mod = types.ModuleType("dummy_mod")

        def dummy_mod__hi():
            return "patched"

        monkey(mod)(dummy_mod__hi)
        assert mod.hi() == "patched"
        assert mod.hi.original is None

    def test_explicit_name_overrides_prefix(self):
        class Target:
            pass

        @monkey(Target, name="renamed")
        def whatever(self):
            return "ok"

        assert Target().renamed() == "ok"
        assert whatever.__name__ == "renamed"

    def test_original_is_none_when_new_attribute(self):
        class Target:
            pass

        @monkey(Target)
        def Target__new_attr(self):
            return 1

        assert Target.new_attr.original is None

    def test_asserts_on_non_class_non_module(self):
        with pytest.raises(AssertionError):
            monkey(42)

    def test_monkey_patch_property(self):
        class Target:
            pass

        @monkey(Target)
        @property
        def Target__prop(self):
            return "prop-value"

        assert Target().prop == "prop-value"
        # fget.original holds the original
        assert Target.prop.fget.original is None

    def test_strips_prefix_from_name(self):
        class Target:
            pass

        @monkey(Target)
        def Target__method(self):
            return "m"

        assert hasattr(Target, "method")
        assert not hasattr(Target, "Target__method")


# ------------------------------ LazyObject ---------------------------------

class TestLazyObject:
    def test_lazy_init_on_attribute_access(self):
        calls = []

        class Real:
            def __init__(self):
                calls.append(1)
                self.x = 10

        lazy = LazyObject(Real)
        assert calls == []
        assert lazy.x == 10
        assert calls == [1]
        # Subsequent accesses don't re-init
        assert lazy.x == 10
        assert calls == [1]

    def test_class_replaced_after_setup(self):
        class Real:
            pass

        lazy = LazyObject(Real)
        assert type(lazy) is LazyObject
        _ = lazy.__class__  # triggers? No—__class__ is found on instance normally
        # Access a normal attribute to trigger setup
        lazy.some_attr = 1
        assert isinstance(lazy, Real)

    def test_setattr_triggers_setup(self):
        calls = []

        class Real:
            def __init__(self):
                calls.append("init")

        lazy = LazyObject(Real)
        lazy.attr = "value"
        assert calls == ["init"]
        assert lazy.attr == "value"

    def test_dict_delegates_after_setup(self):
        class Real:
            def __init__(self):
                self.a = 1

        lazy = LazyObject(Real)
        assert lazy.a == 1
        assert "a" in lazy.__dict__

    def test_init_stored_in_dict(self):
        class Real:
            pass

        lazy = LazyObject(Real)
        assert "_init" in lazy.__dict__
        assert lazy.__dict__["_init"] is Real
