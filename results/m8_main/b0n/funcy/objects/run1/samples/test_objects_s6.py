import pytest

from funcy.objects import cached_property, cached_readonly, wrap_prop, monkey, LazyObject


# ---------------------------------------------------------------------------
# cached_property
# ---------------------------------------------------------------------------

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
        assert len(calls) == 1

    def test_stores_in_instance_dict(self):
        class A:
            @cached_property
            def value(self):
                return 7

        a = A()
        assert 'value' not in a.__dict__
        a.value
        assert a.__dict__['value'] == 7

    def test_caching_per_instance(self):
        class A:
            def __init__(self, v):
                self.v = v

            @cached_property
            def value(self):
                return self.v * 2

        a = A(1)
        b = A(5)
        assert a.value == 2
        assert b.value == 10
        assert a.value == 2

    def test_access_on_class_returns_descriptor(self):
        class A:
            @cached_property
            def value(self):
                return 1

        assert isinstance(A.__dict__['value'], cached_property)
        assert A.value is A.__dict__['value']

    def test_fset_fdel_are_none(self):
        assert cached_property.fset is None
        assert cached_property.fdel is None

    def test_doc_preserved(self):
        class A:
            @cached_property
            def value(self):
                """my doc"""
                return 1

        assert A.value.__doc__ == "my doc"
        assert A().value == 1

    def test_missing_doc(self):
        class A:
            @cached_property
            def value(self):
                return 1

        # getattr(fget, '__doc__') returns None for no docstring
        assert A.value.__doc__ is None

    def test_fget_attribute(self):
        def getter(self):
            return 3

        cp = cached_property(getter)
        assert cp.fget is getter


# ---------------------------------------------------------------------------
# cached_readonly
# ---------------------------------------------------------------------------

class TestCachedReadonly:
    def test_basic_caching(self):
        calls = []

        class A:
            @cached_readonly
            def value(self):
                calls.append(1)
                return 99

        a = A()
        assert a.value == 99
        assert a.value == 99
        assert len(calls) == 1
        assert a.__dict__['value'] == 99

    def test_access_on_class_returns_descriptor(self):
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
        with pytest.raises(AttributeError, match="property is read-only"):
            a.value = 5

    def test_set_raises_even_after_cached(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        a = A()
        assert a.value == 1
        with pytest.raises(AttributeError):
            a.value = 2

    def test_set_via_class_dict_after_cache(self):
        # Directly putting a value into instance dict makes __get__ short-circuit
        class A:
            @cached_readonly
            def value(self):
                return 1

        a = A()
        a.__dict__['value'] = 123
        assert a.value == 123


# ---------------------------------------------------------------------------
# wrap_prop
# ---------------------------------------------------------------------------

class _Ctx:
    """Context manager recording enter/exit counts."""

    def __init__(self):
        self.enters = 0
        self.exits = 0

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *exc):
        self.exits += 1
        return False


class TestWrapProp:
    def test_get_is_wrapped(self):
        ctx = _Ctx()

        class A:
            def _get(self):
                return 5

            prop = wrap_prop(ctx)(property(_get))

        a = A()
        assert a.prop == 5
        assert ctx.enters == 1
        assert ctx.exits == 1

    def test_access_on_class_returns_wrapper(self):
        ctx = _Ctx()

        class A:
            prop = wrap_prop(ctx)(property(lambda self: 1))

        # Class access returns the descriptor (WrapperProp instance)
        cls_attr = A.prop
        assert cls_attr is A.__dict__['prop']
        assert ctx.enters == 0
        assert ctx.exits == 0

    def test_repr_delegates_to_wrapped_prop(self):
        ctx = _Ctx()
        p = property(lambda self: 1)
        wrapper = wrap_prop(ctx)(p)
        assert repr(wrapper) == repr(p)

    def test_set_is_wrapped(self):
        ctx = _Ctx()
        store = {}

        def getter(self):
            return store.get('v')

        def setter(self, value):
            store['v'] = value

        class A:
            prop = wrap_prop(ctx)(property(getter, setter))

        a = A()
        a.prop = 10
        assert store['v'] == 10
        assert ctx.enters == 1
        assert ctx.exits == 1

        assert a.prop == 10
        assert ctx.enters == 2
        assert ctx.exits == 2

    def test_delete_is_wrapped(self):
        ctx = _Ctx()
        deleted = []

        def getter(self):
            return 1

        def deleter(self):
            deleted.append(True)

        class A:
            prop = wrap_prop(ctx)(property(getter, None, deleter))

        a = A()
        del a.prop
        assert deleted == [True]
        assert ctx.enters == 1
        assert ctx.exits == 1

    def test_no_setter_no_set_method(self):
        ctx = _Ctx()

        class A:
            prop = wrap_prop(ctx)(property(lambda self: 1))

        a = A()
        with pytest.raises(AttributeError):
            a.prop = 5
        # ctx should not have been entered for the failed set
        assert ctx.enters == 0

    def test_returns_new_descriptor_per_decorator_call(self):
        ctx = _Ctx()
        decorator = wrap_prop(ctx)
        p1 = property(lambda self: 1)
        p2 = property(lambda self: 2)
        w1 = decorator(p1)
        w2 = decorator(p2)
        assert w1 is not w2


# ---------------------------------------------------------------------------
# monkey
# ---------------------------------------------------------------------------

class TestMonkey:
    def test_patch_class(self):
        class A:
            def method(self):
                return "original"

        @monkey(A)
        def method(self):
            return "patched"

        assert A().method() == "patched"

    def test_original_accessible(self):
        class A:
            def method(self):
                return "original"

        @monkey(A)
        def method(self):
            return "patched"

        assert method.original is not None
        # Bind the original to an instance
        a = A()
        assert method.original(a) == "original"

    def test_original_none_when_new_attr(self):
        class A:
            pass

        @monkey(A)
        def brand_new(self):
            return "new"

        assert brand_new.original is None
        assert A().brand_new() == "new"

    def test_prefix_stripping(self):
        class A:
            pass

        @monkey(A)
        def A__do_it(self):
            return 1

        assert hasattr(A, 'do_it')
        assert not hasattr(A, 'A__do_it')
        assert A().do_it() == 1
        # __name__ is mutated
        assert A.do_it.__name__ == 'do_it'

    def test_prefix_not_stripped_if_not_present(self):
        class A:
            pass

        @monkey(A)
        def something(self):
            return 2

        assert hasattr(A, 'something')
        assert A.something.__name__ == 'something'

    def test_explicit_name(self):
        class A:
            pass

        @monkey(A, name='custom')
        def whatever(self):
            return 3

        assert hasattr(A, 'custom')
        assert not hasattr(A, 'whatever')
        assert A().custom() == 3

    def test_monkey_patch_module(self):
        import types
        mod = types.ModuleType('test_module')

        @monkey(mod)
        def some_func():
            return 4

        assert mod.some_func is some_func
        assert mod.some_func() == 4

    def test_returns_original_value(self):
        class A:
            pass

        def func(self):
            return 1

        returned = monkey(A)(func)
        assert returned is func

    def test_property_support(self):
        class A:
            pass

        def getter(self):
            return 10

        @monkey(A)
        @property
        def A__prop(self):
            return getter(self)

        a = A()
        assert a.prop == 10
        # original set from property's fget, which had no original -> None
        assert A.prop.fget.original is None

    def test_property_original(self):
        class A:
            @property
            def prop(self):
                return "orig"

        @monkey(A)
        @property
        def prop(self):
            return "patched"

        a = A()
        assert a.prop == "patched"

        class B:
            @property
            def prop(self):
                return "orig"

        @monkey(B)
        @property
        def B__prop(self):
            return "patched"

        b = B()
        assert b.prop == "patched"

    def test_invalid_target_raises_assertion(self):
        with pytest.raises(AssertionError,
                           match="Attempting to monkey patch non-class and non-module"):
            monkey(42)

    def test_invalid_target_string_raises(self):
        with pytest.raises(AssertionError):
            monkey("not a class")


# ---------------------------------------------------------------------------
# LazyObject
# ---------------------------------------------------------------------------

class _Target:
    def __init__(self, value):
        self.value = value
        self.extra = []

    def get(self):
        return self.value * 2


class TestLazyObject:
    def test_getattr_sets_up(self):
        calls = []

        def init():
            calls.append(1)
            return _Target(5)

        lo = LazyObject(init)
        assert calls == []
        assert lo.value == 5
        assert calls == [1]
        # After setup, further access is direct
        assert lo.value == 5
        assert calls == [1]

    def test_setattr_triggers_setup(self):
        calls = []

        def init():
            calls.append(1)
            return _Target(1)

        lo = LazyObject(init)
        lo.value = 42
        assert calls == [1]
        assert lo.value == 42

    def test_class_becomes_target_class(self):
        lo = LazyObject(lambda: _Target(3))
        assert lo.__class__ is LazyObject
        lo.value  # trigger setup
        assert lo.__class__ is _Target
        assert isinstance(lo, _Target)

    def test_dict_replaced(self):
        lo = LazyObject(lambda: _Target(3))
        lo.value
        assert lo.__dict__ == {'value': 3, 'extra': []}

    def test_method_call_after_setup(self):
        lo = LazyObject(lambda: _Target(7))
        assert lo.get() == 14

    def test_init_stored_in_private_dict(self):
        init = lambda: _Target(1)
        lo = LazyObject(init)
        assert lo.__dict__ == {'_init': init}
        assert lo._init is init

    def test_setup_uses_init_once(self):
        calls = []

        def init():
            calls.append(1)
            return _Target(2)

        lo = LazyObject(init)
        lo.value
        lo.value
        lo.extra
        assert calls == [1]

    def test_can_set_new_attribute_after_setup(self):
        lo = LazyObject(lambda: _Target(1))
        lo.value  # setup
        lo.new_attr = 99
        assert lo.new_attr == 99
        assert lo.__dict__['new_attr'] == 99
