import pytest

from funcy.objects import (
    cached_property,
    cached_readonly,
    wrap_prop,
    monkey,
    LazyObject,
)


# ---------------------------------------------------------------------------
# cached_property
# ---------------------------------------------------------------------------

def test_cached_property_caches_value():
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
    # stored in instance __dict__ under function name
    assert a.__dict__['value'] == 42


def test_cached_property_class_access_returns_descriptor():
    class A:
        @cached_property
        def value(self):
            return 1

    assert isinstance(A.value, cached_property)
    assert A.value.fget is A.__dict__['value'].fget


def test_cached_property_doc():
    class A:
        @cached_property
        def value(self):
            """my doc"""
            return 1

    assert A.value.__doc__ == "my doc"


def test_cached_property_fset_fdel_are_none():
    class A:
        @cached_property
        def value(self):
            return 1

    assert cached_property.fset is None
    assert cached_property.fdel is None


def test_cached_property_computed_per_instance():
    class A:
        @cached_property
        def value(self):
            return []

    a1, a2 = A(), A()
    a1.value  # both accessed; note the code shadows the attribute with cached result
    # verify each instance caches independently
    a = A()
    a.value.append(1)  # placeholder to ensure it's a list
    assert a.value == [1]


# ---------------------------------------------------------------------------
# cached_readonly
# ---------------------------------------------------------------------------

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


def test_cached_readonly_reads_existing_dict_value():
    class A:
        @cached_readonly
        def value(self):
            raise AssertionError("should not be called")

    a = A()
    a.__dict__['value'] = "preset"
    assert a.value == "preset"


def test_cached_readonly_set_raises():
    class A:
        @cached_readonly
        def value(self):
            return 1

    a = A()
    with pytest.raises(AttributeError) as exc:
        a.value = 5
    assert str(exc.value) == "property is read-only"


def test_cached_readonly_class_access_returns_descriptor():
    class A:
        @cached_readonly
        def value(self):
            return 1

    assert isinstance(A.value, cached_readonly)


# ---------------------------------------------------------------------------
# wrap_prop
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False


def test_wrap_prop_get_wraps_context():
    ctx = _Ctx()

    class A:
        _v = 5

        @wrap_prop(ctx)
        @property
        def value(self):
            return self._v

    a = A()
    assert a.value == 5
    assert ctx.entered == 1
    assert ctx.exited == 1


def test_wrap_prop_class_access_returns_wrapper():
    ctx = _Ctx()

    class A:
        @wrap_prop(ctx)
        @property
        def value(self):
            return 1

    wrapped = A.value
    assert wrapped is not None
    assert ctx.entered == 0  # not accessed via instance


def test_wrap_prop_repr():
    ctx = _Ctx()

    class A:
        @wrap_prop(ctx)
        @property
        def value(self):
            return 1

    assert isinstance(repr(A.value), str)


def test_wrap_prop_set_wraps_context():
    ctx = _Ctx()
    stored = {}

    class A:
        @wrap_prop(ctx)
        @property
        def value(self):
            return stored.get('v')

        @value.setter
        def value(self, val):
            stored['v'] = val

    a = A()
    a.value = 10
    assert stored['v'] == 10
    assert ctx.entered == 1
    assert ctx.exited == 1


def test_wrap_prop_delete_wraps_context():
    ctx = _Ctx()

    class A:
        def __init__(self):
            self._v = 1

        @wrap_prop(ctx)
        @property
        def value(self):
            return self._v

        @value.deleter
        def value(self):
            del self._v

    a = A()
    del a.value
    assert not hasattr(a, '_v')
    assert ctx.entered == 1
    assert ctx.exited == 1


# ---------------------------------------------------------------------------
# monkey
# ---------------------------------------------------------------------------

def test_monkey_patches_class():
    class A:
        def combine(self):
            return 1

    @monkey(A)
    def A__combine(self):
        return 2

    assert A().combine() == 2


def test_monkey_original_attribute():
    class A:
        def act(self):
            return "orig"

    @monkey(A)
    def A__act(self):
        return "new"

    assert A.__dict__['act'].original is not None
    assert A().act() == "new"
    # original is the previously bound function
    assert A.__dict__['act'].original(1) == "orig"


def test_monkey_original_none_when_new():
    class A:
        pass

    @monkey(A)
    def A__newmethod(self):
        return "x"

    assert A.__dict__['newmethod'].original is None
    assert A.__dict__['newmethod'].__name__ == "newmethod"


def test_monkey_explicit_name():
    class A:
        pass

    @monkey(A, 'custom')
    def whatever(self):
        return 5

    assert A().custom() == 5
    assert A.__dict__['custom'].__name__ == "custom"


def test_monkey_with_property():
    class A:
        pass

    @monkey(A)
    @property
    def A__prop(self):
        return 99

    assert A().prop == 99
    # function name adjusted
    assert A.__dict__['prop'].fget.__name__ == "prop"


def test_monkey_patches_module():
    import types
    mod = types.ModuleType('mymod')

    @monkey(mod)
    def mymod__func():
        return 3

    assert mod.func() == 3


def test_monkey_asserts_on_non_class_non_module():
    with pytest.raises(AssertionError) as exc:
        monkey(42)
    assert "Attempting to monkey patch non-class and non-module" in str(exc.value)


def test_monkey_returns_value():
    class A:
        pass

    def fn(self):
        return 1

    decorated = monkey(A)(fn)
    assert decorated is fn


def test_monkey_class_name_prefix_cut_without_cls_name():
    class A:
        pass

    def some_func(self):
        return "v"

    monkey(A)(some_func)
    assert A().some_func() == "v"


# ---------------------------------------------------------------------------
# LazyObject
# ---------------------------------------------------------------------------

class Thing:
    def __init__(self, val):
        self.val = val

    def get(self):
        return self.val


def test_lazy_object_init_and_setup():
    obj = LazyObject(lambda: Thing(5))
    assert isinstance(obj._init, type(lambda: None))
    assert obj.get() == 5
    assert obj.val == 5
    assert isinstance(obj, Thing)


def test_lazy_object_setattr_triggers_setup():
    obj = LazyObject(lambda: Thing(1))
    obj.val = 10
    assert isinstance(obj, Thing)
    assert obj.val == 10


def test_lazy_object_setup_only_once():
    calls = []

    def init():
        calls.append(1)
        return Thing(1)

    obj = LazyObject(init)
    obj.get()
    obj.get()
    assert len(calls) == 1


def test_lazy_object_class_changes():
    obj = LazyObject(lambda: Thing(1))
    assert type(obj) is LazyObject
    obj.get()
    assert type(obj) is Thing
