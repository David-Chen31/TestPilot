import pytest

from funcy.objects import cached_property, cached_readonly, wrap_prop, monkey, LazyObject


# ---------------------------------------------------------------- cached_property

def test_cached_property_basic():
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
    # stored on the instance dict under the function name
    assert a.__dict__['value'] == 42


def test_cached_property_class_access_returns_descriptor():
    class A:
        @cached_property
        def value(self):
            return 1

    assert isinstance(A.value, cached_property)
    assert A.value.fget is not None
    assert A.value.fset is None
    assert A.value.fdel is None


def test_cached_property_doc():
    class A:
        @cached_property
        def value(self):
            "some doc"
            return 1

    assert A.value.__doc__ == "some doc"


def test_cached_property_no_doc():
    def f(self):
        return 1

    cp = cached_property(f)
    assert cp.__doc__ is None


def test_cached_property_different_instances():
    class A:
        @cached_property
        def value(self):
            return object()

    a, b = A(), A()
    assert a.value is a.value
    assert b.value is b.value
    assert a.value is not b.value


def test_cached_property_caches_none():
    calls = []

    class A:
        @cached_property
        def value(self):
            calls.append(1)
            return None

    a = A()
    assert a.value is None
    assert a.value is None
    assert calls == [1]


# ---------------------------------------------------------------- cached_readonly

def test_cached_readonly_basic():
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
    assert a.__dict__['value'] == 7


def test_cached_readonly_class_access_returns_descriptor():
    class A:
        @cached_readonly
        def value(self):
            return 1

    assert isinstance(A.value, cached_readonly)


def test_cached_readonly_set_raises():
    class A:
        @cached_readonly
        def value(self):
            return 1

    a = A()
    with pytest.raises(AttributeError):
        a.value = 5


def test_cached_readonly_direct_dict_write_bypasses():
    # The check is on instance __dict__, so pre-seeding __dict__ is returned.
    class A:
        @cached_readonly
        def value(self):
            return 1

    a = A()
    a.__dict__['value'] = 99
    assert a.value == 99


def test_cached_readonly_inherits_cached_property():
    assert issubclass(cached_readonly, cached_property)


# ---------------------------------------------------------------- wrap_prop

class Ctx:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False


def test_wrap_prop_get():
    ctx = Ctx()

    class A:
        @wrap_prop(ctx)
        @property
        def value(self):
            return 5

    a = A()
    assert a.value == 5
    assert ctx.entered == 1
    assert ctx.exited == 1


def test_wrap_prop_class_access_returns_wrapper():
    ctx = Ctx()

    class A:
        @wrap_prop(ctx)
        @property
        def value(self):
            return 5

    # class access returns the wrapper itself, not invoking prop
    result = A.value
    assert ctx.entered == 0
    assert type(result).__name__ == 'WrapperProp'


def test_wrap_prop_repr():
    ctx = Ctx()

    class A:
        @wrap_prop(ctx)
        @property
        def value(self):
            return 5

    prop = A.__dict__['value']
    inner = prop.__class__.__mro__  # ensure class exists
    assert repr(prop).startswith('<property object')


def test_wrap_prop_set():
    ctx = Ctx()
    store = {}

    class A:
        @wrap_prop(ctx)
        @property
        def value(self):
            return store.get('v')

        @value.setter
        def value(self, v):
            store['v'] = v

    a = A()
    a.value = 10
    assert store['v'] == 10
    assert ctx.entered == 1
    assert ctx.exited == 1


def test_wrap_prop_delete():
    ctx = Ctx()
    store = {'v': 1}

    class A:
        @wrap_prop(ctx)
        @property
        def value(self):
            return store.get('v')

        @value.deleter
        def value(self):
            del store['v']

    a = A()
    del a.value
    assert 'v' not in store
    assert ctx.entered == 1
    assert ctx.exited == 1


def test_wrap_prop_setter_calls_underlying():
    # check __set__ forwards args (name, value)
    ctx = Ctx()
    calls = []

    class P:
        def __get__(self, instance, type=None):
            return 'get'

        def __set__(self, name, value):
            calls.append((name, value))

    wrapper = wrap_prop(ctx)(P())
    wrapper.__set__('x', 3)
    assert calls == [('x', 3)]
    assert ctx.entered == 1


def test_wrap_prop_no_setter_has_no_set():
    ctx = Ctx()

    class P:
        def __get__(self, instance, type=None):
            return 'get'

    wrapper = wrap_prop(ctx)(P())
    assert hasattr(P, '__set__') is False
    # WrapperProp class only defines __set__ conditionally at class creation;
    # P lacks __set__ so WrapperProp should not have one.
    assert '__set__' not in type(wrapper).__dict__


# ---------------------------------------------------------------- monkey

def test_monkey_class_prefix_strip():
    class MyClass:
        pass

    @monkey(MyClass)
    def MyClass__method(self):
        return 'hi'

    assert MyClass().method() == 'hi'
    assert MyClass.method.__name__ == 'method'


def test_monkey_with_explicit_name():
    class MyClass:
        pass

    @monkey(MyClass, name='renamed')
    def whatever():
        return 1

    assert MyClass.renamed() == 1
    assert MyClass.renamed.__name__ == 'renamed'


def test_monkey_original_saved():
    class MyClass:
        def method(self):
            return 'orig'

    orig = MyClass.method

    @monkey(MyClass)
    def MyClass__method(self):
        return 'new'

    assert MyClass().method() == 'new'
    assert MyClass.method.original is orig


def test_monkey_original_none_when_new():
    class MyClass:
        pass

    @monkey(MyClass)
    def MyClass__fresh():
        return 1

    assert MyClass.fresh.original is None


def test_monkey_returns_value():
    class MyClass:
        pass

    def fn():
        return 1

    result = monkey(MyClass)(fn)
    assert result is fn


def test_monkey_property():
    class MyClass:
        pass

    @monkey(MyClass)
    @property
    def MyClass__prop(self):
        return 42

    # value returned is the property itself
    assert isinstance(MyClass.prop, property)
    assert MyClass().prop == 42
    assert MyClass.prop.fget.__name__ == 'prop'


def test_monkey_module():
    import types
    mod = types.ModuleType('tmp_mod')

    @monkey(mod)
    def tmp_mod__func():
        return 'x'

    assert mod.func() == 'x'


def test_monkey_non_class_non_module_raises():
    with pytest.raises(AssertionError):
        monkey(42)


def test_monkey_function_raises():
    def f():
        pass

    with pytest.raises(AssertionError):
        monkey(f)


# ---------------------------------------------------------------- LazyObject

def test_lazy_object_init_and_setup():
    calls = []

    class Target:
        def __init__(self):
            calls.append(1)
            self.value = 5

        def method(self):
            return 'm'

    class Inner:
        value = 10

        def method(self):
            return 'm'

    def make():
        calls.append('made')
        return Inner()

    lazy = LazyObject(make)
    assert lazy.__dict__ == {'_init': make}

    assert lazy.value == 10
    assert calls == ['made']
    assert lazy.method() == 'm'


def test_lazy_object_becomes_class_of_wrapped():
    class Target:
        def __init__(self):
            self.x = 1

    lazy = LazyObject(Target)
    assert isinstance(lazy, LazyObject)
    lazy.x
    assert isinstance(lazy, Target)


def test_lazy_object_setattr_triggers_setup():
    class Target:
        def __init__(self):
            self.created = True

    lazy = LazyObject(Target)
    lazy.newattr = 3
    assert lazy.newattr == 3
    assert lazy.created is True


def test_lazy_object_setup_only_once():
    count = []

    class Target:
        def __init__(self):
            count.append(1)
            self.value = 1

    lazy = LazyObject(Target)
    assert lazy.value == 1
    assert lazy.value == 1
    assert len(count) == 1


def test_lazy_object_dict_replaced():
    class Target:
        def __init__(self):
            self.value = 'v'

    lazy = LazyObject(Target)
    lazy.value
    assert lazy.__dict__ == {'value': 'v'}
    assert '_init' not in lazy.__dict__
