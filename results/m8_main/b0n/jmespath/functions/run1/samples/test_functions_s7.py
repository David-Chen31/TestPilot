import pytest

from jmespath import exceptions
from jmespath.functions import (
    signature,
    FunctionRegistry,
    Functions,
    TYPES_MAP,
    REVERSE_TYPES_MAP,
)


class _VisitResult:
    def __init__(self, mapping, default=None):
        self.mapping = mapping
        self.default = default

    def visit(self, expression, element):
        return self.mapping.get(element, self.default)


def _expref(mapping, default=None):
    class _Expr:
        expression = "dummy"
        visit = _VisitResult(mapping, default).visit
    return _Expr()


# ---------- signature decorator ----------

def test_signature_attaches_signature_tuple():
    @signature({'types': ['number']}, {'types': []})
    def foo(a, b):
        return a

    assert foo.signature == ({'types': ['number']}, {'types': []})
    assert foo() is None or True  # placeholder no-op; foo requires args


def test_signature_returns_same_function_object():
    def foo():
        return 1

    decorated = signature({'types': []})(foo)
    assert decorated is foo
    assert decorated.signature == ({'types': []},)


def test_signature_empty():
    @signature()
    def foo():
        return 2

    assert foo.signature == ()
    assert foo() == 2


# ---------- FunctionRegistry / Functions table ----------

def test_function_table_populated_for_known_functions():
    expected = {
        'abs', 'avg', 'not_null', 'to_array', 'to_string', 'to_number',
        'contains', 'length', 'ends_with', 'starts_with', 'reverse',
        'ceil', 'floor', 'join', 'map', 'max', 'merge', 'min', 'sort',
        'sum', 'keys', 'values', 'type', 'sort_by', 'min_by', 'max_by',
    }
    assert set(Functions.FUNCTION_TABLE.keys()) == expected


def test_function_table_entry_shape():
    entry = Functions.FUNCTION_TABLE['abs']
    assert set(entry.keys()) == {'function', 'signature'}
    assert entry['signature'] == ({'types': ['number']},)
    assert callable(entry['function'])


def test_function_registry_ignores_methods_without_signature():
    class Custom(metaclass=FunctionRegistry):
        def _func_no_signature(self, x):
            return x

        @signature({'types': []})
        def _func_has_signature(self, x):
            return x

        @signature({'types': []})
        def not_prefixed(self, x):
            return x

    assert set(Custom.FUNCTION_TABLE.keys()) == {'has_signature'}


def test_function_registry_strips_prefix():
    class Custom(metaclass=FunctionRegistry):
        @signature({'types': []})
        def _func_foo(self, x):
            return x

    assert 'foo' in Custom.FUNCTION_TABLE


# ---------- call_function / validation ----------

def test_call_function_unknown():
    f = Functions()
    with pytest.raises(exceptions.UnknownFunctionError):
        f.call_function('nope', [])


def test_call_function_abs():
    f = Functions()
    assert f.call_function('abs', [-5]) == 5
    assert f.call_function('abs', [5]) == 5


def test_call_function_arity_error():
    f = Functions()
    with pytest.raises(exceptions.ArityError):
        f.call_function('abs', [1, 2])


def test_call_function_variadic_too_few():
    f = Functions()
    # not_null requires at least 1 argument
    with pytest.raises(exceptions.VariadictArityError):
        f.call_function('not_null', [])


def test_call_function_type_error():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('abs', ['x'])


def test_variadic_merge():
    f = Functions()
    assert f.call_function('merge', [{'a': 1}, {'b': 2}, {'a': 3}]) == {'a': 3, 'b': 2}


# ---------- direct function behaviors ----------

@pytest.fixture
def funcs():
    return Functions()


def test_abs(funcs):
    assert funcs._func_abs(-3) == 3
    assert funcs._func_abs(0) == 0


def test_avg(funcs):
    assert funcs._func_avg([1, 2, 3]) == 2
    assert funcs._func_avg([]) is None


def test_not_null(funcs):
    assert funcs._func_not_null(None, None, 5, 6) == 5
    assert funcs._func_not_null(None, None) is None


def test_to_array(funcs):
    assert funcs._func_to_array([1, 2]) == [1, 2]
    assert funcs._func_to_array(1) == [1]
    assert funcs._func_to_array(None) == [None]


def test_to_string(funcs):
    assert funcs._func_to_string("abc") == "abc"
    assert funcs._func_to_string([1, 2]) == "[1,2]"
    assert funcs._func_to_string({'a': 1}) == '{"a":1}'
    assert funcs._func_to_string(5) == "5"


def test_to_number(funcs):
    assert funcs._func_to_number([1]) is None
    assert funcs._func_to_number({}) is None
    assert funcs._func_to_number(True) is None
    assert funcs._func_to_number(None) is None
    assert funcs._func_to_number(5) == 5
    assert funcs._func_to_number(5.5) == 5.5
    assert funcs._func_to_number("10") == 10
    assert funcs._func_to_number("1.5") == 1.5
    assert funcs._func_to_number("abc") is None


def test_contains(funcs):
    assert funcs._func_contains("foobar", "oba") is True
    assert funcs._func_contains("foobar", "zzz") is False
    assert funcs._func_contains([1, 2, 3], 2) is True
    assert funcs._func_contains([1, 2, 3], 9) is False


def test_length(funcs):
    assert funcs._func_length("abc") == 3
    assert funcs._func_length([1, 2]) == 2
    assert funcs._func_length({'a': 1}) == 1


def test_ends_starts_with(funcs):
    assert funcs._func_ends_with("foobar", "bar") is True
    assert funcs._func_ends_with("foobar", "foo") is False
    assert funcs._func_starts_with("foobar", "foo") is True
    assert funcs._func_starts_with("foobar", "bar") is False


def test_reverse(funcs):
    assert funcs._func_reverse("abc") == "cba"
    assert funcs._func_reverse([1, 2, 3]) == [3, 2, 1]


def test_ceil_floor(funcs):
    assert funcs._func_ceil(1.2) == 2
    assert funcs._func_floor(1.8) == 1


def test_join(funcs):
    assert funcs._func_join(",", ["a", "b"]) == "a,b"
    assert funcs._func_join("", ["a", "b"]) == "ab"


def test_map(funcs):
    expref = _expref({'a': 1, 'b': 2, 'c': 3})
    assert funcs._func_map(expref, ['a', 'b', 'c']) == [1, 2, 3]


def test_max_min(funcs):
    assert funcs._func_max([1, 5, 3]) == 5
    assert funcs._func_max([]) is None
    assert funcs._func_min([4, 2, 8]) == 2
    assert funcs._func_min([]) is None


def test_sort(funcs):
    assert funcs._func_sort([3, 1, 2]) == [1, 2, 3]
    assert funcs._func_sort(["b", "a"]) == ["a", "b"]


def test_sum(funcs):
    assert funcs._func_sum([1, 2, 3]) == 6
    assert funcs._func_sum([]) == 0


def test_keys_values(funcs):
    assert funcs._func_keys({'a': 1, 'b': 2}) == ['a', 'b']
    assert funcs._func_values({'a': 1, 'b': 2}) == [1, 2]


def test_type(funcs):
    assert funcs._func_type("abc") == "string"
    assert funcs._func_type(True) == "boolean"
    assert funcs._func_type([1]) == "array"
    assert funcs._func_type({'a': 1}) == "object"
    assert funcs._func_type(1.5) == "number"
    assert funcs._func_type(3) == "number"
    assert funcs._func_type(None) == "null"


# ---------- sort_by / min_by / max_by ----------

def test_sort_by_empty(funcs):
    expref = _expref({})
    assert funcs._func_sort_by([], expref) == []


def test_sort_by_numbers(funcs):
    data = [{'a': 3}, {'a': 1}, {'a': 2}]
    expref = _expref({id(d): d['a'] for d in data})
    result = funcs._func_sort_by(data, expref)
    assert [d['a'] for d in result] == [1, 2, 3]


def test_sort_by_invalid_required_type(funcs):
    data = [{'a': True}]
    expref = _expref({id(d): d['a'] for d in data})
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._func_sort_by(data, expref)


def test_min_by(funcs):
    data = [{'a': 3}, {'a': 1}, {'a': 2}]
    expref = _expref({id(d): d['a'] for d in data})
    assert funcs._func_min_by(data, expref) == data[1]
    assert funcs._func_min_by([], expref) is None


def test_max_by(funcs):
    data = [{'a': 3}, {'a': 1}, {'a': 2}]
    expref = _expref({id(d): d['a'] for d in data})
    assert funcs._func_max_by(data, expref) == data[0]
    assert funcs._func_max_by([], expref) is None


def test_min_by_wrong_type(funcs):
    data = [{'a': [1]}]
    expref = _expref({id(d): d['a'] for d in data})
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._func_min_by(data, expref)


def test_create_key_func_valid(funcs):
    expref = _expref({id({'a': 1}): 1})
    keyfunc = funcs._create_key_func(expref, ['number'], 'test')
    assert keyfunc({'a': 1}) == 1


def test_create_key_func_invalid(funcs):
    expref = _expref({id({'a': 'x'}): 'x'})
    keyfunc = funcs._create_key_func(expref, ['number'], 'test')
    with pytest.raises(exceptions.JMESPathTypeError):
        keyfunc({'a': 'x'})


# ---------- type checking internals ----------

def test_convert_to_jmespath_type():
    f = Functions()
    assert f._convert_to_jmespath_type('str') == 'string'
    assert f._convert_to_jmespath_type('int') == 'number'
    assert f._convert_to_jmespath_type('bool') == 'boolean'
    assert f._convert_to_jmespath_type('NoneType') == 'null'
    assert f._convert_to_jmespath_type('SomethingElse') == 'unknown'


def test_get_allowed_pytypes():
    f = Functions()
    allowed, subtypes = f._get_allowed_pytypes(['string'])
    assert allowed == ['unicode', 'str']
    assert subtypes == []


def test_get_allowed_pytypes_with_subtype():
    f = Functions()
    allowed, subtypes = f._get_allowed_pytypes(['array-number'])
    assert 'list' in allowed
    assert 'float' in subtypes[0]
    assert 'int' in subtypes[0]


def test_type_check_empty_allowed_types_skipped():
    f = Functions()
    # types empty -> no check, should not raise
    f._type_check([object()], [{'types': []}], 'whatever')


def test_type_check_single_failure():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f._type_check_single("string", ['number'], 'fn')


def test_type_check_single_bool_not_number():
    f = Functions()
    # bool should not be accepted as a number
    with pytest.raises(exceptions.JMESPathTypeError):
        f._type_check_single(True, ['number'], 'fn')


def test_subtype_check_single_allowed():
    f = Functions()
    f._subtype_check([1, 2, 3], [('float', 'int', 'long')], ['array-number'], 'fn')


def test_subtype_check_single_forbidden():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f._subtype_check([1, 'x'], [('float', 'int', 'long')], ['array-number'], 'fn')


def test_subtype_check_dynamic_first_bad():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f._subtype_check([True], [('float', 'int', 'long'), ('str', 'unicode')],
                         ['array-number', 'array-string'], 'fn')


def test_subtype_check_dynamic_mismatch():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f._subtype_check([1, 'x'], [('float', 'int', 'long'), ('str', 'unicode')],
                         ['array-number', 'array-string'], 'fn')


def test_subtype_check_dynamic_empty():
    f = Functions()
    # empty list with multiple subtypes should not raise
    f._subtype_check([], [('float', 'int', 'long'), ('str', 'unicode')],
                     ['array-number', 'array-string'], 'fn')


def test_validate_arguments_branch(funcs):
    # signature with variadic last element, enough args
    funcs._validate_arguments([1], [{'types': [], 'variadic': True}], 'fn')
    # signature with variadic, not enough
    with pytest.raises(exceptions.VariadictArityError):
        funcs._validate_arguments([], [{'types': [], 'variadic': True}], 'fn')</br>
