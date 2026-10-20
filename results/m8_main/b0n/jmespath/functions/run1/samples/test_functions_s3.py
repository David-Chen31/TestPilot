import json
import math

import pytest

from jmespath import exceptions
from jmespath import functions
from jmespath.functions import (
    signature,
    FunctionRegistry,
    Functions,
    TYPES_MAP,
    REVERSE_TYPES_MAP,
)


# ---------------------------------------------------------------------------
# Helper: a concrete Functions subclass to test each _func_* method.
# ---------------------------------------------------------------------------

class Funcs(Functions):
    pass


@pytest.fixture
def funcs():
    return Funcs()


# ---------------------------------------------------------------------------
# signature decorator
# ---------------------------------------------------------------------------

def test_signature_attaches_signature():
    @signature({'types': ['number']})
    def foo(self, arg):
        return arg

    assert foo.signature == ({'types': ['number']},)


def test_signature_multiple_arguments():
    @signature({'types': ['string']}, {'types': ['number']})
    def foo(self, a, b):
        return None

    assert foo.signature == ({'types': ['string']}, {'types': ['number']})


def test_signature_returns_same_function():
    def foo(self):
        return 1

    decorated = signature({'types': []})(foo)
    assert decorated is foo


# ---------------------------------------------------------------------------
# Type maps
# ---------------------------------------------------------------------------

def test_types_map_contents():
    assert TYPES_MAP['bool'] == 'boolean'
    assert TYPES_MAP['list'] == 'array'
    assert TYPES_MAP['dict'] == 'object'
    assert TYPES_MAP['float'] == 'number'
    assert TYPES_MAP['int'] == 'number'
    assert TYPES_MAP['str'] == 'string'
    assert TYPES_MAP['NoneType'] == 'null'


def test_reverse_types_map_contents():
    assert REVERSE_TYPES_MAP['boolean'] == ('bool',)
    assert REVERSE_TYPES_MAP['number'] == ('float', 'int', 'long')
    assert REVERSE_TYPES_MAP['string'] == ('unicode', 'str')


# ---------------------------------------------------------------------------
# FunctionRegistry / FUNCTION_TABLE population
# ---------------------------------------------------------------------------

def test_function_table_populated_for_known_functions():
    table = Functions.FUNCTION_TABLE
    # Every registered function should have function + signature.
    for spec in table.values():
        assert 'function' in spec
        assert 'signature' in spec


def test_function_table_contains_expected_names():
    table = Functions.FUNCTION_TABLE
    expected = {
        'abs', 'avg', 'not_null', 'to_array', 'to_string', 'to_number',
        'contains', 'length', 'ends_with', 'starts_with', 'reverse',
        'ceil', 'floor', 'join', 'map', 'max', 'merge', 'min', 'sort',
        'sum', 'keys', 'values', 'type', 'sort_by', 'min_by', 'max_by',
    }
    assert expected.issubset(set(table.keys()))


def test_function_table_strips_func_prefix():
    for name in Functions.FUNCTION_TABLE:
        assert not name.startswith('_func_')


def test_custom_subclass_registers_new_function():
    class Custom(Functions):
        @signature({'types': ['number']})
        def _func_double(self, arg):
            return arg * 2

    assert 'double' in Custom.FUNCTION_TABLE
    spec = Custom.FUNCTION_TABLE['double']
    assert spec['signature'] == ({'types': ['number']},)
    assert Custom().call_function('double', [21]) == 42


def test_registry_ignores_undecorated_func_methods():
    class Custom(Functions):
        def _func_nosig(self, arg):
            return arg

    assert 'nosig' not in Custom.FUNCTION_TABLE


# ---------------------------------------------------------------------------
# call_function / arity / unknown function
# ---------------------------------------------------------------------------

def test_call_function_unknown_raises(funcs):
    with pytest.raises(exceptions.UnknownFunctionError) as exc:
        funcs.call_function('doesnotexist', [])
    assert 'doesnotexist' in str(exc.value)


def test_call_function_arity_error(funcs):
    with pytest.raises(exceptions.ArityError):
        funcs.call_function('abs', [])


def test_call_function_extra_args_arity_error(funcs):
    with pytest.raises(exceptions.ArityError):
        funcs.call_function('abs', [1, 2])


def test_variadic_too_few_args(funcs):
    # not_null requires at least one argument (len(signature)==1).
    with pytest.raises(exceptions.VariadictArityError):
        funcs.call_function('not_null', [])


def test_variadic_accepts_many_args(funcs):
    assert funcs.call_function('not_null', [None, None, 'x']) == 'x'


def test_call_function_type_error(funcs):
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs.call_function('abs', ['not-a-number'])


# ---------------------------------------------------------------------------
# _get_allowed_pytypes / type-checking helpers
# ---------------------------------------------------------------------------

def test_get_allowed_pytypes_plain():
    allowed, subtypes = funcs._get_allowed_pytypes(['number'])
    assert set(allowed) == {'float', 'int', 'long'}
    assert subtypes == []


def test_get_allowed_pytypes_with_subtype():
    allowed, subtypes = funcs._get_allowed_pytypes(['array-number'])
    assert 'list' in allowed
    assert subtypes == [('float', 'int', 'long')]


def test_convert_to_jmespath_type_known(funcs):
    assert funcs._convert_to_jmespath_type('int') == 'number'
    assert funcs._convert_to_jmespath_type('str') == 'string'
    assert funcs._convert_to_jmespath_type('NoneType') == 'null'


def test_convert_to_jmespath_type_unknown(funcs):
    assert funcs._convert_to_jmespath_type('MyType') == 'unknown'


def test_type_check_single_bool_not_number(funcs):
    # bool must not be treated as number due to name-based checks.
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._type_check_single(True, ['number'], 'test')


def test_subtype_check_single_allowed(funcs):
    # Should not raise.
    funcs._subtype_check([1, 2, 3], [('float', 'int', 'long')],
                         ['array-number'], 'test')


def test_subtype_check_single_disallowed(funcs):
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._subtype_check([1, 'x'], [('float', 'int', 'long')],
                             ['array-number'], 'test')


def test_subtype_check_dynamic_first_invalid(funcs):
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._subtype_check([True], [('float', 'int', 'long'),
                                      ('str', 'unicode')],
                             ['array-number', 'array-string'], 'test')


def test_subtype_check_dynamic_homogeneous(funcs):
    # First is int -> number; all good.
    funcs._subtype_check([1, 2, 3], [('float', 'int', 'long'),
                                     ('str', 'unicode')],
                         ['array-number', 'array-string'], 'test')


def test_subtype_check_dynamic_mismatch(funcs):
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._subtype_check([1, 'x'], [('float', 'int', 'long'),
                                        ('str', 'unicode')],
                             ['array-number', 'array-string'], 'test')


# ---------------------------------------------------------------------------
# Numeric functions
# ---------------------------------------------------------------------------

def test_abs(funcs):
    assert funcs._func_abs(-5) == 5
    assert funcs._func_abs(3.5) == 3.5


def test_avg(funcs):
    assert funcs._func_avg([1, 2, 3]) == 2
    assert funcs._func_avg([]) is None


def test_ceil(funcs):
    assert funcs._func_ceil(1.2) == 2
    assert funcs._func_ceil(3) == 3


def test_floor(funcs):
    assert funcs._func_floor(1.8) == 1
    assert funcs._func_floor(3) == 3


def test_sum(funcs):
    assert funcs._func_sum([1, 2, 3]) == 6
    assert funcs._func_sum([]) == 0


def test_max(funcs):
    assert funcs._func_max([1, 3, 2]) == 3
    assert funcs._func_max([]) is None


def test_min(funcs):
    assert funcs._func_min([1, 3, 2]) == 1
    assert funcs._func_min([]) is None


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def test_not_null(funcs):
    assert funcs._func_not_null(None, None, 'a', 'b') == 'a'
    assert funcs._func_not_null(None, None) is None


def test_to_array(funcs):
    assert funcs._func_to_array([1, 2]) == [1, 2]
    assert funcs._func_to_array('x') == ['x']
    assert funcs._func_to_array(None) == [None]


def test_to_string(funcs):
    assert funcs._func_to_string('abc') == 'abc'
    assert funcs._func_to_string([1, 2]) == '[1,2]'
    assert funcs._func_to_string({'a': 1}) == '{"a":1}'
    assert funcs._func_to_string(5) == '5'


def test_to_number(funcs):
    assert funcs._func_to_number(5) == 5
    assert funcs._func_to_number(5.5) == 5.5
    assert funcs._func_to_number('5') == 5
    assert funcs._func_to_number('5.5') == 5.5
    assert funcs._func_to_number('abc') is None
    assert funcs._func_to_number(True) is None
    assert funcs._func_to_number(None) is None
    assert funcs._func_to_number([1]) is None
    assert funcs._func_to_number({'a': 1}) is None


# ---------------------------------------------------------------------------
# String / array functions
# ---------------------------------------------------------------------------

def test_contains(funcs):
    assert funcs._func_contains('foobar', 'bar') is True
    assert funcs._func_contains('foobar', 'baz') is False
    assert funcs._func_contains([1, 2, 3], 2) is True
    assert funcs._func_contains([1, 2, 3], 5) is False


def test_length(funcs):
    assert funcs._func_length('abc') == 3
    assert funcs._func_length([1, 2]) == 2
    assert funcs._func_length({'a': 1, 'b': 2}) == 2


def test_ends_with(funcs):
    assert funcs._func_ends_with('foobar', 'bar') is True
    assert funcs._func_ends_with('foobar', 'foo') is False


def test_starts_with(funcs):
    assert funcs._func_starts_with('foobar', 'foo') is True
    assert funcs._func_starts_with('foobar', 'bar') is False


def test_reverse(funcs):
    assert funcs._func_reverse('abc') == 'cba'
    assert funcs._func_reverse([1, 2, 3]) == [3, 2, 1]


def test_join(funcs):
    assert funcs._func_join(',', ['a', 'b', 'c']) == 'a,b,c'
    assert funcs._func_join('', ['a', 'b']) == 'ab'


def test_keys_and_values(funcs):
    d = {'a': 1, 'b': 2}
    assert sorted(funcs._func_keys(d)) == ['a', 'b']
    assert sorted(funcs._func_values(d)) == [1, 2]


def test_sort(funcs):
    assert funcs._func_sort([3, 1, 2]) == [1, 2, 3]
    assert funcs._func_sort(['c', 'a', 'b']) == ['a', 'b', 'c']


# ---------------------------------------------------------------------------
# type()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [
    ('x', 'string'),
    (True, 'boolean'),
    (False, 'boolean'),
    ([1], 'array'),
    ({'a': 1}, 'object'),
    (1.5, 'number'),
    (1, 'number'),
    (None, 'null'),
])
def test_type(funcs, value, expected):
    assert funcs._func_type(value) == expected


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def test_merge(funcs):
    assert funcs._func_merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
    assert funcs._func_merge({'a': 1}, {'a': 2}) == {'a': 2}


# ---------------------------------------------------------------------------
# Expref-based functions
# ---------------------------------------------------------------------------

class FakeExpref:
    """Minimal stand-in for a compiled expref used by map/sort_by etc."""

    def __init__(self, expression):
        self.expression = expression

    def visit(self, expression, element):
        return expression(element)


def test_map(funcs):
    expref = FakeExpref(lambda x: x * 2)
    assert funcs._func_map(expref, [1, 2, 3]) == [2, 4, 6]


def test_map_empty(funcs):
    expref = FakeExpref(lambda x: x)
    assert funcs._func_map(expref, []) == []


def test_sort_by(funcs):
    expref = FakeExpref(lambda x: x['k'])
    array = [{'k': 3}, {'k': 1}, {'k': 2}]
    assert funcs._func_sort_by(array, expref) == [
        {'k': 1}, {'k': 2}, {'k': 3}]


def test_sort_by_empty_returns_same(funcs):
    expref = FakeExpref(lambda x: x)
    assert funcs._func_sort_by([], expref) == []


def test_sort_by_invalid_key_type(funcs):
    expref = FakeExpref(lambda x: True)  # boolean key not allowed
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._func_sort_by([{'a': 1}], expref)


def test_sort_by_mixed_key_types(funcs):
    expref = FakeExpref(lambda x: x['k'])
    array = [{'k': 1}, {'k': 'x'}]
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._func_sort_by(array, expref)


def test_min_by(funcs):
    expref = FakeExpref(lambda x: x['k'])
    array = [{'k': 3}, {'k': 1}, {'k': 2}]
    assert funcs._func_min_by(array, expref) == {'k': 1}


def test_min_by_empty(funcs):
    expref = FakeExpref(lambda x: x)
    assert funcs._func_min_by([], expref) is None


def test_max_by(funcs):
    expref = FakeExpref(lambda x: x['k'])
    array = [{'k': 3}, {'k': 1}, {'k': 2}]
    assert funcs._func_max_by(array, expref) == {'k': 3}


def test_max_by_empty(funcs):
    expref = FakeExpref(lambda x: x)
    assert funcs._func_max_by([], expref) is None


def test_min_by_invalid_key_type(funcs):
    expref = FakeExpref(lambda x: [1])  # array not allowed as key
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._func_min_by([{'a': 1}], expref)


def test_max_by_invalid_key_type(funcs):
    expref = FakeExpref(lambda x: None)  # null not allowed as key
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs._func_max_by([{'a': 1}], expref)


def test_create_key_func_ok(funcs):
    expref = FakeExpref(lambda x: x['k'])
    keyfunc = funcs._create_key_func(expref, ['number', 'string'], 'test')
    assert keyfunc({'k': 5}) == 5


def test_create_key_func_bad_type(funcs):
    expref = FakeExpref(lambda x: True)
    keyfunc = funcs._create_key_func(expref, ['number'], 'test')
    with pytest.raises(exceptions.JMESPathTypeError):
        keyfunc({'k': True})
