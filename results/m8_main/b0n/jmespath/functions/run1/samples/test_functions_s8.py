import json

import pytest

from jmespath import exceptions
from jmespath.functions import (
    signature,
    FunctionRegistry,
    Functions,
    TYPES_MAP,
    REVERSE_TYPES_MAP,
)


# ---------------------------------------------------------------------------
# signature decorator
# ---------------------------------------------------------------------------

def test_signature_records_arguments():
    @signature({'types': ['number']})
    def foo(self, arg):
        return arg

    assert foo.signature == ({'types': ['number']},)


def test_signature_multiple_arguments():
    @signature({'types': ['string']}, {'types': [], 'variadic': True})
    def bar(self, *args):
        return args

    assert bar.signature == ({'types': ['string']},
                             {'types': [], 'variadic': True})


def test_signature_returns_same_function():
    def foo(self):
        return 1

    decorated = signature()(foo)
    assert decorated is foo
    assert decorated.signature == ()


# ---------------------------------------------------------------------------
# type maps
# ---------------------------------------------------------------------------

def test_types_map_basic_entries():
    assert TYPES_MAP['str'] == 'string'
    assert TYPES_MAP['unicode'] == 'string'
    assert TYPES_MAP['int'] == 'number'
    assert TYPES_MAP['float'] == 'number'
    assert TYPES_MAP['long'] == 'number'
    assert TYPES_MAP['bool'] == 'boolean'
    assert TYPES_MAP['list'] == 'array'
    assert TYPES_MAP['dict'] == 'object'
    assert TYPES_MAP['NoneType'] == 'null'
    assert TYPES_MAP['_Expression'] == 'expref'


def test_reverse_types_map_entries():
    assert REVERSE_TYPES_MAP['boolean'] == ('bool',)
    assert REVERSE_TYPES_MAP['number'] == ('float', 'int', 'long')
    assert REVERSE_TYPES_MAP['string'] == ('unicode', 'str')
    assert REVERSE_TYPES_MAP['null'] == ('NoneType',)
    assert REVERSE_TYPES_MAP['expref'] == ('_Expression',)
    assert 'list' in REVERSE_TYPES_MAP['array']
    assert 'dict' in REVERSE_TYPES_MAP['object']


# ---------------------------------------------------------------------------
# FunctionRegistry
# ---------------------------------------------------------------------------

def test_function_registry_collects_decorated_functions():
    class MyFuncs(metaclass=FunctionRegistry):
        @signature({'types': ['number']})
        def _func_double(self, arg):
            return arg * 2

    assert 'double' in MyFuncs.FUNCTION_TABLE
    spec = MyFuncs.FUNCTION_TABLE['double']
    assert spec['signature'] == ({'types': ['number']},)
    assert spec['function'].__name__ == '_func_double'


def test_function_registry_ignores_undecorated():
    class MyFuncs(metaclass=FunctionRegistry):
        def _func_undeclared(self, arg):
            return arg

        def not_a_func(self):
            return None

    assert MyFuncs.FUNCTION_TABLE == {}


def test_function_registry_ignores_non_prefix():
    class MyFuncs(metaclass=FunctionRegistry):
        @signature({'types': ['number']})
        def other_name(self, arg):
            return arg

    assert MyFuncs.FUNCTION_TABLE == {}


def test_functions_class_has_registered_builtins():
    table = Functions.FUNCTION_TABLE
    for name in ['abs', 'avg', 'contains', 'length', 'max', 'min', 'sum',
                 'sort', 'keys', 'values', 'merge', 'join', 'reverse',
                 'ceil', 'floor', 'type', 'to_array', 'to_string',
                 'to_number', 'not_null', 'starts_with', 'ends_with',
                 'map', 'sort_by', 'min_by', 'max_by']:
        assert name in table


# ---------------------------------------------------------------------------
# call_function / unknown function
# ---------------------------------------------------------------------------

def test_call_function_unknown():
    f = Functions()
    with pytest.raises(exceptions.UnknownFunctionError) as exc:
        f.call_function('does_not_exist', [])
    assert 'does_not_exist' in str(exc.value)


def test_call_function_abs():
    f = Functions()
    assert f.call_function('abs', [-3]) == 3
    assert f.call_function('abs', [3.5]) == 3.5


def test_call_function_abs_wrong_type():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('abs', ['foo'])


def test_call_function_arity_error():
    f = Functions()
    with pytest.raises(exceptions.ArityError):
        f.call_function('abs', [1, 2])


def test_call_function_variadic_arity_error():
    f = Functions()
    with pytest.raises(exceptions.VariadictArityError):
        f.call_function('not_null', [])


# ---------------------------------------------------------------------------
# individual builtin functions
# ---------------------------------------------------------------------------

def test_avg():
    f = Functions()
    assert f.call_function('avg', [[1, 2, 3]]) == 2.0
    assert f.call_function('avg', [[]]) is None


def test_avg_wrong_element_type():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('avg', [['a', 'b']])


def test_not_null():
    f = Functions()
    assert f.call_function('not_null', [None, None, 5]) == 5
    assert f.call_function('not_null', [None, 3, 5]) == 3
    assert f.call_function('not_null', [None]) is None


def test_to_array():
    f = Functions()
    assert f.call_function('to_array', [[1, 2]]) == [1, 2]
    assert f.call_function('to_array', [1]) == [1]
    assert f.call_function('to_array', ['foo']) == ['foo']
    assert f.call_function('to_array', [None]) == [None]


def test_to_string():
    f = Functions()
    assert f.call_function('to_string', ['foo']) == 'foo'
    assert f.call_function('to_string', [1]) == '1'
    assert f.call_function('to_string', [{'a': 1}]) == '{"a":1}'
    assert f.call_function('to_string', [[1, 2]]) == '[1,2]'


def test_to_number():
    f = Functions()
    assert f.call_function('to_number', [1]) == 1
    assert f.call_function('to_number', [1.5]) == 1.5
    assert f.call_function('to_number', ['42']) == 42
    assert f.call_function('to_number', ['3.14']) == 3.14
    assert f.call_function('to_number', ['abc']) is None
    assert f.call_function('to_number', [None]) is None
    assert f.call_function('to_number', [True]) is None
    assert f.call_function('to_number', [[1]]) is None
    assert f.call_function('to_number', [{'a': 1}]) is None


def test_contains():
    f = Functions()
    assert f.call_function('contains', ['foobar', 'bar']) is True
    assert f.call_function('contains', ['foobar', 'baz']) is False
    assert f.call_function('contains', [[1, 2, 3], 2]) is True
    assert f.call_function('contains', [[1, 2, 3], 5]) is False


def test_length():
    f = Functions()
    assert f.call_function('length', ['foobar']) == 6
    assert f.call_function('length', [[1, 2, 3]]) == 3
    assert f.call_function('length', [{'a': 1, 'b': 2}]) == 2


def test_length_wrong_type():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('length', [5])


def test_ends_with_and_starts_with():
    f = Functions()
    assert f.call_function('ends_with', ['foobar', 'bar']) is True
    assert f.call_function('ends_with', ['foobar', 'foo']) is False
    assert f.call_function('starts_with', ['foobar', 'foo']) is True
    assert f.call_function('starts_with', ['foobar', 'bar']) is False


def test_reverse():
    f = Functions()
    assert f.call_function('reverse', ['abc']) == 'cba'
    assert f.call_function('reverse', [[1, 2, 3]]) == [3, 2, 1]


def test_ceil_floor():
    f = Functions()
    assert f.call_function('ceil', [1.2]) == 2
    assert f.call_function('floor', [1.8]) == 1
    # ints are allowed for numbers
    assert f.call_function('ceil', [3]) == 3
    assert f.call_function('floor', [3]) == 3


def test_join():
    f = Functions()
    assert f.call_function('join', [',', ['a', 'b', 'c']]) == 'a,b,c'
    assert f.call_function('join', ['', ['a', 'b']]) == 'ab'
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('join', [',', [1, 2]])


def test_max_min():
    f = Functions()
    assert f.call_function('max', [[1, 5, 3]]) == 5
    assert f.call_function('max', [['a', 'c', 'b']]) == 'c'
    assert f.call_function('max', [[]]) is None
    assert f.call_function('min', [[1, 5, 3]]) == 1
    assert f.call_function('min', [['a', 'c', 'b']]) == 'a'
    assert f.call_function('min', [[]]) is None


def test_max_mixed_types():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('max', [[1, 'a']])


def test_sort():
    f = Functions()
    assert f.call_function('sort', [[3, 1, 2]]) == [1, 2, 3]
    assert f.call_function('sort', [['c', 'a', 'b']]) == ['a', 'b', 'c']


def test_sum():
    f = Functions()
    assert f.call_function('sum', [[1, 2, 3]]) == 6
    assert f.call_function('sum', [[]]) == 0


def test_keys_values():
    f = Functions()
    assert f.call_function('keys', [{'a': 1, 'b': 2}]) == ['a', 'b']
    assert f.call_function('values', [{'a': 1, 'b': 2}]) == [1, 2]


def test_merge():
    f = Functions()
    assert f.call_function('merge', [{'a': 1}, {'b': 2}]) == {'a': 1, 'b': 2}
    assert f.call_function('merge', [{'a': 1}, {'a': 2}]) == {'a': 2}
    assert f.call_function('merge', [{}]) == {}


def test_type():
    f = Functions()
    assert f.call_function('type', ['foo']) == 'string'
    assert f.call_function('type', [True]) == 'boolean'
    assert f.call_function('type', [[1, 2]]) == 'array'
    assert f.call_function('type', [{'a': 1}]) == 'object'
    assert f.call_function('type', [1]) == 'number'
    assert f.call_function('type', [1.5]) == 'number'
    assert f.call_function('type', [None]) == 'null'


# ---------------------------------------------------------------------------
# expref-based functions
# ---------------------------------------------------------------------------

class FakeExpression(object):
    def __init__(self, func):
        self.expression = func

    def visit(self, expression, value):
        return expression(value)


def test_map():
    f = Functions()
    expref = FakeExpression(lambda x: x * 2)
    assert f.call_function('map', [expref, [1, 2, 3]]) == [2, 4, 6]
    assert f.call_function('map', [expref, []]) == []


def test_sort_by():
    f = Functions()
    expref = FakeExpression(lambda x: x['age'])
    result = f.call_function('sort_by', [[{'age': 3}, {'age': 1}], expref])
    assert result == [{'age': 1}, {'age': 3}]


def test_sort_by_empty():
    f = Functions()
    expref = FakeExpression(lambda x: x['age'])
    assert f.call_function('sort_by', [[], expref]) == []


def test_sort_by_wrong_key_type():
    f = Functions()
    expref = FakeExpression(lambda x: x['val'])
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('sort_by', [[{'val': True}], expref])


def test_sort_by_inconsistent_key_type():
    f = Functions()
    expref = FakeExpression(lambda x: x['val'])
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('sort_by', [[{'val': 1}, {'val': 'a'}], expref])


def test_min_by_max_by():
    f = Functions()
    expref = FakeExpression(lambda x: x['v'])
    data = [{'v': 3}, {'v': 1}, {'v': 2}]
    assert f.call_function('min_by', [data, expref]) == {'v': 1}
    assert f.call_function('max_by', [data, expref]) == {'v': 3}
    assert f.call_function('min_by', [[], expref]) is None
    assert f.call_function('max_by', [[], expref]) is None


def test_min_by_wrong_key_type():
    f = Functions()
    expref = FakeExpression(lambda x: x['v'])
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('min_by', [[{'v': True}], expref])


def test_max_by_wrong_key_type():
    f = Functions()
    expref = FakeExpression(lambda x: x['v'])
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('max_by', [[{'v': None}], expref])


# ---------------------------------------------------------------------------
# helper / internal but reaching via public-ish behavior
# ---------------------------------------------------------------------------

def test_convert_to_jmespath_type():
    f = Functions()
    assert f._convert_to_jmespath_type('str') == 'string'
    assert f._convert_to_jmespath_type('int') == 'number'
    assert f._convert_to_jmespath_type('bool') == 'boolean'
    assert f._convert_to_jmespath_type('nonsense') == 'unknown'


def test_get_allowed_pytypes():
    f = Functions()
    allowed, subtypes = f._get_allowed_pytypes(['array-number'])
    assert 'list' in allowed
    assert subtypes == [('float', 'int', 'long')]


def test_get_allowed_pytypes_no_subtype():
    f = Functions()
    allowed, subtypes = f._get_allowed_pytypes(['string'])
    assert allowed == ['unicode', 'str']
    assert subtypes == []


def test_type_check_allows_empty_types():
    f = Functions()
    # empty types means no type checking
    f._type_check([object()], [{'types': []}], 'anything')


def test_type_check_single_raises():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f._type_check_single('string', ['number'], 'test')


def test_subtype_check_dynamic_multiple_types():
    f = Functions()
    # Multiple allowed subtypes, consistent first element type -> ok
    f._subtype_check(
        [1, 2, 3],
        [('float', 'int', 'long'), ('unicode', 'str')],
        ['array-number', 'array-string'],
        'test',
    )


def test_subtype_check_dynamic_mismatch():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f._subtype_check(
            [1, 'a'],
            [('float', 'int', 'long'), ('unicode', 'str')],
            ['array-number', 'array-string'],
            'test',
        )


def test_subtype_check_unknown_first_type():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f._subtype_check(
            [True],
            [('float', 'int', 'long'), ('unicode', 'str')],
            ['array-number', 'array-string'],
            'test',
        )


def test_bool_is_not_number_for_type_check():
    # Bool is a subclass of int in Python but jmespath considers
    # it a distinct type, so it should fail a number check.
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('abs', [True])
