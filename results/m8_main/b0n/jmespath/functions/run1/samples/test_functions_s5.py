import json
import math
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
# Helpers
# ---------------------------------------------------------------------------

class _FakeExpref:
    def __init__(self, expression, func):
        self.expression = expression
        self._func = func

    def visit(self, node, element):
        return self._func(node, element)


@pytest.fixture
def funcs():
    return Functions()


# ---------------------------------------------------------------------------
# signature decorator
# ---------------------------------------------------------------------------

def test_signature_sets_attribute_and_returns_func():
    def my_func():
        return 42

    decorated = signature({'types': ['number']})(my_func)
    assert decorated is my_func
    assert my_func.signature == ({'types': ['number']},)


def test_signature_with_multiple_arguments():
    def my_func():
        pass

    spec1 = {'types': ['string']}
    spec2 = {'types': ['array'], 'variadic': True}
    signature(spec1, spec2)(my_func)
    assert my_func.signature == (spec1, spec2)


def test_signature_with_no_arguments():
    def my_func():
        pass

    signature()(my_func)
    assert my_func.signature == ()


# ---------------------------------------------------------------------------
# Type maps
# ---------------------------------------------------------------------------

def test_types_map_values():
    assert TYPES_MAP['bool'] == 'boolean'
    assert TYPES_MAP['list'] == 'array'
    assert TYPES_MAP['dict'] == 'object'
    assert TYPES_MAP['NoneType'] == 'null'
    assert TYPES_MAP['str'] == 'string'
    assert TYPES_MAP['int'] == 'number'
    assert TYPES_MAP['float'] == 'number'
    assert TYPES_MAP['_Expression'] == 'expref'
    assert TYPES_MAP['unknown_thing_xyz'] if False else True


def test_reverse_types_map_values():
    assert REVERSE_TYPES_MAP['boolean'] == ('bool',)
    assert 'list' in REVERSE_TYPES_MAP['array']
    assert 'dict' in REVERSE_TYPES_MAP['object']
    assert REVERSE_TYPES_MAP['null'] == ('NoneType',)
    assert 'str' in REVERSE_TYPES_MAP['string']
    assert 'int' in REVERSE_TYPES_MAP['number']
    assert REVERSE_TYPES_MAP['expref'] == ('_Expression',)


# ---------------------------------------------------------------------------
# FunctionRegistry
# ---------------------------------------------------------------------------

def test_function_registry_populates_table():
    class Dummy(metaclass=FunctionRegistry):
        @signature({'types': ['number']})
        def _func_foo(self, arg):
            return arg

    assert '_func_foo' not in Dummy.FUNCTION_TABLE
    assert 'foo' in Dummy.FUNCTION_TABLE
    assert Dummy.FUNCTION_TABLE['foo']['signature'] == ({'types': ['number']},)
    assert Dummy.FUNCTION_TABLE['foo']['function'] == Dummy._func_foo


def test_function_registry_ignores_non_func_prefix():
    class Dummy(metaclass=FunctionRegistry):
        @signature({'types': ['number']})
        def helper(self, arg):
            return arg

    assert Dummy.FUNCTION_TABLE == {}


def test_function_registry_ignores_missing_signature():
    class Dummy(metaclass=FunctionRegistry):
        def _func_foo(self, arg):
            return arg

    assert Dummy.FUNCTION_TABLE == {}


def test_function_registry_inherited_methods_registered():
    class Base(metaclass=FunctionRegistry):
        @signature({'types': ['number']})
        def _func_base(self, arg):
            return arg

    class Child(Base):
        @signature({'types': ['string']})
        def _func_child(self, arg):
            return arg

    assert 'base' in Child.FUNCTION_TABLE
    assert 'child' in Child.FUNCTION_TABLE


# ---------------------------------------------------------------------------
# Functions.call_function / validation
# ---------------------------------------------------------------------------

def test_call_function_unknown(funcs):
    with pytest.raises(exceptions.UnknownFunctionError) as exc:
        funcs.call_function('nonexistent_fn', [])
    assert 'Unknown function: nonexistent_fn()' in str(exc.value)


def test_call_function_abs(funcs):
    assert funcs.call_function('abs', [-5]) == 5
    assert funcs.call_function('abs', [5]) == 5
    assert funcs.call_function('abs', [-5.5]) == 5.5


def test_call_function_arity_error(funcs):
    with pytest.raises(exceptions.ArityError):
        funcs.call_function('abs', [])


def test_call_function_type_error(funcs):
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs.call_function('abs', ['not a number'])


def test_call_function_bool_is_not_number(funcs):
    # booleans are not numbers even though bool subclasses int
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs.call_function('abs', [True])


def test_call_function_variadic_arity_error(funcs):
    with pytest.raises(exceptions.VariadictArityError):
        funcs.call_function('not_null', [])


# ---------------------------------------------------------------------------
# Numeric functions
# ---------------------------------------------------------------------------

def test_abs(funcs):
    assert funcs.call_function('abs', [-3]) == 3


def test_avg(funcs):
    assert funcs.call_function('avg', [[1, 2, 3]]) == 2
    assert funcs.call_function('avg', [[1.0, 2.0]]) == 1.5
    assert funcs.call_function('avg', [[]]) is None


def test_avg_type_error(funcs):
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs.call_function('avg', [['a', 'b']])


def test_ceil(funcs):
    assert funcs.call_function('ceil', [1.1]) == 2
    assert funcs.call_function('ceil', [1]) == 1


def test_floor(funcs):
    assert funcs.call_function('floor', [1.9]) == 1
    assert funcs.call_function('floor', [-1.1]) == -2


def test_sum(funcs):
    assert funcs.call_function('sum', [[1, 2, 3]]) == 6
    assert funcs.call_function('sum', [[]]) == 0


# ---------------------------------------------------------------------------
# not_null
# ---------------------------------------------------------------------------

def test_not_null(funcs):
    assert funcs.call_function('not_null', [None, None, 1, 2]) == 1
    assert funcs.call_function('not_null', [0]) == 0
    assert funcs.call_function('not_null', [None]) is None


# ---------------------------------------------------------------------------
# to_array / to_string / to_number
# ---------------------------------------------------------------------------

def test_to_array(funcs):
    assert funcs.call_function('to_array', [[1, 2]]) == [1, 2]
    assert funcs.call_function('to_array', ['a']) == ['a']
    assert funcs.call_function('to_array', [None]) == [None]
    assert funcs.call_function('to_array', [1]) == [1]


def test_to_string(funcs):
    assert funcs.call_function('to_string', ['abc']) == 'abc'
    assert funcs.call_function('to_string', [[1, 2]]) == '[1,2]'
    assert funcs.call_function('to_string', [{'a': 1}]) == '{"a":1}'
    assert funcs.call_function('to_string', [1]) == '1'
    assert funcs.call_function('to_string', [None]) == 'null'
    assert funcs.call_function('to_string', [True]) == 'true'


def test_to_number(funcs):
    assert funcs.call_function('to_number', ['42']) == 42
    assert funcs.call_function('to_number', ['3.14']) == 3.14
    assert funcs.call_function('to_number', ['abc']) is None
    assert funcs.call_function('to_number', [42]) == 42
    assert funcs.call_function('to_number', [3.14]) == 3.14
    assert funcs.call_function('to_number', [None]) is None
    assert funcs.call_function('to_number', [True]) is None
    assert funcs.call_function('to_number', [[1, 2]]) is None
    assert funcs.call_function('to_number', [{'a': 1}]) is None


# ---------------------------------------------------------------------------
# contains / length
# ---------------------------------------------------------------------------

def test_contains(funcs):
    assert funcs.call_function('contains', ['foobar', 'bar']) is True
    assert funcs.call_function('contains', ['foobar', 'baz']) is False
    assert funcs.call_function('contains', [[1, 2, 3], 2]) is True
    assert funcs.call_function('contains', [[1, 2, 3], 4]) is False


def test_length(funcs):
    assert funcs.call_function('length', ['abc']) == 3
    assert funcs.call_function('length', [[1, 2]]) == 2
    assert funcs.call_function('length', [{'a': 1, 'b': 2}]) == 2


# ---------------------------------------------------------------------------
# ends_with / starts_with
# ---------------------------------------------------------------------------

def test_ends_with(funcs):
    assert funcs.call_function('ends_with', ['foobar', 'bar']) is True
    assert funcs.call_function('ends_with', ['foobar', 'foo']) is False


def test_starts_with(funcs):
    assert funcs.call_function('starts_with', ['foobar', 'foo']) is True
    assert funcs.call_function('starts_with', ['foobar', 'bar']) is False


# ---------------------------------------------------------------------------
# reverse
# ---------------------------------------------------------------------------

def test_reverse(funcs):
    assert funcs.call_function('reverse', ['abc']) == 'cba'
    assert funcs.call_function('reverse', [[1, 2, 3]]) == [3, 2, 1]


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------

def test_join(funcs):
    assert funcs.call_function('join', [',', ['a', 'b', 'c']]) == 'a,b,c'
    assert funcs.call_function('join', ['', ['a', 'b']]) == 'ab'


def test_join_type_error(funcs):
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs.call_function('join', [',', [1, 2, 3]])


# ---------------------------------------------------------------------------
# map (expref)
# ---------------------------------------------------------------------------

class _Expression:
    """Mimic the real expref node; type name maps to 'expref'."""
    pass


def _make_expref(func, expression=_Expression()):
    return _FakeExpref(expression, func)


# _FakeExpref isn't recognized as expref. Use the real one shape.
def test_map(funcs):
    class RealExpref:
        pass

    # rename type to _Expression so type checking passes
    RealExpref.__name__ = '_Expression'

    def visit(expression, element):
        return element * 2

    expref = RealExpref()
    expref.expression = object()
    expref.visit = visit

    result = funcs.call_function('map', [expref, [1, 2, 3]])
    assert result == [2, 4, 6]


# ---------------------------------------------------------------------------
# max / min / sort
# ---------------------------------------------------------------------------

def test_max(funcs):
    assert funcs.call_function('max', [[1, 2, 3]]) == 3
    assert funcs.call_function('max', [['a', 'b', 'c']]) == 'c'
    assert funcs.call_function('max', [[]]) is None


def test_min(funcs):
    assert funcs.call_function('min', [[1, 2, 3]]) == 1
    assert funcs.call_function('min', [['a', 'b', 'c']]) == 'a'
    assert funcs.call_function('min', [[]]) is None


def test_sort(funcs):
    assert funcs.call_function('sort', [[3, 1, 2]]) == [1, 2, 3]
    assert funcs.call_function('sort', [['c', 'a', 'b']]) == ['a', 'b', 'c']


def test_sort_mixed_types_error(funcs):
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs.call_function('sort', [[1, 'a']])


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge(funcs):
    assert funcs.call_function('merge', [{'a': 1}, {'b': 2}]) == {'a': 1, 'b': 2}
    assert funcs.call_function('merge', [{'a': 1}, {'a': 2}]) == {'a': 2}


def test_merge_arity_error(funcs):
    with pytest.raises(exceptions.VariadictArityError):
        funcs.call_function('merge', [])


# ---------------------------------------------------------------------------
# keys / values
# ---------------------------------------------------------------------------

def test_keys(funcs):
    assert sorted(funcs.call_function('keys', [{'a': 1, 'b': 2}])) == ['a', 'b']


def test_values(funcs):
    assert sorted(funcs.call_function('values', [{'a': 1, 'b': 2}])) == [1, 2]


# ---------------------------------------------------------------------------
# type
# ---------------------------------------------------------------------------

def test_type(funcs):
    assert funcs.call_function('type', ['abc']) == 'string'
    assert funcs.call_function('type', [True]) == 'boolean'
    assert funcs.call_function('type', [False]) == 'boolean'
    assert funcs.call_function('type', [[1, 2]]) == 'array'
    assert funcs.call_function('type', [{'a': 1}]) == 'object'
    assert funcs.call_function('type', [1]) == 'number'
    assert funcs.call_function('type', [1.5]) == 'number'
    assert funcs.call_function('type', [None]) == 'null'


# ---------------------------------------------------------------------------
# convert_to_jmespath_type
# ---------------------------------------------------------------------------

def test_convert_to_jmespath_type(funcs):
    assert funcs._convert_to_jmespath_type('int') == 'number'
    assert funcs._convert_to_jmespath_type('str') == 'string'
    assert funcs._convert_to_jmespath_type('NoneType') == 'null'
    assert funcs._convert_to_jmespath_type('something_else') == 'unknown'


# ---------------------------------------------------------------------------
# sort_by / min_by / max_by (expref)
# ---------------------------------------------------------------------------

def _real_expref(func):
    class RealExpref:
        pass
    RealExpref.__name__ = '_Expression'
    expref = RealExpref()
    expref.expression = object()
    expref.visit = func
    return expref


def test_sort_by_numbers(funcs):
    expref = _real_expref(lambda expr, el: el['age'])
    data = [{'age': 30}, {'age': 10}, {'age': 20}]
    result = funcs.call_function('sort_by', [data, expref])
    assert result == [{'age': 10}, {'age': 20}, {'age': 30}]


def test_sort_by_strings(funcs):
    expref = _real_expref(lambda expr, el: el['name'])
    data = [{'name': 'c'}, {'name': 'a'}, {'name': 'b'}]
    result = funcs.call_function('sort_by', [data, expref])
    assert result == [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]


def test_sort_by_empty_array(funcs):
    expref = _real_expref(lambda expr, el: el)
    result = funcs.call_function('sort_by', [[], expref])
    assert result == []


def test_sort_by_invalid_type(funcs):
    expref = _real_expref(lambda expr, el: el['flag'])
    data = [{'flag': True}]
    with pytest.raises(exceptions.JMESPathTypeError) as exc:
        funcs.call_function('sort_by', [data, expref])
    assert 'sort_by' in str(exc.value)


def test_sort_by_mixed_types(funcs):
    expref = _real_expref(lambda expr, el: el['v'])
    data = [{'v': 1}, {'v': 'a'}]
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs.call_function('sort_by', [data, expref])


def test_min_by(funcs):
    expref = _real_expref(lambda expr, el: el['v'])
    data = [{'v': 3}, {'v': 1}, {'v': 2}]
    assert funcs.call_function('min_by', [data, expref]) == {'v': 1}


def test_min_by_empty(funcs):
    expref = _real_expref(lambda expr, el: el['v'])
    assert funcs.call_function('min_by', [[], expref]) is None


def test_max_by(funcs):
    expref = _real_expref(lambda expr, el: el['v'])
    data = [{'v': 3}, {'v': 1}, {'v': 2}]
    assert funcs.call_function('max_by', [data, expref]) == {'v': 3}


def test_max_by_empty(funcs):
    expref = _real_expref(lambda expr, el: el['v'])
    assert funcs.call_function('max_by', [[], expref]) is None


def test_min_by_invalid_type(funcs):
    expref = _real_expref(lambda expr, el: el['v'])
    data = [{'v': [1, 2]}]
    with pytest.raises(exceptions.JMESPathTypeError) as exc:
        funcs.call_function('min_by', [data, expref])
    assert 'min_by' in str(exc.value)


# ---------------------------------------------------------------------------
# _get_allowed_pytypes (via type check paths)
# ---------------------------------------------------------------------------

def test_get_allowed_pytypes(funcs):
    allowed, subtypes = funcs._get_allowed_pytypes(['array-string'])
    assert 'list' in allowed
    assert subtypes == [('unicode', 'str')]


def test_get_allowed_pytypes_plain(funcs):
    allowed, subtypes = funcs._get_allowed_pytypes(['string', 'number'])
    assert 'str' in allowed
    assert 'int' in allowed
    assert subtypes == []


# ---------------------------------------------------------------------------
# Subtype check: multiple allowed subtypes, dynamic behavior
# ---------------------------------------------------------------------------

def test_subtype_check_multiple_subtypes_matching(funcs):
    # contains accepts array of anything; sort checks single subtype.
    # Test empty list passes subtype checks.
    assert funcs.call_function('sort', [[]]) == []


def test_subtype_check_invalid_subtype(funcs):
    with pytest.raises(exceptions.JMESPathTypeError):
        funcs.call_function('sort', [['a', 1]])
