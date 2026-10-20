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


class TestSignatureDecorator:
    def test_signature_sets_attribute(self):
        @signature({'types': ['string']})
        def foo(x):
            return x

        assert foo.signature == ({'types': ['string']},)

    def test_signature_multiple_arguments(self):
        @signature({'types': ['string']}, {'types': ['number']})
        def foo(a, b):
            return a, b

        assert foo.signature == (
            {'types': ['string']},
            {'types': ['number']},
        )

    def test_signature_returns_same_function(self):
        def foo(x):
            return x

        decorated = signature({'types': []})(foo)
        assert decorated is foo


class TestTypeMaps:
    def test_types_map_basic(self):
        assert TYPES_MAP['bool'] == 'boolean'
        assert TYPES_MAP['str'] == 'string'
        assert TYPES_MAP['dict'] == 'object'
        assert TYPES_MAP['NoneType'] == 'null'
        assert TYPES_MAP['_Expression'] == 'expref'

    def test_reverse_types_map(self):
        assert REVERSE_TYPES_MAP['boolean'] == ('bool',)
        assert REVERSE_TYPES_MAP['string'] == ('unicode', 'str')
        assert REVERSE_TYPES_MAP['object'] == ('dict', 'OrderedDict')
        assert 'int' in REVERSE_TYPES_MAP['number']


class TestFunctionRegistry:
    def test_registry_populates_function_table(self):
        class Foo(metaclass=FunctionRegistry):
            @signature({'types': ['number']})
            def _func_double(self, x):
                return x * 2

        assert 'double' in Foo.FUNCTION_TABLE
        assert Foo.FUNCTION_TABLE['double']['signature'] == ({'types': ['number']},)
        assert Foo.FUNCTION_TABLE['double']['function'] is Foo._func_double

    def test_registry_ignores_methods_without_signature(self):
        class Foo(metaclass=FunctionRegistry):
            def _func_nosig(self, x):
                return x

        assert 'nosig' not in Foo.FUNCTION_TABLE

    def test_registry_ignores_non_func_prefix(self):
        class Foo(metaclass=FunctionRegistry):
            @signature({'types': []})
            def regular_method(self, x):
                return x

        assert Foo.FUNCTION_TABLE == {}


class TestCallFunction:
    def setup_method(self):
        self.f = Functions()

    def test_unknown_function_raises(self):
        with pytest.raises(exceptions.UnknownFunctionError):
            self.f.call_function('no_such_fn', [])

    def test_abs(self):
        assert self.f.call_function('abs', [-5]) == 5
        assert self.f.call_function('abs', [-5.5]) == 5.5

    def test_abs_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('abs', ['not a number'])

    def test_avg(self):
        assert self.f.call_function('avg', [[1, 2, 3]]) == 2
        assert self.f.call_function('avg', [[1.5, 2.5]]) == 2.0

    def test_avg_empty(self):
        assert self.f.call_function('avg', [[]]) is None

    def test_avg_bad_element(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('avg', [[1, 'a']])

    def test_not_null(self):
        assert self.f.call_function('not_null', [None, None, 5]) == 5
        assert self.f.call_function('not_null', [None]) is None
        assert self.f.call_function('not_null', [1, 2]) == 1

    def test_not_null_arity(self):
        with pytest.raises(exceptions.VariadictArityError):
            self.f.call_function('not_null', [])

    def test_to_array(self):
        assert self.f.call_function('to_array', [[1, 2]]) == [1, 2]
        assert self.f.call_function('to_array', [1]) == [1]
        assert self.f.call_function('to_array', ['a']) == ['a']
        assert self.f.call_function('to_array', [None]) == [None]

    def test_to_string(self):
        assert self.f.call_function('to_string', ['abc']) == 'abc'
        assert self.f.call_function('to_string', [1]) == '1'
        assert self.f.call_function('to_string', [[1, 2]]) == '[1,2]'
        assert self.f.call_function('to_string', [{'a': 1}]) == '{"a":1}'
        assert self.f.call_function('to_string', [None]) == 'null'

    def test_to_number(self):
        assert self.f.call_function('to_number', ['42']) == 42
        assert self.f.call_function('to_number', ['3.5']) == 3.5
        assert self.f.call_function('to_number', [7]) == 7
        assert self.f.call_function('to_number', [7.5]) == 7.5
        assert self.f.call_function('to_number', ['abc']) is None
        assert self.f.call_function('to_number', [[1]]) is None
        assert self.f.call_function('to_number', [{'a': 1}]) is None
        assert self.f.call_function('to_number', [True]) is None
        assert self.f.call_function('to_number', [None]) is None

    def test_contains(self):
        assert self.f.call_function('contains', ['foobar', 'bar']) is True
        assert self.f.call_function('contains', ['foobar', 'baz']) is False
        assert self.f.call_function('contains', [[1, 2, 3], 2]) is True
        assert self.f.call_function('contains', [[1, 2, 3], 4]) is False

    def test_contains_bad_subject(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('contains', [1, 1])

    def test_length(self):
        assert self.f.call_function('length', ['abcd']) == 4
        assert self.f.call_function('length', [[1, 2, 3]]) == 3
        assert self.f.call_function('length', [{'a': 1, 'b': 2}]) == 2

    def test_length_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('length', [1])

    def test_ends_with(self):
        assert self.f.call_function('ends_with', ['foobar', 'bar']) is True
        assert self.f.call_function('ends_with', ['foobar', 'foo']) is False

    def test_starts_with(self):
        assert self.f.call_function('starts_with', ['foobar', 'foo']) is True
        assert self.f.call_function('starts_with', ['foobar', 'bar']) is False

    def test_starts_with_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('starts_with', [1, 'a'])

    def test_reverse(self):
        assert self.f.call_function('reverse', ['abc']) == 'cba'
        assert self.f.call_function('reverse', [[1, 2, 3]]) == [3, 2, 1]

    def test_ceil_floor(self):
        assert self.f.call_function('ceil', [1.2]) == 2
        assert self.f.call_function('floor', [1.8]) == 1

    def test_join(self):
        assert self.f.call_function('join', [',', ['a', 'b', 'c']]) == 'a,b,c'
        assert self.f.call_function('join', ['', ['a', 'b']]) == 'ab'

    def test_join_bad_element(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('join', [',', ['a', 1]])

    def test_max_min(self):
        assert self.f.call_function('max', [[1, 2, 3]]) == 3
        assert self.f.call_function('min', [[1, 2, 3]]) == 1
        assert self.f.call_function('max', [['a', 'b']]) == 'b'
        assert self.f.call_function('min', [['a', 'b']]) == 'a'

    def test_max_min_empty(self):
        assert self.f.call_function('max', [[]]) is None
        assert self.f.call_function('min', [[]]) is None

    def test_merge(self):
        result = self.f.call_function('merge', [{'a': 1}, {'b': 2}, {'a': 3}])
        assert result == {'a': 3, 'b': 2}

    def test_merge_single(self):
        assert self.f.call_function('merge', [{'a': 1}]) == {'a': 1}

    def test_merge_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('merge', [[1]])

    def test_sort(self):
        assert self.f.call_function('sort', [[3, 1, 2]]) == [1, 2, 3]
        assert self.f.call_function('sort', [['c', 'a', 'b']]) == ['a', 'b', 'c']

    def test_sort_mixed_types_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('sort', [[1, 'a']])

    def test_sum(self):
        assert self.f.call_function('sum', [[1, 2, 3]]) == 6
        assert self.f.call_function('sum', [[]]) == 0

    def test_sum_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('sum', [['a']])

    def test_keys_values(self):
        obj = {'a': 1, 'b': 2}
        assert sorted(self.f.call_function('keys', [obj])) == ['a', 'b']
        assert sorted(self.f.call_function('values', [obj])) == [1, 2]

    def test_type(self):
        assert self.f.call_function('type', ['abc']) == 'string'
        assert self.f.call_function('type', [True]) == 'boolean'
        assert self.f.call_function('type', [[1]]) == 'array'
        assert self.f.call_function('type', [{'a': 1}]) == 'object'
        assert self.f.call_function('type', [1]) == 'number'
        assert self.f.call_function('type', [1.5]) == 'number'
        assert self.f.call_function('type', [None]) == 'null'


class FakeExpression:
    def __init__(self):
        self.expression = 'fake'

    def visit(self, expression, element):
        key = getattr(self, 'key', None)
        if key is not None:
            return element[key]
        return element


class TestExprefFunctions:
    def setup_method(self):
        self.f = Functions()

    def test_map(self):
        expref = FakeExpression()
        result = self.f.call_function('map', [expref, [1, 2, 3]])
        assert result == [1, 2, 3]

    def test_sort_by_on_empty(self):
        expref = FakeExpression()
        assert self.f.call_function('sort_by', [expref, []]) == []

    def test_sort_by_with_key(self):
        expref = FakeExpression()
        expref.key = 'k'
        data = [{'k': 3}, {'k': 1}, {'k': 2}]
        result = self.f.call_function('sort_by', [expref, data])
        assert result == [{'k': 1}, {'k': 2}, {'k': 3}]

    def test_sort_by_invalid_first_type(self):
        expref = FakeExpression()
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('sort_by', [expref, [[1, 2]]])

    def test_min_by_max_by(self):
        expref = FakeExpression()
        expref.key = 'k'
        data = [{'k': 3}, {'k': 1}, {'k': 2}]
        assert self.f.call_function('min_by', [expref, data]) == {'k': 1}
        assert self.f.call_function('max_by', [expref, data]) == {'k': 3}

    def test_min_by_max_by_empty(self):
        expref = FakeExpression()
        assert self.f.call_function('min_by', [expref, []]) is None
        assert self.f.call_function('max_by', [expref, []]) is None

    def test_min_by_bad_key_type(self):
        expref = FakeExpression()
        expref.key = 'k'
        data = [{'k': [1, 2]}]
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('min_by', [expref, data])


class TestValidationHelpers:
    def setup_method(self):
        self.f = Functions()

    def test_arity_error(self):
        with pytest.raises(exceptions.ArityError):
            self.f.call_function('abs', [1, 2])

    def test_arity_error_too_few(self):
        with pytest.raises(exceptions.ArityError):
            self.f.call_function('abs', [])

    def test_variadic_arity_error(self):
        with pytest.raises(exceptions.VariadictArityError):
            self.f.call_function('merge', [])

    def test_convert_to_jmespath_type(self):
        assert self.f._convert_to_jmespath_type('bool') == 'boolean'
        assert self.f._convert_to_jmespath_type('str') == 'string'
        assert self.f._convert_to_jmespath_type('dict') == 'object'
        assert self.f._convert_to_jmespath_type('NoneType') == 'null'
        assert self.f._convert_to_jmespath_type('unknown_type') == 'unknown'

    def test_get_allowed_pytypes(self):
        allowed, subtypes = self.f._get_allowed_pytypes(['array-number', 'string'])
        assert 'list' in allowed
        assert 'str' in allowed
        assert subtypes == [('float', 'int', 'long')]

    def test_type_check_single_passes(self):
        # Should not raise for valid types.
        self.f._type_check_single(1, ['number'], 'test')
        self.f._type_check_single('abc', ['string'], 'test')

    def test_type_check_single_raises(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f._type_check_single([], ['string'], 'test')
