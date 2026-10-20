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
# signature / FunctionRegistry
# ---------------------------------------------------------------------------

class TestSignatureDecorator:
    def test_attaches_signature_attribute(self):
        @signature({'types': ['number']}, {'types': ['string']})
        def foo():
            pass

        assert foo.signature == ({'types': ['number']},
                                 {'types': ['string']})

    def test_empty_signature(self):
        @signature()
        def foo():
            pass

        assert foo.signature == ()

    def test_returns_original_function(self):
        def foo():
            return 42

        decorated = signature({'types': []})(foo)
        assert decorated is foo
        assert decorated() == 42


class TestFunctionRegistry:
    def test_registry_populates_function_table(self):
        class MyFuncs(metaclass=FunctionRegistry):
            @signature({'types': ['number']})
            def _func_double(self, arg):
                return arg * 2

        assert 'double' in MyFuncs.FUNCTION_TABLE
        entry = MyFuncs.FUNCTION_TABLE['double']
        assert entry['signature'] == ({'types': ['number']},)
        assert entry['function'].__name__ == '_func_double'

    def test_registry_ignores_non_func_prefixed(self):
        class MyFuncs(metaclass=FunctionRegistry):
            @signature({'types': []})
            def double(self, arg):
                return arg * 2

        assert 'double' not in MyFuncs.FUNCTION_TABLE
        assert MyFuncs.FUNCTION_TABLE == {}

    def test_registry_ignores_func_without_signature(self):
        class MyFuncs(metaclass=FunctionRegistry):
            def _func_nosig(self, arg):
                return arg

        assert 'nosig' not in MyFuncs.FUNCTION_TABLE

    def test_inherits_function_table_from_base(self):
        class Base(metaclass=FunctionRegistry):
            @signature({'types': []})
            def _func_base_fn(self, arg):
                return arg

        class Child(Base):
            @signature({'types': []})
            def _func_child_fn(self, arg):
                return arg

        # Child's table is repopulated with all _func_ methods (inherited too)
        assert 'base_fn' in Child.FUNCTION_TABLE
        assert 'child_fn' in Child.FUNCTION_TABLE

    def test_functions_class_has_populated_table(self):
        # The metaclass populates FUNCTION_TABLE with all _func_ methods
        # defined on the Functions class.
        table = Functions.FUNCTION_TABLE
        for name in ('abs', 'avg', 'ceil', 'floor', 'sum', 'max', 'min',
                     'sort', 'keys', 'values', 'type', 'map', 'sort_by',
                     'min_by', 'max_by', 'merge', 'not_null', 'join',
                     'contains', 'length', 'reverse', 'ends_with',
                     'starts_with', 'to_array', 'to_string', 'to_number'):
            assert name in table
        # Each entry has function and signature keys.
        assert set(table['abs'].keys()) == {'function', 'signature'}


# ---------------------------------------------------------------------------
# call_function / dispatch
# ---------------------------------------------------------------------------

class TestCallFunction:
    def setup_method(self):
        self.f = Functions()

    def test_unknown_function(self):
        with pytest.raises(exceptions.UnknownFunctionError) as exc:
            self.f.call_function('does_not_exist', [])
        assert 'Unknown function: does_not_exist()' in str(exc.value)

    def test_abs(self):
        assert self.f.call_function('abs', [-5]) == 5
        assert self.f.call_function('abs', [5]) == 5

    def test_abs_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('abs', ['5'])

    def test_abs_bool_rejected(self):
        # booleans are not numbers in jmespath type system
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('abs', [True])

    def test_arity_error(self):
        with pytest.raises(exceptions.ArityError):
            self.f.call_function('abs', [])

    def test_arity_error_too_many(self):
        with pytest.raises(exceptions.ArityError):
            self.f.call_function('abs', [1, 2])


# ---------------------------------------------------------------------------
# Variadic functions
# ---------------------------------------------------------------------------

class TestVariadic:
    def setup_method(self):
        self.f = Functions()

    def test_not_null_returns_first_non_null(self):
        assert self.f.call_function('not_null', [None, 1, 2]) == 1

    def test_not_null_all_null(self):
        assert self.f.call_function('not_null', [None, None]) is None

    def test_not_null_variadic_arity_error(self):
        with pytest.raises(exceptions.VariadictArityError):
            self.f.call_function('not_null', [])

    def test_merge(self):
        result = self.f.call_function('merge', [{'a': 1}, {'b': 2}])
        assert result == {'a': 1, 'b': 2}

    def test_merge_override(self):
        result = self.f.call_function('merge', [{'a': 1}, {'a': 2}])
        assert result == {'a': 2}

    def test_merge_arity_error(self):
        with pytest.raises(exceptions.VariadictArityError):
            self.f.call_function('merge', [])

    def test_merge_type_error_non_dict_first_arg(self):
        # The variadic signature only declares a single {'types': ['object']}
        # entry, so _type_check validates index 0 as an object.  A list at
        # position 0 therefore raises JMESPathTypeError.
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('merge', [[1, 2], {'b': 2}])

    def test_merge_ignores_type_of_first_arg(self):
        # Only index 0 is under the signature check; extra args bypass
        # validation because _type_check iterates over len(signature).
        # Here {'a': 1} is valid at index 0; list is merged as pairs.
        result = self.f.call_function('merge', [{'a': 1}, [('b', 2)]])
        assert result == {'a': 1, 'b': 2}


# ---------------------------------------------------------------------------
# Simple numeric / collection functions
# ---------------------------------------------------------------------------

class TestFunctions:
    def setup_method(self):
        self.f = Functions()

    def test_avg(self):
        assert self.f.call_function('avg', [[1, 2, 3]]) == 2
        assert self.f.call_function('avg', [[1, 2]]) == 1.5

    def test_avg_empty(self):
        assert self.f.call_function('avg', [[]]) is None

    def test_avg_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('avg', [['a', 'b']])

    def test_ceil(self):
        assert self.f.call_function('ceil', [1.2]) == 2

    def test_floor(self):
        assert self.f.call_function('floor', [1.8]) == 1

    def test_sum(self):
        assert self.f.call_function('sum', [[1, 2, 3]]) == 6

    def test_sum_empty(self):
        assert self.f.call_function('sum', [[]]) == 0

    def test_sum_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('sum', [['a']])

    def test_max(self):
        assert self.f.call_function('max', [[1, 3, 2]]) == 3
        assert self.f.call_function('max', [['a', 'c', 'b']]) == 'c'

    def test_max_empty(self):
        assert self.f.call_function('max', [[]]) is None

    def test_min(self):
        assert self.f.call_function('min', [[3, 1, 2]]) == 1
        assert self.f.call_function('min', [['c', 'a', 'b']]) == 'a'

    def test_min_empty(self):
        assert self.f.call_function('min', [[]]) is None

    def test_min_mixed_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('min', [[1, 'a']])

    def test_sort(self):
        assert self.f.call_function('sort', [[3, 1, 2]]) == [1, 2, 3]
        assert self.f.call_function('sort', [['b', 'a']]) == ['a', 'b']

    def test_sort_mixed_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('sort', [[1, 'a']])

    def test_keys(self):
        assert self.f.call_function('keys', [{'a': 1, 'b': 2}]) == ['a', 'b']

    def test_values(self):
        assert self.f.call_function('values', [{'a': 1, 'b': 2}]) == [1, 2]

    def test_length_strings_and_collections(self):
        assert self.f.call_function('length', ['abcd']) == 4
        assert self.f.call_function('length', [[1, 2, 3]]) == 3
        assert self.f.call_function('length', [{'a': 1}]) == 1

    def test_length_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('length', [5])

    def test_contains(self):
        assert self.f.call_function('contains', ['abcd', 'bc']) is True
        assert self.f.call_function('contains', ['abcd', 'zz']) is False
        assert self.f.call_function('contains', [[1, 2, 3], 2]) is True

    def test_contains_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('contains', [5, 2])

    def test_ends_with(self):
        assert self.f.call_function('ends_with', ['abcd', 'cd']) is True
        assert self.f.call_function('ends_with', ['abcd', 'ab']) is False

    def test_starts_with(self):
        assert self.f.call_function('starts_with', ['abcd', 'ab']) is True
        assert self.f.call_function('starts_with', ['abcd', 'cd']) is False

    def test_reverse_string(self):
        assert self.f.call_function('reverse', ['abc']) == 'cba'

    def test_reverse_array(self):
        assert self.f.call_function('reverse', [[1, 2, 3]]) == [3, 2, 1]

    def test_reverse_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('reverse', [5])

    def test_join(self):
        assert self.f.call_function('join', [',', ['a', 'b', 'c']]) == 'a,b,c'

    def test_join_empty(self):
        assert self.f.call_function('join', [',', []]) == ''

    def test_join_subtype_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('join', [',', [1, 2]])


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

class TestConversions:
    def setup_method(self):
        self.f = Functions()

    def test_to_array_list(self):
        assert self.f.call_function('to_array', [[1, 2]]) == [1, 2]

    def test_to_array_scalar(self):
        assert self.f.call_function('to_array', [1]) == [1]
        assert self.f.call_function('to_array', ['a']) == ['a']
        assert self.f.call_function('to_array', [None]) == [None]

    def test_to_string_string(self):
        assert self.f.call_function('to_string', ['abc']) == 'abc'

    def test_to_string_number(self):
        assert self.f.call_function('to_string', [5]) == '5'

    def test_to_string_object(self):
        assert self.f.call_function('to_string', [{'a': 1}]) == '{"a":1}'

    def test_to_string_array(self):
        assert self.f.call_function('to_string', [[1, 2]]) == '[1,2]'

    def test_to_string_null(self):
        assert self.f.call_function('to_string', [None]) == 'null'

    def test_to_number_int(self):
        assert self.f.call_function('to_number', ['5']) == 5

    def test_to_number_float(self):
        assert self.f.call_function('to_number', ['5.5']) == 5.5

    def test_to_number_invalid(self):
        assert self.f.call_function('to_number', ['abc']) is None

    def test_to_number_from_number(self):
        assert self.f.call_function('to_number', [5]) == 5
        assert self.f.call_function('to_number', [5.5]) == 5.5

    def test_to_number_from_list_dict_bool(self):
        assert self.f.call_function('to_number', [[1]]) is None
        assert self.f.call_function('to_number', [{'a': 1}]) is None
        assert self.f.call_function('to_number', [True]) is None

    def test_to_number_from_null(self):
        assert self.f.call_function('to_number', [None]) is None

    def test_type_function(self):
        assert self.f.call_function('type', ['a']) == 'string'
        assert self.f.call_function('type', [True]) == 'boolean'
        assert self.f.call_function('type', [[1]]) == 'array'
        assert self.f.call_function('type', [{'a': 1}]) == 'object'
        assert self.f.call_function('type', [1]) == 'number'
        assert self.f.call_function('type', [1.5]) == 'number'
        assert self.f.call_function('type', [None]) == 'null'


# ---------------------------------------------------------------------------
# expref-based functions (with a fake expression object)
# ---------------------------------------------------------------------------

class FakeExpression:
    """Stand-in for _Expression that the type checker recognizes as expref.

    The type checker matches by exact type().__name__ == '_Expression', so we
    name the class accordingly.
    """

    def __init__(self, func):
        self.expression = func

    def visit(self, expression, value):
        return expression(value)


FakeExpression.__name__ = '_Expression'


class TestExprefFunctions:
    def setup_method(self):
        self.f = Functions()

    def test_map(self):
        expref = FakeExpression(lambda x: x + 1)
        assert self.f.call_function('map', [expref, [1, 2, 3]]) == [2, 3, 4]

    def test_map_empty(self):
        expref = FakeExpression(lambda x: x + 1)
        assert self.f.call_function('map', [expref, []]) == []

    def test_map_type_error_on_array(self):
        expref = FakeExpression(lambda x: x)
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('map', [expref, 'notarray'])

    def test_sort_by(self):
        expref = FakeExpression(lambda x: x['age'])
        array = [{'age': 3}, {'age': 1}, {'age': 2}]
        result = self.f.call_function('sort_by', [array, expref])
        assert result == [{'age': 1}, {'age': 2}, {'age': 3}]

    def test_sort_by_empty(self):
        expref = FakeExpression(lambda x: x)
        assert self.f.call_function('sort_by', [[], expref]) == []

    def test_sort_by_invalid_type(self):
        expref = FakeExpression(lambda x: True)
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('sort_by', [[{'a': 1}], expref])

    def test_min_by(self):
        expref = FakeExpression(lambda x: x['v'])
        array = [{'v': 3}, {'v': 1}, {'v': 2}]
        assert self.f.call_function('min_by', [array, expref]) == {'v': 1}

    def test_min_by_empty(self):
        expref = FakeExpression(lambda x: x)
        assert self.f.call_function('min_by', [[], expref]) is None

    def test_max_by(self):
        expref = FakeExpression(lambda x: x['v'])
        array = [{'v': 3}, {'v': 1}, {'v': 2}]
        assert self.f.call_function('max_by', [array, expref]) == {'v': 3}

    def test_max_by_empty(self):
        expref = FakeExpression(lambda x: x)
        assert self.f.call_function('max_by', [[], expref]) is None

    def test_max_by_bad_type(self):
        expref = FakeExpression(lambda x: [1])
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f.call_function('max_by', [[{'a': 1}], expref])


# ---------------------------------------------------------------------------
# Type-check internals
# ---------------------------------------------------------------------------

class TestTypeCheckInternals:
    def setup_method(self):
        self.f = Functions()

    def test_convert_to_jmespath_type(self):
        assert self.f._convert_to_jmespath_type('int') == 'number'
        assert self.f._convert_to_jmespath_type('str') == 'string'
        assert self.f._convert_to_jmespath_type('dict') == 'object'
        assert self.f._convert_to_jmespath_type('NoneType') == 'null'
        assert self.f._convert_to_jmespath_type('bool') == 'boolean'
        assert self.f._convert_to_jmespath_type('list') == 'array'
        assert self.f._convert_to_jmespath_type('weird') == 'unknown'

    def test_get_allowed_pytypes_plain(self):
        allowed, subtypes = self.f._get_allowed_pytypes(['number', 'string'])
        assert 'int' in allowed
        assert 'float' in allowed
        assert 'str' in allowed
        assert 'unicode' in allowed
        assert subtypes == []

    def test_get_allowed_pytypes_subtype(self):
        allowed, subtypes = self.f._get_allowed_pytypes(['array-number'])
        assert 'list' in allowed
        assert subtypes == [REVERSE_TYPES_MAP['number']]

    def test_type_check_single_ok(self):
        # Should not raise
        self.f._type_check_single(5, ['number'], 'test')

    def test_type_check_single_raises(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f._type_check_single('a', ['number'], 'test')

    def test_subtype_check_single_ok(self):
        self.f._subtype_check(
            [1, 2, 3], [REVERSE_TYPES_MAP['number']], ['array-number'], 'test')

    def test_subtype_check_single_raises(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f._subtype_check(
                [1, 'a'], [REVERSE_TYPES_MAP['number']],
                ['array-number'], 'test')

    def test_subtype_check_multi_dynamic_ok(self):
        # Two subtype options: number or string; all numbers -> ok
        allowed_subtypes = [REVERSE_TYPES_MAP['number'],
                            REVERSE_TYPES_MAP['string']]
        self.f._subtype_check([1, 2, 3], allowed_subtypes,
                              ['array-number', 'array-string'], 'test')

    def test_subtype_check_multi_dynamic_mixed_raises(self):
        allowed_subtypes = [REVERSE_TYPES_MAP['number'],
                            REVERSE_TYPES_MAP['string']]
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f._subtype_check([1, 'a'], allowed_subtypes,
                                  ['array-number', 'array-string'], 'test')

    def test_subtype_check_multi_empty_ok(self):
        # Empty current -> no check performed
        self.f._subtype_check([], [REVERSE_TYPES_MAP['number']],
                              ['array-number'], 'test')

    def test_subtype_check_multi_dynamic_first_element_unknown_raises(self):
        # Cover branch at line 151/156: first element's type does not match
        # any of the allowed subtypes; the for/else raises JMESPathTypeError.
        # Use two subtype options whose element types won't match the first
        # element.
        class _Weird:
            pass

        # _Weird is neither number nor string, but the top-level type in
        # _type_check_single would normally reject it.  Here we call
        # _subtype_check directly to exercise the dynamic branch.
        allowed_subtypes = [REVERSE_TYPES_MAP['number'],
                            REVERSE_TYPES_MAP['string']]
        with pytest.raises(exceptions.JMESPathTypeError) as exc:
            self.f._subtype_check([_Weird(), 1], allowed_subtypes,
                                  ['array-number', 'array-string'], 'fn')
        err = exc.value
        # Inspect the exception's string representation: it should reference
        # the function name and the offending type.  (JMESPathTypeError does
        # not expose an ``actual`` attribute on all versions.)
        message = str(err)
        assert 'fn' in message
        assert '_Weird' in message

    def test_subtype_check_multi_dynamic_second_element_unknown_raises(self):
        # First element picks 'number' branch; later element fails subtype.
        # Exercises the inner per-element check after 'allowed' is set.
        allowed_subtypes = [REVERSE_TYPES_MAP['number'],
                            REVERSE_TYPES_MAP['string']]
        with pytest.raises(exceptions.JMESPathTypeError):
            self.f._subtype_check([1, 'a'], allowed_subtypes,
                                  ['array-number', 'array-string'], 'fn')


# ---------------------------------------------------------------------------
# Module-level maps sanity
# ---------------------------------------------------------------------------

class TestTypeMaps:
    def test_types_map_entries(self):
        assert TYPES_MAP['int'] == 'number'
        assert TYPES_MAP['str'] == 'string'
        assert TYPES_MAP['_Expression'] == 'expref'

    def test_reverse_types_map_entries(self):
        assert REVERSE_TYPES_MAP['number'] == ('float', 'int', 'long')
        assert REVERSE_TYPES_MAP['expref'] == ('_Expression',)
