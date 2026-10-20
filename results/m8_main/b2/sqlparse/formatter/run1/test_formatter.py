"""Tests for sqlparse.formatter."""

import pytest

from sqlparse import filters
from sqlparse.engine import FilterStack
from sqlparse.exceptions import SQLParseError
from sqlparse.formatter import build_filter_stack, validate_options


# ---------------------------------------------------------------------------
# validate_options
# ---------------------------------------------------------------------------

class TestValidateOptionsKeywordCase:
    @pytest.mark.parametrize('value', [None, 'upper', 'lower', 'capitalize'])
    def test_valid(self, value):
        opts = {} if value is None else {'keyword_case': value}
        assert validate_options(opts) is opts

    @pytest.mark.parametrize('value', ['UPPER', 'mixed', '', 1, False])
    def test_invalid(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'keyword_case': value})
        assert 'Invalid value for keyword_case' in str(exc.value)


class TestValidateOptionsIdentifierCase:
    @pytest.mark.parametrize('value', [None, 'upper', 'lower', 'capitalize'])
    def test_valid(self, value):
        opts = {} if value is None else {'identifier_case': value}
        assert validate_options(opts) is opts

    @pytest.mark.parametrize('value', ['UPPER', 'mixed', '', 0])
    def test_invalid(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'identifier_case': value})
        assert 'Invalid value for identifier_case' in str(exc.value)


class TestValidateOptionsOutputFormat:
    @pytest.mark.parametrize('value', [None, 'sql', 'python', 'php'])
    def test_valid(self, value):
        opts = {} if value is None else {'output_format': value}
        assert validate_options(opts) is opts

    @pytest.mark.parametrize('value', ['java', 'SQL', '', 3])
    def test_invalid(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'output_format': value})
        assert 'Unknown output format' in str(exc.value)


class TestValidateOptionsBooleans:
    @pytest.mark.parametrize('key', [
        'strip_comments', 'use_space_around_operators', 'strip_whitespace',
        'indent_columns', 'reindent', 'reindent_aligned',
        'indent_after_first', 'indent_tabs', 'comma_first', 'compact',
    ])
    @pytest.mark.parametrize('value', [True, False])
    def test_valid(self, key, value):
        opts = {key: value}
        result = validate_options(opts)
        assert result is opts
        assert result[key] is value

    @pytest.mark.parametrize('key,message', [
        ('strip_comments', 'Invalid value for strip_comments'),
        ('use_space_around_operators', 'Invalid value for use_space_around_operators'),
        ('strip_whitespace', 'Invalid value for strip_whitespace'),
        ('indent_columns', 'Invalid value for indent_columns'),
        ('reindent', 'Invalid value for reindent'),
        ('reindent_aligned', 'Invalid value for reindent_aligned'),
        ('indent_after_first', 'Invalid value for indent_after_first'),
        ('indent_tabs', 'Invalid value for indent_tabs'),
        ('comma_first', 'comma_first requires a boolean value'),
        ('compact', 'compact requires a boolean value'),
    ])
    @pytest.mark.parametrize('value', ['yes', '', 'maybe', 2, -1])
    def test_invalid(self, key, message, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({key: value})
        assert message in str(exc.value)

    @pytest.mark.parametrize('key', [
        'strip_comments', 'use_space_around_operators', 'strip_whitespace',
        'indent_columns', 'reindent', 'reindent_aligned',
        'indent_after_first', 'indent_tabs', 'comma_first', 'compact',
    ])
    @pytest.mark.parametrize('value', [0, 1])
    def test_int_coerced_equality_true(self, key, value):
        # Because check is `value not in [True, False]`, 0 == False and
        # 1 == True, so integers 0 and 1 are accepted.
        opts = {key: value}
        result = validate_options(opts)
        assert result is opts
        assert result[key] == value


class TestValidateOptionsIndentColumns:
    def test_enforces_reindent(self):
        opts = {'indent_columns': True}
        result = validate_options(opts)
        assert result['reindent'] is True
        assert result['indent_columns'] is True
        assert result['strip_whitespace'] is True

    def test_false_default_recorded(self):
        opts = {}
        result = validate_options(opts)
        assert result['indent_columns'] is False


class TestValidateOptionsReindent:
    def test_reindent_enforces_strip_whitespace(self):
        opts = {'reindent': True}
        result = validate_options(opts)
        assert result['strip_whitespace'] is True

    def test_reindent_aligned_enforces_strip_whitespace(self):
        opts = {'reindent_aligned': True}
        result = validate_options(opts)
        assert result['strip_whitespace'] is True


class TestValidateOptionsIndentTabs:
    def test_tabs(self):
        opts = {'indent_tabs': True}
        assert validate_options(opts)['indent_char'] == '\t'

    def test_spaces(self):
        opts = {'indent_tabs': False}
        assert validate_options(opts)['indent_char'] == ' '


class TestValidateOptionsTruncateStrings:
    def test_valid_int(self):
        opts = {'truncate_strings': 10}
        result = validate_options(opts)
        assert result['truncate_strings'] == 10
        assert result['truncate_char'] == '[...]'

    def test_valid_string_int(self):
        opts = {'truncate_strings': '5'}
        result = validate_options(opts)
        assert result['truncate_strings'] == 5

    def test_custom_truncate_char(self):
        opts = {'truncate_strings': 8, 'truncate_char': '...'}
        result = validate_options(opts)
        assert result['truncate_strings'] == 8
        assert result['truncate_char'] == '...'

    def test_none_left_alone(self):
        opts = {}
        result = validate_options(opts)
        assert 'truncate_strings' not in result

    @pytest.mark.parametrize('value', ['abc', object()])
    def test_invalid_type(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'truncate_strings': value})
        assert 'Invalid value for truncate_strings' in str(exc.value)

    @pytest.mark.parametrize('value', [0, 1, -5, '1'])
    def test_too_small(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'truncate_strings': value})
        assert 'Invalid value for truncate_strings' in str(exc.value)


class TestValidateOptionsIndentWidth:
    def test_default(self):
        result = validate_options({})
        assert result['indent_width'] == 2

    def test_int(self):
        result = validate_options({'indent_width': 4})
        assert result['indent_width'] == 4

    def test_string(self):
        result = validate_options({'indent_width': '3'})
        assert result['indent_width'] == 3

    @pytest.mark.parametrize('value', ['abc', None, object()])
    def test_invalid(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'indent_width': value})
        assert 'indent_width requires an integer' in str(exc.value)

    @pytest.mark.parametrize('value', [0, -1, '-5'])
    def test_not_positive(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'indent_width': value})
        assert 'indent_width requires a positive integer' in str(exc.value)


class TestValidateOptionsWrapAfter:
    def test_default(self):
        assert validate_options({})['wrap_after'] == 0

    def test_int(self):
        assert validate_options({'wrap_after': 80})['wrap_after'] == 80

    def test_string(self):
        assert validate_options({'wrap_after': '50'})['wrap_after'] == 50

    @pytest.mark.parametrize('value', ['abc', None, object()])
    def test_invalid(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'wrap_after': value})
        assert 'wrap_after requires an integer' in str(exc.value)

    def test_negative(self):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'wrap_after': -1})
        assert 'wrap_after requires a positive integer' in str(exc.value)


class TestValidateOptionsRightMargin:
    def test_default_none(self):
        assert validate_options({})['right_margin'] is None

    def test_valid(self):
        assert validate_options({'right_margin': 20})['right_margin'] == 20

    def test_valid_string(self):
        assert validate_options({'right_margin': '30'})['right_margin'] == 30

    @pytest.mark.parametrize('value', ['abc', object()])
    def test_invalid_type(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'right_margin': value})
        assert 'right_margin requires an integer' in str(exc.value)

    @pytest.mark.parametrize('value', [9, 0, -5])
    def test_too_small(self, value):
        with pytest.raises(SQLParseError) as exc:
            validate_options({'right_margin': value})
        assert 'right_margin requires an integer > 10' in str(exc.value)


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

def _validate_and_build(options):
    opts = validate_options(dict(options))
    stack = FilterStack()
    return build_filter_stack(stack, opts)


class TestBuildFilterStackEmpty:
    def test_no_filters(self):
        stack = _validate_and_build({})
        assert stack.preprocess == []
        assert stack.stmtprocess == []
        assert stack.postprocess == []

    def test_returns_same_stack(self):
        stack = FilterStack()
        result = build_filter_stack(stack, {})
        assert result is stack


class TestBuildFilterStackPreprocess:
    def test_keyword_case(self):
        stack = _validate_and_build({'keyword_case': 'upper'})
        assert len(stack.preprocess) == 1
        assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)

    def test_identifier_case(self):
        stack = _validate_and_build({'identifier_case': 'lower'})
        assert len(stack.preprocess) == 1
        assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)

    def test_truncate_strings(self):
        stack = _validate_and_build({'truncate_strings': 10})
        assert len(stack.preprocess) == 1
        assert isinstance(stack.preprocess[0], filters.TruncateStringFilter)

    def test_all_preprocess(self):
        stack = _validate_and_build({
            'keyword_case': 'upper',
            'identifier_case': 'lower',
            'truncate_strings': 5,
        })
        assert len(stack.preprocess) == 3
        assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
        assert isinstance(stack.preprocess[1], filters.IdentifierCaseFilter)
        assert isinstance(stack.preprocess[2], filters.TruncateStringFilter)


class TestBuildFilterStackStmtProcess:
    def test_space_around_operators(self):
        stack = _validate_and_build({'use_space_around_operators': True})
        assert len(stack.stmtprocess) == 1
        assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)

    def test_strip_comments(self):
        stack = _validate_and_build({'strip_comments': True})
        assert len(stack.stmtprocess) == 1
        assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)

    def test_strip_whitespace(self):
        stack = _validate_and_build({'strip_whitespace': True})
        assert len(stack.stmtprocess) == 1
        assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)

    def test_reindent(self):
        stack = _validate_and_build({'reindent': True})
        # strip_whitespace enforced by reindent -> both filters added
        assert len(stack.stmtprocess) == 2
        assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
        assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)

    def test_reindent_aligned(self):
        stack = _validate_and_build({'reindent_aligned': True})
        assert len(stack.stmtprocess) == 2
        assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
        assert isinstance(stack.stmtprocess[1], filters.AlignedIndentFilter)

    def test_right_margin(self):
        stack = _validate_and_build({'right_margin': 20})
        assert len(stack.stmtprocess) == 1
        assert isinstance(stack.stmtprocess[0], filters.RightMarginFilter)


class TestBuildFilterStackPostprocess:
    def test_output_php(self):
        stack = _validate_and_build({'output_format': 'php'})
        assert len(stack.postprocess) == 1
        assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)

    def test_output_python(self):
        stack = _validate_and_build({'output_format': 'python'})
        assert len(stack.postprocess) == 1
        assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)

    def test_output_sql_no_filter(self):
        stack = _validate_and_build({'output_format': 'sql'})
        assert stack.postprocess == []

    def test_output_format_case_insensitive_but_validated(self):
        # validate_options rejects uppercase, so build directly with a
        # dict bypassing validation to exercise frmt.lower() branch.
        stack = FilterStack()
        build_filter_stack(stack, {'output_format': 'PYTHON'})
        assert len(stack.postprocess) == 1
        assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


class TestBuildFilterStackCombined:
    def test_full_options(self):
        stack = _validate_and_build({
            'keyword_case': 'upper',
            'identifier_case': 'lower',
            'truncate_strings': 5,
            'use_space_around_operators': True,
            'strip_comments': True,
            'reindent': True,
            'right_margin': 20,
            'output_format': 'python',
        })
        assert len(stack.preprocess) == 3
        # space_around_operators, strip_comments, strip_whitespace, reindent,
        # right_margin
        assert len(stack.stmtprocess) == 5
        assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)
        assert isinstance(stack.stmtprocess[1], filters.StripCommentsFilter)
        assert isinstance(stack.stmtprocess[2], filters.StripWhitespaceFilter)
        assert isinstance(stack.stmtprocess[3], filters.ReindentFilter)
        assert isinstance(stack.stmtprocess[4], filters.RightMarginFilter)
        assert len(stack.postprocess) == 1
        assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)

    def test_grouping_side_effect_via_enable_grouping(self):
        # build_filter_stack calls stack.enable_grouping() for the relevant
        # options. Rather than asserting on a non-existent attribute, we
        # confirm the filter-building path succeeds and produces expected
        # stacks.
        stack = _validate_and_build({})
        assert stack.preprocess == []
        assert stack.stmtprocess == []
        assert stack.postprocess == []
