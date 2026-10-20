"""Unit tests for sqlparse.formatter."""

import pytest

from sqlparse import filters
from sqlparse.exceptions import SQLParseError
from sqlparse.formatter import build_filter_stack, validate_options


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyStack:
    """Minimal stand-in for filters.FilterStack."""

    def __init__(self):
        self.preprocess = []
        self.stmtprocess = []
        self.postprocess = []
        self.grouping_enabled = False

    def enable_grouping(self):
        self.grouping_enabled = True


# ---------------------------------------------------------------------------
# validate_options: valid inputs
# ---------------------------------------------------------------------------

def test_validate_options_empty_dict():
    opts = validate_options({})
    assert opts['indent_char'] == ' '
    assert opts['indent_width'] == 2
    assert opts['wrap_after'] == 0
    assert opts['indent_columns'] is False
    assert opts['indent_after_first'] is False
    assert opts['comma_first'] is False
    assert opts['compact'] is False
    assert opts['right_margin'] is None


def test_validate_options_returns_same_object():
    opts = {'keyword_case': 'upper'}
    result = validate_options(opts)
    assert result is opts


@pytest.mark.parametrize('case', ['upper', 'lower', 'capitalize'])
def test_validate_options_valid_keyword_case(case):
    opts = validate_options({'keyword_case': case})
    assert opts['keyword_case'] == case


@pytest.mark.parametrize('case', ['upper', 'lower', 'capitalize'])
def test_validate_options_valid_identifier_case(case):
    opts = validate_options({'identifier_case': case})
    assert opts['identifier_case'] == case


@pytest.mark.parametrize('fmt', ['sql', 'python', 'php'])
def test_validate_options_valid_output_format(fmt):
    opts = validate_options({'output_format': fmt})
    assert opts['output_format'] == fmt


@pytest.mark.parametrize('case', [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_keyword_case_none_allowed(case):
    validate_options({'keyword_case': case})  # no raise


# ---------------------------------------------------------------------------
# validate_options: invalid keyword/identifier/output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', ['UPPER', 'Upper', 'mixed', 1, True, ''])
def test_validate_options_invalid_keyword_case(case):
    with pytest.raises(SQLParseError, match='Invalid value for keyword_case'):
        validate_options({'keyword_case': case})


@pytest.mark.parametrize('case', ['UPPER', 'Upper', 'mixed', 1, True, ''])
def test_validate_options_invalid_identifier_case(case):
    with pytest.raises(SQLParseError,
                       match='Invalid value for identifier_case'):
        validate_options({'identifier_case': case})


@pytest.mark.parametrize('fmt', ['SQL', 'javascript', 'html', 1, ''])
def test_validate_options_invalid_output_format(fmt):
    with pytest.raises(SQLParseError, match='Unknown output format'):
        validate_options({'output_format': fmt})


# ---------------------------------------------------------------------------
# validate_options: booleans
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('key', [
    'strip_comments',
    'use_space_around_operators',
    'strip_whitespace',
    'indent_columns',
    'indent_after_first',
    'indent_tabs',
])
def test_validate_options_boolean_defaults_ok(key):
    # default (missing) and explicit booleans should be accepted
    validate_options({})
    validate_options({key: True})
    validate_options({key: False})


@pytest.mark.parametrize('key', [
    'strip_comments',
    'use_space_around_operators',
    'strip_whitespace',
    'indent_columns',
    'indent_after_first',
    'indent_tabs',
])
@pytest.mark.parametrize('value', ['yes', 1, 0, [], 'True'])
def test_validate_options_boolean_invalid(key, value):
    with pytest.raises(SQLParseError):
        validate_options({key: value})


def test_validate_options_reindent_invalid():
    with pytest.raises(SQLParseError, match='Invalid value for reindent'):
        validate_options({'reindent': 'yes'})


def test_validate_options_reindent_aligned_invalid():
    with pytest.raises(SQLParseError,
                       match='Invalid value for reindent_aligned'):
        validate_options({'reindent_aligned': 'yes'})


def test_validate_options_comma_first_invalid():
    with pytest.raises(SQLParseError,
                       match='comma_first requires a boolean value'):
        validate_options({'comma_first': 'yes'})


def test_validate_options_compact_invalid():
    with pytest.raises(SQLParseError,
                       match='compact requires a boolean value'):
        validate_options({'compact': 'yes'})


# ---------------------------------------------------------------------------
# validate_options: truncate_strings
# ---------------------------------------------------------------------------

def test_validate_options_truncate_strings_none():
    opts = validate_options({})
    assert 'truncate_strings' not in opts


def test_validate_options_truncate_strings_valid():
    opts = validate_options({'truncate_strings': 10})
    assert opts['truncate_strings'] == 10
    assert opts['truncate_char'] == '[...]'


def test_validate_options_truncate_strings_string_number():
    opts = validate_options({'truncate_strings': '10'})
    assert opts['truncate_strings'] == 10


def test_validate_options_truncate_strings_custom_char():
    opts = validate_options({'truncate_strings': 5, 'truncate_char': '***'})
    assert opts['truncate_strings'] == 5
    assert opts['truncate_char'] == '***'


@pytest.mark.parametrize('value', ['abc', [1], {}, 1.5])
def test_validate_options_truncate_strings_invalid_type(value):
    with pytest.raises(SQLParseError,
                       match='Invalid value for truncate_strings'):
        validate_options({'truncate_strings': value})


@pytest.mark.parametrize('value', [1, 0, -5, '1'])
def test_validate_options_truncate_strings_too_small(value):
    with pytest.raises(SQLParseError,
                       match='Invalid value for truncate_strings'):
        validate_options({'truncate_strings': value})


# ---------------------------------------------------------------------------
# validate_options: indent_columns / reindent side effects
# ---------------------------------------------------------------------------

def test_validate_options_indent_columns_enforces_reindent():
    opts = validate_options({'indent_columns': True})
    assert opts['indent_columns'] is True
    assert opts['reindent'] is True
    assert opts['strip_whitespace'] is True


def test_validate_options_reindent_enforces_strip_whitespace():
    opts = validate_options({'reindent': True})
    assert opts['reindent'] is True
    assert opts['strip_whitespace'] is True


def test_validate_options_reindent_aligned_enforces_strip_whitespace():
    opts = validate_options({'reindent_aligned': True})
    assert opts['reindent_aligned'] is True
    assert opts['strip_whitespace'] is True


# ---------------------------------------------------------------------------
# validate_options: indent_tabs / indent_char
# ---------------------------------------------------------------------------

def test_validate_options_indent_tabs_true():
    opts = validate_options({'indent_tabs': True})
    assert opts['indent_char'] == '\t'


def test_validate_options_indent_tabs_false():
    opts = validate_options({'indent_tabs': False})
    assert opts['indent_char'] == ' '


# ---------------------------------------------------------------------------
# validate_options: indent_width
# ---------------------------------------------------------------------------

def test_validate_options_indent_width_default():
    opts = validate_options({})
    assert opts['indent_width'] == 2


@pytest.mark.parametrize('value,expected', [(4, 4), ('3', 3), (1, 1)])
def test_validate_options_indent_width_valid(value, expected):
    opts = validate_options({'indent_width': value})
    assert opts['indent_width'] == expected


@pytest.mark.parametrize('value', ['abc', None, []])
def test_validate_options_indent_width_invalid_type(value):
    with pytest.raises(SQLParseError, match='indent_width requires an integer'):
        validate_options({'indent_width': value})


@pytest.mark.parametrize('value', [0, -1, '-3'])
def test_validate_options_indent_width_non_positive(value):
    with pytest.raises(SQLParseError,
                       match='indent_width requires a positive integer'):
        validate_options({'indent_width': value})


# ---------------------------------------------------------------------------
# validate_options: wrap_after
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [(0, 0), (10, 10), ('5', 5)])
def test_validate_options_wrap_after_valid(value, expected):
    opts = validate_options({'wrap_after': value})
    assert opts['wrap_after'] == expected


@pytest.mark.parametrize('value', ['abc', None, []])
def test_validate_options_wrap_after_invalid_type(value):
    with pytest.raises(SQLParseError, match='wrap_after requires an integer'):
        validate_options({'wrap_after': value})


@pytest.mark.parametrize('value', [-1, '-5'])
def test_validate_options_wrap_after_negative(value):
    with pytest.raises(SQLParseError,
                       match='wrap_after requires a positive integer'):
        validate_options({'wrap_after': value})


# ---------------------------------------------------------------------------
# validate_options: right_margin
# ---------------------------------------------------------------------------

def test_validate_options_right_margin_none():
    opts = validate_options({})
    assert opts['right_margin'] is None


@pytest.mark.parametrize('value,expected', [(10, 10), ('20', 20), (100, 100)])
def test_validate_options_right_margin_valid(value, expected):
    opts = validate_options({'right_margin': value})
    assert opts['right_margin'] == expected


@pytest.mark.parametrize('value', ['abc', None, []])
def test_validate_options_right_margin_invalid_type(value):
    # None is handled before conversion (skipped), so use a value that
    # reaches int() and fails. But None is special-cased, so it won't raise.
    if value is None:
        pytest.skip("None is explicitly allowed for right_margin")
    with pytest.raises(SQLParseError, match='right_margin requires an integer'):
        validate_options({'right_margin': value})


@pytest.mark.parametrize('value', [0, 5, 9, '-1'])
def test_validate_options_right_margin_too_small(value):
    with pytest.raises(SQLParseError,
                       match='right_margin requires an integer > 10'):
        validate_options({'right_margin': value})


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

def test_build_filter_stack_empty():
    stack = DummyStack()
    result = build_filter_stack(stack, {})
    assert result is stack
    assert stack.preprocess == []
    assert stack.stmtprocess == []
    assert stack.postprocess == []
    assert stack.grouping_enabled is False


def test_build_filter_stack_keyword_case():
    stack = DummyStack()
    build_filter_stack(stack, {'keyword_case': 'upper'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert stack.grouping_enabled is False


def test_build_filter_stack_identifier_case():
    stack = DummyStack()
    build_filter_stack(stack, {'identifier_case': 'lower'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)


def test_build_filter_stack_truncate_strings():
    stack = DummyStack()
    build_filter_stack(stack, {'truncate_strings': 10, 'truncate_char': '..'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.TruncateStringFilter)


def test_build_filter_stack_use_space_around_operators():
    stack = DummyStack()
    build_filter_stack(stack, {'use_space_around_operators': True})
    assert stack.grouping_enabled is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)


def test_build_filter_stack_strip_comments():
    stack = DummyStack()
    build_filter_stack(stack, {'strip_comments': True})
    assert stack.grouping_enabled is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)


def test_build_filter_stack_strip_whitespace():
    stack = DummyStack()
    build_filter_stack(stack, {'strip_whitespace': True})
    assert stack.grouping_enabled is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)


def test_build_filter_stack_reindent_adds_strip_and_reindent():
    stack = DummyStack()
    opts = validate_options({'reindent': True})
    build_filter_stack(stack, opts)
    assert stack.grouping_enabled is True
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)


def test_build_filter_stack_reindent_passes_options():
    stack = DummyStack()
    opts = validate_options({
        'reindent': True,
        'indent_tabs': True,
        'indent_width': 4,
        'indent_after_first': True,
        'indent_columns': True,
        'wrap_after': 20,
        'comma_first': True,
        'compact': True,
    })
    build_filter_stack(stack, opts)
    reindent = [f for f in stack.stmtprocess
                if isinstance(f, filters.ReindentFilter)][0]
    assert reindent.char == '\t'
    assert reindent.width == 4
    assert reindent.indent_after_first is True
    assert reindent.indent_columns is True
    assert reindent.wrap_after == 20
    assert reindent.comma_first is True
    assert reindent.compact is True


def test_build_filter_stack_reindent_aligned():
    stack = DummyStack()
    opts = validate_options({'reindent_aligned': True, 'indent_tabs': True})
    build_filter_stack(stack, opts)
    assert stack.grouping_enabled is True
    assert isinstance(stack.stmtprocess[-1], filters.AlignedIndentFilter)
    assert stack.stmtprocess[-1].char == '\t'


def test_build_filter_stack_right_margin():
    stack = DummyStack()
    opts = validate_options({'right_margin': 40})
    build_filter_stack(stack, opts)
    assert stack.grouping_enabled is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.RightMarginFilter)


def test_build_filter_stack_output_python():
    stack = DummyStack()
    build_filter_stack(stack, {'output_format': 'python'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


def test_build_filter_stack_output_php():
    stack = DummyStack()
    build_filter_stack(stack, {'output_format': 'php'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_output_sql_adds_no_postprocess():
    stack = DummyStack()
    build_filter_stack(stack, {'output_format': 'sql'})
    assert stack.postprocess == []


def test_build_filter_stack_combined():
    stack = DummyStack()
    opts = validate_options({
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'strip_comments': True,
        'strip_whitespace': True,
        'output_format': 'python',
    })
    build_filter_stack(stack, opts)
    assert len(stack.preprocess) == 2
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert isinstance(stack.preprocess[1], filters.IdentifierCaseFilter)
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)
    # strip_comments + strip_whitespace
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)
    assert isinstance(stack.stmtprocess[1], filters.StripWhitespaceFilter)


def test_build_filter_stack_returns_stack():
    stack = DummyStack()
    assert build_filter_stack(stack, {}) is stack
