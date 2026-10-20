"""Tests for sqlparse.formatter."""

import pytest

from sqlparse import filters
from sqlparse.exceptions import SQLParseError
from sqlparse.formatter import validate_options, build_filter_stack


# ---------------------------------------------------------------------------
# validate_options: defaults and return value
# ---------------------------------------------------------------------------

def test_validate_options_returns_same_dict_object():
    options = {}
    result = validate_options(options)
    assert result is options


def test_validate_options_defaults():
    options = validate_options({})
    # Booleans default to False and are normalized in the dict.
    assert options['strip_comments'] is False
    assert options['use_space_around_operators'] is False
    assert options['strip_whitespace'] is False
    assert options['indent_columns'] is False
    assert options['indent_after_first'] is False
    assert options['comma_first'] is False
    assert options['compact'] is False
    # Numeric defaults.
    assert options['indent_width'] == 2
    assert options['wrap_after'] == 0
    # indent_char derived from indent_tabs default (False).
    assert options['indent_char'] == ' '
    # right_margin default None.
    assert options['right_margin'] is None
    # truncate_strings default means no truncate keys added.
    assert 'truncate_strings' not in options
    assert 'truncate_char' not in options
    # reindent not added unless given.
    assert 'reindent' not in options


@pytest.mark.parametrize('key', ['keyword_case', 'identifier_case'])
def test_validate_options_case_valid_values(key):
    for value in [None, 'upper', 'lower', 'capitalize']:
        options = validate_options({key: value})
        assert options[key] == value


@pytest.mark.parametrize('key', ['keyword_case', 'identifier_case'])
def test_validate_options_case_invalid(key):
    with pytest.raises(SQLParseError):
        validate_options({key: 'bogus'})


def test_validate_options_keyword_case_error_message():
    with pytest.raises(SQLParseError, match='Invalid value for keyword_case'):
        validate_options({'keyword_case': 'mix'})


def test_validate_options_identifier_case_error_message():
    with pytest.raises(SQLParseError,
                       match='Invalid value for identifier_case'):
        validate_options({'identifier_case': 123})


# ---------------------------------------------------------------------------
# output_format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value', [None, 'sql', 'python', 'php'])
def test_validate_options_output_format_valid(value):
    options = validate_options({'output_format': value})
    assert options['output_format'] == value


def test_validate_options_output_format_invalid():
    with pytest.raises(SQLParseError, match='Unknown output format'):
        validate_options({'output_format': 'ruby'})


# ---------------------------------------------------------------------------
# Boolean options
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('key', ['strip_comments',
                                 'use_space_around_operators',
                                 'strip_whitespace',
                                 'indent_columns',
                                 'indent_after_first',
                                 'comma_first',
                                 'compact'])
@pytest.mark.parametrize('value', [True, False])
def test_validate_options_boolean_valid(key, value):
    options = validate_options({key: value})
    assert options[key] is value


@pytest.mark.parametrize('key', ['strip_comments',
                                 'use_space_around_operators',
                                 'strip_whitespace',
                                 'indent_columns',
                                 'indent_after_first',
                                 'comma_first',
                                 'compact'])
def test_validate_options_boolean_invalid(key):
    with pytest.raises(SQLParseError):
        validate_options({key: 'yes'})


def test_validate_options_reindent_valid():
    options = validate_options({'reindent': True})
    assert options['reindent'] is True
    # reindent enforces strip_whitespace.
    assert options['strip_whitespace'] is True


def test_validate_options_reindent_invalid():
    with pytest.raises(SQLParseError, match='Invalid value for reindent'):
        validate_options({'reindent': 1})


def test_validate_options_reindent_aligned_valid():
    options = validate_options({'reindent_aligned': True})
    assert options['strip_whitespace'] is True


def test_validate_options_reindent_aligned_invalid():
    with pytest.raises(SQLParseError):
        validate_options({'reindent_aligned': 'no'})


def test_validate_options_indent_columns_enforces_reindent():
    options = validate_options({'indent_columns': True})
    assert options['indent_columns'] is True
    assert options['reindent'] is True
    # reindent then enforces strip_whitespace.
    assert options['strip_whitespace'] is True


# ---------------------------------------------------------------------------
# truncate_strings
# ---------------------------------------------------------------------------

def test_validate_options_truncate_strings_valid():
    options = validate_options({'truncate_strings': 5})
    assert options['truncate_strings'] == 5
    assert options['truncate_char'] == '[...]'


def test_validate_options_truncate_strings_custom_char():
    options = validate_options({'truncate_strings': 5, 'truncate_char': '..'})
    assert options['truncate_char'] == '..'


def test_validate_options_truncate_strings_string_number():
    options = validate_options({'truncate_strings': '10'})
    assert options['truncate_strings'] == 10


def test_validate_options_truncate_strings_none_skips():
    options = validate_options({'truncate_strings': None})
    assert 'truncate_strings' not in options


@pytest.mark.parametrize('value', [0, 1, -5])
def test_validate_options_truncate_strings_too_small(value):
    with pytest.raises(SQLParseError,
                       match='Invalid value for truncate_strings'):
        validate_options({'truncate_strings': value})


@pytest.mark.parametrize('value', ['abc', object()])
def test_validate_options_truncate_strings_not_int(value):
    with pytest.raises(SQLParseError,
                       match='Invalid value for truncate_strings'):
        validate_options({'truncate_strings': value})


# ---------------------------------------------------------------------------
# indent_tabs / indent_char
# ---------------------------------------------------------------------------

def test_validate_options_indent_tabs_true():
    options = validate_options({'indent_tabs': True})
    assert options['indent_char'] == '\t'


def test_validate_options_indent_tabs_false():
    options = validate_options({'indent_tabs': False})
    assert options['indent_char'] == ' '


def test_validate_options_indent_tabs_invalid():
    with pytest.raises(SQLParseError, match='Invalid value for indent_tabs'):
        validate_options({'indent_tabs': 'nope'})


# ---------------------------------------------------------------------------
# indent_width
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [(4, 4), ('8', 8), (1, 1)])
def test_validate_options_indent_width_valid(value, expected):
    options = validate_options({'indent_width': value})
    assert options['indent_width'] == expected


@pytest.mark.parametrize('value', ['abc', None, object()])
def test_validate_options_indent_width_not_int(value):
    with pytest.raises(SQLParseError, match='indent_width requires an integer'):
        validate_options({'indent_width': value})


@pytest.mark.parametrize('value', [0, -1, -10])
def test_validate_options_indent_width_not_positive(value):
    with pytest.raises(SQLParseError,
                       match='indent_width requires a positive integer'):
        validate_options({'indent_width': value})


# ---------------------------------------------------------------------------
# wrap_after
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [(0, 0), (5, 5), ('10', 10)])
def test_validate_options_wrap_after_valid(value, expected):
    options = validate_options({'wrap_after': value})
    assert options['wrap_after'] == expected


@pytest.mark.parametrize('value', ['abc', None, object()])
def test_validate_options_wrap_after_not_int(value):
    with pytest.raises(SQLParseError, match='wrap_after requires an integer'):
        validate_options({'wrap_after': value})


def test_validate_options_wrap_after_negative():
    with pytest.raises(SQLParseError,
                       match='wrap_after requires a positive integer'):
        validate_options({'wrap_after': -1})


# ---------------------------------------------------------------------------
# right_margin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [(10, 10), (80, 80), ('50', 50)])
def test_validate_options_right_margin_valid(value, expected):
    options = validate_options({'right_margin': value})
    assert options['right_margin'] == expected


def test_validate_options_right_margin_none():
    options = validate_options({'right_margin': None})
    assert options['right_margin'] is None


def test_validate_options_right_margin_too_small():
    with pytest.raises(SQLParseError,
                       match='right_margin requires an integer > 10'):
        validate_options({'right_margin': 9})


@pytest.mark.parametrize('value', ['abc', object()])
def test_validate_options_right_margin_not_int(value):
    with pytest.raises(SQLParseError,
                       match='right_margin requires an integer'):
        validate_options({'right_margin': value})


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

class FakeStack:
    """Minimal stand-in for filters.FilterStack to record interactions."""

    def __init__(self):
        self.preprocess = []
        self.stmtprocess = []
        self.postprocess = []
        self.grouping_enabled = 0

    def enable_grouping(self):
        self.grouping_enabled += 1


def test_build_filter_stack_returns_stack():
    stack = FakeStack()
    result = build_filter_stack(stack, {})
    assert result is stack


def test_build_filter_stack_empty_options_no_filters():
    stack = FakeStack()
    build_filter_stack(stack, {})
    assert stack.preprocess == []
    assert stack.stmtprocess == []
    assert stack.postprocess == []
    assert stack.grouping_enabled == 0


def test_build_filter_stack_keyword_case():
    stack = FakeStack()
    build_filter_stack(stack, {'keyword_case': 'upper'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)


def test_build_filter_stack_identifier_case():
    stack = FakeStack()
    build_filter_stack(stack, {'identifier_case': 'lower'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)


def test_build_filter_stack_truncate_strings():
    stack = FakeStack()
    build_filter_stack(stack, {'truncate_strings': 5, 'truncate_char': '...'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.TruncateStringFilter)


def test_build_filter_stack_truncate_strings_no_char():
    # No truncate_char key -> no truncate filter appended.
    stack = FakeStack()
    build_filter_stack(stack, {'truncate_strings': 5})
    assert stack.preprocess == []


def test_build_filter_stack_use_space_around_operators():
    stack = FakeStack()
    build_filter_stack(stack, {'use_space_around_operators': True})
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)


def test_build_filter_stack_strip_comments():
    stack = FakeStack()
    build_filter_stack(stack, {'strip_comments': True})
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)


def test_build_filter_stack_strip_whitespace():
    stack = FakeStack()
    build_filter_stack(stack, {'strip_whitespace': True})
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)


def test_build_filter_stack_reindent_adds_strip_and_reindent():
    stack = FakeStack()
    opts = validate_options({'reindent': True})
    build_filter_stack(stack, opts)
    kinds = [type(f) for f in stack.stmtprocess]
    assert filters.StripWhitespaceFilter in kinds
    assert filters.ReindentFilter in kinds
    # Reindent comes after StripWhitespace.
    assert kinds.index(filters.StripWhitespaceFilter) < \
        kinds.index(filters.ReindentFilter)


def test_build_filter_stack_reindent_aligned():
    stack = FakeStack()
    opts = validate_options({'reindent_aligned': True})
    build_filter_stack(stack, opts)
    kinds = [type(f) for f in stack.stmtprocess]
    assert filters.AlignedIndentFilter in kinds


def test_build_filter_stack_right_margin():
    stack = FakeStack()
    opts = validate_options({'right_margin': 20})
    build_filter_stack(stack, opts)
    kinds = [type(f) for f in stack.stmtprocess]
    assert filters.RightMarginFilter in kinds


def test_build_filter_stack_output_php():
    stack = FakeStack()
    build_filter_stack(stack, {'output_format': 'php'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_output_python():
    stack = FakeStack()
    build_filter_stack(stack, {'output_format': 'python'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


def test_build_filter_stack_output_sql_no_postprocess():
    stack = FakeStack()
    build_filter_stack(stack, {'output_format': 'sql'})
    assert stack.postprocess == []


def test_build_filter_stack_full_combination():
    stack = FakeStack()
    opts = validate_options({
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'truncate_strings': 3,
        'use_space_around_operators': True,
        'strip_comments': True,
        'reindent': True,
        'output_format': 'python',
    })
    build_filter_stack(stack, opts)
    # Three preprocess filters: keyword, identifier, truncate.
    assert len(stack.preprocess) == 3
    pre_kinds = [type(f) for f in stack.preprocess]
    assert pre_kinds == [
        filters.KeywordCaseFilter,
        filters.IdentifierCaseFilter,
        filters.TruncateStringFilter,
    ]
    # Statement filters include operators, comments, strip whitespace,
    # reindent.
    stmt_kinds = [type(f) for f in stack.stmtprocess]
    assert filters.SpacesAroundOperatorsFilter in stmt_kinds
    assert filters.StripCommentsFilter in stmt_kinds
    assert filters.StripWhitespaceFilter in stmt_kinds
    assert filters.ReindentFilter in stmt_kinds
    # One postprocess filter for python output.
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)
    # Grouping enabled multiple times.
    assert stack.grouping_enabled >= 1
