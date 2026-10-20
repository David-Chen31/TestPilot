import pytest

from sqlparse import filters
from sqlparse.exceptions import SQLParseError
from sqlparse.formatter import validate_options, build_filter_stack


# ---------------------------------------------------------------------------
# validate_options
# ---------------------------------------------------------------------------

def test_validate_options_empty_returns_defaults():
    opts = validate_options({})
    assert opts['indent_columns'] is False
    assert opts['indent_after_first'] is False
    assert opts['indent_char'] == ' '
    assert opts['indent_width'] == 2
    assert opts['wrap_after'] == 0
    assert opts['comma_first'] is False
    assert opts['compact'] is False
    assert opts['right_margin'] is None
    # No truncate keys added when not provided
    assert 'truncate_strings' not in opts
    assert 'truncate_char' not in opts


def test_validate_options_returns_same_dict():
    opts = {'keyword_case': 'upper'}
    result = validate_options(opts)
    assert result is opts


@pytest.mark.parametrize('good', [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_keyword_case_valid(good):
    opts = validate_options({'keyword_case': good})
    assert opts.get('keyword_case') == good


@pytest.mark.parametrize('bad', ['UPPER', 'Capitalize', 'foo', 1, True])
def test_validate_options_keyword_case_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'keyword_case': bad})
    assert 'Invalid value for keyword_case' in str(exc.value)


@pytest.mark.parametrize('good', [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_identifier_case_valid(good):
    validate_options({'identifier_case': good})


@pytest.mark.parametrize('bad', ['foo', 'UPPER', 3])
def test_validate_options_identifier_case_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'identifier_case': bad})
    assert 'Invalid value for identifier_case' in str(exc.value)


@pytest.mark.parametrize('good', [None, 'sql', 'python', 'php'])
def test_validate_options_output_format_valid(good):
    validate_options({'output_format': good})


@pytest.mark.parametrize('bad', ['SQL', 'ruby', 'PYTHON', 42])
def test_validate_options_output_format_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'output_format': bad})
    assert 'Unknown output format' in str(exc.value)


@pytest.mark.parametrize('good', [True, False])
def test_validate_options_strip_comments_valid(good):
    validate_options({'strip_comments': good})


@pytest.mark.parametrize('bad', [0, 1, 'yes', None])
def test_validate_options_strip_comments_invalid(bad):
    # Default is False so should pass; only non-default invalid values raise.

    if bad is None:
        # None is not True/False -> error
        with pytest.raises(SQLParseError):
            validate_options({'strip_comments': bad})
    else:
        with pytest.raises(SQLParseError) as exc:
            validate_options({'strip_comments': bad})
        assert 'Invalid value for strip_comments' in str(exc.value)


@pytest.mark.parametrize('bad', [0, 1, 'x'])
def test_validate_options_use_space_around_operators_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'use_space_around_operators': bad})
    assert 'Invalid value for use_space_around_operators' in str(exc.value)


@pytest.mark.parametrize('bad', [0, 1, 'x'])
def test_validate_options_strip_whitespace_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'strip_whitespace': bad})
    assert 'Invalid value for strip_whitespace' in str(exc.value)


# truncate_strings
def test_validate_options_truncate_strings_none_skips():
    opts = validate_options({})
    assert 'truncate_strings' not in opts


def test_validate_options_truncate_strings_valid_int():
    opts = validate_options({'truncate_strings': 5})
    assert opts['truncate_strings'] == 5
    assert opts['truncate_char'] == '[...]'


def test_validate_options_truncate_strings_string_coerced():
    opts = validate_options({'truncate_strings': '10'})
    assert opts['truncate_strings'] == 10


def test_validate_options_truncate_strings_custom_char():
    opts = validate_options({'truncate_strings': 3, 'truncate_char': '>>'})
    assert opts['truncate_char'] == '>>'


@pytest.mark.parametrize('bad', ['abc', [1], 2.5, ''])
def test_validate_options_truncate_strings_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'truncate_strings': bad})
    assert 'truncate_strings' in str(exc.value)


@pytest.mark.parametrize('bad', [1, 0, -5, '1', '0'])
def test_validate_options_truncate_strings_too_small(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'truncate_strings': bad})
    assert 'truncate_strings' in str(exc.value)


# indent_columns
def test_validate_options_indent_columns_true_enforces_reindent():
    opts = validate_options({'indent_columns': True})
    assert opts['indent_columns'] is True
    assert opts['reindent'] is True


def test_validate_options_indent_columns_false():
    opts = validate_options({'indent_columns': False})
    assert opts['indent_columns'] is False
    assert 'reindent' not in opts


@pytest.mark.parametrize('bad', [0, 1, 'yes'])
def test_validate_options_indent_columns_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_columns': bad})
    assert 'Invalid value for indent_columns' in str(exc.value)


# reindent
def test_validate_options_reindent_true_enforces_strip_whitespace():
    opts = validate_options({'reindent': True})
    assert opts['strip_whitespace'] is True


def test_validate_options_reindent_false():
    opts = validate_options({'reindent': False})
    assert 'strip_whitespace' not in opts


@pytest.mark.parametrize('bad', [0, 1, 'x'])
def test_validate_options_reindent_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'reindent': bad})
    assert 'Invalid value for reindent' in str(exc.value)


# reindent_aligned
def test_validate_options_reindent_aligned_true_sets_strip_whitespace():
    opts = validate_options({'reindent_aligned': True})
    assert opts['strip_whitespace'] is True


@pytest.mark.parametrize('bad', [0, 1, 'x'])
def test_validate_options_reindent_aligned_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'reindent_aligned': bad})
    assert 'Invalid value for reindent' in str(exc.value)


# indent_after_first
def test_validate_options_indent_after_first_set():
    opts = validate_options({'indent_after_first': True})
    assert opts['indent_after_first'] is True
    opts = validate_options({'indent_after_first': False})
    assert opts['indent_after_first'] is False


@pytest.mark.parametrize('bad', [0, 1, 'x'])
def test_validate_options_indent_after_first_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_after_first': bad})
    assert 'Invalid value for indent_after_first' in str(exc.value)


# indent_tabs
def test_validate_options_indent_tabs_true():
    opts = validate_options({'indent_tabs': True})
    assert opts['indent_char'] == '\t'


def test_validate_options_indent_tabs_false():
    opts = validate_options({'indent_tabs': False})
    assert opts['indent_char'] == ' '


@pytest.mark.parametrize('bad', [0, 1, 'x'])
def test_validate_options_indent_tabs_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_tabs': bad})
    assert 'Invalid value for indent_tabs' in str(exc.value)


# indent_width
def test_validate_options_indent_width_default_and_set():
    assert validate_options({})['indent_width'] == 2
    assert validate_options({'indent_width': 4})['indent_width'] == 4
    assert validate_options({'indent_width': '8'})['indent_width'] == 8


@pytest.mark.parametrize('bad', ['abc', [1], 2.5, ''])
def test_validate_options_indent_width_invalid_type(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': bad})
    assert 'indent_width requires an integer' in str(exc.value)


@pytest.mark.parametrize('bad', [0, -1, '-3'])
def test_validate_options_indent_width_nonpositive(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': bad})
    assert 'indent_width requires a positive integer' in str(exc.value)


# wrap_after
def test_validate_options_wrap_after_default_and_set():
    assert validate_options({})['wrap_after'] == 0
    assert validate_options({'wrap_after': 10})['wrap_after'] == 10
    assert validate_options({'wrap_after': '5'})['wrap_after'] == 5


@pytest.mark.parametrize('bad', ['abc', [1], 2.5])
def test_validate_options_wrap_after_invalid_type(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': bad})
    assert 'wrap_after requires an integer' in str(exc.value)


@pytest.mark.parametrize('bad', [-1, '-3'])
def test_validate_options_wrap_after_negative(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': bad})
    assert 'wrap_after requires a positive integer' in str(exc.value)


# comma_first
def test_validate_options_comma_first_valid():
    assert validate_options({'comma_first': True})['comma_first'] is True
    assert validate_options({'comma_first': False})['comma_first'] is False


@pytest.mark.parametrize('bad', [0, 1, 'x'])
def test_validate_options_comma_first_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'comma_first': bad})
    assert 'comma_first requires a boolean' in str(exc.value)


# compact
def test_validate_options_compact_valid():
    assert validate_options({'compact': True})['compact'] is True
    assert validate_options({'compact': False})['compact'] is False


@pytest.mark.parametrize('bad', [0, 1, 'x'])
def test_validate_options_compact_invalid(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'compact': bad})
    assert 'compact requires a boolean' in str(exc.value)


# right_margin
def test_validate_options_right_margin_default_none():
    assert validate_options({})['right_margin'] is None


def test_validate_options_right_margin_valid():
    assert validate_options({'right_margin': 10})['right_margin'] == 10
    assert validate_options({'right_margin': '20'})['right_margin'] == 20


@pytest.mark.parametrize('bad', ['abc', [1], 2.5])
def test_validate_options_right_margin_invalid_type(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'right_margin': bad})
    assert 'right_margin requires an integer' in str(exc.value)


@pytest.mark.parametrize('bad', [0, 9, -5, '9'])
def test_validate_options_right_margin_too_small(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'right_margin': bad})
    assert 'right_margin requires an integer > 10' in str(exc.value)


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

class _MockStack:
    def __init__(self):
        self.preprocess = []
        self.stmtprocess = []
        self.postprocess = []
        self.grouping_enabled = False

    def enable_grouping(self):
        self.grouping_enabled = True


def test_build_filter_stack_empty():
    stack = _MockStack()
    result = build_filter_stack(stack, {})
    assert result is stack
    assert stack.preprocess == []
    assert stack.stmtprocess == []
    assert stack.postprocess == []
    assert stack.grouping_enabled is False


def test_build_filter_stack_keyword_case():
    stack = _MockStack()
    build_filter_stack(stack, {'keyword_case': 'upper'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)


def test_build_filter_stack_identifier_case():
    stack = _MockStack()
    build_filter_stack(stack, {'identifier_case': 'lower'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)


def test_build_filter_stack_truncate_strings():
    stack = _MockStack()
    build_filter_stack(stack, {'truncate_strings': 5, 'truncate_char': '..'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.TruncateStringFilter)


def test_build_filter_stack_use_space_around_operators():
    stack = _MockStack()
    build_filter_stack(stack, {'use_space_around_operators': True})
    assert stack.grouping_enabled is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)


def test_build_filter_stack_strip_comments():
    stack = _MockStack()
    build_filter_stack(stack, {'strip_comments': True})
    assert stack.grouping_enabled is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)


def test_build_filter_stack_strip_whitespace():
    stack = _MockStack()
    build_filter_stack(stack, {'strip_whitespace': True})
    assert stack.grouping_enabled is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)


def test_build_filter_stack_reindent():
    stack = _MockStack()
    opts = validate_options({'reindent': True})
    build_filter_stack(stack, opts)
    # StripWhitespaceFilter then ReindentFilter
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)


def test_build_filter_stack_reindent_aligned():
    stack = _MockStack()
    opts = validate_options({'reindent_aligned': True})
    build_filter_stack(stack, opts)
    # StripWhitespaceFilter then AlignedIndentFilter
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.AlignedIndentFilter)


def test_build_filter_stack_right_margin():
    stack = _MockStack()
    build_filter_stack(stack, {'right_margin': 20})
    assert stack.grouping_enabled is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.RightMarginFilter)


def test_build_filter_stack_output_format_php():
    stack = _MockStack()
    build_filter_stack(stack, {'output_format': 'php'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_output_format_python():
    stack = _MockStack()
    build_filter_stack(stack, {'output_format': 'python'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


def test_build_filter_stack_output_format_sql():
    stack = _MockStack()
    build_filter_stack(stack, {'output_format': 'sql'})
    assert stack.postprocess == []


def test_build_filter_stack_combined():
    stack = _MockStack()
    opts = validate_options({
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'truncate_strings': 5,
        'strip_comments': True,
        'reindent': True,
        'right_margin': 30,
        'output_format': 'python',
    })
    build_filter_stack(stack, opts)
    assert len(stack.preprocess) == 3
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert isinstance(stack.preprocess[1], filters.IdentifierCaseFilter)
    assert isinstance(stack.preprocess[2], filters.TruncateStringFilter)

    # strip_comments (grouping + filter), strip_whitespace via reindent,
    # reindent, right_margin
    assert stack.grouping_enabled is True
    types = [type(f) for f in stack.stmtprocess]
    assert filters.StripCommentsFilter in types
    assert filters.StripWhitespaceFilter in types
    assert filters.ReindentFilter in types
    assert filters.RightMarginFilter in types

    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)
