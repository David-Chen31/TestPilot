import pytest

from sqlparse.formatter import validate_options, build_filter_stack
from sqlparse.exceptions import SQLParseError
from sqlparse import filters


class DummyStack:
    def __init__(self):
        self.preprocess = []
        self.stmtprocess = []
        self.postprocess = []
        self.grouping_enabled = 0

    def enable_grouping(self):
        self.grouping_enabled += 1


# ---------------------------------------------------------------------------
# validate_options
# ---------------------------------------------------------------------------

def test_validate_options_empty_dict_defaults():
    options = validate_options({})
    assert options['indent_columns'] is False
    assert options['indent_after_first'] is False
    assert options['indent_char'] == ' '
    assert options['indent_width'] == 2
    assert options['wrap_after'] == 0
    assert options['comma_first'] is False
    assert options['compact'] is False
    assert options['right_margin'] is None
    # No truncate options added when truncate_strings not provided
    assert 'truncate_strings' not in options
    assert 'truncate_char' not in options
    assert 'reindent' not in options


def test_validate_options_returns_same_dict():
    options = {'keyword_case': 'upper'}
    result = validate_options(options)
    assert result is options


@pytest.mark.parametrize('kwcase', [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_keyword_case_valid(kwcase):
    options = validate_options({'keyword_case': kwcase})
    assert options['keyword_case'] == kwcase


def test_validate_options_keyword_case_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'keyword_case': 'title'})
    assert 'Invalid value for keyword_case' in str(exc.value)


@pytest.mark.parametrize('idcase', [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_identifier_case_valid(idcase):
    options = validate_options({'identifier_case': idcase})
    assert options['identifier_case'] == idcase


def test_validate_options_identifier_case_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'identifier_case': 'mixed'})
    assert 'Invalid value for identifier_case' in str(exc.value)


@pytest.mark.parametrize('ofrmt', [None, 'sql', 'python', 'php'])
def test_validate_options_output_format_valid(ofrmt):
    options = validate_options({'output_format': ofrmt})
    assert options['output_format'] == ofrmt


def test_validate_options_output_format_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'output_format': 'ruby'})
    assert 'Unknown output format' in str(exc.value)


@pytest.mark.parametrize('val', [True, False])
def test_validate_options_strip_comments_valid(val):
    options = validate_options({'strip_comments': val})
    assert options['strip_comments'] is val


def test_validate_options_strip_comments_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'strip_comments': 1})
    assert 'Invalid value for strip_comments' in str(exc.value)


@pytest.mark.parametrize('val', [True, False])
def test_validate_options_space_around_operators_valid(val):
    options = validate_options({'use_space_around_operators': val})
    assert options['use_space_around_operators'] is val


def test_validate_options_space_around_operators_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'use_space_around_operators': 'yes'})
    assert 'Invalid value for use_space_around_operators' in str(exc.value)


@pytest.mark.parametrize('val', [True, False])
def test_validate_options_strip_whitespace_valid(val):
    options = validate_options({'strip_whitespace': val})
    assert options['strip_whitespace'] is val


def test_validate_options_strip_whitespace_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'strip_whitespace': 'yes'})
    assert 'Invalid value for strip_whitespace' in str(exc.value)


def test_validate_options_truncate_strings_none():
    options = validate_options({'truncate_strings': None})
    assert 'truncate_strings' not in options


@pytest.mark.parametrize('value,expected', [('5', 5), (10, 10), ('2', 2)])
def test_validate_options_truncate_strings_valid(value, expected):
    options = validate_options({'truncate_strings': value})
    assert options['truncate_strings'] == expected
    assert options['truncate_char'] == '[...]'


def test_validate_options_truncate_strings_custom_char():
    options = validate_options({'truncate_strings': 3, 'truncate_char': '***'})
    assert options['truncate_strings'] == 3
    assert options['truncate_char'] == '***'


@pytest.mark.parametrize('bad', ['abc', {}, []])
def test_validate_options_truncate_strings_invalid_type(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'truncate_strings': bad})
    assert 'Invalid value for truncate_strings' in str(exc.value)


@pytest.mark.parametrize('bad', [1, 0, -5, '1'])
def test_validate_options_truncate_strings_too_small(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'truncate_strings': bad})
    assert 'Invalid value for truncate_strings' in str(exc.value)


@pytest.mark.parametrize('val', [True, False])
def test_validate_options_indent_columns_valid(val):
    options = validate_options({'indent_columns': val})
    assert options['indent_columns'] is val


def test_validate_options_indent_columns_enforces_reindent():
    options = validate_options({'indent_columns': True})
    assert options['indent_columns'] is True
    assert options['reindent'] is True


def test_validate_options_indent_columns_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_columns': 'yes'})
    assert 'Invalid value for indent_columns' in str(exc.value)


@pytest.mark.parametrize('val', [True, False])
def test_validate_options_reindent_valid(val):
    options = validate_options({'reindent': val})
    assert options['reindent'] is val


def test_validate_options_reindent_enforces_strip_whitespace():
    options = validate_options({'reindent': True})
    assert options['strip_whitespace'] is True


def test_validate_options_reindent_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'reindent': 1})
    assert 'Invalid value for reindent' in str(exc.value)


@pytest.mark.parametrize('val', [True, False])
def test_validate_options_reindent_aligned_valid(val):
    options = validate_options({'reindent_aligned': val})
    assert options['reindent_aligned'] is val


def test_validate_options_reindent_aligned_enforces_strip_whitespace():
    options = validate_options({'reindent_aligned': True})
    assert options['strip_whitespace'] is True


def test_validate_options_reindent_aligned_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'reindent_aligned': 'yes'})
    assert 'Invalid value for reindent_aligned' in str(exc.value)


@pytest.mark.parametrize('val', [True, False])
def test_validate_options_indent_after_first_valid(val):
    options = validate_options({'indent_after_first': val})
    assert options['indent_after_first'] is val


def test_validate_options_indent_after_first_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_after_first': 'x'})
    assert 'Invalid value for indent_after_first' in str(exc.value)


def test_validate_options_indent_tabs_true():
    options = validate_options({'indent_tabs': True})
    assert options['indent_char'] == '\t'


def test_validate_options_indent_tabs_false():
    options = validate_options({'indent_tabs': False})
    assert options['indent_char'] == ' '


def test_validate_options_indent_tabs_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_tabs': 1})
    assert 'Invalid value for indent_tabs' in str(exc.value)


@pytest.mark.parametrize('val,expected', [('4', 4), (8, 8), (2, 2)])
def test_validate_options_indent_width_valid(val, expected):
    options = validate_options({'indent_width': val})
    assert options['indent_width'] == expected


@pytest.mark.parametrize('bad', ['abc', [], {}])
def test_validate_options_indent_width_invalid_type(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': bad})
    assert 'indent_width requires an integer' in str(exc.value)


@pytest.mark.parametrize('bad', [0, -1, '-5'])
def test_validate_options_indent_width_too_small(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': bad})
    assert 'indent_width requires a positive integer' in str(exc.value)


@pytest.mark.parametrize('val,expected', [('10', 10), (0, 0), (5, 5)])
def test_validate_options_wrap_after_valid(val, expected):
    options = validate_options({'wrap_after': val})
    assert options['wrap_after'] == expected


@pytest.mark.parametrize('bad', ['abc', [], {}])
def test_validate_options_wrap_after_invalid_type(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': bad})
    assert 'wrap_after requires an integer' in str(exc.value)


@pytest.mark.parametrize('bad', [-1, '-5'])
def test_validate_options_wrap_after_negative(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': bad})
    assert 'wrap_after requires a positive integer' in str(exc.value)


@pytest.mark.parametrize('val', [True, False])
def test_validate_options_comma_first_valid(val):
    options = validate_options({'comma_first': val})
    assert options['comma_first'] is val


def test_validate_options_comma_first_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'comma_first': 'yes'})
    assert 'comma_first requires a boolean value' in str(exc.value)


@pytest.mark.parametrize('val', [True, False])
def test_validate_options_compact_valid(val):
    options = validate_options({'compact': val})
    assert options['compact'] is val


def test_validate_options_compact_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'compact': 1})
    assert 'compact requires a boolean value' in str(exc.value)


def test_validate_options_right_margin_none():
    options = validate_options({'right_margin': None})
    assert options['right_margin'] is None


@pytest.mark.parametrize('val,expected', [('10', 10), (20, 20), (100, 100)])
def test_validate_options_right_margin_valid(val, expected):
    options = validate_options({'right_margin': val})
    assert options['right_margin'] == expected


@pytest.mark.parametrize('bad', ['abc', [], {}])
def test_validate_options_right_margin_invalid_type(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'right_margin': bad})
    assert 'right_margin requires an integer' in str(exc.value)


@pytest.mark.parametrize('bad', [0, 9, -5, '5'])
def test_validate_options_right_margin_too_small(bad):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'right_margin': bad})
    assert 'right_margin requires an integer > 10' in str(exc.value)


def test_validate_options_chained_indent_columns_enables_reindent_strip():
    options = validate_options({'indent_columns': True, 'reindent': True})
    assert options['reindent'] is True
    assert options['strip_whitespace'] is True
    assert options['indent_columns'] is True


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

def test_build_filter_stack_empty_options():
    stack = DummyStack()
    result = build_filter_stack(stack, {})
    assert result is stack
    assert stack.preprocess == []
    assert stack.stmtprocess == []
    assert stack.postprocess == []
    assert stack.grouping_enabled == 0


def test_build_filter_stack_keyword_case():
    stack = DummyStack()
    build_filter_stack(stack, {'keyword_case': 'upper'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)


def test_build_filter_stack_identifier_case():
    stack = DummyStack()
    build_filter_stack(stack, {'identifier_case': 'lower'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)


def test_build_filter_stack_truncate_strings():
    stack = DummyStack()
    build_filter_stack(stack, {
        'truncate_strings': 5,
        'truncate_char': '[...]',
    })
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.TruncateStringFilter)


def test_build_filter_stack_truncate_strings_missing_char_raises():
    stack = DummyStack()
    with pytest.raises(KeyError):
        build_filter_stack(stack, {'truncate_strings': 5})


def test_build_filter_stack_space_around_operators():
    stack = DummyStack()
    build_filter_stack(stack, {'use_space_around_operators': True})
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)


def test_build_filter_stack_strip_comments():
    stack = DummyStack()
    build_filter_stack(stack, {'strip_comments': True})
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)


def test_build_filter_stack_strip_whitespace():
    stack = DummyStack()
    build_filter_stack(stack, {'strip_whitespace': True})
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)


def test_build_filter_stack_reindent_alone_creates_two_filters():
    stack = DummyStack()
    options = {
        'reindent': True,
        'indent_char': ' ',
        'indent_width': 2,
        'indent_after_first': False,
        'indent_columns': False,
        'wrap_after': 0,
        'comma_first': False,
        'compact': False,
    }
    build_filter_stack(stack, options)
    # StripWhitespaceFilter then ReindentFilter
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)
    # enable_grouping called twice
    assert stack.grouping_enabled == 2


def test_build_filter_stack_reindent_with_strip_whitespace():
    stack = DummyStack()
    options = {
        'reindent': True,
        'strip_whitespace': True,
        'indent_char': '\t',
        'indent_width': 4,
        'indent_after_first': True,
        'indent_columns': True,
        'wrap_after': 80,
        'comma_first': True,
        'compact': True,
    }
    build_filter_stack(stack, options)
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)
    assert stack.grouping_enabled == 2


def test_build_filter_stack_reindent_missing_options_raises():
    stack = DummyStack()
    with pytest.raises(KeyError):
        build_filter_stack(stack, {'reindent': True})


def test_build_filter_stack_reindent_aligned():
    stack = DummyStack()
    build_filter_stack(stack, {'reindent_aligned': True, 'indent_char': ' '})
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.AlignedIndentFilter)


def test_build_filter_stack_reindent_aligned_missing_char_raises():
    stack = DummyStack()
    with pytest.raises(KeyError):
        build_filter_stack(stack, {'reindent_aligned': True})


def test_build_filter_stack_right_margin():
    stack = DummyStack()
    build_filter_stack(stack, {'right_margin': 80})
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.RightMarginFilter)


def test_build_filter_stack_output_format_php():
    stack = DummyStack()
    build_filter_stack(stack, {'output_format': 'php'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_output_format_python():
    stack = DummyStack()
    build_filter_stack(stack, {'output_format': 'python'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


def test_build_filter_stack_output_format_sql_no_filter():
    stack = DummyStack()
    build_filter_stack(stack, {'output_format': 'sql'})
    assert stack.postprocess == []


def test_build_filter_stack_output_format_case_insensitive():
    stack = DummyStack()
    build_filter_stack(stack, {'output_format': 'PHP'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_combined_full():
    stack = DummyStack()
    options = validate_options({
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'truncate_strings': 5,
        'use_space_around_operators': True,
        'strip_comments': True,
        'strip_whitespace': True,
        'reindent': True,
        'reindent_aligned': True,
        'right_margin': 80,
        'output_format': 'python',
    })
    build_filter_stack(stack, options)
    assert len(stack.preprocess) == 3
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert isinstance(stack.preprocess[1], filters.IdentifierCaseFilter)
    assert isinstance(stack.preprocess[2], filters.TruncateStringFilter)
    # stmtprocess: SpacesAroundOperators, StripComments, StripWhitespace,
    # Reindent, AlignedIndent, RightMargin = 6
    assert len(stack.stmtprocess) == 6
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)
    assert isinstance(stack.stmtprocess[1], filters.StripCommentsFilter)
    assert isinstance(stack.stmtprocess[2], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[3], filters.ReindentFilter)
    assert isinstance(stack.stmtprocess[4], filters.AlignedIndentFilter)
    assert isinstance(stack.stmtprocess[5], filters.RightMarginFilter)
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


def test_build_filter_stack_custom_stack_with_real_filterstack():
    from sqlparse.filters import FilterStack
    stack = FilterStack()
    build_filter_stack(stack, {'keyword_case': 'lower'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
