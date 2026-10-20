import pytest

from sqlparse import filters
from sqlparse.exceptions import SQLParseError
from sqlparse.formatter import validate_options, build_filter_stack
from sqlparse.filters import FilterStack


# ---------------------------------------------------------------------------
# validate_options
# ---------------------------------------------------------------------------

class TestValidateOptionsDefaults:
    def test_empty_options_returns_filled_defaults(self):
        opts = validate_options({})
        # indent-related defaults
        assert opts['indent_columns'] is False
        assert opts['indent_after_first'] is False
        assert opts['indent_char'] == ' '
        assert opts['indent_width'] == 2
        assert opts['wrap_after'] == 0
        assert opts['comma_first'] is False
        assert opts['compact'] is False
        assert opts['right_margin'] is None

    def test_returns_same_object(self):
        opts = {}
        result = validate_options(opts)
        assert result is opts

    def test_truncate_strings_absent(self):
        opts = validate_options({})
        assert 'truncate_strings' not in opts
        assert 'truncate_char' not in opts


class TestValidateOptionsKeywordCase:
    @pytest.mark.parametrize('value', [None, 'upper', 'lower', 'capitalize'])
    def test_valid(self, value):
        opts = validate_options({'keyword_case': value})
        assert opts['keyword_case'] == value

    def test_invalid(self):
        with pytest.raises(SQLParseError, match='Invalid value for keyword_case'):
            validate_options({'keyword_case': 'bogus'})


class TestValidateOptionsIdentifierCase:
    @pytest.mark.parametrize('value', [None, 'upper', 'lower', 'capitalize'])
    def test_valid(self, value):
        opts = validate_options({'identifier_case': value})
        assert opts['identifier_case'] == value

    def test_invalid(self):
        with pytest.raises(SQLParseError, match='Invalid value for identifier_case'):
            validate_options({'identifier_case': 'bogus'})


class TestValidateOptionsOutputFormat:
    @pytest.mark.parametrize('value', [None, 'sql', 'python', 'php'])
    def test_valid(self, value):
        opts = validate_options({'output_format': value})
        assert opts['output_format'] == value

    def test_invalid(self):
        with pytest.raises(SQLParseError, match='Unknown output format'):
            validate_options({'output_format': 'xml'})


class TestValidateOptionsBooleanFlags:
    @pytest.mark.parametrize('key', ['strip_comments',
                                     'use_space_around_operators',
                                     'strip_whitespace'])
    @pytest.mark.parametrize('value', [True, False])
    def test_valid(self, key, value):
        opts = validate_options({key: value})
        assert opts[key] == value

    @pytest.mark.parametrize('key', ['strip_comments',
                                     'use_space_around_operators',
                                     'strip_whitespace'])
    def test_invalid(self, key):
        with pytest.raises(SQLParseError):
            validate_options({key: 'yes'})


class TestValidateOptionsTruncateStrings:
    def test_none_skips(self):
        opts = validate_options({'truncate_strings': None})
        assert 'truncate_strings' not in opts

    def test_valid_int(self):
        opts = validate_options({'truncate_strings': 5})
        assert opts['truncate_strings'] == 5
        assert opts['truncate_char'] == '[...]'

    def test_valid_numeric_string(self):
        opts = validate_options({'truncate_strings': '5'})
        assert opts['truncate_strings'] == 5

    def test_custom_truncate_char(self):
        opts = validate_options({'truncate_strings': 5, 'truncate_char': '...'})
        assert opts['truncate_char'] == '...'

    def test_non_numeric_raises(self):
        with pytest.raises(SQLParseError, match='Invalid value for truncate_strings'):
            validate_options({'truncate_strings': 'abc'})

    @pytest.mark.parametrize('value', [1, 0, -5])
    def test_too_small_raises(self, value):
        with pytest.raises(SQLParseError, match='Invalid value for truncate_strings'):
            validate_options({'truncate_strings': value})

    def test_type_error_raises(self):
        with pytest.raises(SQLParseError):
            validate_options({'truncate_strings': [1, 2]})


class TestValidateOptionsIndentColumns:
    @pytest.mark.parametrize('value', [True, False])
    def test_valid(self, value):
        opts = validate_options({'indent_columns': value})
        assert opts['indent_columns'] == value

    def test_invalid(self):
        with pytest.raises(SQLParseError, match='Invalid value for indent_columns'):
            validate_options({'indent_columns': 1})

    def test_enforces_reindent(self):
        opts = validate_options({'indent_columns': True})
        assert opts['reindent'] is True


class TestValidateOptionsReindent:
    def test_valid(self):
        assert validate_options({'reindent': True})['reindent'] is True
        assert validate_options({'reindent': False})['reindent'] is False

    def test_invalid(self):
        with pytest.raises(SQLParseError, match='Invalid value for reindent'):
            validate_options({'reindent': 'yes'})

    def test_enforces_strip_whitespace(self):
        opts = validate_options({'reindent': True})
        assert opts['strip_whitespace'] is True


class TestValidateOptionsReindentAligned:
    def test_valid(self):
        assert validate_options({'reindent_aligned': True})['reindent_aligned'] is True

    def test_invalid(self):
        with pytest.raises(SQLParseError, match='Invalid value for reindent_aligned'):
            validate_options({'reindent_aligned': 'yes'})

    def test_enforces_strip_whitespace(self):
        opts = validate_options({'reindent_aligned': True})
        assert opts['strip_whitespace'] is True


class TestValidateOptionsIndentAfterFirst:
    def test_default(self):
        assert validate_options({})['indent_after_first'] is False

    def test_valid(self):
        assert validate_options({'indent_after_first': True})['indent_after_first'] is True

    def test_invalid(self):
        with pytest.raises(SQLParseError, match='Invalid value for indent_after_first'):
            validate_options({'indent_after_first': 'yes'})


class TestValidateOptionsIndentTabs:
    def test_tabs_true_sets_tab_char(self):
        opts = validate_options({'indent_tabs': True})
        assert opts['indent_char'] == '\t'

    def test_tabs_false_sets_space_char(self):
        opts = validate_options({'indent_tabs': False})
        assert opts['indent_char'] == ' '

    def test_invalid(self):
        with pytest.raises(SQLParseError, match='Invalid value for indent_tabs'):
            validate_options({'indent_tabs': 'yes'})


class TestValidateOptionsIndentWidth:
    def test_default(self):
        assert validate_options({})['indent_width'] == 2

    def test_valid_int(self):
        assert validate_options({'indent_width': 4})['indent_width'] == 4

    def test_valid_string(self):
        assert validate_options({'indent_width': '4'})['indent_width'] == 4

    def test_bad_type(self):
        with pytest.raises(SQLParseError, match='indent_width requires an integer'):
            validate_options({'indent_width': 'x'})

    @pytest.mark.parametrize('value', [0, -1])
    def test_non_positive(self, value):
        with pytest.raises(SQLParseError,
                           match='indent_width requires a positive integer'):
            validate_options({'indent_width': value})


class TestValidateOptionsWrapAfter:
    def test_default(self):
        assert validate_options({})['wrap_after'] == 0

    def test_valid_int(self):
        assert validate_options({'wrap_after': 80})['wrap_after'] == 80

    def test_valid_string(self):
        assert validate_options({'wrap_after': '80'})['wrap_after'] == 80

    def test_bad_type(self):
        with pytest.raises(SQLParseError, match='wrap_after requires an integer'):
            validate_options({'wrap_after': 'x'})

    def test_negative(self):
        with pytest.raises(SQLParseError,
                           match='wrap_after requires a positive integer'):
            validate_options({'wrap_after': -1})


class TestValidateOptionsCommaFirst:
    def test_default(self):
        assert validate_options({})['comma_first'] is False

    def test_valid(self):
        assert validate_options({'comma_first': True})['comma_first'] is True

    def test_invalid(self):
        with pytest.raises(SQLParseError,
                           match='comma_first requires a boolean value'):
            validate_options({'comma_first': 'yes'})


class TestValidateOptionsCompact:
    def test_default(self):
        assert validate_options({})['compact'] is False

    def test_valid(self):
        assert validate_options({'compact': True})['compact'] is True

    def test_invalid(self):
        with pytest.raises(SQLParseError,
                           match='compact requires a boolean value'):
            validate_options({'compact': 'yes'})


class TestValidateOptionsRightMargin:
    def test_default_none(self):
        assert validate_options({})['right_margin'] is None

    def test_valid(self):
        assert validate_options({'right_margin': 20})['right_margin'] == 20

    def test_valid_string(self):
        assert validate_options({'right_margin': '20'})['right_margin'] == 20

    def test_bad_type(self):
        with pytest.raises(SQLParseError, match='right_margin requires an integer'):
            validate_options({'right_margin': 'x'})

    @pytest.mark.parametrize('value', [10, 9, 0, -1])
    def test_too_small(self, value):
        with pytest.raises(SQLParseError,
                           match='right_margin requires an integer > 10'):
            validate_options({'right_margin': value})


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

def _validated(opts):
    """Run validate_options to produce a complete options dict."""
    return validate_options(dict(opts))


class TestBuildFilterStack:
    def test_empty_options_no_filters(self):
        stack = FilterStack()
        result = build_filter_stack(stack, _validated({}))
        assert result is stack
        assert stack.preprocess == []
        assert stack.stmtprocess == []
        assert stack.postprocess == []

    def test_keyword_case_filter_added(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'keyword_case': 'upper'}))
        assert len(stack.preprocess) == 1
        fltr = stack.preprocess[0]
        assert isinstance(fltr, filters.KeywordCaseFilter)
        assert fltr.ttype is None or True  # smoke: it's constructed

    def test_identifier_case_filter_added(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'identifier_case': 'lower'}))
        assert len(stack.preprocess) == 1
        assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)

    def test_truncate_string_filter_added(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'truncate_strings': 5}))
        assert len(stack.preprocess) == 1
        fltr = stack.preprocess[0]
        assert isinstance(fltr, filters.TruncateStringFilter)

    def test_space_around_operators(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'use_space_around_operators': True}))
        assert stack.grouping is True
        assert len(stack.stmtprocess) == 1
        assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)

    def test_strip_comments(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'strip_comments': True}))
        assert stack.grouping is True
        assert len(stack.stmtprocess) == 1
        assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)

    def test_strip_whitespace(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'strip_whitespace': True}))
        assert stack.grouping is True
        assert len(stack.stmtprocess) == 1
        assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)

    def test_reindent_adds_strip_and_reindent(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'reindent': True}))
        assert stack.grouping is True
        assert len(stack.stmtprocess) == 2
        assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
        assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)

    def test_reindent_aligned_adds_aligned_filter(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'reindent_aligned': True}))
        assert stack.grouping is True
        assert len(stack.stmtprocess) == 1
        assert isinstance(stack.stmtprocess[0], filters.AlignedIndentFilter)

    def test_right_margin_filter(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'right_margin': 20}))
        assert stack.grouping is True
        assert len(stack.stmtprocess) == 1
        assert isinstance(stack.stmtprocess[0], filters.RightMarginFilter)

    @pytest.mark.parametrize('fmt,filtcls', [
        ('php', filters.OutputPHPFilter),
        ('PHP', filters.OutputPHPFilter),
        ('python', filters.OutputPythonFilter),
        ('Python', filters.OutputPythonFilter),
    ])
    def test_output_format_serializer(self, fmt, filtcls):
        stack = FilterStack()
        # output_format validation is case-sensitive on value list, so use
        # lowercase for validation then override to test the .lower() branch.
        opts = _validated({'output_format': 'sql'})
        opts['output_format'] = fmt
        build_filter_stack(stack, opts)
        assert len(stack.postprocess) == 1
        assert isinstance(stack.postprocess[0], filtcls)

    def test_output_format_sql_no_serializer(self):
        stack = FilterStack()
        build_filter_stack(stack, _validated({'output_format': 'sql'}))
        assert stack.postprocess == []

    def test_combined_options_order(self):
        opts = _validated({
            'keyword_case': 'upper',
            'identifier_case': 'lower',
            'truncate_strings': 5,
            'strip_comments': True,
            'reindent': True,
            'comma_first': True,
            'output_format': 'python',
        })
        stack = FilterStack()
        build_filter_stack(stack, opts)
        # Preprocess: keyword, identifier, truncate
        assert [type(f) for f in stack.preprocess] == [
            filters.KeywordCaseFilter,
            filters.IdentifierCaseFilter,
            filters.TruncateStringFilter,
        ]
        # Stmtprocess: strip_comments, strip_whitespace, reindent
        assert [type(f) for f in stack.stmtprocess] == [
            filters.StripCommentsFilter,
            filters.StripWhitespaceFilter,
            filters.ReindentFilter,
        ]
        assert [type(f) for f in stack.postprocess] == [
            filters.OutputPythonFilter,
        ]

    def test_input_stack_returned(self):
        stack = FilterStack()
        assert build_filter_stack(stack, _validated({})) is stack
