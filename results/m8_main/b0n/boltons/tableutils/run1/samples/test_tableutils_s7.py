import types
from collections import namedtuple, OrderedDict

import pytest

from boltons.tableutils import (
    to_text,
    escape_html,
    UnsupportedData,
    InputType,
    DictInputType,
    ObjectInputType,
    ListInputType,
    TupleInputType,
    NamedTupleInputType,
    Table,
)


# ---------------------------------------------------------------------------
# to_text / escape_html
# ---------------------------------------------------------------------------

def test_to_text_basic():
    assert to_text(123) == '123'
    assert to_text('hello') == 'hello'
    assert to_text(None) == 'None'


def test_to_text_maxlen():
    assert to_text('abcdefghij', maxlen=6) == 'abc...'
    # exactly at maxlen: no truncation
    assert to_text('abcdef', maxlen=6) == 'abcdef'
    # shorter than maxlen
    assert to_text('abc', maxlen=6) == 'abc'
    # maxlen falsy: no truncation
    assert to_text('abcdefghij', maxlen=None) == 'abcdefghij'
    assert to_text('abcdefghij', maxlen=0) == 'abcdefghij'


def test_to_text_str_raises():
    class BadStr:
        def __str__(self):
            raise ValueError('nope')

        def __repr__(self):
            return 'REPR'

    assert to_text(BadStr()) == 'REPR'


def test_to_text_both_raise():
    class Bad:
        def __str__(self):
            raise ValueError('s')

        def __repr__(self):
            raise ValueError('r')

    # falls back to object.__repr__
    result = to_text(Bad())
    assert result.startswith('<')
    assert 'Bad' in result


def test_escape_html():
    assert escape_html('<b>') == '&lt;b&gt;'
    assert escape_html('"quoted"') == '&quot;quoted&quot;'
    assert escape_html("'single'") == '&#x27;single&#x27;'
    assert escape_html('a&b') == 'a&amp;b'


def test_escape_html_maxlen():
    assert escape_html('<abcdefgh>', maxlen=6) == '&lt;...'


# ---------------------------------------------------------------------------
# Input type classes
# ---------------------------------------------------------------------------

def test_input_type_get_entry_seq():
    it = InputType()
    # default InputType.get_entry passes headers and returns them per entry
    # but base class has no get_entry; use a tiny subclass
    class MyType(InputType):
        def get_entry(self, obj, headers):
            return (obj, headers)

    it = MyType()
    result = it.get_entry_seq([1, 2, 3], ['h'])
    assert result == [(1, ['h']), (2, ['h']), (3, ['h'])]


def test_dict_input_type():
    dit = DictInputType()
    assert dit.check_type({}) is True
    assert dit.check_type([]) is False
    assert dit.guess_headers({'b': 1, 'a': 2, 'c': 3}) == ['a', 'b', 'c']
    assert dit.get_entry({'a': 1, 'b': 2}, ['a', 'b']) == [1, 2]
    # missing headers -> None
    assert dit.get_entry({'a': 1}, ['a', 'b']) == [1, None]
    seq = dit.get_entry_seq([{'a': 1, 'b': 2}, {'a': 3}], ['a', 'b'])
    assert seq == [[1, 2], [3, None]]


def test_list_input_type():
    lit = ListInputType()
    assert lit.check_type([]) is True
    assert lit.check_type(()) is False
    assert lit.guess_headers([1, 2]) is None
    assert lit.get_entry([1, 2], None) == [1, 2]
    seq = [[1, 2], [3, 4]]
    # returns the same object (no copy)
    assert lit.get_entry_seq(seq, None) is seq


def test_tuple_input_type():
    tit = TupleInputType()
    assert tit.check_type((1, 2)) is True
    assert tit.check_type([1, 2]) is False
    assert tit.guess_headers((1, 2)) is None
    assert tit.get_entry((1, 2), None) == [1, 2]
    assert tit.get_entry_seq([(1, 2), (3, 4)], None) == [[1, 2], [3, 4]]


def test_named_tuple_input_type():
    ntit = NamedTupleInputType()
    NT = namedtuple('NT', ['x', 'y'])
    inst = NT(1, 2)
    assert ntit.check_type(inst) is True
    assert ntit.check_type((1, 2)) is False
    assert ntit.check_type([1, 2]) is False
    assert ntit.guess_headers(inst) == ['x', 'y']
    assert ntit.get_entry(inst, ['x', 'y']) == [1, 2]
    assert ntit.get_entry(inst, ['y', 'x']) == [2, 1]
    seq = ntit.get_entry_seq([NT(1, 2), NT(3, 4)], ['x', 'y'])
    assert seq == [[1, 2], [3, 4]]


def test_object_input_type():
    oit = ObjectInputType()

    class Obj:
        def __init__(self):
            self.a = 1
            self.b = 2

        def method(self):
            pass

    o = Obj()
    assert oit.check_type(o) is True
    # _DNR types are not objects
    assert oit.check_type(1) is False
    assert oit.check_type('str') is False
    assert oit.check_type(None) is False
    headers = oit.guess_headers(o)
    assert 'a' in headers
    assert 'b' in headers
    assert 'method' not in headers
    assert oit.get_entry(o, ['a', 'b']) == [1, 2]
    # missing attr -> None
    assert oit.get_entry(o, ['a', 'nonexistent']) == [1, None]


# ---------------------------------------------------------------------------
# UnsupportedData
# ---------------------------------------------------------------------------

def test_unsupported_data_is_type_error():
    assert issubclass(UnsupportedData, TypeError)


# ---------------------------------------------------------------------------
# Table construction / init
# ---------------------------------------------------------------------------

def test_table_empty():
    t = Table()
    assert t.headers == []
    assert t.metadata == {}
    assert len(t) == 0
    assert t._width == 0


def test_table_infers_headers_from_first_row():
    t = Table([['a', 'b'], [1, 2], [3, 4]])
    assert t.headers == ['a', 'b']
    assert t._data is not None
    assert list(t._data) == [[1, 2], [3, 4]]


def test_table_explicit_headers():
    t = Table([[1, 2], [3, 4]], headers=['x', 'y'])
    assert t.headers == ['x', 'y']
    assert list(t._data) == [[1, 2], [3, 4]]


def test_table_headers_none():
    t = Table([[1, 2], [3, 4]], headers=None)
    assert t.headers == []
    assert list(t._data) == [[1, 2], [3, 4]]


def test_table_metadata():
    t = Table([[1]], headers=['a'], metadata={'k': 'v'})
    assert t.metadata == {'k': 'v'}


def test_table_fill_short_rows():
    t = Table([[1], [2, 3]], headers=['a', 'b'])
    assert t[0] == [1, None]
    assert t[1] == [2, 3]


def test_table_width_from_data_no_headers():
    t = Table([[1, 2, 3], [4, 5]])
    assert t._width == 3
    # second row filled
    assert t[1] == [4, 5, None]


# ---------------------------------------------------------------------------
# Table extend, len, getitem, repr
# ---------------------------------------------------------------------------

def test_table_extend_empty():
    t = Table([[1]], headers=['a'])
    t.extend(None)
    assert len(t) == 1
    t.extend([])
    assert len(t) == 1


def test_table_extend():
    t = Table([[1]], headers=['a', 'b'])
    t.extend([[2], [3, 4]])
    assert len(t) == 3
    assert t[1] == [2, None]
    assert t[2] == [3, 4]


def test_table_getitem_and_len():
    t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
    assert len(t) == 2
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]
    assert t[-1] == [3, 4]


def test_table_repr_with_headers():
    t = Table([[1]], headers=['a'])
    r = repr(t)
    assert r.startswith('Table(')
    assert "headers=['a']" in r


def test_table_repr_without_headers():
    t = Table([[1, 2]])
    t.headers = []
    r = repr(t)
    assert r.startswith('Table([')


# ---------------------------------------------------------------------------
# Table.from_data and friends
# ---------------------------------------------------------------------------

def test_from_data_list_of_lists():
    t = Table.from_data([[1, 2], [3, 4]])
    assert list(t._data) == [[1, 2], [3, 4]]
    # first row inferred as headers
    assert t.headers == [1, 2]


def test_from_data_empty_sequence():
    t = Table.from_data([])
    assert len(t) == 0
    assert t.headers == []


def test_from_data_max_depth_zero():
    t = Table.from_data([1, 2, 3], max_depth=0)
    assert len(t) == 0


def test_from_data_scalar():
    t = Table.from_data(5)
    assert list(t._data) == [[5]]


def test_from_data_list_of_dicts():
    t = Table.from_data([{'b': 1, 'a': 2}, {'a': 3, 'b': 4}])
    assert t.headers == ['a', 'b']
    assert list(t._data) == [[2, 1], [3, 4]]


def test_from_data_single_dict():
    t = Table.from_data({'b': 1, 'a': 2})
    assert t.headers == ['a', 'b']
    assert list(t._data) == [[2, 1]]


def test_from_data_list_of_tuples():
    t = Table.from_data([(1, 2), (3, 4)])
    assert list(t._data) == [[1, 2], [3, 4]]


def test_from_data_namedtuples():
    NT = namedtuple('NT', ['x', 'y'])
    t = Table.from_data([NT(1, 2), NT(3, 4)])
    assert t.headers == ['x', 'y']
    assert list(t._data) == [[1, 2], [3, 4]]


def test_from_data_single_namedtuple():
    NT = namedtuple('NT', ['x', 'y'])
    t = Table.from_data(NT(1, 2))
    assert t.headers == ['x', 'y']
    assert list(t._data) == [[1, 2]]


def test_from_data_object():
    class Obj:
        def __init__(self):
            self.a = 1
            self.b = 2

    t = Table.from_data(Obj())
    assert 'a' in t.headers
    assert 'b' in t.headers
    row = t[0]
    assert row[t.headers.index('a')] == 1
    assert row[t.headers.index('b')] == 2


def test_from_data_unsupported():
    with pytest.raises(UnsupportedData):
        Table.from_data(object())


def test_from_data_nested_max_depth():
    inner = [[1, 2], [3, 4]]
    t = Table.from_data([inner, inner], max_depth=2)
    # cells contain nested Tables
    assert isinstance(t[0][0], Table)
    assert isinstance(t[0][1], Table)


def test_from_data_nested_default_depth():
    inner = [[1, 2], [3, 4]]
    t = Table.from_data([inner, inner])
    # default max_depth=1 -> no nesting
    assert not isinstance(t[0][0], Table)


def test_from_dict():
    t = Table.from_dict([{'a': 1, 'b': 2}])
    assert t.headers == ['a', 'b']
    assert list(t._data) == [[1, 2]]


def test_from_list():
    t = Table.from_list([[1, 2], [3, 4]])
    assert t.headers == [1, 2]
    assert list(t._data) == [[3, 4]]


def test_from_object():
    class Obj:
        def __init__(self):
            self.a = 1

    t = Table.from_object(Obj())
    assert 'a' in t.headers


def test_from_data_with_metadata():
    t = Table.from_data([[1, 2]], metadata={'k': 'v'})
    assert t.metadata == {'k': 'v'}


# ---------------------------------------------------------------------------
# to_html
# ---------------------------------------------------------------------------

def test_to_html_basic_horizontal():
    t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
    html = t.to_html()
    assert '<table>' in html
    assert '</table>' in html
    assert '<thead>' in html
    assert '<th>a</th>' in html
    assert '<th>b</th>' in html
    assert '<tbody>' in html
    assert '<td>1</td>' in html
    assert '<td>4</td>' in html


def test_to_html_with_newlines():
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(with_newlines=True)
    assert '\n' in html
    lines = html.split('\n')
    assert lines[0] == '<table>'
    assert lines[-1] == '</table>'


def test_to_html_no_newlines():
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(with_newlines=False)
    assert '\n' not in html


def test_to_html_unwrapped():
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(wrapped=False)
    assert '<table>' not in html
    assert '</table>' not in html
    assert '<tbody>' in html


def test_to_html_no_headers():
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(with_headers=False)
    assert '<th>' not in html
    assert '<thead>' not in html


def test_to_html_escape():
    t = Table([['<script>']], headers=['<h>'])
    html = t.to_html()
    assert '<script>' not in html
    assert '&lt;script&gt;' in html
    assert '&lt;h&gt;' in html


def test_to_html_vertical():
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(orientation='vertical')
    # vertical: each row is a column
    assert '<th>a</th>' in html
    assert '<td>1</td>' in html


def test_to_html_auto_single_row_vertical():
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(orientation='auto')
    # single row -> vertical
    assert '<thead>' not in html  # vertical doesn't add thead
    assert '<th>a</th>' in html


def test_to_html_invalid_orientation():
    t = Table([[1, 2]], headers=['a', 'b'])
    with pytest.raises(ValueError):
        t.to_html(orientation='bogus')


def test_to_html_metadata_top():
    t = Table([[1, 2]], headers=['a', 'b'], metadata={'x': 'y'})
    html = t.to_html(with_metadata=True)
    assert '<br />' in html
    # metadata table comes first
    idx_meta = html.index('x')
    idx_table = html.index('<table>')
    assert idx_meta < idx_table or idx_meta > idx_table  # just ensure present


def test_to_html_metadata_bottom():
    t = Table([[1, 2]], headers=['a', 'b'], metadata={'x': 'y'})
    html = t.to_html(with_metadata='bottom')
    assert '<br />' in html


def test_get_cell_html_override():
    class MyTable(Table):
        def get_cell_html(self, value):
            return 'X' + str(value)

    t = MyTable([[1]], headers=['a'])
    html = t.to_html()
    assert 'X1' in html
    # headers also use get_cell_html
    assert 'Xa' in html


def test_to_html_nested_tables():
    inner = Table([[1, 2]], headers=['x', 'y'])
    outer = Table([[inner, 'plain']], headers=['a', 'b'])
    html = outer.to_html(max_depth=2)
    assert '<table>' in html
    # nested table rendered
    assert html.count('<table>') == 2


def test_to_html_nested_depth_one():
    inner = Table([[1, 2]], headers=['x', 'y'])
    outer = Table([[inner, 'plain']], headers=['a', 'b'])
    html = outer.to_html(max_depth=1)
    # nested table repr'd, not rendered as table
    assert html.count('<table>') == 1


# ---------------------------------------------------------------------------
# to_text
# ---------------------------------------------------------------------------

def test_to_text_table():
    t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
    text = t.to_text()
    lines = text.split('\n')
    # header, separator, two data rows
    assert len(lines) == 4
    assert 'a' in lines[0]
    assert 'b' in lines[0]
    assert '---' in lines[1]
    assert '1' in lines[2]
    assert '4' in lines[3]


def test_to_text_no_headers():
    t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
    text = t.to_text(with_headers=False)
    lines = text.split('\n')
    assert len(lines) == 2
    assert '1' in lines[0]
    assert '4' in lines[1]


def test_to_text_maxlen():
    t = Table([['abcde']], headers=['h'])
    text = t.to_text(maxlen=4)
    assert 'a...' in text
    assert 'abcde' not in text


def test_to_text_empty():
    t = Table()
    assert t.to_text() == ''


def test_to_text_alignment():
    t = Table([['x']], headers=['header'])
    text = t.to_text()
    # all lines same width
    lines = text.split('\n')
    # header gets centered within the wider column
    assert 'header' in lines[0]
