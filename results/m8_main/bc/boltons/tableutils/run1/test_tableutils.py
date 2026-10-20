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


# ---------------- to_text ----------------

def test_to_text_basic():
    assert to_text(123) == '123'
    assert to_text('abc') == 'abc'
    assert to_text(None) == 'None'
    assert to_text([1, 2]) == '[1, 2]'


def test_to_text_maxlen():
    assert to_text('abcdefgh', maxlen=6) == 'abc...'
    assert to_text('abc', maxlen=10) == 'abc'
    # exactly maxlen - not truncated
    assert to_text('abcdef', maxlen=6) == 'abcdef'


def test_to_text_maxlen_truncates_at_limit():
    # length must be <= maxlen
    out = to_text('x' * 100, maxlen=10)
    assert out == 'xxxxxxx...'
    assert len(out) == 10


def test_to_text_str_raises():
    class BadStr:
        def __str__(self):
            raise RuntimeError('nope')

        def __repr__(self):
            return 'BadRepr'

    assert to_text(BadStr()) == 'BadRepr'


def test_to_text_str_and_repr_raise():
    class BadBoth:
        def __str__(self):
            raise RuntimeError('nope')

        def __repr__(self):
            raise RuntimeError('norepr')

    out = to_text(BadBoth())
    # falls back to object.__repr__
    assert 'BadBoth object at' in out


# ---------------- escape_html ----------------

def test_escape_html():
    assert escape_html('<b>') == '&lt;b&gt;'
    assert escape_html('"&') == '&quot;&amp;'
    assert escape_html("'") == '&#x27;'


def test_escape_html_maxlen():
    assert escape_html('a' * 20, maxlen=6) == 'aaa...'


# ---------------- UnsupportedData ----------------

def test_unsupported_data_is_typeerror():
    assert issubclass(UnsupportedData, TypeError)
    with pytest.raises(UnsupportedData):
        raise UnsupportedData('x')


# ---------------- InputType base ----------------

def test_input_type_noop_init_and_get_entry_seq():
    it = InputType(1, 2, key='value')
    # get_entry is not defined on base InputType, so get_entry_seq raises
    with pytest.raises(AttributeError):
        it.get_entry_seq([1, 2, 3], ['h'])
    with pytest.raises(AttributeError):
        it.get_entry(1, ['h'])


# ---------------- DictInputType ----------------

def test_dict_input_type_check_type():
    dt = DictInputType()
    assert dt.check_type({})
    assert dt.check_type({'a': 1})
    assert not dt.check_type([1, 2])
    assert not dt.check_type('abc')


def test_dict_input_type_guess_headers_sorted():
    dt = DictInputType()
    assert dt.guess_headers({'b': 1, 'a': 2}) == ['a', 'b']


def test_dict_input_type_get_entry():
    dt = DictInputType()
    assert dt.get_entry({'a': 1, 'b': 2}, ['b', 'a']) == [2, 1]
    assert dt.get_entry({'a': 1}, ['a', 'missing']) == [1, None]


def test_dict_input_type_get_entry_seq():
    dt = DictInputType()
    seq = [{'a': 1, 'b': 2}, {'b': 3}]
    assert dt.get_entry_seq(seq, ['a', 'b']) == [[1, 2], [None, 3]]


# ---------------- ObjectInputType ----------------

def test_object_input_type_check_type():
    ot = ObjectInputType()

    class Foo:
        pass

    assert ot.check_type(Foo())
    assert not ot.check_type(1)
    assert not ot.check_type('s')
    assert not ot.check_type(None)
    # Note: lists are NOT in _DNR and have __class__, so check_type returns True
    assert ot.check_type([1])
    # dicts are also NOT in _DNR, so check_type returns True
    assert ot.check_type({})
    assert not ot.check_type(lambda x: x)


def test_object_input_type_guess_headers_and_get_entry():
    class Foo:
        def __init__(self):
            self.a = 1
            self.b = 'x'

        def method(self):
            pass

    ot = ObjectInputType()
    f = Foo()
    headers = ot.guess_headers(f)
    # only non-callable attrs, no methods
    assert 'a' in headers
    assert 'b' in headers
    assert 'method' not in headers
    values = ot.get_entry(f, ['a', 'b'])
    assert values == [1, 'x']


def test_object_input_type_get_entry_missing_attr_is_none():
    class Foo:
        def __init__(self):
            self.a = 1

    ot = ObjectInputType()
    assert ot.get_entry(Foo(), ['a', 'not_there']) == [1, None]


def test_object_input_type_guess_headers_skips_raising_property():
    class Foo:
        a = 1

        @property
        def broken(self):
            raise ValueError('broken')

    ot = ObjectInputType()
    headers = ot.guess_headers(Foo())
    assert 'a' in headers
    assert 'broken' not in headers


# ---------------- ListInputType ----------------

def test_list_input_type():
    lt = ListInputType()
    assert lt.check_type([1, 2])
    assert not lt.check_type((1, 2))
    assert not lt.check_type('abc')
    assert lt.guess_headers([1, 2]) is None
    assert lt.get_entry([1, 2], None) == [1, 2]
    seq = [[1, 2], [3, 4]]
    # get_entry_seq returns the same object (no copy)
    assert lt.get_entry_seq(seq, None) is seq


# ---------------- TupleInputType ----------------

def test_tuple_input_type():
    tt = TupleInputType()
    assert tt.check_type((1, 2))
    assert not tt.check_type([1, 2])
    assert tt.guess_headers((1, 2)) is None
    assert tt.get_entry((1, 2), None) == [1, 2]
    assert tt.get_entry_seq([(1, 2), (3, 4)], None) == [[1, 2], [3, 4]]


# ---------------- NamedTupleInputType ----------------

def test_named_tuple_input_type():
    from collections import namedtuple

    Point = namedtuple('Point', ['x', 'y'])
    nt = NamedTupleInputType()
    p = Point(1, 2)
    assert nt.check_type(p)
    assert not nt.check_type((1, 2))  # plain tuple not a namedtuple
    assert nt.guess_headers(p) == ['x', 'y']
    assert nt.get_entry(p, ['x', 'y']) == [1, 2]
    assert nt.get_entry(p, ['x', 'missing']) == [1, None]
    assert nt.get_entry_seq([p, Point(3, 4)], ['x', 'y']) == [[1, 2], [3, 4]]


# ---------------- Table basics ----------------

def test_table_empty():
    t = Table()
    assert len(t) == 0
    assert t.headers == []
    assert t.metadata == {}
    assert t._width == 0


def test_table_from_list_of_lists_infers_headers():
    data = [['a', 'b'], [1, 2], [3, 4]]
    t = Table(data)
    assert t.headers == ['a', 'b']
    assert len(t) == 2
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]
    assert t._width == 2


def test_table_headers_explicit():
    t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
    assert t.headers == ['a', 'b']
    assert len(t) == 2
    assert t[0] == [1, 2]


def test_table_fills_short_rows():
    t = Table([[1], [2, 3]], headers=['a', 'b'])
    assert t[0] == [1, None]
    assert t[1] == [2, 3]


def test_table_metadata():
    t = Table([[1]], headers=['a'], metadata={'k': 'v'})
    assert t.metadata == {'k': 'v'}


def test_table_extend():
    t = Table([[1]], headers=['a', 'b'])
    t.extend([[2, 3], [4]])
    assert t[1] == [2, 3]
    assert t[2] == [4, None]


def test_table_extend_empty_is_noop():
    t = Table([[1]], headers=['a'])
    t.extend([])
    assert len(t) == 1


def test_table_len_getitem():
    t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
    assert len(t) == 2
    assert t[0] == [1, 2]
    assert t[-1] == [3, 4]


def test_table_repr_with_headers():
    t = Table([[1, 2]], headers=['a', 'b'])
    r = repr(t)
    assert r.startswith('Table(')
    assert 'headers=' in r
    assert 'data=' in r


def test_table_repr_without_headers():
    # Table([[1, 2]]) treats first row as headers, so use Table() empty
    t = Table()
    r = repr(t)
    assert r.startswith('Table(')
    assert 'headers=' not in r
    assert r == 'Table([])'


# ---------------- Table internals: _set_width(reset) / _fill no-width ------

def test_table_set_width_reset():
    t = Table([[1, 2]], headers=['a', 'b'])
    assert t._width == 2
    t._width = 0
    # reset=True forces recompute from headers
    t._set_width(reset=True)
    assert t._width == 2

    t2 = Table([[1, 2], [3, 4]], headers=[])
    assert t2._width == 2
    t2._width = 0
    t2._set_width(reset=True)
    assert t2._width == 2


def test_table_fill_no_width():
    t = Table()
    assert t._width == 0
    # _fill short-circuits when width is 0, returns None
    assert t._fill() is None
    assert t._data == []


# ---------------- Table.from_dict ----------------

def test_table_from_dict_single():
    t = Table.from_dict({'a': 1, 'b': 2})
    assert t.headers == ['a', 'b']
    assert len(t) == 1
    assert t[0] == [1, 2]


def test_table_from_dict_list():
    t = Table.from_dict([{'a': 1, 'b': 2}, {'b': 3}])
    assert t.headers == ['a', 'b']
    assert len(t) == 2
    assert t[0] == [1, 2]
    assert t[1] == [None, 3]


def test_table_from_dict_explicit_headers():
    t = Table.from_dict([{'a': 1}], headers=['a', 'b'])
    assert t.headers == ['a', 'b']
    assert t[0] == [1, None]


# ---------------- Table.from_list ----------------

def test_table_from_list():
    t = Table.from_list([[1, 2], [3, 4]])
    assert t.headers == []
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


# ---------------- Table.from_object ----------------

def test_table_from_object_single():
    class Foo:
        def __init__(self):
            self.a = 1
            self.b = 2

    t = Table.from_object(Foo())
    assert len(t) == 1
    # guess_headers uses dir() so includes dunder attrs; check a & b present
    assert 'a' in t.headers
    assert 'b' in t.headers
    row = t[0]
    idx_a = t.headers.index('a')
    idx_b = t.headers.index('b')
    assert row[idx_a] == 1
    assert row[idx_b] == 2


# ---------------- Table.from_data ----------------

def test_table_from_data_scalar():
    t = Table.from_data(42)
    # headers becomes [42], data empty
    assert t.headers == [42]
    assert len(t) == 0


def test_table_from_data_empty():
    t = Table.from_data([])
    assert len(t) == 0
    assert t.headers == []


def test_table_from_data_max_depth_zero():
    t = Table.from_data([[1, 2]], max_depth=0, headers=['a'])
    assert len(t) == 0
    assert t.headers == ['a']


def test_table_from_data_namedtuple():
    from collections import namedtuple

    Point = namedtuple('Point', ['x', 'y'])
    t = Table.from_data([Point(1, 2), Point(3, 4)])
    assert t.headers == ['x', 'y']
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_table_from_data_tuple():
    t = Table.from_data((1, 2))
    # tuple gets converted to lists
    assert t[0] == [1, 2]


def test_table_from_data_unsupported_raises():
    # Because ObjectInputType is permissive, we must force an unsupported path.
    # Passing an explicit _data_type=None and data that no input type accepts:
    # Force the failure by creating a data whose first element fails all checks
    # AND is a Sequence so to_check gets set. But ObjectInputType is permissive.
    # Instead use _data_type explicitly? That bypasses the check loop.
    # Use a class that raises in check_type via ObjectInputType? Not feasible.
    # Directly test the UnsupportedData raise by monkeypatching _input_types.
    original = Table._input_types
    try:
        Table._input_types = []
        with pytest.raises(UnsupportedData) as exc:
            Table.from_data([[1, 2]])
        assert 'unsupported data type' in str(exc.value)
    finally:
        Table._input_types = original


def test_table_from_data_set_accepted():
    # A set is not a Sequence. type(set) not in _DNR. Then check_type loop:
    # ObjectInputType.check_type returns True (type not in _DNR, has __class__)
    t = Table.from_data({1, 2, 3})
    assert isinstance(t, Table)


# ---------------- Table.to_text ----------------

def test_to_text_basic_table():
    t = Table([['x', 'y'], ['longer', 'z']], headers=['a', 'b'])
    out = t.to_text()
    lines = out.split('\n')
    # header line, separator, plus 2 data rows
    assert len(lines) == 4
    assert 'a' in lines[0]
    assert 'b' in lines[0]
    assert set(lines[1]) <= {'-', '|', ' '}


def test_to_text_without_headers():
    t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
    out = t.to_text(with_headers=False)
    lines = out.split('\n')
    assert len(lines) == 2
    assert 'a' not in out


def test_to_text_maxlen():
    t = Table([['a' * 30]], headers=['h'])
    out = t.to_text(maxlen=10)
    assert '...' in out


def test_to_text_headers_padded_when_short():
    # When headers are shorter than the row width, the implementation does
    # not pad the header list (the _width is set from headers, then _fill
    # is not applied). Instead iterating data rows whose length exceeds
    # _width raises IndexError inside to_text. Assert the actual behavior.
    t = Table([[1, 2, 3]], headers=['a'])
    with pytest.raises(IndexError):
        t.to_text()


# ---------------- Table.to_html ----------------

def test_to_html_horizontal():
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(orientation='horizontal')
    assert '<table>' in html
    assert '</table>' in html
    assert '<thead>' in html
    assert '<tbody>' in html
    assert '<th>' in html
    assert '<td>' in html


def test_to_html_vertical():
    t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
    html = t.to_html(orientation='vertical')
    assert '<tr>' in html
    assert '<th>' in html
    assert '<td>' in html


def test_to_html_auto_single_row_is_vertical():
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(orientation='auto')
    # single row => vertical orientation
    # vertical emits a <tr> per column, so more than one <tr> in tbody
    assert html.count('<tr>') >= 2


def test_to_html_wrapped_false():
    t = Table([[1]], headers=['a'])
    html = t.to_html(wrapped=False)
    assert '<table>' not in html
    assert '</table>' not in html


def test_to_html_with_newlines():
    t = Table([[1]], headers=['a'])
    html = t.to_html(with_newlines=True)
    assert '\n' in html


def test_to_html_without_headers():
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(with_headers=False)
    assert '<th>' not in html


def test_to_html_escapes():
    t = Table([['<script>']], headers=['a'])
    html = t.to_html()
    assert '&lt;script&gt;' in html
    assert '<script>' not in html


def test_to_html_invalid_orientation():
    t = Table([[1]], headers=['a'])
    with pytest.raises(ValueError) as exc:
        t.to_html(orientation='zzz')
    assert "expected one of 'auto', 'vertical', or 'horizontal'" in str(exc.value)
    assert 'zzz' in str(exc.value)


def test_to_html_with_metadata_top():
    t = Table([[1]], headers=['a'], metadata={'k': 'v'})
    html = t.to_html(with_metadata=True)
    assert '<br />' in html
    assert html.count('<table>') >= 2


def test_to_html_with_metadata_bottom():
    t = Table([[1]], headers=['a'], metadata={'k': 'v'})
    html = t.to_html(with_metadata='bottom')
    assert '<br />' in html
    assert html.count('<table>') >= 2


def test_to_html_nested_table():
    inner = Table([[1]], headers=['a'])
    outer = Table([[inner]], headers=['x'])
    html = outer.to_html(max_depth=2)
    assert '<table>' in html
    assert html.count('<table>') >= 2


def test_get_cell_html_default_escapes():
    t = Table([[1]], headers=['a'])
    assert t.get_cell_html('<b>') == '&lt;b&gt;'
    assert t.get_cell_html('"') == '&quot;'
    assert t.get_cell_html(None) == 'None'


def test_to_html_horizontal_no_headers():
    # exercises the branch where 'headers' is empty in horizontal layout
    t = Table([[1, 2]], headers=['a', 'b'])
    html = t.to_html(orientation='horizontal', with_headers=False)
    assert '<thead>' not in html
    # width smaller than headers => headers list empty; still renders td
    assert '<td>1</td>' in html
    assert '<td>2</td>' in html


def test_to_html_horizontal_headers_padded():
    # headers fewer than width: the implementation's headers padding relies
    # on self._width matching len(self.headers). With a short header list,
    # _width == len(headers) == 1, so only one <th> is emitted even though
    # the row has 3 cells.
    t = Table([[1, 2, 3]], headers=['a'])
    html = t.to_html(orientation='horizontal')
    assert '<th>a</th>' in html
    assert html.count('<th>') == 1


def test_to_html_horizontal_nested_depth():
    inner = Table([[1]], headers=['a'])
    outer = Table([[inner, 'plain']], headers=['x', 'y'])
    html = outer.to_html(orientation='horizontal', max_depth=2)
    assert html.count('<table>') >= 2
    assert 'plain' in html


def test_to_html_vertical_nested_depth():
    inner = Table([[1]], headers=['a'])
    outer = Table([[inner]], headers=['x'])
    html = outer.to_html(orientation='vertical', max_depth=2)
    assert html.count('<table>') >= 2


def test_to_html_with_metadata_bottom_no_newlines():
    t = Table([[1]], headers=['a'], metadata={'k': 'v'})
    html = t.to_html(with_metadata='bottom', with_newlines=False)
    assert '\n' not in html
    assert html.count('<table>') >= 2


# ---------------- max_depth nesting via from_data ----------------

def test_from_data_nested_max_depth():
    inner = {'x': 1}
    data = [[inner]]
    t = Table.from_data(data, headers=['c'], max_depth=2)
    cell = t[0][0]
    assert isinstance(cell, Table)
    assert cell.headers == ['x']
    assert cell[0] == [1]


def test_from_data_nested_max_depth_one():
    # max_depth=1 means no nesting
    inner = {'x': 1}
    t = Table.from_data([[inner]], headers=['c'], max_depth=1)
    cell = t[0][0]
    assert not isinstance(cell, Table)
    assert cell == inner


def test_from_data_nested_depth_dnr_skipped():
    # int cells should be skipped (continue branch).
    data = [[1, {'x': 2}]]
    t = Table.from_data(data, headers=['a', 'b'], max_depth=2)
    assert t[0][0] == 1
    assert isinstance(t[0][1], Table)
    assert t[0][1].headers == ['x']


def test_from_data_nested_depth_unsupported_swallowed():
    # Objects that fail from_data at deeper levels should leave cells intact.
    # We monkeypatch _input_types to cause an UnsupportedData on the inner
    # call, which gets swallowed and the original cell retained.
    class Weird:
        pass

    obj = Weird()
    data = [[obj]]
    original = Table._input_types
    calls = {'n': 0}

    class RejectAll:
        def check_type(self, o):
            return False

    # We need outer to succeed, inner to fail. Use a stateful _input_types
    # that accepts first invocation then rejects.
    class OnceAccept:
        def check_type(self, o):
            calls['n'] += 1
            return calls['n'] <= 1

        def guess_headers(self, obj):
            return ['c']

        def get_entry(self, obj, headers):
            return [obj]

        def get_entry_seq(self, obj_seq, headers):
            return [[c] for c in obj_seq]

    try:
        Table._input_types = [OnceAccept()]
        t = Table.from_data([obj], max_depth=2)
        # inner from_data hits no accepting type -> UnsupportedData swallowed
        assert t[0][0] is obj
    finally:
        Table._input_types = original
