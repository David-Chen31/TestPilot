import pickle
import pytest

from boltons.namedutils import namedtuple, namedlist


# --------------------------------------------------------------------------
# namedtuple basics
# --------------------------------------------------------------------------

def test_namedtuple_docstring():
    Point = namedtuple('Point', ['x', 'y'])
    assert Point.__doc__ == 'Point(x, y)'


def test_namedtuple_fields_attribute():
    Point = namedtuple('Point', ['x', 'y'])
    assert Point._fields == ('x', 'y')


def test_namedtuple_is_tuple_subclass():
    Point = namedtuple('Point', ['x', 'y'])
    assert issubclass(Point, tuple)
    p = Point(1, 2)
    assert isinstance(p, tuple)
    assert tuple(p) == (1, 2)


def test_namedtuple_positional_and_keyword():
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(11, y=22)
    assert p[0] == 11
    assert p[1] == 22
    assert p.x == 11
    assert p.y == 22


def test_namedtuple_unpacking():
    Point = namedtuple('Point', ['x', 'y'])
    x, y = Point(11, 22)
    assert (x, y) == (11, 22)


def test_namedtuple_repr():
    Point = namedtuple('Point', ['x', 'y'])
    assert repr(Point(11, 22)) == 'Point(x=11, y=22)'
    assert repr(Point(x=1, y=2)) == 'Point(x=1, y=2)'


def test_namedtuple_repr_single_field():
    One = namedtuple('One', ['a'])
    assert repr(One(5)) == 'One(a=5)'


def test_namedtuple_equality():
    Point = namedtuple('Point', ['x', 'y'])
    assert Point(1, 2) == Point(1, 2)
    assert Point(1, 2) != Point(1, 3)
    # equal to plain tuple too
    assert Point(1, 2) == (1, 2)


def test_namedtuple_make():
    Point = namedtuple('Point', ['x', 'y'])
    p = Point._make([3, 4])
    assert isinstance(p, Point)
    assert p == (3, 4)


def test_namedtuple_make_wrong_length():
    Point = namedtuple('Point', ['x', 'y'])
    with pytest.raises(TypeError):
        Point._make([1, 2, 3])
    with pytest.raises(TypeError):
        Point._make([1])


def test_namedtuple_make_from_generator():
    Point = namedtuple('Point', ['x', 'y'])
    p = Point._make(iter([7, 8]))
    assert p == (7, 8)


def test_namedtuple_asdict():
    Point = namedtuple('Point', ['x', 'y'])
    d = Point(11, 22)._asdict()
    assert d['x'] == 11
    assert d['y'] == 22
    assert list(d.keys()) == ['x', 'y']


def test_namedtuple_asdict_is_ordereddict():
    Point = namedtuple('Point', ['x', 'y'])
    d = Point(1, 2)._asdict()
    # OrderedDict preserves insertion order
    assert list(d.items()) == [('x', 1), ('y', 2)]


def test_namedtuple_dict_property():
    Point = namedtuple('Point', ['x', 'y'])
    d = Point(11, 22).__dict__
    assert d['x'] == 11
    assert d['y'] == 22


def test_namedtuple_construct_from_dict():
    Point = namedtuple('Point', ['x', 'y'])
    d = {'x': 11, 'y': 22}
    assert Point(**d) == (11, 22)


def test_namedtuple_replace():
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(11, 22)
    p2 = p._replace(x=100)
    assert p2 == (100, 22)
    assert isinstance(p2, Point)
    # original unchanged
    assert p == (11, 22)


def test_namedtuple_replace_unknown_field():
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    with pytest.raises(ValueError):
        p._replace(z=5)


def test_namedtuple_getnewargs():
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    assert p.__getnewargs__() == (1, 2)


def test_namedtuple_pickling():
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    restored = pickle.loads(pickle.dumps(p))
    assert restored == p
    assert isinstance(restored, Point)


def test_namedtuple_field_names_from_string():
    Point = namedtuple('Point', 'x y')
    assert Point._fields == ('x', 'y')
    assert Point(1, 2) == (1, 2)


def test_namedtuple_field_names_from_comma_string():
    Point = namedtuple('Point', 'x, y')
    assert Point._fields == ('x', 'y')


def test_namedtuple_field_names_from_tuple():
    Point = namedtuple('Point', ('x', 'y'))
    assert Point._fields == ('x', 'y')


# --------------------------------------------------------------------------
# namedtuple argument validation / errors
# --------------------------------------------------------------------------

def test_namedtuple_empty_field_name_error():
    with pytest.raises(ValueError):
        namedtuple('Point', [''])


def test_namedtuple_invalid_field_char_error():
    with pytest.raises(ValueError):
        namedtuple('Point', ['x!'])


def test_namedtuple_keyword_field_error():
    with pytest.raises(ValueError):
        namedtuple('Point', ['class'])


def test_namedtuple_keyword_typename_error():
    with pytest.raises(ValueError):
        namedtuple('class', ['x'])


def test_namedtuple_field_starts_with_digit_error():
    with pytest.raises(ValueError):
        namedtuple('Point', ['1x'])


def test_namedtuple_typename_starts_with_digit_error():
    with pytest.raises(ValueError):
        namedtuple('1Point', ['x'])


def test_namedtuple_underscore_field_error():
    with pytest.raises(ValueError):
        namedtuple('Point', ['_x'])


def test_namedtuple_duplicate_field_error():
    with pytest.raises(ValueError):
        namedtuple('Point', ['x', 'x'])


def test_namedtuple_rename_option():
    Point = namedtuple('Point', ['x', 'x', 'class', '_y'], rename=True)
    assert Point._fields == ('x', '_1', '_2', '_3')


def test_namedtuple_rename_preserves_valid_names():
    Point = namedtuple('Point', ['x', 'y'], rename=True)
    assert Point._fields == ('x', 'y')


# --------------------------------------------------------------------------
# namedlist basics
# --------------------------------------------------------------------------

def test_namedlist_docstring():
    Point = namedlist('Point', ['x', 'y'])
    assert Point.__doc__ == 'Point(x, y)'


def test_namedlist_fields_attribute():
    Point = namedlist('Point', ['x', 'y'])
    assert Point._fields == ('x', 'y')


def test_namedlist_is_list_subclass():
    Point = namedlist('Point', ['x', 'y'])
    assert issubclass(Point, list)
    p = Point(1, 2)
    assert isinstance(p, list)
    assert p == [1, 2]


def test_namedlist_positional_and_keyword():
    Point = namedlist('Point', ['x', 'y'])
    p = Point(11, y=22)
    assert p[0] == 11
    assert p[1] == 22
    assert p.x == 11
    assert p.y == 22


def test_namedlist_unpacking():
    Point = namedlist('Point', ['x', 'y'])
    x, y = Point(11, 22)
    assert (x, y) == (11, 22)


def test_namedlist_mutable():
    Point = namedlist('Point', ['x', 'y'])
    p = Point(1, 2)
    p.x = 100
    assert p == [100, 2]
    assert p[0] == 100
    p[1] = 200
    assert p.y == 200
    assert p == [100, 200]


def test_namedlist_repr():
    Point = namedlist('Point', ['x', 'y'])
    assert repr(Point(11, 22)) == 'Point(x=11, y=22)'


def test_namedlist_repr_single():
    One = namedlist('One', ['a'])
    assert repr(One(5)) == 'One(a=5)'


def test_namedlist_equality():
    Point = namedlist('Point', ['x', 'y'])
    assert Point(1, 2) == Point(1, 2)
    assert Point(1, 2) != Point(1, 3)
    assert Point(1, 2) == [1, 2]


def test_namedlist_make():
    Point = namedlist('Point', ['x', 'y'])
    p = Point._make([3, 4])
    assert isinstance(p, Point)
    assert p == [3, 4]


def test_namedlist_make_wrong_length():
    Point = namedlist('Point', ['x', 'y'])
    with pytest.raises(TypeError):
        Point._make([1, 2, 3])
    with pytest.raises(TypeError):
        Point._make([1])


def test_namedlist_asdict():
    Point = namedlist('Point', ['x', 'y'])
    d = Point(11, 22)._asdict()
    assert d['x'] == 11
    assert d['y'] == 22
    assert list(d.items()) == [('x', 11), ('y', 22)]


def test_namedlist_dict_property():
    Point = namedlist('Point', ['x', 'y'])
    d = Point(11, 22).__dict__
    assert d['x'] == 11
    assert d['y'] == 22


def test_namedlist_construct_from_dict():
    Point = namedlist('Point', ['x', 'y'])
    d = {'x': 11, 'y': 22}
    assert Point(**d) == [11, 22]


def test_namedlist_replace():
    Point = namedlist('Point', ['x', 'y'])
    p = Point(11, 22)
    p2 = p._replace(x=100)
    assert p2 == [100, 22]
    assert isinstance(p2, Point)
    # original unchanged
    assert p == [11, 22]


def test_namedlist_replace_unknown_field():
    Point = namedlist('Point', ['x', 'y'])
    p = Point(1, 2)
    with pytest.raises(ValueError):
        p._replace(z=5)


def test_namedlist_getnewargs():
    Point = namedlist('Point', ['x', 'y'])
    p = Point(1, 2)
    assert p.__getnewargs__() == (1, 2)


def test_namedlist_field_names_from_string():
    Point = namedlist('Point', 'x y')
    assert Point._fields == ('x', 'y')
    assert Point(1, 2) == [1, 2]


def test_namedlist_field_names_from_comma_string():
    Point = namedlist('Point', 'x, y')
    assert Point._fields == ('x', 'y')


# --------------------------------------------------------------------------
# namedlist argument validation / errors
# --------------------------------------------------------------------------

def test_namedlist_empty_field_name_error():
    with pytest.raises(ValueError):
        namedlist('Point', [''])


def test_namedlist_invalid_field_char_error():
    with pytest.raises(ValueError):
        namedlist('Point', ['x!'])


def test_namedlist_keyword_field_error():
    with pytest.raises(ValueError):
        namedlist('Point', ['class'])


def test_namedlist_keyword_typename_error():
    with pytest.raises(ValueError):
        namedlist('class', ['x'])


def test_namedlist_field_starts_with_digit_error():
    with pytest.raises(ValueError):
        namedlist('Point', ['1x'])


def test_namedlist_typename_starts_with_digit_error():
    with pytest.raises(ValueError):
        namedlist('1Point', ['x'])


def test_namedlist_underscore_field_error():
    with pytest.raises(ValueError):
        namedlist('Point', ['_x'])


def test_namedlist_duplicate_field_error():
    with pytest.raises(ValueError):
        namedlist('Point', ['x', 'x'])


def test_namedlist_rename_option():
    Point = namedlist('Point', ['x', 'x', 'class', '_y'], rename=True)
    assert Point._fields == ('x', '_1', '_2', '_3')


# --------------------------------------------------------------------------
# module attribute
# --------------------------------------------------------------------------

def test_generated_class_module_attribute():
    import boltons.namedutils as namedutils
    Point = namedutils.namedtuple('Point', ['x', 'y'])
    assert Point.__module__ == __name__


def test_verbose_prints_class_definition(capsys):
    namedtuple('VerbosePoint', ['x', 'y'], verbose=True)
    out = capsys.readouterr().out
    assert 'class VerbosePoint(tuple):' in out
    assert 'VerbosePoint(x, y)' in out


def test_namedlist_verbose_prints_class_definition(capsys):
    namedlist('VerboseList', ['x', 'y'], verbose=True)
    out = capsys.readouterr().out
    assert 'class VerboseList(list):' in out
    assert 'VerboseList(x, y)' in out
