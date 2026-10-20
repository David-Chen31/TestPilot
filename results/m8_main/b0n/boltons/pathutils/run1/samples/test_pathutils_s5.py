import os
import os.path

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_no_changes():
    assert augpath('foo.bar') == 'foo.bar'


def test_augpath_replace_ext():
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'


def test_augpath_add_suffix():
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'


def test_augpath_add_prefix():
    assert augpath('foo.bar', prefix='_') == '_foo.bar'


def test_augpath_replace_base():
    assert augpath('foo.bar', base='baz') == 'baz.bar'


def test_augpath_docstring_example():
    newpath = augpath('foo.bar', suffix='_suff', prefix='pref_',
                      ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_multidot_true_ext():
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'


def test_augpath_multidot_false_ext():
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'


def test_augpath_multidot_true_suffix():
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_multidot_false_suffix():
    assert augpath('foo.tar.gz', suffix='_new', multidot=False) == 'foo.tar_new.gz'


def test_augpath_multidot_no_dot():
    # No dot at all -> whole name is base, ext empty
    assert augpath('foobar', suffix='_x', multidot=True) == 'foobar_x'
    assert augpath('foobar', suffix='_x', multidot=False) == 'foobar_x'


def test_augpath_with_directory():
    result = augpath(os.path.join('a', 'b', 'foo.bar'), ext='.baz')
    assert result == os.path.join('a', 'b', 'foo.baz')


def test_augpath_replace_dpath():
    result = augpath(os.path.join('a', 'foo.bar'), dpath='c')
    assert result == os.path.join('c', 'foo.bar')


def test_augpath_replace_dpath_and_ext():
    result = augpath(os.path.join('a', 'foo.bar'), dpath='c', ext='.baz')
    assert result == os.path.join('c', 'foo.baz')


def test_augpath_no_extension_multidot_false():
    # splitext on 'foo' gives ('foo', '')
    assert augpath('foo', ext='.txt') == 'foo.txt'


def test_augpath_multidot_true_with_leading_dot():
    # '.foo' -> split('.', 1) gives ['', 'foo'] -> base '', ext '.foo'
    assert augpath('.foo', suffix='_x', multidot=True) == '_x.foo'
    # splitext('.foo') yields ('.foo', '')
    assert augpath('.foo', suffix='_x', multidot=False) == '.foo_x'


def test_augpath_empty_ext_replaces_extension():
    assert augpath('foo.bar', ext='') == 'foo'


def test_augpath_empty_base():
    assert augpath('foo.bar', base='') == '.bar'


def test_augpath_all_components():
    result = augpath(os.path.join('d', 'foo.tar.gz'),
                     prefix='p_', suffix='_s', base='base',
                     ext='.e', dpath='x')
    assert result == os.path.join('x', 'p_base_s.e')


def test_augpath_trailing_sep():
    # 'foo.bar/' -> split gives ('foo.bar', '')
    result = augpath('foo.bar' + os.path.sep, ext='.baz')
    assert result == os.path.join('foo.bar', '.baz')


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_home_is_tilde():
    home = os.path.expanduser('~')
    assert shrinkuser(home) == '~'


def test_shrinkuser_custom_home_symbol():
    home = os.path.expanduser('~')
    assert shrinkuser(home, '$HOME') == '$HOME'


def test_shrinkuser_home_subpath():
    home = os.path.expanduser('~')
    assert shrinkuser(os.path.join(home, '1')) == os.path.join('~', '1')


def test_shrinkuser_home_subpath_custom_symbol():
    home = os.path.expanduser('~')
    assert shrinkuser(os.path.join(home, '1'), '$HOME') == os.path.join('$HOME', '1')


def test_shrinkuser_prefix_not_boundary():
    # Path that starts with home but next char is not a separator
    home = os.path.expanduser('~')
    path = home + '1'
    assert shrinkuser(path) == path


def test_shrinkuser_unrelated_path():
    assert shrinkuser('/some/other/path') == '/some/other/path'


def test_shrinkuser_relative_path():
    assert shrinkuser('foo/bar') == os.path.normpath('foo/bar')


def test_shrinkuser_normalizes():
    home = os.path.expanduser('~')
    messy = os.path.join(home, 'a', '..', 'b')
    assert shrinkuser(messy) == os.path.join('~', 'b')


def test_shrinkuser_deeper_subpath():
    home = os.path.expanduser('~')
    result = shrinkuser(os.path.join(home, 'a', 'b', 'c'))
    assert result == os.path.join('~', 'a', 'b', 'c')


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    result = expandpath('~/$SPAM')
    assert result == os.path.expanduser('~/eggs')


def test_expandpath_plain(monkeypatch):
    monkeypatch.delenv('NOTSET_VAR_XYZ', raising=False)
    assert expandpath('foo') == 'foo'


def test_expandpath_tilde(monkeypatch):
    assert expandpath('~') == os.path.expanduser('~')


def test_expandpath_env_in_middle(monkeypatch):
    monkeypatch.setenv('MYDIR', 'abc')
    assert expandpath('/$MYDIR/def') == '/abc/def'


def test_expandpath_multiple_vars(monkeypatch):
    monkeypatch.setenv('A', '1')
    monkeypatch.setenv('B', '2')
    assert expandpath('$A/$B') == '1/2'


def test_expandpath_missing_var(monkeypatch):
    monkeypatch.delenv('DEFINITELY_NOT_SET_VAR', raising=False)
    # expandvars leaves unknown vars with the trailing part removed / as-is
    result = expandpath('x/$DEFINITELY_NOT_SET_VAR/y')
    assert result == os.path.expandvars('x/$DEFINITELY_NOT_SET_VAR/y')


def test_expandpath_tilde_subdir(monkeypatch):
    home = os.path.expanduser('~')
    assert expandpath('~/sub') == os.path.join(home, 'sub')
