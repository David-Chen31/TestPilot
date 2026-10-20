import os
import sys

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_identity():
    assert augpath('foo.bar') == 'foo.bar'


def test_augpath_docstring_example():
    newpath = augpath('foo.bar', '_suff', 'pref_', ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_ext_replacement():
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'


def test_augpath_suffix():
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'


def test_augpath_prefix():
    assert augpath('foo.bar', prefix='_') == '_foo.bar'


def test_augpath_base_replacement():
    assert augpath('foo.bar', base='baz') == 'baz.bar'


def test_augpath_all_components():
    assert augpath('foo.bar', suffix='_s', prefix='p_', base='b', ext='.e') == 'p_b_s.e'


def test_augpath_multidot_true():
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'


def test_augpath_multidot_false():
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'


def test_augpath_multidot_suffix():
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_multidot_no_dot():
    # No dot at all: with multidot, whole name is base and ext is ''
    assert augpath('foo', ext='.bar', multidot=True) == 'foo.bar'
    assert augpath('foo', suffix='_', multidot=True) == 'foo_'


def test_augpath_dpath_replacement():
    result = augpath('dir/foo.bar', dpath='other')
    assert result == os.path.join('other', 'foo.bar')


def test_augpath_dpath_from_original():
    result = augpath(os.path.join('a', 'b', 'foo.txt'))
    assert result == os.path.join('a', 'b', 'foo.txt')


def test_augpath_no_extension():
    assert augpath('foo') == 'foo'
    assert augpath('foo', ext='.txt') == 'foo.txt'
    assert augpath('foo', base='bar') == 'bar'


def test_augpath_hidden_file():
    # dotfile handled by splitext: '.bashrc' -> base='.bashrc', ext=''
    assert augpath('.bashrc') == '.bashrc'


def test_augpath_returns_str():
    assert isinstance(augpath('foo.bar'), str)


def test_augpath_pathlike():
    import pathlib
    p = pathlib.PurePosixPath('foo.bar')
    result = augpath(p, ext='.baz')
    assert result == 'foo.baz'


def test_augpath_empty_path():
    assert augpath('') == ''


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_home_directly():
    home = os.path.expanduser('~')
    assert shrinkuser(home) == '~'


def test_shrinkuser_home_with_suffix_extension():
    home = os.path.expanduser('~')
    # home + '1' is not a subpath boundary; should be unchanged
    assert shrinkuser(home + '1') == home + '1'


def test_shrinkuser_subpath():
    home = os.path.expanduser('~')
    result = shrinkuser(os.path.join(home, '1'))
    assert result == os.path.join('~', '1')


def test_shrinkuser_custom_home():
    home = os.path.expanduser('~')
    result = shrinkuser(os.path.join(home, '1'), '$HOME')
    assert result == os.path.join('$HOME', '1')


def test_shrinkuser_unrelated_path():
    assert shrinkuser('/some/other/path') == '/some/other/path'


def test_shrinkuser_normalizes():
    home = os.path.expanduser('~')
    messy = os.path.join(home, 'a', '..', 'b')
    assert shrinkuser(messy) == os.path.join('~', 'b')


def test_shrinkuser_relative_path():
    assert shrinkuser('foo/bar') == os.path.join('foo', 'bar')


def test_shrinkuser_returns_str():
    assert isinstance(shrinkuser('/tmp/whatever'), str)


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_expands_env_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    result = expandpath('~/$SPAM')
    assert result == os.path.expanduser('~/eggs')


def test_expandpath_plain(monkeypatch):
    monkeypatch.delenv('SPAM', raising=False)
    assert expandpath('foo') == 'foo'


def test_expandpath_tilde_only():
    assert expandpath('~') == os.path.expanduser('~')


def test_expandpath_env_only(monkeypatch):
    monkeypatch.setenv('MYVAR', 'value')
    assert expandpath('$MYVAR') == 'value'


def test_expandpath_missing_env_var(monkeypatch):
    monkeypatch.delenv('NONEXISTENT_VAR_XYZ', raising=False)
    # expandvars leaves unknown vars as-is
    assert expandpath('$NONEXISTENT_VAR_XYZ') == '$NONEXISTENT_VAR_XYZ'


def test_expandpath_returns_str():
    assert isinstance(expandpath('foo'), str)


def test_expandpath_pathlike():
    import pathlib
    p = pathlib.PurePosixPath('foo')
    assert expandpath(p) == 'foo'


def test_expandpath_empty():
    assert expandpath('') == ''
