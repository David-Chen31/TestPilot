import os
import os.path
from os.path import expanduser, join, normpath

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------- augpath

def test_augpath_no_changes():
    assert augpath('foo.bar') == 'foo.bar'
    assert augpath('foo') == 'foo'
    assert augpath('') == ''


def test_augpath_docstring_examples():
    assert augpath('foo.bar') == 'foo.bar'
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'
    assert augpath('foo.bar', prefix='_') == '_foo.bar'
    assert augpath('foo.bar', base='baz') == 'baz.bar'
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_full_example():
    newpath = augpath('foo.bar', '_suff', 'pref_', ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_directory_preserved():
    assert augpath('/a/b/foo.bar', ext='.baz') == '/a/b/foo.baz'
    assert augpath('a/b/foo.bar', suffix='_x') == 'a/b/foo_x.bar'
    assert augpath('a/b/foo.bar', prefix='x_') == 'a/b/x_foo.bar'


def test_augpath_dpath_replacement():
    result = augpath('a/b/foo.bar', dpath='c/d')
    assert result == join('c/d', 'foo.bar')
    result = augpath('/a/b/foo.bar', ext='.txt', dpath='/x/y')
    assert result == join('/x/y', 'foo.txt')


def test_augpath_dpath_none_keeps_original():
    assert augpath('a/b/foo.bar', dpath=None) == join('a/b', 'foo.bar')


def test_augpath_multidot_true():
    # first dot defines extension
    assert augpath('foo.tar.gz', suffix='_x', multidot=True) == 'foo_x.tar.gz'
    assert augpath('foo.tar.gz', prefix='p_', multidot=True) == 'p_foo.tar.gz'
    assert augpath('foo.tar.gz', base='baz', multidot=True) == 'baz.tar.gz'


def test_augpath_multidot_false_default():
    # last dot defines extension
    assert augpath('foo.tar.gz', suffix='_x') == 'foo.tar_x.gz'
    assert augpath('foo.tar.gz', base='baz') == 'baz.gz'


def test_augpath_no_dot_multidot_true():
    assert augpath('foo', multidot=True) == 'foo'
    assert augpath('foo', ext='.bar', multidot=True) == 'foo.bar'
    assert augpath('foo', suffix='_', multidot=True) == 'foo_'


def test_augpath_no_dot_multidot_false():
    assert augpath('foo', ext='.bar') == 'foo.bar'
    assert augpath('foo', suffix='_') == 'foo_'


def test_augpath_all_components():
    result = augpath('a/b/foo.tar.gz', suffix='_s', prefix='p_',
                     ext='.zip', base='newb', dpath='x/y', multidot=False)
    assert result == join('x/y', 'p_newb_s.zip')


def test_augpath_pathlike():
    import pathlib
    p = pathlib.PurePosixPath('a/b/foo.bar')
    assert augpath(p, ext='.baz') == join('a/b', 'foo.baz')


def test_augpath_returns_str():
    assert isinstance(augpath('foo.bar'), str)


# -------------------------------------------------------------- shrinkuser

def test_shrinkuser_home_exact():
    home = expanduser('~')
    assert shrinkuser(home) == '~'


def test_shrinkuser_home_custom_symbol():
    home = expanduser('~')
    assert shrinkuser(home, home='$HOME') == '$HOME'


def test_shrinkuser_under_home():
    home = expanduser('~')
    assert shrinkuser(join(home, '1')) == join('~', '1')
    assert shrinkuser(join(home, 'a', 'b')) == join('~', 'a', 'b')


def test_shrinkuser_under_home_custom_symbol():
    home = expanduser('~')
    assert shrinkuser(join(home, '1'), home='$HOME') == join('$HOME', '1')


def test_shrinkuser_not_under_home():
    home = expanduser('~')
    other = home + '1'
    assert shrinkuser(other) == normpath(other)


def test_shrinkuser_relative_path():
    assert shrinkuser('foo/bar') == normpath('foo/bar')
    assert shrinkuser('foo') == 'foo'


def test_shrinkuser_normalizes():
    assert shrinkuser('foo/./bar') == normpath('foo/./bar')
    assert shrinkuser('foo//bar') == normpath('foo//bar')


def test_shrinkuser_pathlike():
    import pathlib
    home = expanduser('~')
    assert shrinkuser(pathlib.PurePosixPath(join(home, '1'))) == join('~', '1')


def test_shrinkuser_docstring_examples():
    path = expanduser('~')
    assert path != '~'
    assert shrinkuser(path) == '~'
    assert shrinkuser(path + '1') == path + '1'
    assert shrinkuser(path + '/1') == join('~', '1')
    assert shrinkuser(path + '/1', '$HOME') == join('$HOME', '1')


# -------------------------------------------------------------- expandpath

def test_expandpath_simple():
    assert expandpath('foo') == 'foo'


def test_expandpath_no_vars_no_tilde():
    assert expandpath('a/b/c') == 'a/b/c'


def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('$SPAM') == 'eggs'
    assert expandpath('foo/$SPAM/bar') == 'foo/eggs/bar'


def test_expandpath_env_var_braces(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('${SPAM}') == 'eggs'


def test_expandpath_tilde():
    assert expandpath('~') == expanduser('~')
    assert expandpath('~/foo') == expanduser('~/foo')


def test_expandpath_tilde_and_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('~/$SPAM') == expanduser('~/eggs') or \
        expandpath('~/$SPAM') == join(expanduser('~'), 'eggs')


def test_expandpath_docstring(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('~/$SPAM') == expanduser('~/eggs')
    assert expandpath('foo') == 'foo'


def test_expandpath_undefined_var(monkeypatch):
    monkeypatch.delenv('NOSUCHVAR', raising=False)
    # expandvars leaves unknown variables untouched on POSIX
    assert expandpath('$NOSUCHVAR') == '$NOSUCHVAR'


def test_expandpath_pathlike():
    import pathlib
    assert expandpath(pathlib.PurePosixPath('foo/bar')) == 'foo/bar'


def test_expandpath_returns_str():
    assert isinstance(expandpath('foo'), str)
