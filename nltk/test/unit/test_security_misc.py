# -*- coding: utf-8 -*-
"""
Offline regression tests for miscellaneous 3.4.5.post3 security fixes.
"""
from __future__ import absolute_import, unicode_literals

import os
import runpy
import shutil
import sys
import tempfile
import unittest

import mock


class CollocationsCliEvalTest(unittest.TestCase):
    """CVE-2025-71408 / GHSA-848c-c2cx-j7qx"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.marker = os.path.join(self.tmp, 'PWNED')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cli_argument_is_not_evaluated(self):
        payload = "pmi if __import__('os').mkdir(%r) else 0" % self.marker
        saved = sys.argv
        sys.argv = ['collocations', payload, payload]
        try:
            # str(): Py2 runpy rejects unicode module names (unicode_literals)
            runpy.run_module(str('nltk.collocations'), run_name=str('__main__'))
        except Exception:
            pass  # the demo may fail offline; only the side effect matters
        finally:
            sys.argv = saved
        self.assertFalse(os.path.exists(self.marker))

    def test_scorer_lookup(self):
        from nltk.collocations import _scorer_from_name
        from nltk.metrics import BigramAssocMeasures

        self.assertEqual(
            _scorer_from_name('pmi'), BigramAssocMeasures.pmi)
        self.assertIsNone(_scorer_from_name('__class__'))
        self.assertIsNone(_scorer_from_name('pmi(); x'))
        self.assertIsNone(_scorer_from_name(''))


@unittest.skipIf(os.name == 'nt', 'POSIX shell script fixture')
class GraphvizDotSearchPathTest(unittest.TestCase):
    """CVE-2026-78680 / GHSA-6hwm-xvph-95vm"""

    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.cwd = os.path.join(self.tmp, 'cwd')
        self.bindir = os.path.join(self.tmp, 'bin')
        os.makedirs(self.cwd)
        os.makedirs(self.bindir)
        self.marker = os.path.join(self.tmp, 'PLANTED_DOT_RAN')
        self._script(os.path.join(self.cwd, 'dot'), ': > %s' % self.marker)
        self.saved_cwd = os.getcwd()
        self.saved_path = os.environ.get('PATH', '')
        os.chdir(self.cwd)

    def tearDown(self):
        os.chdir(self.saved_cwd)
        os.environ['PATH'] = self.saved_path
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _script(path, body):
        with open(path, 'w') as f:
            # shell builtins only: the tests run with a minimal PATH
            f.write('#!/bin/sh\n%s\nwhile read -r l; do :; done\n'
                    'echo "<svg/>"\n' % body)
        os.chmod(path, 0o755)

    @staticmethod
    def _renderers():
        from nltk.parse import DependencyGraph
        from nltk.translate import AlignedSent

        dg = DependencyGraph('John N 2\nloves V 0\nMary N 2')
        return [dg._repr_svg_, AlignedSent(['a'], ['b'])._repr_svg_]

    def test_planted_dot_in_cwd_is_not_run(self):
        os.environ['PATH'] = os.pathsep.join(['.', ''])
        for render in self._renderers():
            self.assertRaises(Exception, render)
        self.assertFalse(os.path.exists(self.marker))

    def test_dot_on_absolute_path_entry_is_used(self):
        self._script(os.path.join(self.bindir, 'dot'), ':')
        os.environ['PATH'] = os.pathsep.join(['.', self.bindir])
        for render in self._renderers():
            self.assertIn('<svg/>', render())
        self.assertFalse(os.path.exists(self.marker))
