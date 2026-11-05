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
