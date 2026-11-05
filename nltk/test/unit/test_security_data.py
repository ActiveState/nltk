# -*- coding: utf-8 -*-
"""
Offline regression tests for nltk.data / corpus reader path containment
fixes in 3.4.5.post3.
"""
from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

import nltk.data
from nltk.data import find, load


class _TreeTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.data = os.path.join(self.tmp, 'data')
        self.outside = os.path.join(self.tmp, 'outside')
        os.makedirs(os.path.join(self.data, 'corpora'))
        os.makedirs(self.outside)
        self._write(os.path.join(self.data, 'corpora', 'ok.txt'), 'OK')
        self._write(os.path.join(self.data, 'corpora', 'a b.txt'), 'SPACE')
        self.secret = os.path.join(self.outside, 'secret.txt')
        self._write(self.secret, 'SECRET')
        self._saved_path = list(nltk.data.path)
        nltk.data.path[:] = [self.data]
        nltk.data._resource_cache.clear()

    def tearDown(self):
        nltk.data.path[:] = self._saved_path
        nltk.data._resource_cache.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _write(path, text):
        with open(path, 'w') as f:
            f.write(text)


class DataLoadTraversalTest(_TreeTestBase):
    """CVE-2026-54293 / GHSA-p4gq-832x-fm9v and
    CVE-2026-12243 / GHSA-m42h-3232-vpv3"""

    PAYLOADS = [
        '../outside/secret.txt',
        'corpora/../../outside/secret.txt',
        '%2e%2e/outside/secret.txt',
        '..%2foutside%2fsecret.txt',
        '%2E%2E%2Foutside%2Fsecret.txt',
        'corpora/%2e%2e/%2e%2e/outside/secret.txt',
        'nltk:..%2foutside/secret.txt',
        'nltk:%2e%2e/outside/secret.txt',
    ]

    def test_traversal_payloads_are_refused_by_load(self):
        for payload in self.PAYLOADS:
            try:
                data = load(payload, format='raw', cache=False)
            except ValueError:
                continue
            self.fail('%r was not refused (read %r)' % (payload, data))

    def test_encoded_absolute_path_is_refused(self):
        payload = 'nltk:' + self.secret.replace('/', '%2f')
        self.assertRaises(ValueError, load, payload, format='raw', cache=False)

    def test_find_refuses_encoded_traversal(self):
        self.assertRaises(ValueError, find, '%2e%2e/outside/secret.txt')
        self.assertRaises(ValueError, find, '../outside/secret.txt')

    def test_legitimate_names_still_load(self):
        self.assertEqual(load('corpora/ok.txt', format='raw', cache=False), b'OK')
        self.assertEqual(
            load('corpora/a%20b.txt', format='raw', cache=False), b'SPACE')
        self.assertEqual(
            load('nltk:corpora/ok.txt', format='raw', cache=False), b'OK')
        # An explicit absolute file: URL remains the caller's choice.
        self.assertEqual(
            load('file:' + self.secret, format='raw', cache=False), b'SECRET')
