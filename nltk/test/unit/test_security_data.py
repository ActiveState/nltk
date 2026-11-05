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


@unittest.skipUnless(hasattr(os, 'symlink'), 'os.symlink not available')
class CorpusReaderSymlinkTest(_TreeTestBase):
    """CVE-2026-70626 / GHSA-r6gq-whwq-mvg9 and
    CVE-2026-79676 / GHSA-p4rw-rvv2-7xwr"""

    def setUp(self):
        _TreeTestBase.setUp(self)
        self.root = os.path.join(self.tmp, 'corpus')
        os.makedirs(self.root)
        self._write(os.path.join(self.root, 'c.txt'), 'IN ROOT')
        os.symlink(self.secret, os.path.join(self.root, 'a.txt'))
        os.symlink(self.outside, os.path.join(self.root, 'd'))
        os.symlink(
            os.path.join(self.root, 'c.txt'), os.path.join(self.root, 'lnk.txt'))

    def _reader(self, fileids=r'.*\.txt'):
        from nltk.corpus.reader import PlaintextCorpusReader

        return PlaintextCorpusReader(self.root, fileids)

    def test_symlinked_file_outside_root_is_refused(self):
        reader = self._reader(['a.txt'])
        self.assertRaises(ValueError, reader.open, 'a.txt')
        self.assertRaises(ValueError, reader.raw, 'a.txt')

    def test_symlinked_dir_outside_root_is_refused(self):
        reader = self._reader(['d/secret.txt'])
        self.assertRaises(ValueError, reader.raw, 'd/secret.txt')

    def test_escaping_symlinks_are_not_listed(self):
        self.assertEqual(self._reader().fileids(), ['c.txt', 'lnk.txt'])

    def test_in_root_symlink_still_reads(self):
        self.assertEqual(self._reader().raw('lnk.txt'), 'IN ROOT')

    def test_crubadan_language_code_cannot_escape_root(self):
        from nltk.corpus.reader import CrubadanCorpusReader

        root = os.path.join(self.tmp, 'crubadan')
        os.makedirs(root)
        self._write(os.path.join(root, 'table.txt'), '../outside/secret\txxx')
        self._write(os.path.join(self.outside, 'secret-3grams.txt'), '1 abc\n')
        reader = CrubadanCorpusReader(root, r'.*\.txt')
        self.assertRaises(ValueError, reader.lang_freq, 'xxx')

    def test_crubadan_table_symlink_outside_root_is_refused(self):
        from nltk.corpus.reader import CrubadanCorpusReader

        root = os.path.join(self.tmp, 'crubadan2')
        os.makedirs(root)
        table = os.path.join(self.outside, 'table.txt')
        self._write(table, 'en\teng')
        os.symlink(table, os.path.join(root, 'table.txt'))
        self.assertRaises(
            (ValueError, RuntimeError), CrubadanCorpusReader, root, ['table.txt'])
