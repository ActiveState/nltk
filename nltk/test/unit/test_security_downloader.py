# -*- coding: utf-8 -*-
"""
Offline regression tests for downloader security fixes (3.4.5.post3).

Packages are served from ``file://`` URLs so no network access is needed.
"""
from __future__ import absolute_import, unicode_literals

import hashlib
import os
import shutil
import tempfile
import unittest
import zipfile

from six.moves.urllib.request import pathname2url

from nltk.downloader import Downloader, ErrorMessage, Package


def _make_zip(path, members):
    zf = zipfile.ZipFile(path, 'w')
    try:
        for name, data in members:
            zf.writestr(name, data)
    finally:
        zf.close()


def _md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class _DownloaderTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.srv = os.path.join(self.tmp, 'srv')
        self.dl = os.path.join(self.tmp, 'nltk_data')
        os.mkdir(self.srv)
        os.mkdir(self.dl)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _package(self, pkg_id, members, subdir='taggers', **overrides):
        zpath = os.path.join(self.srv, pkg_id + '.zip')
        _make_zip(zpath, members)
        kw = dict(
            id=pkg_id,
            url='file:' + pathname2url(zpath),
            subdir=subdir,
            size=os.path.getsize(zpath),
            unzipped_size=sum(len(d) for _, d in members),
            checksum=_md5(zpath),
        )
        kw.update(overrides)
        return Package(**kw)

    def _install(self, pkg):
        d = Downloader(download_dir=self.dl)
        return list(d._download_package(pkg, self.dl, True))

    @staticmethod
    def _errors(msgs):
        return [m for m in msgs if isinstance(m, ErrorMessage)]


class CrossPackageOwnershipTest(_DownloaderTestBase):
    """CVE-2026-12261 / GHSA-ffj6-66c4-86gw"""

    def test_member_outside_own_package_dir_is_rejected(self):
        victim = os.path.join(self.dl, 'taggers', 'victim_tagger', 'model.json')
        os.makedirs(os.path.dirname(victim))
        with open(victim, 'w') as f:
            f.write('ORIGINAL')
        pkg = self._package(
            'evil', [('victim_tagger/model.json', b'POISONED')])
        errors = self._errors(self._install(pkg))
        self.assertTrue(errors)
        self.assertIn('Cross-package', errors[0].message)
        with open(victim) as f:
            self.assertEqual(f.read(), 'ORIGINAL')

    def test_member_inside_own_package_dir_is_extracted(self):
        pkg = self._package('good', [('good/model.json', b'{}')])
        self.assertEqual(self._errors(self._install(pkg)), [])
        self.assertTrue(
            os.path.isfile(os.path.join(self.dl, 'taggers', 'good', 'model.json')))
