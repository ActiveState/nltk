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


class IntegrityBeforeExtractTest(_DownloaderTestBase):
    """CVE-2026-12259 / CVE-2026-63310 / GHSA-5wp5-5229-5g6q"""

    def _assert_rejected(self, pkg):
        errors = self._errors(self._install(pkg))
        self.assertTrue(errors)
        self.assertIn('Integrity check failed', errors[0].message)
        self.assertFalse(
            os.path.exists(os.path.join(self.dl, 'taggers', 'pkg')))
        self.assertFalse(
            os.path.exists(os.path.join(self.dl, 'taggers', 'pkg.zip')))

    def test_md5_mismatch_is_not_extracted(self):
        self._assert_rejected(self._package(
            'pkg', [('pkg/a.txt', b'x')], checksum='0' * 32))

    def test_sha256_mismatch_is_not_extracted(self):
        # sha256 takes precedence over a (correct) md5 when the index has it.
        self._assert_rejected(self._package(
            'pkg', [('pkg/a.txt', b'x')], sha256_checksum='0' * 64))

    def test_size_mismatch_is_not_extracted(self):
        self._assert_rejected(self._package(
            'pkg', [('pkg/a.txt', b'x')], size=1))

    def test_matching_sha256_is_extracted_and_installed(self):
        pkg = self._package('pkg', [('pkg/a.txt', b'x')])
        pkg.sha256_checksum = _sha256(os.path.join(self.srv, 'pkg.zip'))
        self.assertEqual(self._errors(self._install(pkg)), [])
        self.assertTrue(
            os.path.isfile(os.path.join(self.dl, 'taggers', 'pkg', 'a.txt')))
        d = Downloader(download_dir=self.dl)
        self.assertEqual(d.status(pkg, self.dl), Downloader.INSTALLED)

    def test_status_uses_sha256_when_present(self):
        pkg = self._package('pkg', [('pkg/a.txt', b'x')])
        self.assertEqual(self._errors(self._install(pkg)), [])
        pkg.sha256_checksum = '0' * 64
        d = Downloader(download_dir=self.dl)
        self.assertEqual(d.status(pkg, self.dl), Downloader.STALE)


@unittest.skipUnless(hasattr(os, 'link'), 'os.link not available')
class HardlinkTargetTest(_DownloaderTestBase):
    """CVE-2026-81727 / GHSA-f794-5jv7-7672"""

    def test_extraction_does_not_write_through_hardlink(self):
        outside = os.path.join(self.tmp, 'outside.txt')
        with open(outside, 'w') as f:
            f.write('SECRET')
        pkg_dir = os.path.join(self.dl, 'taggers', 'pkg')
        os.makedirs(pkg_dir)
        os.link(outside, os.path.join(pkg_dir, 'f.txt'))
        pkg = self._package('pkg', [('pkg/f.txt', b'OVERWRITTEN')])
        errors = self._errors(self._install(pkg))
        self.assertTrue(errors)
        self.assertIn('multiply-linked', errors[0].message)
        with open(outside) as f:
            self.assertEqual(f.read(), 'SECRET')

    def test_unzip_helper_also_refuses_hardlinked_target(self):
        from nltk.downloader import unzip
        outside = os.path.join(self.tmp, 'outside.txt')
        with open(outside, 'w') as f:
            f.write('SECRET')
        root = os.path.join(self.tmp, 'root')
        os.makedirs(os.path.join(root, 'pkg'))
        os.link(outside, os.path.join(root, 'pkg', 'f.txt'))
        zpath = os.path.join(self.srv, 'pkg.zip')
        _make_zip(zpath, [('pkg/f.txt', b'OVERWRITTEN')])
        self.assertRaises(Exception, unzip, zpath, root, False)
        with open(outside) as f:
            self.assertEqual(f.read(), 'SECRET')
