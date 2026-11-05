# -*- coding: utf-8 -*-
"""Regression tests for the Group D DoS backports (3.4.5.post3).

Each crafted input took tens of seconds or crashed on unpatched 3.4.5 under
Python 2.7.18; patched code finishes in well under a second.  Bounds are ~10x
the patched time on a slow box and ~10x below the unpatched time.
"""
from __future__ import unicode_literals

import io
import time
import unittest


def _elapsed(fn):
    start = time.time()
    fn()
    return time.time() - start


class TestReviewsFeatures(unittest.TestCase):  # GHSA-fg7f-2386-8897
    def test_parity(self):
        from nltk.corpus.reader.reviews import _find_features
        line = 'battery life[+2], picture quality[-1][u] x[+3]## ok\n'
        self.assertEqual(
            _find_features(line),
            [('battery life', '+2'), ('picture quality', '-1'), ('x', '+3')],
        )

    def test_long_bracketless_lines_are_linear(self):
        from nltk.corpus.reader.reviews import _find_features
        for line in ('word ' * 40000, 'a' * 200000, ('a' * 80 + ' ') * 2500 + '['):
            self.assertLess(_elapsed(lambda: _find_features(line)), 2.0)

    def test_reader_features_on_crafted_line(self):
        import os
        import shutil
        import tempfile
        from nltk.corpus.reader.reviews import ReviewsCorpusReader

        tmp = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp, 'r.txt'), 'w') as f:
                f.write('ok feature[+1]##fine\n' + 'word ' * 8000 + '\n')
            reader = ReviewsCorpusReader(tmp, ['r.txt'])
            t = time.time()
            feats = list(reader.features())
            # unpatched: ~30s for this line
            self.assertLess(time.time() - t, 2.0)
            self.assertEqual(feats, [('ok feature', '+1')])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestTweetTokenizer(unittest.TestCase):  # GHSA-qx2g-xrx7-vfh8
    def test_dotted_run(self):
        from nltk.tokenize.casual import TweetTokenizer
        # unpatched: ~6s at 4000, super-quadratic
        self.assertLess(_elapsed(lambda: TweetTokenizer().tokenize('a.' * 4000)), 2.0)

    def test_urls_and_email_unchanged(self):
        from nltk.tokenize.casual import TweetTokenizer
        toks = TweetTokenizer().tokenize(
            'see https://foo.example.com/a_(b)_c?x=1 mail a.b+c@ex-ample.co.uk')
        self.assertIn('https://foo.example.com/a_(b)_c?x=1', toks)
        self.assertIn('a.b+c@ex-ample.co.uk', toks)
