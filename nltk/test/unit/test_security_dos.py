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


class TestPorterYRun(unittest.TestCase):  # GHSA-ww6m-cw3f-q94g
    def test_outputs(self):
        from nltk.stem.porter import PorterStemmer
        p = PorterStemmer()
        self.assertEqual([p.stem(w) for w in ['ponies', 'happy', 'sky', 'syzygy']],
                         ['poni', 'happi', 'sky', 'syzygi'])

    def test_long_y_run(self):
        from nltk.stem.porter import PorterStemmer
        # unpatched: RuntimeError (recursion) at ~1000 'y'; quadratic below
        self.assertLess(_elapsed(lambda: PorterStemmer().stem('y' * 40000 + 'ness')), 2.0)


class TestXMLFragment(unittest.TestCase):  # GHSA-vp2x-qp44-57v7
    def test_benign(self):
        from nltk.corpus.reader.xmldocs import XMLCorpusView
        v = XMLCorpusView.__new__(XMLCorpusView)
        self.assertEqual(v._read_xml_fragment(io.StringIO('<a><b>x</b></a>z')),
                         '<a><b>x</b></a>z')

    def test_giant_tag(self):
        from nltk.corpus.reader.xmldocs import XMLCorpusView
        v = XMLCorpusView.__new__(XMLCorpusView)
        s = io.StringIO('<a ' + 'x' * 2000000 + '>')
        self.assertLess(_elapsed(lambda: v._read_xml_fragment(s)), 3.0)  # unpatched ~8s+


class TestSexprBlockGrowth(unittest.TestCase):  # same upstream change (7808692d)
    def test_oversized_sexpr_is_linear(self):
        from nltk.corpus.reader.util import read_sexpr_block

        text = '(' + 'a ' * 800000 + ')'  # unpatched: ~8s
        stream = io.StringIO(text)
        t = time.time()
        tokens = read_sexpr_block(stream, block_size=1024)
        self.assertLess(time.time() - t, 1.0)
        self.assertEqual(tokens, [text])


class TestPl196x(unittest.TestCase):  # GHSA-8mpw-7fpc-4gqj
    def _view(self, textids=None):
        from nltk.corpus.reader.pl196x import TEICorpusView
        v = TEICorpusView.__new__(TEICorpusView)
        v._pagesize, v._textids, v._tagged = 4096, textids, False
        v._group_by_sent = v._group_by_para = False
        return v

    def test_benign(self):
        body = '<text id="t1"><p><s><w>a</w><c>,</c></s></p></text>\n'
        self.assertEqual(self._view().read_block(io.StringIO(body)), ['a', ','])

    def test_unclosed_tags(self):
        for tag in ('<p>', '<w>'):  # unpatched: ~15s for 20000 '<p>'
            self.assertLess(_elapsed(lambda: self._view().read_block(
                io.StringIO('<text id="t">' + tag * 200000 + '</text>'))), 3.0)

    def test_textid_filter(self):
        body = ''.join('<text id="t%d">x</text>' % i for i in range(80000))
        self.assertLess(_elapsed(lambda: self._view(set(['z'])).read_block(
            io.StringIO(body))), 3.0)


class TestFeatStructDepth(unittest.TestCase):  # GHSA-cw6x-m8jw-qmrh
    def test_deep_nesting_raises_valueerror(self):
        from nltk.featstruct import FeatStruct
        self.assertRaises(ValueError, FeatStruct, '[a=' * 5000 + '1' + ']' * 5000)
        self.assertEqual(FeatStruct('[a=[b=1]]')['a']['b'], 1)
