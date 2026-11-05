# -*- coding: utf-8 -*-
"""
CVE-2026-78683 / GHSA-rhp5-r9x4-f5g2: TransitionParser.parse() must not
execute code from a crafted model file.
"""
from __future__ import absolute_import

import os
import pickle
import shutil
import sys
import tempfile
import unittest

from nltk.parse import DependencyGraph
from nltk.parse.transitionparser import TransitionParser, _load_model

try:
    import sklearn  # noqa: F401
    import scipy  # noqa: F401
    import numpy  # noqa: F401

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

GOLD = """
Economic  JJ     2      ATT
news  NN     3       SBJ
has       VBD       0       ROOT
little      JJ      5       ATT
effect   NN     3       OBJ
on     IN      5       ATT
financial       JJ       8       ATT
markets    NNS      6       PC
.    .      3       PU
"""


class _Evil(object):
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


class TransitionParserPickleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.marker = os.path.join(self.tmp, 'PWNED')
        self.model = os.path.join(self.tmp, 'model.pkl')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, obj, protocol):
        with open(self.model, 'wb') as f:
            pickle.dump(obj, f, protocol)

    def test_reduce_gadget_is_refused_by_parse(self):
        for protocol in range(0, pickle.HIGHEST_PROTOCOL + 1):
            self._write(_Evil(self.marker), protocol)
            parser = TransitionParser('arc-standard')
            self.assertRaises(
                pickle.UnpicklingError,
                parser.parse,
                [DependencyGraph(GOLD)],
                self.model,
            )
            self.assertFalse(os.path.exists(self.marker))

    def test_dotted_name_and_module_lookalikes_are_refused(self):
        # numpy is allowlisted only for exact names; anything else in the
        # namespace (e.g. numpy.load, or a dotted attribute walk) is not.
        for payload in (
            b"cnumpy\nload\n(S'x'\ntR.",
            b"cnumpy.core.multiarray\n_reconstruct.__globals__\n.",
            b"cposix\nsystem\n(S'true'\ntR.",
        ):
            with open(self.model, 'wb') as f:
                f.write(payload)
            self.assertRaises(pickle.UnpicklingError, _load_model, self.model)

    @unittest.skipUnless(HAVE_SKLEARN, 'needs numpy/scipy/scikit-learn')
    def test_legitimate_model_round_trips(self):
        gold = DependencyGraph(GOLD)
        parser = TransitionParser('arc-standard')
        devnull = open(os.devnull, 'w')
        saved = sys.stdout
        sys.stdout = devnull
        try:
            parser.train([gold], self.model, verbose=False)
            result = parser.parse([gold], self.model)
        finally:
            sys.stdout = saved
            devnull.close()
        self.assertEqual(len(result), 1)

    @unittest.skipUnless(HAVE_SKLEARN, 'needs numpy')
    def test_object_dtype_numpy_scalar_is_refused(self):
        for protocol in (0, 2):
            self._write(_ObjectScalarCall(), protocol)
            self.assertRaises(pickle.UnpicklingError, _load_model, self.model)


class _ObjectScalarCall(object):
    """Pickles as numpy.core.multiarray.scalar(dtype('O'), ...), the nested
    unpickle sink upstream guards in CVE-2026-79657."""

    def __reduce__(self):
        import numpy as np

        return (np.core.multiarray.scalar, (np.dtype(object), b'payload'))
