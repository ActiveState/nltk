# -*- coding: utf-8 -*-
# Natural Language Toolkit: XML parsing that refuses entity declarations
#
# For license information, see LICENSE.TXT
"""
``parse()`` / ``fromstring()`` replacements for ``xml.etree.ElementTree``
that refuse documents declaring XML entities (CVE-2026-78681 /
GHSA-97qj-x29f-37w7: "billion laughs" entity expansion, CWE-776).

``defusedxml`` is used when it is installed; otherwise the document is
first scanned with ``xml.parsers.expat`` (the parser ElementTree itself
uses) and rejected if it declares any entity or references an external
one.  An external DTD reference, the predefined entities (``&amp;`` ...)
and numeric character references are still accepted.
"""
from __future__ import absolute_import

from xml.etree import ElementTree
from xml.parsers import expat

__all__ = ['HAVE_DEFUSEDXML', 'EntitiesForbidden', 'fromstring', 'parse']


class _FallbackEntitiesForbidden(ValueError):
    """A document declares or references an XML entity (stdlib fallback)."""

    def __init__(self, name=''):
        self.name = name
        ValueError.__init__(
            self, 'XML entity declaration forbidden: %s' % (name or '<unnamed>')
        )


try:
    from defusedxml import EntitiesForbidden
    from defusedxml.ElementTree import fromstring as _defused_fromstring
    from defusedxml.ElementTree import parse as _defused_parse

    HAVE_DEFUSEDXML = True
except ImportError:
    HAVE_DEFUSEDXML = False
    EntitiesForbidden = _FallbackEntitiesForbidden


def _reject_entities(data):
    parser = expat.ParserCreate()

    def _forbid_declaration(name, *rest):
        raise _FallbackEntitiesForbidden(name)

    def _forbid_external(*rest):
        raise _FallbackEntitiesForbidden('external')

    parser.EntityDeclHandler = _forbid_declaration
    parser.ExternalEntityRefHandler = _forbid_external
    try:
        parser.Parse(data, True)
    except expat.ExpatError:
        pass  # malformed: let ElementTree report it as before


def fromstring(text):
    """Safe ``ElementTree.fromstring``."""
    if HAVE_DEFUSEDXML:
        return _defused_fromstring(text)
    _reject_entities(text)
    return ElementTree.fromstring(text)


def parse(source):
    """Safe ``ElementTree.parse`` (``source`` is a filename or a file object)."""
    if HAVE_DEFUSEDXML:
        return _defused_parse(source)
    if hasattr(source, 'read'):
        data = source.read()
    else:
        with open(source, 'rb') as f:
            data = f.read()
    _reject_entities(data)
    return ElementTree.ElementTree(ElementTree.fromstring(data))
