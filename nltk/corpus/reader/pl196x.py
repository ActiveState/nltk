# Natural Language Toolkit:
#
# Copyright (C) 2001-2019 NLTK Project
# Author: Piotr Kasprzyk <p.j.kasprzyk@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from six import string_types

from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader


PARA = re.compile(r'<p(?: [^>]*){0,1}>(.*?)</p>')
SENT = re.compile(r'<s(?: [^>]*){0,1}>(.*?)</s>')

TAGGEDWORD = re.compile(r'<([wc](?: [^>]*){0,1}>)(.*?)</[wc]>')
WORD = re.compile(r'<[wc](?: [^>]*){0,1}>(.*?)</[wc]>')

TYPE = re.compile(r'type="(.*?)"')
ANA = re.compile(r'ana="(.*?)"')

TEXTID = re.compile(r'text id="(.*?)"')

_OPEN_P = re.compile(r'<p')
_CLOSE_P = re.compile(r'</p>')
_OPEN_S = re.compile(r'<s')
_CLOSE_S = re.compile(r'</s>')
_OPEN_WC = re.compile(r'<[wc]')
_CLOSE_WC = re.compile(r'</[wc]>')


def _find_elements(s, open_re, close_re, tagged=False):
    """Linear-time equivalent of ``findall`` with PARA / SENT / WORD /
    TAGGEDWORD on a newline-free block.

    Those lazy ``.*?`` patterns rescan to the end of the block from every
    unclosed opening tag, which is O(k*n) (CWE-407).  Once one opening tag
    has no closing tag after it, no later one can either, so stop there.
    """
    out = []
    pos = 0
    n = len(s)
    while True:
        m = open_re.search(s, pos)
        if m is None:
            break
        a = m.end()
        if a >= n or s[a] not in ' >':
            pos = m.start() + 1
            continue
        gt = s.find('>', a)
        if gt < 0:
            break
        c = close_re.search(s, gt + 1)
        if c is None:
            break
        if tagged:
            out.append((s[m.start() + 1 : gt + 1], s[gt + 1 : c.start()]))
        else:
            out.append(s[gt + 1 : c.start()])
        pos = c.end()
    return out


class TEICorpusView(StreamBackedCorpusView):
    def __init__(
        self,
        corpus_file,
        tagged,
        group_by_sent,
        group_by_para,
        tagset=None,
        head_len=0,
        textids=None,
    ):

        self._tagged = tagged
        self._textids = textids

        self._group_by_sent = group_by_sent
        self._group_by_para = group_by_para
        # WARNING -- skip header
        StreamBackedCorpusView.__init__(self, corpus_file, startpos=head_len)

    _pagesize = 4096

    def read_block(self, stream):
        block = stream.readlines(self._pagesize)
        block = concat(block)
        # Running counts: re-counting the whole growing block per line is
        # O(n^2) on a file with no '</text>' (CWE-407).
        open_count = block.count('<text id')
        close_count = block.count('</text>')
        while (open_count > close_count) or (open_count == 0):
            tmp = stream.readline()
            if len(tmp) <= 0:
                break
            block += tmp
            open_count += tmp.count('<text id')
            close_count += tmp.count('</text>')

        block = block.replace('\n', '')

        if self._textids:
            # Keep only requested <text> elements in one forward pass
            # (per-id find + rebuild was O(k*n), CWE-407).
            kept = []
            pos = 0
            while True:
                beg = block.find('<text id', pos)
                if beg < 0:
                    kept.append(block[pos:])
                    break
                q1 = block.find('"', beg)
                q2 = block.find('"', q1 + 1) if q1 >= 0 else -1
                tid = block[q1 + 1 : q2] if 0 <= q1 < q2 else ''
                close = block.find('</text>', beg)
                end = len(block) if close < 0 else close + len('</text>')
                kept.append(block[pos:beg])
                if tid in self._textids:
                    kept.append(block[beg:end])
                pos = end
            block = ''.join(kept)

        output = []
        for para_str in _find_elements(block, _OPEN_P, _CLOSE_P):
            para = []
            for sent_str in _find_elements(para_str, _OPEN_S, _CLOSE_S):
                if not self._tagged:
                    sent = _find_elements(sent_str, _OPEN_WC, _CLOSE_WC)
                else:
                    sent = list(
                        map(
                            self._parse_tag,
                            _find_elements(sent_str, _OPEN_WC, _CLOSE_WC, True),
                        )
                    )
                if self._group_by_sent:
                    para.append(sent)
                else:
                    para.extend(sent)
            if self._group_by_para:
                output.append(para)
            else:
                output.extend(para)
        return output

    def _parse_tag(self, tag_word_tuple):
        (tag, word) = tag_word_tuple
        if tag.startswith('w'):
            tag = ANA.search(tag).group(1)
        else:  # tag.startswith('c')
            tag = TYPE.search(tag).group(1)
        return word, tag


class Pl196xCorpusReader(CategorizedCorpusReader, XMLCorpusReader):
    head_len = 2770

    def __init__(self, *args, **kwargs):
        if 'textid_file' in kwargs:
            self._textids = kwargs['textid_file']
        else:
            self._textids = None

        XMLCorpusReader.__init__(self, *args)
        CategorizedCorpusReader.__init__(self, kwargs)

        self._init_textids()

    def _init_textids(self):
        self._f2t = defaultdict(list)
        self._t2f = defaultdict(list)
        if self._textids is not None:
            with open(self._textids) as fp:
                for line in fp:
                    line = line.strip()
                    file_id, text_ids = line.split(' ', 1)
                    if file_id not in self.fileids():
                        raise ValueError(
                            'In text_id mapping file %s: %s not found'
                            % (self._textids, file_id)
                        )
                    for text_id in text_ids.split(self._delimiter):
                        self._add_textids(file_id, text_id)

    def _add_textids(self, file_id, text_id):
        self._f2t[file_id].append(text_id)
        self._t2f[text_id].append(file_id)

    def _resolve(self, fileids, categories, textids=None):
        tmp = None
        if (
            len(
                filter(
                    lambda accessor: accessor is None, (fileids, categories, textids)
                )
            )
            != 1
        ):

            raise ValueError(
                'Specify exactly one of: fileids, ' 'categories or textids'
            )

        if fileids is not None:
            return fileids, None

        if categories is not None:
            return self.fileids(categories), None

        if textids is not None:
            if isinstance(textids, string_types):
                textids = [textids]
            files = sum((self._t2f[t] for t in textids), [])
            tdict = dict()
            for f in files:
                tdict[f] = set(self._f2t[f]) & set(textids)
            return files, tdict

    def decode_tag(self, tag):
        # to be implemented
        return tag

    def textids(self, fileids=None, categories=None):
        """
        In the pl196x corpus each category is stored in single
        file and thus both methods provide identical functionality. In order
        to accommodate finer granularity, a non-standard textids() method was
        implemented. All the main functions can be supplied with a list
        of required chunks---giving much more control to the user.
        """
        fileids, _ = self._resolve(fileids, categories)
        if fileids is None:
            return sorted(self._t2f)

        if isinstance(fileids, string_types):
            fileids = [fileids]
        return sorted(sum((self._f2t[d] for d in fileids), []))

    def words(self, fileids=None, categories=None, textids=None):
        fileids, textids = self._resolve(fileids, categories, textids)
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]

        if textids:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid),
                        False,
                        False,
                        False,
                        head_len=self.head_len,
                        textids=textids[fileid],
                    )
                    for fileid in fileids
                ]
            )
        else:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid),
                        False,
                        False,
                        False,
                        head_len=self.head_len,
                    )
                    for fileid in fileids
                ]
            )

    def sents(self, fileids=None, categories=None, textids=None):
        fileids, textids = self._resolve(fileids, categories, textids)
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]

        if textids:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid),
                        False,
                        True,
                        False,
                        head_len=self.head_len,
                        textids=textids[fileid],
                    )
                    for fileid in fileids
                ]
            )
        else:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid), False, True, False, head_len=self.head_len
                    )
                    for fileid in fileids
                ]
            )

    def paras(self, fileids=None, categories=None, textids=None):
        fileids, textids = self._resolve(fileids, categories, textids)
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]

        if textids:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid),
                        False,
                        True,
                        True,
                        head_len=self.head_len,
                        textids=textids[fileid],
                    )
                    for fileid in fileids
                ]
            )
        else:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid), False, True, True, head_len=self.head_len
                    )
                    for fileid in fileids
                ]
            )

    def tagged_words(self, fileids=None, categories=None, textids=None):
        fileids, textids = self._resolve(fileids, categories, textids)
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]

        if textids:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid),
                        True,
                        False,
                        False,
                        head_len=self.head_len,
                        textids=textids[fileid],
                    )
                    for fileid in fileids
                ]
            )
        else:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid), True, False, False, head_len=self.head_len
                    )
                    for fileid in fileids
                ]
            )

    def tagged_sents(self, fileids=None, categories=None, textids=None):
        fileids, textids = self._resolve(fileids, categories, textids)
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]

        if textids:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid),
                        True,
                        True,
                        False,
                        head_len=self.head_len,
                        textids=textids[fileid],
                    )
                    for fileid in fileids
                ]
            )
        else:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid), True, True, False, head_len=self.head_len
                    )
                    for fileid in fileids
                ]
            )

    def tagged_paras(self, fileids=None, categories=None, textids=None):
        fileids, textids = self._resolve(fileids, categories, textids)
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]

        if textids:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid),
                        True,
                        True,
                        True,
                        head_len=self.head_len,
                        textids=textids[fileid],
                    )
                    for fileid in fileids
                ]
            )
        else:
            return concat(
                [
                    TEICorpusView(
                        self.abspath(fileid), True, True, True, head_len=self.head_len
                    )
                    for fileid in fileids
                ]
            )

    def xml(self, fileids=None, categories=None):
        fileids, _ = self._resolve(fileids, categories)
        if len(fileids) == 1:
            return XMLCorpusReader.xml(self, fileids[0])
        else:
            raise TypeError('Expected a single file')

    def raw(self, fileids=None, categories=None):
        fileids, _ = self._resolve(fileids, categories)
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]
        return concat([self.open(f).read() for f in fileids])
