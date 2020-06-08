# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import lmdb
import logging
import mwparserfromhell
import pkg_resources
import re
import six
from functools import partial
from uuid import uuid1
import zlib
from contextlib import closing
from six.moves import cPickle as pickle
from multiprocessing.pool import Pool

import pydash
import json
import pdb

logger = logging.getLogger(__name__)

STYLE_RE = re.compile("'''*")

# remove
cdef class Paragraph:
    def __init__(self, unicode text, list wiki_links, bint abstract):
        self.text = text
        self.wiki_links = wiki_links
        self.abstract = abstract

    def __repr__(self):
        if six.PY2:
            return ('<Paragraph %s>' % (self.text[:50] + '...')).encode('utf-8')
        else:
            return '<Paragraph %s>' % (self.text[:50] + '...')

    def __reduce__(self):
        return (self.__class__, (self.text, self.wiki_links, self.abstract))

# remove
cdef class WikiLink:
    def __init__(self, unicode title, unicode text, int32_t start, int32_t end):
        self.title = title
        self.text = text
        self.start = start
        self.end = end

    @property
    def span(self):
        return (self.start, self.end)

    def __repr__(self):
        if six.PY2:
            return ('<WikiLink %s->%s>' % (self.text, self.title)).encode('utf-8')
        else:
            return '<WikiLink %s->%s>' % (self.text, self.title)

    def __reduce__(self):
        return (self.__class__, (self.title, self.text, self.start, self.end))


cdef class Entity:
    def __init__(self, unicode qid, unicode label, unicode description, unicode types, list alias):
        self.qid = qid
        self.label = label
        self.description = description
        self.types = types
        self.alias = alias

    def __repr__(self):
        if six.PY2:
            return ('<Entity %s>' % (self.label)).encode('utf-8')
        else:
            return '<Entity %s>' % (self.label)

    def __reduce__(self):
        return (self.__class__, (self.qid, self.label, self.description, self.types, self.alias))


cdef class DumpDB:
    def __init__(self, db_file):
        self._db_file = db_file
        self._env = lmdb.open(db_file, readonly=True, subdir=False, lock=False, max_dbs=3)
        self._meta_db = self._env.open_db(b'__meta__')
        self._entity_db = self._env.open_db(b'__entity__')

    def __reduce__(self):
        return (self.__class__, (self._db_file,))

    @property
    def uuid(self):
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b'id').decode('utf-8')

    @property
    def dump_file(self):
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b'dump_file').decode('utf-8')

    @property
    def language(self):
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b'language').decode('utf-8')
        
    def entity_size(self):
        with self._env.begin(db=self._entity_db) as txn:
            return txn.stat()['entries']
    
    # remove
    def page_size(self):
        with self._env.begin(db=self._entity_db) as txn:
            return txn.stat()['entries']

    def titles(self):
        cdef bytes key

        with self._env.begin(db=self._entity_db) as txn:
            cur = txn.cursor()
            for key in cur.iternext(values=False):
                yield key.decode('utf-8')
    
    # remove
    def redirects(self):
        cdef bytes key, value

        with self._env.begin(db=self._redirect_db) as txn:
            cur = txn.cursor()
            for (key, value) in iter(cur):
                yield (key.decode('utf-8'), value.decode('utf-8'))
    
    # remove
    cpdef unicode resolve_redirect(self, unicode title):
        with self._env.begin(db=self._redirect_db) as txn:
            value = txn.get(title.encode('utf-8'))
            if value:
                return value.decode('utf-8')
            else:
                return title
    
    # remove
    cpdef is_redirect(self, unicode title):
        with self._env.begin(db=self._redirect_db) as txn:
            value = txn.get(title.encode('utf-8'))

        return bool(value)

    # remove
    cpdef is_disambiguation(self, unicode title):
        with self._env.begin(db=self._page_db) as txn:
            value = txn.get(title.encode('utf-8'))

        if not value:
            return False

        return pickle.loads(zlib.decompress(value))[1]
    
    # remove
    cpdef list get_paragraphs(self, unicode key):
        cdef bytes value

        with self._env.begin(db=self._page_db) as txn:
            value = txn.get(key.encode('utf-8'))
            if not value:
                raise KeyError(key)

        return self._deserialize_paragraphs(value)
    
    # remove
    cdef list _deserialize_paragraphs(self, bytes value):
        cdef list ret, wiki_links

        ret = []
        for obj in pickle.loads(zlib.decompress(value))[0]:
            wiki_links = [WikiLink(*args) for args in obj[1]]
            ret.append(Paragraph(obj[0], wiki_links, obj[2]))

        return ret
    
    cpdef Entity get_entity(self, unicode key):
        cdef bytes value

        with self._env.begin(db=self._entity_db) as txn:
            value = txn.get(key.encode('utf-8'))
            if not value:
                raise KeyError(key)

        return self._deserialize_entity(value)
    
    cdef Entity _deserialize_entity(self, bytes value):
        cdef list ret, wiki_links
        
        # [qid, label, descriptions, types, aliases]
        ret = pickle.loads(zlib.decompress(value))

        return Entity(*ret)

    @staticmethod
    def build(dump_reader, out_file, pool_size, chunk_size, preprocess_func=None,
              init_map_size=500000000, buffer_size=3000, language='en'):
        with closing(lmdb.open(out_file, subdir=False, map_async=True, map_size=init_map_size,
                               max_dbs=2)) as env:
            map_size = [init_map_size]
            meta_db = env.open_db(b'__meta__')
            with env.begin(db=meta_db, write=True) as txn:
                txn.put(b'id', six.text_type(uuid1().hex).encode('utf-8'))
                txn.put(b'dump_file', dump_reader.dump_file.encode('utf-8'))
                txn.put(b'language', language.encode('utf-8'))
                txn.put(b'version', six.text_type(
                    pkg_resources.get_distribution('wikidata2vec').version).encode('utf-8')
                )

            entity_db = env.open_db(b'__entity__')

            def write_db(db, data):
                while len(data) > 0:
                    try:
                        with env.begin(db=db, write=True) as txn:
                            txn.cursor().putmulti(data)
                        break
                        
                    except lmdb.MapFullError:
                        map_size[0] *= 2
                        env.set_mapsize(map_size[0])

                    except lmdb.BadValsizeError as error:
                        st = str(error).index('#')
                        ed = str(error).index(':')
                        idx = int(str(error)[st+1:ed])
                        logger.error(error)
                        logger.error('remove element %d', idx)
                        data.pop(idx)

            with closing(Pool(pool_size)) as pool:
                entity_buf = []
                f = partial(_parse, preprocess_func=preprocess_func, language=language)
                for ret in pool.imap_unordered(f, dump_reader, chunksize=chunk_size):
                    if ret:
                        entity_buf.append(ret[1])

                    if len(entity_buf) == buffer_size:
                        write_db(entity_db, entity_buf)
                        entity_buf = []

                if entity_buf:
                    write_db(entity_db, entity_buf)


def _parse(line, preprocess_func, language):
    cdef dict entity
    cdef unicode qid, label, descriptions, types
    cdef list aliases, ret

    if preprocess_func is None:
        preprocess_func = lambda x: x
    
    try:
        entity = json.loads(line.rstrip(',\n'))
        # entity dict_keys(['lastrevid', 'claims', 'labels', 'descriptions', 'id', 'sitelinks', 'type', 'aliases'])
    except json.decoder.JSONDecodeError:
        return None
    
    qid = pydash.get(entity, 'id')
    types = pydash.get(entity, 'type')
    label = pydash.get(entity, 'labels.' + language + '.value')
    descriptions = pydash.get(entity, 'descriptions.' + language + '.value')
    aliases = pydash.get(entity, 'aliases.' + language)
    if aliases is not None:
        aliases = [i['value'] for i in aliases]

    if (qid is None) or (label is None) or (not label):
        return None
        
    ret = [qid, label, descriptions, types, aliases]
    return ('entity', ((label.encode('utf-8'), zlib.compress(pickle.dumps(ret, protocol=-1)))))
