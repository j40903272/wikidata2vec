# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import joblib
import logging
import multiprocessing
import numpy as np
import pkg_resources
import time
import six
import six.moves.cPickle as pickle
from collections import Counter, defaultdict
from contextlib import closing
from functools import partial
from itertools import chain
from marisa_trie import Trie, RecordTrie
from multiprocessing.pool import Pool
from tqdm import tqdm
from uuid import uuid1

from .dump_db cimport DumpDB, Paragraph, WikiLink, Entity
from .utils.tokenizer.token cimport Token


logger = logging.getLogger(__name__)


cdef class Item:
    def __init__(self, int32_t index, int32_t count, int32_t doc_count):
        self.index = index
        self.count = count
        self.doc_count = doc_count


cdef class Word(Item):
    def __init__(self, unicode text, int32_t index, int32_t count, int32_t doc_count):
        super(Word, self).__init__(index, count, doc_count)
        self.text = text

    def __repr__(self):
        if six.PY2:
            return b'<Word %s>' % self.text.encode('utf-8')
        else:
            return '<Word %s>' % self.text

    def __reduce__(self):
        return (self.__class__, (self.text, self.index, self.count, self.doc_count))
    

# remove
# cdef class Entity(Item):
#     def __init__(self, unicode title, int32_t index, int32_t count, int32_t doc_count):
#         super(Entity, self).__init__(index, count, doc_count)
#         self.title = title

#     def __repr__(self):
#         if six.PY2:
#             return b'<Entity %s>' % self.title.encode('utf-8')
#         else:
#             return '<Entity %s>' % self.title

#     def __reduce__(self):
#         return (self.__class__, (self.title, self.index, self.count, self.doc_count))


cdef class Dictionary:
    def __init__(self, entity2idx, dict qid2label, dict label2qid, label2alias, alias2label, #qid_counter, label_counter, alias_counter, 
                          unicode language, bint lowercase, dict build_params, unicode uuid=''):
        
        self._entity2idx = entity2idx
        self._qid2label = qid2label
        self._label2qid = label2qid
        self._label2alias = label2alias
        self._alias2label = alias2label
        
#         self._qid_counter = qid_counter
#         self._label_counter = label_counter
#         self._alias_counter = alias_counter
        
        self.uuid = uuid
        self.language = language
        self.lowercase = lowercase
        self.build_params = build_params
    
    # remove
    @property
    def entity_offset(self):
        return self._entity_offset
    
    #remove
    @property
    def word_size(self):
        return len(self._word_dict)
    
    @property
    def entity_size(self):
        return len(self._entity2idx)
    
    def __len__(self):
        return len(self._entity2idx)

    def __iter__(self):
        return self.entities()

    def entities(self):
        cdef unicode title
        cdef int32_t index

        for (title, index) in six.iteritems(self._entity2idx):
            yield title
    
    # return label, possible alias entity label
    cpdef get_entity(self, unicode word=None, unicode qid=None, int32_t index=-1, default=None):
        cdef set candidates
        cdef unicode label
        cdef list alias
        
        if qid is not None:
            label = self.get_entity_by_qid(qid)
            return label, []
        
        if index != -1:  # cannot assign default value None to int32_t
            label = self.get_entity_by_index(index)
            return label, []
        
        if word is not None:
            label, alias = self.get_entity_by_word(word)
            return label, alias
    
    cpdef get_entity_by_word(self, unicode word):
        cdef unicode label = None
        cdef list alias = []
        
        if self.lowercase:
            word = word.lower()
        
        if word in self._entity2idx:
            label = word
            
        if word in self._alias2label:
            alias = self._alias2label[word]
            
        return label, alias
    
    cpdef get_entity_by_qid(self, unicode qid):
        cdef unicode label
        
        label = self._qid2label[qid]
        return label
    
    cpdef get_entity_by_index(self, int32_t index):
        cdef unicode label
        
        label = self._entity2idx.restore_key(index)
        return label

    cpdef int32_t get_entity_index(self, unicode title):
        cdef int32_t index

        if self.lowercase:
            title = title.lower()
        index = self._entity2idx[title]
        return index
        
        
####################### remove #############################################
    cpdef get_word(self, unicode word, default=None):
        cdef int32_t index

        index = self.get_word_index(word)
        if index == -1:
            return default
        else:
            return Word(word, index, *self._word_stats[index])
    
    cpdef int32_t get_word_index(self, unicode word):
        try:
            return self._word_dict[word]
        except KeyError:
            return -1
    
    cpdef Item get_item_by_index(self, int32_t index):
        if index < self._entity_offset:
            return self.get_word_by_index(index)
        else:
            return self.get_entity_by_index(index)
    
    cpdef Word get_word_by_index(self, int32_t index):
        cdef unicode word

        word = self._word_dict.restore_key(index)
        return Word(word, index, *self._word_stats[index])
    
###########################################################################  

    @staticmethod
    def build(dump_db, tokenizer, lowercase, min_word_count, min_entity_count, min_paragraph_len,
              category, disambi, pool_size, chunk_size, progressbar=True):
        start_time = time.time()

        logger.info('Processing WikiData entities...')

        alias_counter = Counter()
        label_counter = Counter()
        qid_counter = Counter()
        
        label2qid = dict()
        qid2label = dict()
        alias2label = defaultdict(list)
        label2alias = defaultdict(list)#dict()

        with closing(Pool(pool_size, initializer=init_worker, initargs=(dump_db, tokenizer))) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                f = partial(_process_entity, lowercase=lowercase)
                for qid, label, alias in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    
                    alias_counter.update(alias)
                    label_counter.update([label])
                    qid_counter.update([qid])
                    
                    qid2label[qid] = label
                    label2qid [label] = qid
                    label2alias[label] = alias
                    if alias is not None:
                        for i in alias:
                            alias2label[i].append(label)
                    
                    bar.update(1)
        
        entity2idx = Trie(label2qid.keys())
        
        build_params = dict(
            dump_db=dump_db.uuid,
            dump_file=dump_db.dump_file,
            build_time=time.time() - start_time,
            version=pkg_resources.get_distribution('wikidata2vec').version
        )

        uuid = six.text_type(uuid1().hex)

        logger.info('%d entities and %d aliases are indexed in the dictionary', len(entity2idx), len(alias_counter))

        return Dictionary(entity2idx, qid2label, label2qid, label2alias, alias2label, #qid_counter, label_counter, alias_counter, 
                          dump_db.language, lowercase, build_params, uuid)

    def save(self, out_file):
        joblib.dump(self.serialize(), out_file)

    def serialize(self, shared_array=False):
        return dict(
            entity2idx=self._entity2idx.tobytes(),
            qid2label = self._qid2label,
            label2qid = self._label2qid,
            label2alias = self._label2alias,
            alias2label = self._alias2label,
#             qid_counter = self._qid_counter,
#             label_counter = self._label_counter,
#             alias_counter = self._alias_counter,
            meta=dict(uuid=self.uuid,
                      language=self.language,
                      lowercase=self.lowercase,
                      build_params=self.build_params)
        )

    @staticmethod
    def load(target, mmap=True):
        entity2idx = Trie()

        if not isinstance(target, dict):
            if mmap:
                target = joblib.load(target, mmap_mode='r')
            else:
                target = joblib.load(target)
        
        st = time.time()
        entity2idx.frombytes(target['entity2idx'])
        print('entity2idx {}'.format(st - time.time()))
        
        st = time.time()
        qid2label = target['qid2label']
        print('entity2idx {}'.format(st - time.time()))
        
        st = time.time()
        label2qid = target['label2qid']
        print('entity2idx {}'.format(st - time.time()))
        
        st = time.time()
        label2alias = target['label2alias']
        print('entity2idx {}'.format(st - time.time()))
        
        st = time.time()
        alias2label = target['alias2label']
        print('entity2idx {}'.format(st - time.time()))

        
#         qid_counter = target['qid_counter']
#         label_counter = target['label_counter']
#         alias_counter = target['alias_counter']
        
        return Dictionary(entity2idx, qid2label, label2qid, label2alias, alias2label, **target['meta'])#qid_counter, label_counter, alias_counter, 


cdef DumpDB _dump_db = None


def init_worker(dump_db, tokenizer):
    global _dump_db
    _dump_db = dump_db


def _process_entity(unicode title, bint lowercase):
    cdef unicode qid, label
    cdef list alias
    cdef Entity entity

    entity = _dump_db.get_entity(title)
    
    qid, label, alias = entity.qid, entity.label, entity.alias
    if lowercase:
        label = label.lower()
        if alias is not None:
            alias = [i.lower() for i in alias]
        
    return qid, label, alias
        
        