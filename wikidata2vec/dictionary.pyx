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
import os
import sys
import six.moves.cPickle as pickle
from collections import Counter, defaultdict
from contextlib import closing
from functools import partial
from itertools import chain
from marisa_trie import Trie, RecordTrie, BytesTrie
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
    def __init__(self, entity2idx, qid2label, label2qid, label2alias, alias2label, #qid_counter, label_counter, alias_counter, 
                          unicode language, bint lowercase, dict build_params, unicode uuid=''):
        
        self._entity2idx = entity2idx
        self._qid2label = qid2label
        self._label2qid = label2qid
        self._label2alias = label2alias
        self._alias2label = alias2label
        
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
    
    @property
    def entity2idx(self):
        return self._entity2idx
    
    @property
    def qid2label(self):
        return self._qid2label
    
    @property
    def label2qid(self):
        return self._label2qid
    
    @property
    def alias2label(self):
        return self._alias2label
    
    @property
    def label2alias(self):
        return self._label2alias
    
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
            alias = [i.decode('utf-8') for i in self._alias2label[word]]
            
        return label, alias
    
    cpdef unicode get_entity_by_qid(self, unicode qid):
        cdef unicode label
        
        label = self._qid2label[qid][0].decode('utf-8')
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
    
    cpdef unicode get_entity_qid(self, unicode title):
        cdef unicode qid

        if self.lowercase:
            title = title.lower()
        qid = self._label2qid[title][0].decode('utf-8')
        return qid
    
    cpdef list get_entity_alias(self, unicode title):
        cdef list alias

        if self.lowercase:
            title = title.lower()
        alias = [i.decode('utf-8') for i in self._label2alias[title]]
        return alias
        
        
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
        
        label2qid_list = list()
        qid2label_list = list()
        alias2label_list = list()
        label2alias_list = list()
        
        with closing(Pool(pool_size, initializer=init_worker, initargs=(dump_db, tokenizer))) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                f = partial(_process_entity, lowercase=lowercase)
                for qid2label, label2qid, label2alias, alias2label in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    
                    label2qid_list.extend(label2qid)
                    qid2label_list.extend(qid2label)
                    alias2label_list.extend(alias2label)
                    label2alias_list.extend(label2alias)
                    
                    bar.update(1)
                    
        logger.info('Building Internal Data Structure...')
        
        entity2idx = Trie([label for label, qid in label2qid_list])
        qid2label = BytesTrie(qid2label_list)
        label2qid = BytesTrie(label2qid_list)
        label2alias = BytesTrie(label2alias_list)
        alias2label = BytesTrie(alias2label_list)
        
        
        logger.info('%d entities and %d aliases are indexed in the dictionary', len(entity2idx), len(alias2label))
        logger.info('Size : entity2idx %d, qid2label %d, label2qid %d, alias2label %d, label2alias %d', 
                    sys.getsizeof(entity2idx), sys.getsizeof(qid2label), sys.getsizeof(label2qid), 
                    sys.getsizeof(alias2label), sys.getsizeof(label2alias))

        logger.info('Building Dictionary...')
        
        build_params = dict(
            dump_db=dump_db.uuid,
            dump_file=dump_db.dump_file,
            build_time=time.time() - start_time,
            version=pkg_resources.get_distribution('wikidata2vec').version
        )
        uuid = six.text_type(uuid1().hex)
        
        return Dictionary(entity2idx, qid2label, label2qid, label2alias, alias2label, #qid_counter, label_counter, alias_counter, 
                          dump_db.language, lowercase, build_params, uuid)
    
    ####
    #
    # save      : 
    # serialize : 
    # load      : 30.0 s
    # dada_load : 217 ms
    # dada_save : 5    s
    #
    ####
    
    def save(self, out_file):
        logger.info('Saving Dictionary...')
        os.makedirs(out_file, exist_ok = True)
        self._entity2idx.save(os.path.join(out_file, 'entity2idx.marisa'))
        self._qid2label.save(os.path.join(out_file, 'qid2label.marisa'))
        self._label2qid.save(os.path.join(out_file, 'label2qid.marisa'))
        self._label2alias.save(os.path.join(out_file, 'label2alias.marisa'))
        self._alias2label.save(os.path.join(out_file, 'alias2label.marisa'))
        meta=dict(uuid=self.uuid,
                      language=self.language,
                      lowercase=self.lowercase,
                      build_params=self.build_params)
        joblib.dump(meta, os.path.join(out_file, 'meta.pkl'))
        logger.info('Saving Complete')

    
    @staticmethod
    def load(target, mmap=True):
        logger.info('Loading Dictionary...')
        if mmap:
            entity2idx = Trie().mmap(os.path.join(target, 'entity2idx.marisa'))
            qid2label = BytesTrie().mmap(os.path.join(target, 'qid2label.marisa'))
            label2qid = BytesTrie().mmap(os.path.join(target, 'label2qid.marisa'))
            label2alias = BytesTrie().mmap(os.path.join(target, 'label2alias.marisa'))
            alias2label = BytesTrie().mmap(os.path.join(target, 'alias2label.marisa'))
        else:
            entity2idx = Trie().load(os.path.join(target, 'entity2idx.marisa'))
            qid2label = BytesTrie().load(os.path.join(target, 'qid2label.marisa'))
            label2qid = BytesTrie().load(os.path.join(target, 'label2qid.marisa'))
            label2alias = BytesTrie().load(os.path.join(target, 'label2alias.marisa'))
            alias2label = BytesTrie().load(os.path.join(target, 'alias2label.marisa'))
            
        meta = joblib.load(os.path.join(target, 'meta.pkl'))
        logger.info('Loading Complete')
        return Dictionary(entity2idx, qid2label, label2qid, label2alias, alias2label, **meta)
    
####################### remove ############################################# 
#     def save(self, out_file):
#         logger.info('Saving Dictionary...')
#         joblib.dump(self.serialize(), out_file)
#         logger.info('Saving Complete')

#     def serialize(self, shared_array=False):
#         logger.info('Serializing Dictionary...')
#         return dict(
#             entity2idx=self._entity2idx.tobytes(),
#             qid2label = self._qid2label.tobytes(),
#             label2qid = self._label2qid.tobytes(),
#             label2alias = self._label2alias.tobytes(),
#             alias2label = self._alias2label.tobytes(),
#             meta=dict(uuid=self.uuid,
#                       language=self.language,
#                       lowercase=self.lowercase,
#                       build_params=self.build_params)
#         )
#         logger.info('Serializing Complete')

#     @staticmethod
#     def load(target, mmap=True):
#         logger.info('Loading Dictionary...')
        
#         st = time.time()
#         if not isinstance(target, dict):
#             if mmap:
#                 target = joblib.load(target, mmap_mode='r')
#             else:
#                 target = joblib.load(target)
#         print('joblib {}'.format(time.time() - st))
        
#         st = time.time()
#         entity2idx = Trie().frombytes(target['entity2idx'])
#         print('entity2idx {}'.format(time.time() - st))
        
#         st = time.time()
#         qid2label = BytesTrie().frombytes(target['qid2label'])
#         print('qid2label {}'.format(time.time() - st))
        
#         st = time.time()
#         label2qid = BytesTrie().frombytes(target['label2qid'])
#         print('label2qid {}'.format(time.time() - st))
        
#         st = time.time()
#         label2alias = BytesTrie().frombytes(target['label2alias'])
#         print('label2alias {}'.format(time.time() - st))
        
#         st = time.time()
#         alias2label = BytesTrie().frombytes(target['alias2label'])
#         print('alias2label {}'.format(time.time() - st))
        
#         logger.info('Loading Complete')
#         return Dictionary(entity2idx, qid2label, label2qid, label2alias, alias2label, **target['meta'])
####################### remove #############################################

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
        
    qid2label = [(qid, label.encode('utf-8'))]
    label2qid = [(label, qid.encode('utf-8'))]
    label2alias = [(label, i.encode('utf-8')) for i in alias] if alias is not None else []
    alias2label = [(i, label.encode('utf-8')) for i in alias] if alias is not None else []
    
    return qid2label, label2qid, label2alias, alias2label
        
        