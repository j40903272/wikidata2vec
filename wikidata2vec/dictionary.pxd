# -*- coding: utf-8 -*-
# License: Apache License 2.0

cimport numpy as np
from libc.stdint cimport int32_t


cdef class Item:
    cdef readonly int32_t index
    cdef readonly int32_t count
    cdef readonly int32_t doc_count


cdef class Word(Item):
    cdef readonly unicode text


# cdef class Entity(Item):
#     cdef readonly unicode title


cdef class Dictionary:
    cdef readonly unicode uuid
    cdef readonly unicode language
    cdef readonly bint lowercase
    cdef readonly dict build_params
    
    cdef readonly _entity2idx 
    cdef readonly _qid2label
    cdef readonly _label2qid
    cdef readonly _label2alias
    cdef readonly _alias2label
    
    cpdef get_entity(self, unicode word=?, unicode qid=?, int32_t index=?, default=?)
    cpdef int32_t get_entity_index(self, unicode)
    cpdef unicode get_entity_qid(self, unicode)
    cpdef list get_entity_alias(self, unicode)
    cpdef get_entity_by_index(self, int32_t)
    cpdef get_entity_by_word(self, unicode)
    cpdef unicode get_entity_by_qid(self, unicode)
    
    # remove
    cpdef get_word(self, unicode, default=?)
    cpdef Item get_item_by_index(self, int32_t)
    cpdef Word get_word_by_index(self, int32_t)
    cpdef int32_t get_word_index(self, unicode)