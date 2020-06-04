# -*- coding: utf-8 -*-
# License: Apache License 2.0

from libc.stdint cimport int32_t

cdef class DumpDB:
    cdef _db_file
    cdef _env
    cdef _meta_db
    cdef _entity_db

    cpdef unicode resolve_redirect(self, unicode)
    cpdef is_redirect(self, unicode)
    cpdef is_disambiguation(self, unicode)
    cpdef list get_paragraphs(self, unicode)
    cdef list _deserialize_paragraphs(self, bytes)


cdef class Paragraph:
    cdef readonly unicode text
    cdef readonly list wiki_links
    cdef readonly bint abstract


cdef class WikiLink:
    cdef readonly unicode title
    cdef readonly unicode text
    cdef readonly int32_t start
    cdef readonly int32_t end
    
    
cdef class Entity:
    cdef readonly unicode qid
    cdef readonly list alias
    cdef readonly unicode label
    cdef readonly unicode types
    cdef readonly unicode descriptions

