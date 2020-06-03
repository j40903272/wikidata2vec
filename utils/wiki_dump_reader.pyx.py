# -*- coding: utf-8 -*-
# License: Apache License 2.0

from __future__ import unicode_literals

import bz2
import logging

logger = logging.getLogger(__name__)


class WikiDataDumpReader(object):
    def __init__(self, dump_file):
        self._dump_file = dump_file

    @property
    def dump_file(self):
        return self._dump_file

    def __iter__(self):
        with bz2.open(self.dump_file, mode='rt') as f:
            c = 0
            f.read(2) # skip first two bytes: "{\n"
            for line in f:
                try:
                    yield json.loads(line.rstrip(',\n'))
                except json.decoder.JSONDecodeError:
                    continue

                c += 1
                if c % 100000 == 0:
                    logger.info('Processed: %d lines', c)


cdef unicode _normalize_title(unicode title):
    return (title[0].upper() + title[1:]).replace('_', ' ')