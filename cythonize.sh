#!/bin/bash

cython --cplus wikidata2vec/*.pyx
cython --cplus wikidata2vec/utils/*.pyx
cython --cplus wikidata2vec/utils/tokenizer/*.pyx
cython --cplus wikidata2vec/utils/sentence_detector/*.pyx
