#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Anna Jancso, January 2018

import os
import re
import random
from lxml import etree
from string import punctuation


def get_blocked_charpos(config):
    """Return hash with blocked character positions per PMID."""
    dir_path = config["paths"]["craft_xml"]
    dicts = config["craft_dicts"]

    blocked_charpos = {}

    for ann in get_annotation_dict(dir_path, iter_craft_xml, dicts):
        pmid, sspan, espan, d, _ = ann

        if pmid in blocked_charpos:
            for c in range(int(sspan), int(espan) + 1):
                blocked_charpos[pmid].add(c)
        else:
            blocked_charpos[pmid] = {
                c for c in range(int(sspan), int(espan) + 1)}

    return blocked_charpos


def iter_craft_nn(config):
    """Yield normal nouns of CRAFT training corpus."""
    pmids_path = config["paths"]["training_pmids"]
    pmids = get_pmids_from_file(pmids_path)
    dir_path = config["paths"]["craft_txt"]
    blocked_charpos = get_blocked_charpos(config)

    punct = {p for p in punctuation}

    words = []

    for fname in os.listdir(dir_path):
        pmid, ext = os.path.splitext(fname)

        if ext == ".txt":
            if pmid in pmids:

                with open(os.path.join(dir_path, fname)) as f:
                    text = f.read()[:-2000]

                    for match in re.finditer(r"\S+", text):
                        if (match.start() not in blocked_charpos[pmid]
                                and match.end() not in blocked_charpos[pmid]):
                            word = match.group()

                            if word[-1] in punct:
                                word = word[:-1]
                            if word and word[0] in punct:
                                word = word[1:]

                            if not word:
                                continue

                            if word in punct:
                                continue

                            if re.search(r"\d", word):
                                continue

                            words.append(word)

    random.shuffle(words)

    for word in words:
        yield word


def iter_craft_xml(path):
    """Yield all annotations of a CRAFT XML.

    args:
        path (str): path to an XML file
    yields:
        (pmid, n-gram, start span of n-gram, end span of n-gram, entity ID)
    """
    tree = etree.parse(path)
    root = tree.getroot()
    pmid = os.path.splitext(root.get('textSource'))[0]
    for ann in root.iter('annotation'):
        spans = [span for span in ann.iter('span')]
        n_grams = ann.find('spannedText').text.split(' ... ')

        for i, span in enumerate(spans):
            start_span = span.get('start')
            end_span = span.get('end')
            n_gram = n_grams[i]
            yield (pmid, n_gram, start_span, end_span, '')


def iter_oger_tsv(path):
    """Yield all annotations of an OGER tsv.

    args:
        path (str): path to a tsv file
    yields:
        (pmid, n-gram, start span of n-gram, end span of n-gram, entity ID)
    """
    with open(path) as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            pmid = fields[0]
            start_span = fields[2]
            end_span = fields[3]
            n_gram = fields[4]
            entity_id = fields[6]
            yield (pmid, n_gram, start_span, end_span, entity_id)


def iter_annotations(dir_path, func):
    """Yield all annotations of all files in a directory.

    args:
        dir_path (str): path to directory with files containing annotations
        func (func): function yielding annotations of a file
    yields:
        (pmid, n-gram, start span of n-gram, end span of n-gram, entity ID)
    """
    for fname in os.listdir(dir_path):
        file_path = os.path.join(dir_path, fname)
        for ann in func(file_path):
            yield ann


def get_annotation_dict(dir_path, func, dicts):
    """Returns an annotation hash.

    args:
        dir_path (str): path directory containing dicts and their annotations
        func (func): function yielding annotations of a file
        dicts (iterable): dictionaries to be used

    Each ontology contains all annotations of all XML's:
    returns:
        dictionary: (pmid, start span of n-gram,
                     end span of n-gram, ontology, entity ID): n-gram
    """
    ann_dict = {}
    # go through ontologies
    for d in dicts:
        dict_path = os.path.join(dir_path, d)
        for pmid, n_gram, sspan, espan, entity_id in iter_annotations(
                dict_path, func):
            ann_dict[(pmid, sspan, espan, d, entity_id)] = n_gram

    return ann_dict


def get_pmids_from_file(path):
    """Return PMID's from a file.

    args:
        path (str): path to the file
    returns:
        set of PMID's
    """
    pmids = set()
    with open(path) as f:
        for line in f:
            pmids.add(line.rstrip('\n'))

    return pmids
