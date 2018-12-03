#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
OGER postfilter for DNN disambiguation.
'''


import os
import sys

import tensorflow as tf

from oger.doc import document

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from classifier import Classifier
from feature_extractor import FeatureExtractor
from predict import PatientPredictor


# Setup (note: importing this module will take some time).
CONFIG_PATH = os.path.join(HERE, 'configs.ini')
CLF = Classifier(CONFIG_PATH)
CLF.restore_model()
FEX = FeatureExtractor(CONFIG_PATH)
PRED = PatientPredictor(
    CLF.classifier,
    # As a side effect, this loads all the lazy resources in FEX:
    tf.data.Dataset.from_tensor_slices(FEX.features([''])).output_types)
# Trigger loading the graph through a first dummy prediction.
PRED.predict([FEX.features([''])])

# TODO: find a better way to specify the mapping from the model's predictions
# to the labels used by OGER.
CLASSES = [None, 'cell', 'gene/protein', 'chemical', 'organism', 'BPMF',
           'cellular_component', 'sequence']


def disambiguate(content):
    '''
    Disambiguate and filter entities with a DNN.
    '''
    # Collect all terms.
    terms = set(e.text for e in content.iter_entities())
    probs = dict(_etype_probs(terms))
    for sentence in content.get_subelements(document.Sentence):
        sentence.entities = list(_filter_entities(sentence.entities, probs))


def _etype_probs(terms):
    '''
    Get all class probabilities for each term.
    '''
    for term in terms:
        features = FEX.features([term])
        (pred,) = PRED.predict([features])
        yield term, sorted(zip(pred['probabilities'], CLASSES),
                           reverse=True, key=lambda x: x[0])


def _filter_entities(entities, probs):
    removables = set(_ruled_out(entities, probs))
    return (e for e in entities if e not in removables)


def _ruled_out(entities, probs):
    '''
    Iterate over the subset of entities that are irrelevant.

    There are two reasons for removal:
      1. the model predicted None (not a bio-entity)
      2. there is a colocated entity with a type with higher score.
    '''
    for group in _colocated(entities):
        term = group[0].text
        prob_dist = probs[term]
        if prob_dist[0][1] is None:
            # Not a bio-entity.
            yield from group
        elif len(group) > 1:
            types = set(e.type for e in group)
            if len(types) > 1:
                # Ambiguous group. Iterate over the not-best.
                best = None
                for _, t in prob_dist:
                    if t in types:
                        best = t
                        break
                if best is None:
                    # Be robust agains unexpected entity types.
                    continue  # jump to the next group
                for e in group:
                    if e.type != best:
                        yield e


def _colocated(entities):
    '''
    Iterate over groups of annotations that completely overlap.
    '''
    entities = iter(entities)
    try:
        first = next(entities)
    except StopIteration:
        return

    group = [first]
    last = first.start, first.end
    for entity in entities:
        if (entity.start, entity.end) == last:
            # There is an overlap.
            group.append(entity)
        else:
            yield group
            group = [entity]
        last = entity.start, entity.end
    yield group
