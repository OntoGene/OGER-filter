#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Anna Jancso, January 2018

import configparser as cp

import numpy as np
import hunspell
from gensim.models.keyedvectors import KeyedVectors


class FeatureExtractor:

    FEATURE_FUNCS = [
        'in_common_dict',
        'is_stop_word',
        'word_embeddings'
    ]

    def __init__(self, config_path):
        """Initialize feature functions and config file.

        Args:
            config_path (str): path to configuration file
        """
        self.config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
        self.config.read(config_path)
        self.feature_funcs = list(self.config['features'])
        self.column_names = [f + str(i)
                             for f in self.feature_funcs
                             for i in range(self.config.getint('features', f))]

        self.hobj = None
        self.stopws = None
        self.word_embedds = None

    def in_common_dict(self, n_gram):
        """Return whether the word occurs in a common dictionary."""
        # if hunspell has not been loaded yet
        if self.hobj is None:
            dic_path = self.config['paths']['hunspell_dic']
            aff_path = self.config['paths']['hunspell_aff']
            self.hobj = hunspell.HunSpell(dic_path, aff_path)

        try:
            in_dict = self.hobj.spell(n_gram)
        except UnicodeEncodeError:
            return 0
        else:
            return int(in_dict)

    def is_stop_word(self, n_gram):
        """Return whether the n_gram can also be a stopword."""
        # if stopwords have not been loaded yet
        if self.stopws is None:
            with open(self.config['paths']['stopwords']) as f:
                self.stopws = set(s.strip() for s in f)

        return int(n_gram in self.stopws)

    def word_embeddings(self, n_gram):
        """Return the word embedding of the n_gram."""
        if self.word_embedds is None:
            model_path = self.config["paths"]["word_embeddings"]
            self.word_embedds = KeyedVectors.load_word2vec_format(
                model_path, binary=True)

        n_gram = n_gram.split()
        embeddings = np.array([self.word_embedds[tok]
                               for tok in n_gram
                               if tok in self.word_embedds.vocab])
        if embeddings.size:
            mean = embeddings.mean(axis=0)
            return list(mean)
        else:
            dim = self.word_embedds.vectors.shape[1]
            return dim*[0.0]

    def iter_feature_values(self, n_gram):
        """Call all feature methods and yield their values."""
        for feature_fn in self.feature_funcs:
            feature_val = getattr(self, feature_fn)(n_gram)

            if isinstance(feature_val, list):
                for val in feature_val:
                    yield val
            else:
                yield feature_val

    def features(self, n_grams):
        """Construct feature dictionaries for multiple n-grams."""
        vals = zip(*(self.iter_feature_values(ng) for ng in n_grams))
        features = {name: list(v) for name, v in zip(self.column_names, vals)}
        return features
