#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Repeated predictions without graph rebuilding.

Based on https://github.com/marcsto/rl/blob/master/src/fast_predict2.py.
'''


import tensorflow as tf


class PatientPredictor:
    '''
    A wrapper around classifier.predict which avoids regenerating the graph.
    '''

    def __init__(self, estimator, output_types):
        self._current_batch = None
        self._closed = False

        input_fn = self._get_input_fn(output_types)
        self._predictions = estimator.predict(input_fn=input_fn)

    def _generator(self):
        while not self._closed:
            yield from self._current_batch

    def predict(self, feature_batch):
        '''
        Predict labels for the given features.

        feature_batch must be a sequence of feature dictionaries.
        '''
        self._current_batch = feature_batch
        results = [next(self._predictions) for _ in range(len(feature_batch))]
        return results

    def close(self):
        '''
        Release the tf graph.
        '''
        self._closed = True
        try:
            next(self._predictions)
        except StopIteration:
            pass

    def _get_input_fn(self, output_types):
        def _input_fn():
            dataset = tf.data.Dataset.from_generator(self._generator,
                                                     output_types=output_types)
            return dataset.batch(1)
        return _input_fn
