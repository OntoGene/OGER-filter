#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Anna Jancso, January 2018

import configparser as cp
import shutil
import os
import tensorflow as tf
from feature_extractor import FeatureExtractor
import pandas as pd


class Classifier:
    """Feed-forward neural network classifier."""

    def __init__(self, config_path='configs.ini'):
        """Get a config parser and read config file.

        config_path (str): path to INI file containing configurations for NN
        """
        self.config_path = config_path
        self.config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
        self.config.read(config_path)
        self.classifier = None

        tf.logging.set_verbosity(tf.logging.INFO)

        # [feature1, feature2, ..., featureN, label]
        self.column_names = self.get_column_names()

    def get_column_names(self):
        """Get column names."""
        columns = FeatureExtractor(self.config_path).column_names + ['label']
        return columns

    def load_data(self):
        """Load the training data."""
        train_path = self.config['paths']['training_data']

        train = pd.read_csv(train_path, names=self.column_names, header=0)
        train_x, train_y = train, train.pop('label')

        return train_x, train_y

    def train_input_fn(self, features, labels):
        """Input function for training."""
        n_epochs = int(self.config['parameters']['n_epochs'])
        batch_size = int(self.config['parameters']['batch_size'])
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        dataset = dataset.shuffle(10000).repeat(count=n_epochs)
        dataset = dataset.batch(batch_size)

        return dataset

    def create_model(self):
        """Create a new model."""
        model_path = self.config['paths']['model']
        # if a model already exists, remove it first
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)

        self.build_model()

    def restore_model(self):
        """Restore a trained model."""
        self.build_model()

    def build_model(self):
        """Build or restore neural network model."""
        model_path = self.config['paths']['model']
        n_hidden_layers = int(self.config['parameters']['n_hidden_layers'])
        n_hidden_neurons = int(self.config['parameters']['n_hidden_neurons'])
        n_output_neurons = len(self.config['classes'])

        checkpoint_config = tf.estimator.RunConfig(
            save_checkpoints_secs=2*60,
            keep_checkpoint_max=10,
        )

        feature_columns = [
            tf.feature_column.numeric_column(name)
            for name in self.column_names[:-1]]

        self.classifier = tf.estimator.DNNClassifier(
                            feature_columns=feature_columns,
                            hidden_units=([n_hidden_neurons])*n_hidden_layers,
                            n_classes=n_output_neurons,
                            model_dir=model_path,
                            config=checkpoint_config)

    def train_model(self):
        """Train the model."""
        train_x, train_y = self.load_data()

        self.classifier.train(
            input_fn=lambda: self.train_input_fn(train_x, train_y),
            steps=None)

    def predict_from_ngram(self, ngram):
        """Predict class from a ngram.

        args:
            ngram (str): n-gram
        """
        feat_extr = FeatureExtractor(self.config_path)
        feat_val_list = [val for val in feat_extr.iter_feature_values(ngram)]
        return self.predict_from_feat_val_list(feat_val_list)

    def predict_from_feat_val_list(self, feat_val_list):
        """Predict class from a list of feature values.

        args:
            feat_val_list (list): list of feature values
        """
        features = {}
        for i, f in enumerate(feat_val_list):
            name = self.column_names[i]
            features[name] = [f]

        predictions = self.classifier.predict(
            input_fn=lambda: self.eval_input_fn(features))

        for pred_dict in predictions:
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            print(class_id, probability)

    def eval_input_fn(self, features):
        """An input function for evaluation or prediction.

        args:
            features (dict): {feature-name: [value1, value2, ...],...}
        """
        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = dataset.batch(128)

        return dataset


def main():
    classifier = Classifier('CRAFT.ini')
    classifier.create_model()
    classifier.train_model()


if __name__ == '__main__':
    main()
