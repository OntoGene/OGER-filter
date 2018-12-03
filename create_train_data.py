#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Anna Jancso, January 2018

import configparser as cp
from feature_extractor import FeatureExtractor
from reader import get_pmids_from_file, get_annotation_dict, iter_craft_xml
from reader import iter_craft_nn


class TrainingDataCreator:

    def __init__(self, config_path='configs.ini'):
        self.config_path = config_path
        self.config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
        self.config.optionxform = str
        self.config.read(config_path)


class CRAFTTrainingDataCreator(TrainingDataCreator):

    def create_train_data(self):
        """Create a CSV training file of the CRAFT corpus."""
        craft_path = self.config['paths']['craft_xml']
        dicts = self.config['craft_dicts']
        training_path = self.config['paths']['training_data']
        training_pmids_path = self.config['paths']['training_pmids']
        feature_funcs = list(self.config['features'])

        training_pmids = get_pmids_from_file(training_pmids_path)
        ann_dict = get_annotation_dict(craft_path, iter_craft_xml, dicts)

        # open test and training files for writing
        with open(training_path, 'w') as trainf:

            # create and write header of file
            header = ','.join(
                        f + (int(self.config['features'][f])-1)*','
                        for f in feature_funcs) + ',label\n'
            trainf.write(header)

            # create a feature extractor
            fextr = FeatureExtractor(self.config_path)

            # include CRAFT annotations of training PMIDs
            for ann in ann_dict:
                pmid, _, _, label, _ = ann
                n_gram = ann_dict[ann]
                # encode label
                label = self.config['classes'][label]

                if pmid in training_pmids:
                    # get stringified list of feature values
                    fiterator = (str(val)
                                 for val in fextr.iter_feature_values(
                                 n_gram))

                    # get whole string of feature values + label
                    fstring = ','.join(fiterator) + ',' + label + '\n'

                    trainf.write(fstring)

            # include corpus of normal nouns
            if int(self.config['other']['include_nn']):
                nn_limit = int(self.config['other']['nn_limit'])
                niterator = iter_craft_nn(self.config)

                for _ in range(nn_limit):
                    # get stringified list of feature values
                    fiterator = (str(val)
                                 for val in fextr.iter_feature_values(
                                 next(niterator)))

                    # get whole string of feature values + label
                    fstring = ','.join(fiterator) + ',' + '0' + '\n'
                    trainf.write(fstring)


def main():
    train_creator = CRAFTTrainingDataCreator('CRAFT.ini')
    train_creator.create_train_data()


if __name__ == "__main__":
    main()
