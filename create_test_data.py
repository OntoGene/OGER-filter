#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Anna Jancso, January 2018

from classifier import Classifier
import configparser as cp
from feature_extractor import FeatureExtractor
from reader import get_annotation_dict, get_pmids_from_file
from reader import iter_craft_xml, iter_oger_tsv


class TestDataCreator:

    def __init__(self, config_path='configs.ini'):
        self.config_path = config_path
        self.config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
        self.config.optionxform = str
        self.config.read(config_path)

    def write_to_file(self, ann_iterator, is_gold):
        """Write the annotations to a file.

        args:
            ann_iterator (iterator): yields annotations
            is_gold (bool): annotations are gold
        """
        if is_gold:
            test_path = self.config['paths']['gold_test_data']
        else:
            test_path = self.config['paths']['system_test_data']

        # open test file for writing
        with open(test_path, 'w') as testf:

            # sort by doc-ID (-> necessary for evaluation script!)
            for ann in sorted(ann_iterator()):
                # get string for writing
                wstring = "\t".join(str(i) for i in ann) + '\n'

                # write to test file
                testf.write(wstring)


class CRAFTTestDataCreator(TestDataCreator):

    def iter_gold_test_data(self):
        """Iter gold annotations on the CRAFT corpus.

        yields:
            (pmid, sspan, espan, n_gram, label, entity ID)-tuples
        """
        craft_path = self.config['paths']['craft_xml']
        dicts = self.config['craft_dicts']
        test_pmids_path = self.config['paths']['test_pmids']

        # get CRAFT annotations
        ann_dict = get_annotation_dict(craft_path, iter_craft_xml, dicts)
        # get test PMID's
        test_pmids = get_pmids_from_file(test_pmids_path)

        # go through all OGER annotations
        for ann in ann_dict:
            pmid, sspan, espan, label, entity_id = ann
            n_gram = ann_dict[ann]

            # only consider test PMID's
            if pmid in test_pmids:
                yield (pmid, sspan, espan, n_gram, label, entity_id)

    def iter_oger_test_data(self):
        """Iter annotations by OGER on the CRAFT corpus.

        yields:
            (pmid, sspan, espan, n_gram, label, entity ID)-tuples
        """
        oger_path = self.config['paths']['oger_tsv']
        dicts = self.config['craft_dicts']
        test_pmids_path = self.config['paths']['test_pmids']

        # get OGER annotations
        ann_dict = get_annotation_dict(oger_path, iter_oger_tsv, dicts)
        # get test PMID's
        test_pmids = get_pmids_from_file(test_pmids_path)

        # go through all OGER annotations
        for ann in ann_dict:
            pmid, sspan, espan, label, entity_id = ann
            n_gram = ann_dict[ann]

            # only consider test PMID's
            if pmid in test_pmids:
                yield (pmid, sspan, espan, n_gram, label, entity_id)

    def iter_oger_nn_test_data(self):
        """Iter annotations of OGER filtered by the NN on the CRAFT corpus.

        yields:
            tuple: (pmid, sspan, espan, n_gram, label, entity ID)
        """
        # perform concept recognition or not
        cr = int(self.config['other']['concept_recognition'])

        # load the classifier
        c = Classifier(self.config_path)
        c.restore_model()

        # create a feature extractor
        fextr = FeatureExtractor(self.config_path)

        # back mapping from integer to ontology
        mapping = {int(self.config['classes'][o]):
                   o for o in self.config['classes']}

        # initialize feature names with empty arrays
        features = {}
        for name in c.column_names[:-1]:
            features[name] = []

        # lists of term data
        tlists = []

        # go through all OGER test annotations
        for pmid, sspan, espan, n_gram, label, entity_id in \
                self.iter_oger_test_data():
            # get list of feature values
            for i, val in enumerate(fextr.iter_feature_values(n_gram)):
                name = c.column_names[i]
                features[name].append(val)

            tlists.append((pmid, sspan, espan, n_gram, label, entity_id))

        predictions = c.classifier.predict(
                            input_fn=lambda: c.eval_input_fn(features))

        # get predictions and zip them with other annotation data
        for i, pred_dict in enumerate(predictions):
            # list of (probability, entity type label)-tuples
            probs = []
            # go through the probabilities of the entity types
            for index, p in enumerate(pred_dict['probabilities']):
                # append (probability, entity type label)
                probs.append((p, mapping[index]))

            # sort tuples by probability in decreasing order
            probs = sorted(probs, reverse=True)

            # labels to consider, by default only the one with highest prob
            labels = [probs[0][1]]

            # check if difference between the highest and second-highest
            # probability is smaller than 0.3
            threshold = float(self.config['parameters']['threshold'])
            prob_diff = probs[0][0] - probs[1][0]
            if prob_diff < threshold:
                labels.append(probs[1][1])

            # go through entity type labels
            for label in labels:
                # ignore entity types classified as normal nouns
                if label != 'nn':
                    # check if concept recognition should be performed
                    if cr:
                        # ignore entity types where OGER and NN give different
                        # labels
                        if label == tlists[i][4]:
                            yield tlists[i]
                    else:
                        yield tlists[i][:4] + (label,) + (tlists[i][-1],)


def main():
    """Create test data files."""
    test_creator = CRAFTTestDataCreator('CRAFT.ini')
    test_creator.write_to_file(test_creator.iter_gold_test_data, True)
    test_creator.write_to_file(test_creator.iter_oger_nn_test_data, False)


if __name__ == "__main__":
    main()
