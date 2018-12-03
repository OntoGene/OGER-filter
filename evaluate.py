#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Anna Jancso, January 2018

import subprocess
import configparser as cp


def print_matrix(gold_func, sys_func, rounding=2, in_percentages=True):
    """Print matrix of recall, precision and f1-scores for all entity types.

    args:
        gold_func (generator):
            yields (pmid, sspan, espan, n_gram, label) of gold standard
        sys_func (generator):
            yields (pmid, sspan, espan, n_gram, label) of system
        rounding (int): results are rounded to n digits from the decimal point
        in_percentages (bool): results are given as percentages or as fractions
    """

    if in_percentages:
        percentage = 100
    else:
        percentage = 1

    gold = set(gold_func())
    sys = set(sys_func())

    both = gold.intersection(sys)
    only_sys = sys - gold
    only_gold = gold - sys

    label_hash = {}

    for s, name in [(both, "TP"), (only_gold, "FN"), (only_sys, "FP")]:
        for ann in s:
            label = ann[-1]

            if label in label_hash:
                label_hash[label][name] += 1
            else:
                label_hash[label] = {"TP": 0, "FP": 0, "FN": 0}
                label_hash[label][name] += 1

    print("\t\t".join(["label", "recall", "precision", "f1-score"]))
    for label in label_hash:
        TPs = label_hash[label]["TP"]
        FNs = label_hash[label]["FN"]
        FPs = label_hash[label]["FP"]

        if (TPs + FNs) == 0:
            recall = 0
        else:
            recall = TPs/(TPs + FNs)

        if (TPs + FPs) == 0:
            precision = 0
        else:
            precision = TPs/(TPs + FPs)

        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = (2*precision*recall)/(precision + recall)

        print("\t\t".join([label,
                           str(round(percentage*recall, rounding)),
                           str(round(percentage*precision, rounding)),
                           str(round(percentage*f1, rounding))]))

    total_TPs = len(both)
    total_FNs = len(only_gold)
    total_FPs = len(only_sys)

    if (total_TPs + total_FNs) == 0:
        total_recall = 0
    else:
        total_recall = total_TPs/(total_TPs + total_FNs)

    if (total_TPs + total_FPs) == 0:
        total_precision = 0
    else:
        total_precision = total_TPs/(total_TPs + total_FPs)

    if (total_precision + total_recall) == 0:
        total_f1 = 0
    else:
        total_f1 = \
            (2*total_precision*total_recall)/(total_precision + total_recall)

    print("\t\t".join(["total",
                       str(round(percentage*total_recall, rounding)),
                       str(round(percentage*total_precision, rounding)),
                       str(round(percentage*total_f1, rounding))]))


def evaluate_with_lenz_script(config_path='configs.ini'):
    config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
    config.optionxform = str
    config.read(config_path)

    gold = config['paths']['gold_test_data']
    system = config['paths']['system_test_data']
    measure = config['evaluation']['measure']
    avg = config['evaluation']['average']
    gfields = config['evaluation']['gold_fields']
    sfields = config['evaluation']['system_fields']

    template = """

    python3 term_coverage.py \
        -g {gold} \
        -a {system} \
        --{measure} \
        --{avg} \
        -G "{gfields}" \
        -A "{sfields}" \
        -prf

    """

    cmd = template.format(gold=gold,
                          system=system,
                          measure=measure,
                          avg=avg,
                          gfields=gfields,
                          sfields=sfields)

    evaluation = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE)

    print(evaluation.stdout.decode("utf-8"))


def main():
    evaluate_with_lenz_script('CRAFT.ini')


if __name__ == "__main__":
    main()
