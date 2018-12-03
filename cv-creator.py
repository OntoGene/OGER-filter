"""Split data for cross-validation."""

# Anna Jancso

import os
import shutil

orig_train_ids_path = 'shuffled'

ids = []

for line in open(orig_train_ids_path):
    line = line.rstrip()
    ids.append(line)

folds = 5

folds_ids = [([], []) for i in range(folds)]

for i, id_ in enumerate(ids):
    test_fold = i % folds
    folds_ids[test_fold][1].append(id_)

    for fold in range(folds):
        if fold != test_fold:
            folds_ids[fold][0].append(id_)


base_dir = 'cross-validation'

if os.path.isdir(base_dir):
    shutil.rmtree(base_dir)

os.mkdir(base_dir)

for i, fold in enumerate(folds_ids):
    fold_dir = os.path.join(base_dir, 'fold'+str(i))
    os.mkdir(fold_dir)

    train_ids_filename = 'train_ids'
    train_ids_filepath = os.path.join(fold_dir, train_ids_filename)
    train_ids_file = open(train_ids_filepath, 'w')

    for id_ in sorted(fold[0]):
        train_ids_file.write(id_)
        train_ids_file.write('\n')

    test_ids_filename = 'test_ids'
    test_ids_filepath = os.path.join(fold_dir, test_ids_filename)
    test_ids_file = open(test_ids_filepath, 'w')

    for id_ in sorted(fold[1]):
        test_ids_file.write(id_)
        test_ids_file.write('\n')

    train_ids_file.close()
    test_ids_file.close()
