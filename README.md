# OGER-filter

Plugin for [OGER](https://github.com/OntoGene/OGER) to filter and re-classify
some of its annotations using a feed-forward neural network.

### Prerequisites
The OGER-filter runs on Python 3 only.

The following third-party libraries need to be installed (e.g. via pip):

* gensim
* hunspell
* lxml
* tensorflow

### Required Resources

* [CRAFT corpus](http://bionlp-corpora.sourceforge.net/CRAFT/)
* [Biomedical Word Embeddings](https://github.com/cambridgeltl/BioNLP-2016)
* Hunspell Dictionary (e.g. the default US dictionary)
* [NLTK's Stopword list](https://www.nltk.org/nltk_data/)

All paths to these resources must be given in a ini file. An example ini
file is provided (see CRAFT.ini). In addition, a list of PMIDs has to be
specified both for training and testing.

### Example Usage

```python
from create_train_data import CRAFTTrainingDataCreator
from create_test_data import CRAFTTestDataCreator
from classifier import Classifier
from evaluate import evaluate_with_lenz_script


config_path = 'CRAFT.ini'

# --------------------------------------
print("Creating CRAFT training data...")
train_creator = CRAFTTrainingDataCreator(config_path)
train_creator.create_train_data()

# --------------------------------------
print("Training CRAFT classifier...")
classifier = Classifier(config_path)
classifier.create_model()
classifier.train_model()

# --------------------------------------
print("Creating CRAFT test data...")
test_creator = CRAFTTestDataCreator(config_path)
test_creator.write_to_file(test_creator.iter_gold_test_data, True)
test_creator.write_to_file(test_creator.iter_oger_nn_test_data, False)

# --------------------------------------
evaluate_with_lenz_script(config_path)
```
