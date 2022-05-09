"""This file kind of shows what i feel like would be nice to work with on a high level. Feel free to change."""
from nli_dataset import NLIDataset
from evaluation import evaluate_model
from baselines import RandomChoiceClassifier, OverlapClassifier

INSTANCES_PATH = './data/jsonl/dev.jsonl'
LABEL_PATH = './data/jsonl/dev-labels.lst'

test_data = NLIDataset(INSTANCES_PATH, LABEL_PATH)

model = RandomChoiceClassifier()
model.fit(test_data) # big meme

print("Random choice: ", evaluate_model(model, test_data))

model = OverlapClassifier(n=2)
model.fit(test_data) # so much fitting going on

print("ngram overlap: ", evaluate_model(model, test_data))