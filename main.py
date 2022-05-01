"""This file kind of shows what i feel like would be nice to work with on a high level. Feel free to change."""
from nli_dataset import NLIDataset
from evaluation import evaluate_model
from baselines import RandomChoiceClassifier

INSTANCES_PATH = './data/jsonl/dev.jsonl'
LABEL_PATH = './data/jsonl/dev-labels.lst'

test_data = NLIDataset(INSTANCES_PATH, LABEL_PATH)

# Make it iterable somehow (probably without loading it into memory completely)
for instance in test_data:
	print(instance) # This is a simple dataclass
	break

model = RandomChoiceClassifier()
model.fit(test_data) # big meme

print(evaluate_model(model, test_data))