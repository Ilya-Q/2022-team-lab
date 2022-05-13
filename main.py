"""This file kind of shows what i feel like would be nice to work with on a high level. Feel free to change."""
from nli_dataset import NLIDataset
from evaluation import evaluate_model
from baselines import RandomChoiceClassifier, BoWOverlapClassifier, MajorityClassifier

TRAIN_INSTANCES_PATH = './data/jsonl/train.jsonl'
TRAIN_LABEL_PATH = './data/jsonl/train-labels.lst'
TEST_INSTANCES_PATH = './data/jsonl/dev.jsonl'
TEST_LABEL_PATH = './data/jsonl/dev-labels.lst'

train_data = NLIDataset(TRAIN_INSTANCES_PATH, TRAIN_LABEL_PATH)
test_data = NLIDataset(TEST_INSTANCES_PATH, TEST_LABEL_PATH)

# Make it iterable somehow (probably without loading it into memory completely)
for instance in test_data:
	print(instance) # This is a simple dataclass
	break

model = BoWOverlapClassifier()
model.fit(train_data)

model2 = MajorityClassifier()
model2.fit(train_data)

model3 = RandomChoiceClassifier()
model3.fit(train_data)

print('BoW overlap classifier:', evaluate_model(model, test_data))
print('Majority classifier:', evaluate_model(model2, test_data))
print('Random choice classifier:', evaluate_model(model3, test_data))