"""This file kind of shows what i feel like would be nice to work with on a high level. Feel free to change."""
from nli_dataset import NLIDataset

DATASET_PATH = './data/jsonl/dev.jsonl'

train_data = NLIDataset(DATASET_PATH)

# Make it iterable somehow (probably without loading it into memory completely)
for instance in train_data:
	print(instance) # This is a simple dataclass