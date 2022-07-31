import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Author: Patrick Bareiß

@dataclass(frozen=True)
class DataInstance:
	id: str
	obs1: str
	obs2: str
	hyp1: str
	hyp2: str
	label: int

class NLIDataset:

	def __init__(self, data_path: str, labels_path: str):
		self.data_path = data_path
		self.labels_path = labels_path

	def __iter__(self):
		with open(self.data_path, 'r') as data_file:
			with open(self.labels_path, 'r') as labels_file:
				for data_str, label_str in zip(data_file, labels_file):
					data = json.loads(data_str)
					label = int(label_str)
					yield DataInstance(id=data['story_id'], obs1=data['obs1'], obs2=data['obs2'], hyp1=data['hyp1'], hyp2=data['hyp2'], label=label)