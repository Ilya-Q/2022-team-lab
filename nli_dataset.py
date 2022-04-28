from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass(frozen=True)
class DataInstance:
	id: str
	obs1: str
	obs2: str
	hyp1: str
	hyp2: str
	label: int
	
	@staticmethod
	def from_json(json_dict: dict) -> 'DataInstance':
		return DataInstance(
			story_id=json_dict['story_id'],
			obs1=json_dict['obs1'],
			obs2=json_dict['obs2'],
			hyp1=json_dict['hyp1'],
			hyp2=json_dict['hyp2'],
			label=json_dict['label'] # This is actually part of a seperate label file (ugh) so yeah this won't work out of the box just parsing dev.jsonl
		)

class NLIDataset:
	pass

	# Or perhaps just make this as the constructor, not like we have many different formats...
	@staticmethod
	def from_jsonl(jsonl_file: str) -> 'NLIDataset':
		pass