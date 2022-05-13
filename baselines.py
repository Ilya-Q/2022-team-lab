import abc
import random
from embeddings import BoWEmbedding
from typing import Set
import tokenizers


class BaseClassifier(abc.ABC):
	@abc.abstractmethod
	def fit(self, data):
		pass

	@abc.abstractmethod
	def predict(self, instance):
		pass

class MajorityClassifier(BaseClassifier):
	def fit(self, data):
		first_class_count = 0
		second_class_count = 0
		for instance in data:
			if instance.label == 1:
				first_class_count += 1
			else:
				second_class_count += 1
		if first_class_count > second_class_count:	
			self.prediction = 1
		else:
			self.prediction = 2

	def predict(self, instance):
		return self.prediction

class BoWOverlapClassifier:
	def fit(self, data):
		pass

	def predict(self, instance):
		obs_embedding = BoWEmbedding(instance.obs1 + ' ' + instance.obs2)
		hyp1_embedding = BoWEmbedding(instance.hyp1)
		hyp2_embedding = BoWEmbedding(instance.hyp2)
		if obs_embedding.similarity(hyp1_embedding) > obs_embedding.similarity(hyp2_embedding):
			return 1
		else:
			return 2

class RandomChoiceClassifier(BaseClassifier):
	def fit(self, data):
		pass

	def predict(self, instance) -> int:
		return random.choice([1, 2])

class OverlapClassifier(BaseClassifier):
	tokenizer: tokenizers.BaseTokenizer
	n: int

	def __init__(self, *, n=2, tokenizer=None) -> None:
		super().__init__()
		self.n = n
		if tokenizer is not None:
			self.tokenizer = tokenizer
		else:
			self.tokenizer = tokenizers.SplitTokenizer()

	def fit(self, data):
		pass

	def predict(self, instance) -> int:
		def ngrams(sent) -> Set[str]:
			sent = ['<S>'] + sent + ['</S>']
			ret = set()
			for i in range(self.n):
				slices = [sent[j:] for j in range(i)]
				ret |= set(zip(*slices))
			return ret
		obs = ngrams(self.tokenizer(instance.obs1)) | ngrams(self.tokenizer(instance.obs2))
		hyp1 = ngrams(self.tokenizer(instance.hyp1))
		hyp2 = ngrams(self.tokenizer(instance.hyp2))
		overlap1 = obs & hyp1
		overlap2 = obs & hyp2
		if len(overlap1) >= len(overlap2):
			return 1
		else:
			return 2