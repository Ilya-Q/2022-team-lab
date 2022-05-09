import abc
import random
import tokenizers

class BaseClassifier(abc.ABC):
	@abc.abstractmethod
	def fit(self, data):
		pass

	@abc.abstractmethod
	def predict(self, instance):
		pass

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
		def ngrams(sent) -> set[str]:
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

				
