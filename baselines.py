import random

class RandomChoiceClassifier:
	def fit(self, data):
		pass

	def predict(self, instance):
		return random.choice([1, 2])