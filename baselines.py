import random
from embeddings import BoWEmbedding

class RandomChoiceClassifier:
	def fit(self, data):
		# COmpute hash here and use that as random seed
		pass

	def predict(self, instance):
		return random.choice([1, 2])

class MajorityClassifier:
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