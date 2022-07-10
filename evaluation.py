from tqdm import tqdm
from datasets import load_dataset, load_metric
from abc import ABC, abstractmethod

class Evaluator(ABC):

	def __init__(self):
		pass

	@abstractmethod
	def evaluate(self, model):
		pass

	def __call__(self, model):
		return self.evaluate(model)

class AccuracyEvaluator(Evaluator):

	def __init__(self, data):
		self.data = data

	def evaluate(self, model):
		total_count = 0
		correct_count = 0
		for instance in tqdm(self.data):
			total_count += 1
			prediction = model.predict(instance)
			if prediction == instance.label:
				correct_count += 1

		accuracy = correct_count / total_count
		return {'accuracy': accuracy}


class GroupEvaluator(Evaluator):
	
	def __init__(self, evaluators):
		self.evaluators = evaluators

	def evaluate(self, model):
		results = {}
		for evaluator in self.evaluators:
			results.update(evaluator.evaluate(model))
		return results

class SentimentEvaluator(Evaluator):

	POS_SENTENCE = 'The reviewer liked this movie.'
	NEG_SENTENCE = 'The reviewer did not like this movie.'
	
	def __init__(self):
		self.sst2_test_data = load_dataset('glue', 'sst2')['validation']
		self.sst2_metric = load_metric('glue', 'sst2')
		#from sentence_transformers import SentenceTransformer
		#self.sus = SentenceTransformer('all-MiniLM-L6-v2')

	def _predict_sentiment(self, model, instance):
		"""
		from sentence_transformers import util
		
		sent_embed = self.sus.encode(instance['sentence'])

		pos_embed = self.sus.encode(self.POS_SENTENCE)
		neg_embed = self.sus.encode(self.NEG_SENTENCE)

		if util.cos_sim(sent_embed, pos_embed) > 0:
			return 1
		else:
			return 0
		"""

		embedder = model.embedder

		pos_embed = embedder.occurs_after(
			embedder.encode(instance['sentence'], convert_to_tensor=True),
			-embedder.encode(self.NEG_SENTENCE, convert_to_tensor=True)
		)

		neg_embed = embedder.occurs_after(
			embedder.encode(instance['sentence'], convert_to_tensor=True),
			embedder.encode(self.NEG_SENTENCE, convert_to_tensor=True)
		)

		if embedder.consistent(pos_embed) > embedder.consistent(neg_embed):
			return 1
		else:
			return 0

	def evaluate(self, model):
		for instance in tqdm(self.sst2_test_data):
			prediction = self._predict_sentiment(model, instance)
			
			self.sst2_metric.add(predictions=prediction, references=instance['label'])

		return self.sst2_metric.compute()
		

class NLIEvaluator(Evaluator):
	
	def __init__(self):
		self.mnli_test_data = load_dataset('glue', 'mnli')['validation_matched']
		self.mnli_metric = load_metric('glue', 'mnli')

	def _predict_entailment(self, model, instance, threshold=0.5):
		embedder = model.embedder

		entail_embed = embedder.occurs_after(
			embedder.encode(instance['premise'], convert_to_tensor=True),
			embedder.encode(instance['hypothesis'], convert_to_tensor=True)
		)

		#print(embedder.consistent(entail_embed))
		if embedder.consistent(entail_embed) > threshold:
			return 0
		elif embedder.consistent(entail_embed) < -threshold:
			return 2
		else:
			return 1

	def evaluate(self, model):
		for instance in tqdm(self.mnli_test_data):
			prediction = self._predict_entailment(model, instance)
			self.mnli_metric.add(predictions=prediction, references=instance['label'])

		return self.mnli_metric.compute()
