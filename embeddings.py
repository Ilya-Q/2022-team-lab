class BoWEmbedding:
	def __init__(self, text):
		self.text = text
		self.bow_entries = {}
		self._init_bow_entries()

	def _init_bow_entries(self):
		for word in self.text.split():
			word = word.lower()
			if not word.isalpha():
				continue
			if word not in self.bow_entries:
				self.bow_entries[word] = 1
			else:
				self.bow_entries[word] += 1

	def similarity(self, other):
		common_words = set(self.bow_entries.keys()) & set(other.bow_entries.keys())
		if len(common_words) == 0:
			return 0
		else:
			return sum(self.bow_entries[word] * other.bow_entries[word] for word in common_words) / (len(self.bow_entries) * len(other.bow_entries))