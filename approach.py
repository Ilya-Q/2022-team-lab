import torch
import numpy as np
from baselines import BaseClassifier
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from nli_dataset import NLIDataset
from evaluation import evaluate_model
from functools import cached_property

TRAIN_INSTANCES_PATH = './data/jsonl/train.jsonl'
TRAIN_LABEL_PATH = './data/jsonl/train-labels.lst'
TEST_INSTANCES_PATH = './data/jsonl/dev.jsonl'
TEST_LABEL_PATH = './data/jsonl/dev-labels.lst'

train_data = NLIDataset(TRAIN_INSTANCES_PATH, TRAIN_LABEL_PATH)
test_data = NLIDataset(TEST_INSTANCES_PATH, TEST_LABEL_PATH)

class SentenceEmbedClassifier(BaseClassifier):
    def __init__(self, embedder_path='./sentence_embed/') -> None:
        super().__init__()
        self.embedder_path = embedder_path

    def fit(self, data):
        self.embedder.fit(data)

    @cached_property
    def embedder(self):
        return SentenceEmbedder(self.embedder_path)

    def occurs_after(self, sentence1, sentence2):
        return self.embedder.occurs_after(sentence1, sentence2)

    def consistent(self, sentence):
        return self.embedder.consistent(sentence)

    def embed(self, sentence):
        return self.embedder.embed(sentence)

    def predict(self, instance):
        # Embed the two observations and hypothesis to torch tensors
        obs1_embed, obs2_embed, hyp1_embed, hyp2_embed = [self.embed(sentence) for sentence in [instance.obs1, instance.obs2, instance.hyp1, instance.hyp2]]
        
        story1_embed = self.occurs_after(self.occurs_after(obs1_embed, hyp1_embed), hyp2_embed)
        story2_embed = self.occurs_after(self.occurs_after(obs1_embed, hyp2_embed), hyp2_embed)

        if self.consistent(story1_embed) > self.consistent(story2_embed):
            return 1
        else:
            return 2

# This should probably move to a separate file...
class SentenceEmbedder:
    def __init__(self, embedder_path):
        super().__init__()
        self.embedder_path = embedder_path

    def fit(self, data):
        new_model = SentenceTransformer('bert-base-nli-mean-tokens')

        train_samples = [InputExample(texts=[instance.obs1, instance.obs2, instance.hyp1, instance.hyp2], label=float(instance.label)) for instance in data]
        train_dataloader = DataLoader(train_samples, batch_size=16, shuffle=True)
        train_loss = losses.CosineSimilarityLoss(new_model) # THIS IS THE MAIN THING WE WOULD LIKE TO CHANGE SO THAT IT IS NOT COSINE SIMILARITY

        new_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

        new_model.save(self.embedder_path)


    @cached_property
    def embedder(self):
        return SentenceTransformer(embedder_path)

    def occurs_after(self, sentence1, sentence2):
        return self.embedder.encode(sentence1) @ self.embedder.encode(sentence2)

    def consistent(self, sentence):
        return self.embedder.encode(sentence) @ self.embedder.encode(sentence)

    def embed(self, sentence):
        return self.embedder.encode(sentence)

if __name__ == '__main__':
    model = SentenceEmbedClassifier(embedder_path='./data/embeddings')
    model.fit(train_data)

    print('Approach:', evaluate_model(model, test_data))
