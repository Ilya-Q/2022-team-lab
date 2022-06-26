import torch
import torch.nn as nn
import numpy as np
from baselines import BaseClassifier
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
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

class KroneckerLoss(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()

    def occurs_after(self, embed1, embed2):
        return torch.kron(embed1, embed2)

    def consistent(self, embed):
        return embed @ torch.tile(torch.tensor([1., -1.]).cuda(), (len(embed) // 2,)) # Can also vary the pattern (i.e. 1 -1 -1 1 or 1 1 -1 -1)

    def forward(self, sentence_features, labels):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        total_loss = torch.tensor(0., requires_grad=True).cuda()
        for i, label in enumerate(labels):
            obs1, obs2, hyp1, hyp2 = embeddings[0][i], embeddings[1][i], embeddings[2][i], embeddings[3][i]
            first_score = self.consistent(self.occurs_after(self.occurs_after(obs1, hyp1), obs2))
            second_score = self.consistent(self.occurs_after(self.occurs_after(obs1, hyp2), obs2))

            if label == 1:
                #loss = second_score - first_score
                loss = self.loss(torch.stack([first_score, second_score]), torch.tensor([1., -1.]).cuda())
            elif label == 2:
                loss = self.loss(torch.stack([first_score, second_score]), torch.tensor([-1., 1.]).cuda())
                #loss = first_score - second_score

            total_loss += loss

            # Add terms that obs1 -> obs2 should be more consistent than obs2 -> obs1 for example (also with hypotheses)

            # Add terms that obs across things should generally have similar consistency

        return total_loss

        
        

# This should probably move to a separate file...
class SentenceEmbedder:
    def __init__(self, embedder_path):
        super().__init__()
        self.embedder_path = embedder_path

    def fit(self, data):
        word_embedding_model = models.Transformer('bert-base-uncased')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

        new_model = SentenceTransformer(modules=[
            word_embedding_model,
            pooling_model,
            dense_model
        ])

        train_samples = [InputExample(texts=[instance.obs1, instance.obs2, instance.hyp1, instance.hyp2], label=float(instance.label)) for instance in data]
        train_dataloader = DataLoader(train_samples, batch_size=16, shuffle=True)
        train_loss = KroneckerLoss(new_model) # THIS IS THE MAIN THING WE WOULD LIKE TO CHANGE SO THAT IT IS NOT COSINE SIMILARITY

        new_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

        new_model.save(self.embedder_path)


    @cached_property
    def embedder(self):
        return SentenceTransformer(self.embedder_path)

    # TODO: Decide: Should these be based on sentences or on embeddings? (i.e. do we embed them inside this method or does the user have to do it before)
    def occurs_after(self, sentence1, sentence2):
        return np.kron(sentence1, sentence2)

    def consistent(self, sentence):
        return sentence @ np.tile([1, -1], len(sentence) // 2) # Can also vary the pattern (i.e. 1 -1 -1 1 or 1 1 -1 -1)

    def embed(self, sentence):
        return self.embedder.encode(sentence)

if __name__ == '__main__':
    model = SentenceEmbedClassifier(embedder_path='./data/embeddings')
    model.fit(train_data)

    print('Approach:', evaluate_model(model, test_data))
