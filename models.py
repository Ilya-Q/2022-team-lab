import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models
from sbert_modules.CNNConsistency import CNNConsistency

class BaseSentenceEmbedder(SentenceTransformer, abc.ABC):
    def __init__(self, *args, **kwargs):
        if "backbone" in kwargs:
            bb_name = kwargs["backbone"]
            del kwargs["backbone"]
            kwargs["modules"] = self._build_modules_list(models.Transformer(bb_name))
        super().__init__(*args, **kwargs)
        self._seq_modules = [module for module in self._modules.keys() if module not in self._nonseq_modules]

    def forward(self, input):
        for module_name in self._seq_modules:
            input = self._modules[module_name](input)
        return input

    @abc.abstractmethod
    def _build_modules_list(self, backbone):
        pass

    @abc.abstractmethod
    def occurs_after(self, sentence1, sentence2):
        pass

    @abc.abstractmethod
    def consistent(self, sentence):
        pass 

class KroeneckerSentenceEmbedder(BaseSentenceEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_modules_list(self, backbone):
        pooling_model = models.Pooling(backbone.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
        return [backbone, pooling_model, dense_model]

    def occurs_after(self, sentence1, sentence2):
        return torch.kron(sentence1, sentence2)

    def consistent(self, sentence):
        return sentence @ torch.tile(torch.tensor([1., -1.]).to(self._target_device), (len(sentence) // 2,)) # Can also vary the pattern (i.e. 1 -1 -1 1 or 1 1 -1 -1)

class SimpleSentenceEmbedder(BaseSentenceEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_modules_list(self, backbone):
        pooling_model = models.Pooling(backbone.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
        return [backbone, pooling_model, dense_model]

    def occurs_after(self, sentence1, sentence2):
        return sentence1 + sentence2

    def consistent(self, sentence):
        return sentence.sum()

class MatrixSentenceEmbedder(BaseSentenceEmbedder):
    def __init__(self, *args, **kwargs):
        self._nonseq_modules = {'consistency_cnn'}
        super().__init__(*args, **kwargs)
        self.consistency_cnn = CNNConsistency()
    
    def _build_modules_list(self, backbone):
        pooling_model = models.Pooling(backbone.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
        return [backbone, pooling_model, dense_model]

    def occurs_after(self, sentence1, sentence2):
        return torch.flatten(
            torch.matmul(
                torch.reshape(sentence1, (16,16)),
                torch.reshape(sentence2, (16,16))
            )
        )
          
    def consistent(self, sentence):
        return self.consistency_cnn(sentence)


