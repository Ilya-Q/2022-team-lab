import abc
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models

class BaseSentenceEmbedder(SentenceTransformer, abc.ABC):
    def __init__(self, *args, **kwargs):
        if "backbone" in kwargs:
            bb_name = kwargs["backbone"]
            del kwargs["backbone"]
            kwargs["modules"] = self._build_modules_list(models.Transformer(bb_name))
        super().__init__(*args, **kwargs)

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

    # TODO: Decide: Should these be based on sentences or on embeddings? (i.e. do we embed them inside this method or does the user have to do it before)
    def occurs_after(self, sentence1, sentence2):
        return torch.kron(sentence1, sentence2)

    def consistent(self, sentence):
        return sentence @ torch.tile(torch.tensor([1., -1.]).to(self._target_device), (len(sentence) // 2,)) # Can also vary the pattern (i.e. 1 -1 -1 1 or 1 1 -1 -1)

