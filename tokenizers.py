import abc
from typing import List

class BaseTokenizer(abc.ABC):
    def __call__(self, sent: str) -> List[str]:
        return self.tokenize(sent)

    @abc.abstractmethod
    def tokenize(self, sent: str) -> List[str]:
        pass

class SplitTokenizer(BaseTokenizer):
    def tokenize(self, sent: str) -> List[str]:
        return sent.split()