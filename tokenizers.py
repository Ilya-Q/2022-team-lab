import abc

class BaseTokenizer(abc.ABC):
    def __call__(self, sent: str) -> list[str]:
        return self.tokenize(sent)

    @abc.abstractmethod
    def tokenize(self, sent: str) -> list[str]:
        pass

class SplitTokenizer(BaseTokenizer):
    def tokenize(self, sent: str) -> list[str]:
        return sent.split()