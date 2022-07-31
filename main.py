from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from classifiers import SentenceEmbedClassifier, RandomChoiceClassifier, OverlapClassifier, MajorityClassifier
from nli_dataset import NLIDataset
from evaluation import AccuracyEvaluator, SentimentEvaluator, NLIEvaluator
import argparse
from models import KroeneckerSentenceEmbedder, SimpleSentenceEmbedder, MatrixSentenceEmbedder, MatrixSentenceEmbedderSimpleConsistency
from losses import MSELoss, CELoss

import fire

# Author: Patrick Bareiß & Ilya Kuryanov

TRAIN_INSTANCES_PATH = './data/jsonl/train.jsonl'
TRAIN_LABEL_PATH = './data/jsonl/train-labels.lst'
TEST_INSTANCES_PATH = './data/jsonl/dev.jsonl'
TEST_LABEL_PATH = './data/jsonl/dev-labels.lst'

train_data = NLIDataset(TRAIN_INSTANCES_PATH, TRAIN_LABEL_PATH)
test_data = NLIDataset(TEST_INSTANCES_PATH, TEST_LABEL_PATH)

class CommandLineInterface:

    def __init__(self,
        model_path='./data/embeddings',
        model_type='kronecker',
        train_data_instances_path=TRAIN_INSTANCES_PATH,
        train_data_label_path=TRAIN_LABEL_PATH,
        test_data_instances_path=TEST_INSTANCES_PATH,
        test_data_label_path=TEST_LABEL_PATH
    ):
        self.model_path = model_path
        self.model_type = model_type

        self.train_data = NLIDataset(train_data_instances_path, train_data_label_path)
        self.test_data = NLIDataset(test_data_instances_path, test_data_label_path)

    def _create_model(self, *args, **kwargs):
        if self.model_type == 'kronecker':
            return KroeneckerSentenceEmbedder(*args, **kwargs)
        elif self.model_type == 'simple':
            return SimpleSentenceEmbedder(*args, **kwargs)
        elif self.model_type == 'matrix':
            return MatrixSentenceEmbedder(*args, **kwargs)
        elif self.model_type == 'matrix-simple-consistency':
            return MatrixSentenceEmbedderSimpleConsistency(*args, **kwargs)
        else:
            raise RuntimeError(f'Model type "{self.model_type}" not found.')
        
    def _create_loss(self, loss_type, *args, **kwargs):
        if loss_type == 'ce':
            return CELoss(*args, **kwargs)
        elif loss_type == 'mse':
            return MSELoss(*args, **kwargs)
        else:
            raise RuntimeError(f'Loss type "{loss_type}" not found.')

    def eval(self, task="anli", baseline=None):
        if baseline is None:
            model = self._create_model(self.model_path)
            classifier = SentenceEmbedClassifier(model)
        else:
            if baseline == 'random':
                classifier = RandomChoiceClassifier()
            elif baseline == 'overlap':
                classifier = OverlapClassifier(n=2)
            elif baseline == 'majority':
                classifier = MajorityClassifier()
                classifier.fit(self.train_data)
            else:
                raise RuntimeError(f'Baseline "{baseline}" not found.')
        
        if task == "anli":
            evaluate_model = AccuracyEvaluator(self.test_data)
        elif task == "sentiment":
            evaluate_model = SentimentEvaluator()
        elif task == 'nli':
            evaluate_model = NLIEvaluator()
        else:
            raise RuntimeError(f'Evaluation task "{task}" not found.')

        print(f'Result on task "{task}":', evaluate_model(classifier))

    def train(self, backbone='bert-large-uncased', loss_type='ce', device='cuda', epochs=1, batch_size=16):
        model = self._create_model(
            backbone=backbone,
            device=device
        )
        train_samples = [InputExample(
            texts=[
                instance.obs1,
                instance.obs2,
                instance.hyp1,
                instance.hyp2
            ],
            label=float(instance.label)) for instance in self.train_data
        ]
        train_dataloader = DataLoader(
            train_samples,
            batch_size=batch_size,
            shuffle=True
        )
        train_loss = self._create_loss(loss_type, model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100
        )
        model.save(self.model_path)

    def __call__(self, train=False, eval=True, **kwargs):
        if train:
            self.train(**kwargs)

        if eval:
            self.eval(**kwargs)
        
if __name__ == '__main__':
    fire.Fire(CommandLineInterface)
