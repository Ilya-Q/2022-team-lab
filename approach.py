from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from classifiers import SentenceEmbedClassifier
from nli_dataset import NLIDataset
from evaluation import AccuracyEvaluator, SentimentEvaluator, NLIEvaluator
import argparse
from models import KroeneckerSentenceEmbedder, SimpleSentenceEmbedder, MatrixSentenceEmbedder
from losses import MSELoss, CELoss

import fire

TRAIN_INSTANCES_PATH = './data/jsonl/train.jsonl'
TRAIN_LABEL_PATH = './data/jsonl/train-labels.lst'
TEST_INSTANCES_PATH = './data/jsonl/dev.jsonl'
TEST_LABEL_PATH = './data/jsonl/dev-labels.lst'

train_data = NLIDataset(TRAIN_INSTANCES_PATH, TRAIN_LABEL_PATH)
test_data = NLIDataset(TEST_INSTANCES_PATH, TEST_LABEL_PATH)

# might be a good idea to just have training and evaluation in different files
# especially since we already have evaluation.py
# when we come up with more losses and models, these may also be specified as args

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
        else:
            raise RuntimeError(f'Model type "{self.model_type}" not found.')
        
    def _create_loss(self, loss_type, *args, **kwargs):
        if loss_type == 'ce':
            return CELoss(*args, **kwargs)
        elif loss_type == 'mse':
            return MSELoss(*args, **kwargs)
        else:
            raise RuntimeError(f'Loss type "{loss_type}" not found.')

    def eval(self):
        model = self._create_model(self.model_path)
        classifier = SentenceEmbedClassifier(model)

        #evaluate_model = AccuracyEvaluator(self.test_data)
        #evaluate_model = SentimentEvaluator()
        evaluate_model = NLIEvaluator()

        print('Approach:', evaluate_model(classifier))

    def train(self, backbone='bert-large-uncased', loss_type='ce', device='cuda'):
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
            batch_size=16,
            shuffle=True
        )
        train_loss = self._create_loss(loss_type, model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
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
