from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from classifiers import SentenceEmbedClassifier
from nli_dataset import NLIDataset
from evaluation import evaluate_model
import argparse
from models import KroeneckerSentenceEmbedder
from losses import MSELoss

TRAIN_INSTANCES_PATH = './data/jsonl/train.jsonl'
TRAIN_LABEL_PATH = './data/jsonl/train-labels.lst'
TEST_INSTANCES_PATH = './data/jsonl/dev.jsonl'
TEST_LABEL_PATH = './data/jsonl/dev-labels.lst'

train_data = NLIDataset(TRAIN_INSTANCES_PATH, TRAIN_LABEL_PATH)
test_data = NLIDataset(TEST_INSTANCES_PATH, TEST_LABEL_PATH)

# might be a good idea to just have training and evaluation in different files
# especially since we already have evaluation.py
parser = argparse.ArgumentParser()
parser.add_argument('--train', default=False, action='store_true') 
parser.add_argument('--no-train', dest='train', action='store_false') 
parser.add_argument('--eval', default=True, action='store_true') 
parser.add_argument('--no-eval', dest='eval', action='store_false') 
parser.add_argument('--model_path', default='./data/embeddings')
parser.add_argument('--backbone', default='bert-large-uncased')
parser.add_argument('--device', default='cuda')
# when we come up with more losses and models, these may also be specified as args

        
if __name__ == '__main__':
    args = parser.parse_args()
    if args.train:
        model = KroeneckerSentenceEmbedder(
            backbone=args.backbone,
            device=args.device
        )
        train_samples = [InputExample(
            texts=[
                instance.obs1,
                instance.obs2,
                instance.hyp1,
                instance.hyp2
            ],
            label=float(instance.label)) for instance in train_data
        ]
        train_dataloader = DataLoader(
            train_samples,
            batch_size=16,
            shuffle=True
        )
        train_loss = MSELoss(model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100
        )
        model.save(args.model_path)
    else:
        model = KroeneckerSentenceEmbedder(args.model_path)

    if args.eval:
        classifier = SentenceEmbedClassifier(model)
        print('Approach:', evaluate_model(classifier, test_data))
