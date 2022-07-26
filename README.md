# An exploration of noncommutative sentence embeddings for Abductive Natural Language Reasoning

---

## Overview

This repository contains the code for the Team Laboratory project done by Patrick Bareiss and Ilya Kuryanov.

TODO: BRIEF TASK AND APPROACH DESCRIPTION GOES HERE

## Prerequisites

To use the code, first [download](https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip) the ART dataset from AllenAI, and unzip it into the `./data` directory:
```
$ wget https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip
$ mkdir data && unzip alphanli-train-dev.zip -d data
```

You will also need to install the dependencies from `requirements.txt`:
```
$ pip install -r requirements.txt
```

## Usage

`main.py` provides the CLI both for training different models and for evaluating them.

The syntax for training models looks as follows:
```
$ python main.py train --loss_type ce --model_type matrix --model_path path/to/saved/model --device cuda --backbone bert-base-uncased --batch_size 16 --epochs 1
```

There are three options for `--model_type` (see the paper for more details):
- `simple` is the baseline commutative model that uses vector summation as the combination operator and elementwise summation for consistency checking
- `kronecker` uses the Kronecker product as the combination operator and alternating summation for consistency checking
- `matrix` uses matrix multiplication as the combination operator and a 2-dimensional CNN for consistency checking

`--loss_type` can be either `ce` for cross-entropy or `mse` for mean square error.

`--backbone` is the base SBert or HuggingFace model for the base embeddings that will be fine-tuned.

This is the command to evaulate a trained model:
```
python main.py eval --model_type kronecker --model_path path/to/saved/model
```

Note that the `--model_type` must be specified both in evaluation and training.

The `--task` argument allows for evaluation on tasks other than aNLI, and can be either `sentiment`, to evaluate on the SST2 dataset, or `nli`, to evaluate on MNLI. Again, see the paper for more detail.

You can also specify a `--baseline` argument instead of `--model_type` and `--model_path` to compare the results to a `random`, `majority`, or `overlap` baseline.