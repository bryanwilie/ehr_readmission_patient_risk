import os
import time
import shutil
import collections
from typing import Any, NewType
import logging
import copy
import math

from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from datasets import load_metric
from tqdm.std import Bar
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    EvalPrediction,
    default_data_collator,
    HfArgumentParser
)
import wandb

from src.data_utils import BioDataset, EHRDataset, EHREpisodeDataset
from src.modules.trainer import MultilabelTrainer
from src.utils.load_dataset import load_tokenized_dataset
from src.utils.load_model import load_model_and_tokenizer
from src.utils.args_helper import DataTrainingArguments, ModelArguments, EHRTrainingArguments


def main(model_args, data_args, training_args):
    
    # Model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # Dataset
    encodings, labels = load_tokenized_dataset(data_args, tokenizer)

    train_dataset = BioDataset(encodings, labels)
    val_dataset = BioDataset(encodings, labels)
    test_dataset = BioDataset(encodings, labels)
    
    # Train
    trainer = MultilabelTrainer(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        train_dataset=train_dataset,     # training dataset
        eval_dataset=val_dataset,        # evaluation dataset
        args=training_args,              # training arguments, defined above 
        tokenizer=tokenizer
    )
    trainer.train()
    

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EHRTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)