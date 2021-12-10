import os
import time
import shutil
import collections
from typing import Any, NewType
import logging
import copy
import math
import json

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

from src.data_utils import BioDataset, EHRDataset, EHREpisodeDataset
from src.modules.trainer import MultilabelTrainer
from src.utils.load_dataset import load_tokenized_dataset
from src.utils.load_model import load_model_and_tokenizer
from src.utils.args_helper import DataTrainingArguments, ModelArguments, EHRTrainingArguments


def main(model_args, data_args, training_args):
    
    # Model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, data_args)
    
    # Dataset Preparation
    ## Load Metadata

    ### Load patient_id
    train_patient_id_list = json.load(open(data_args.train_path,'r'))
    valid_patient_id_list = json.load(open(data_args.valid_path,'r'))
    test_patient_id_list = json.load(open(data_args.test_path,'r'))

    ## Instantiate dataset
    train_dataset = EHREpisodeDataset(data_args.patient_base_path, train_patient_id_list, tokenizer, data_args.max_text_len, data_args.max_duration, use_tabular=data_args.data_type=='text_tabular')
    valid_dataset = EHREpisodeDataset(data_args.patient_base_path, valid_patient_id_list, tokenizer, data_args.max_text_len, data_args.max_duration, use_tabular=data_args.data_type=='text_tabular')
    test_dataset = EHREpisodeDataset(data_args.patient_base_path, test_patient_id_list, tokenizer, data_args.max_text_len, data_args.max_duration, use_tabular=data_args.data_type=='text_tabular')
    
    # Train
    trainer = MultilabelTrainer(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        train_dataset=train_dataset,     # training dataset
        eval_dataset=valid_dataset,        # evaluation dataset
        args=training_args,              # training arguments, defined above 
        tokenizer=tokenizer
    )
    trainer.train()
    

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EHRTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)