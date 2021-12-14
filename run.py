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
from src.modules.multi_label_trainer import MultilabelTrainer
from src.utils.load_dataset import load_tokenized_dataset
from src.utils.load_model import load_model_and_tokenizer
from src.utils.args_helper import DataTrainingArguments, ModelArguments, EHRTrainingArguments
from datasets import load_metric

def main(model_args, data_args, training_args):
    
    # Model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, data_args)
    
    # Dataset Preparation
    ### Load patient_id
    train_patient_id_list = json.load(open(data_args.train_path,'r'))
    valid_patient_id_list = json.load(open(data_args.valid_path,'r'))
    test_patient_id_list = json.load(open(data_args.test_path,'r'))

    ## Instantiate dataset
    train_dataset = EHREpisodeDataset(data_args.patient_base_path, train_patient_id_list, tokenizer, data_args.max_text_len, data_args.max_duration, use_tabular=data_args.dataset_type=='text_tabular')
    valid_dataset = EHREpisodeDataset(data_args.patient_base_path, valid_patient_id_list, tokenizer, data_args.max_text_len, data_args.max_duration, use_tabular=data_args.dataset_type=='text_tabular')
    test_dataset = EHREpisodeDataset(data_args.patient_base_path, test_patient_id_list, tokenizer, data_args.max_text_len, data_args.max_duration, use_tabular=data_args.dataset_type=='text_tabular')
    
    ## Define Metrics
    acc_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    num_multi_labels = model.config.num_multi_labels
    tags = {0: 'diag', 1: 'readmission', 2: 'mortality'}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        rem_logits = logits
        metrics = {}
        for i, num_label in enumerate(num_multi_labels):
            class_logits = rem_logits[:,:num_label]
            rem_logits = rem_logits[:,num_label:]

            class_pred =  np.argmax(class_logits, axis=-1)
            acc = acc_metric.compute(predictions=class_pred, references=labels[:,i])
            f1 = f1_metric.compute(predictions=class_pred, references=labels[:,i], average="macro")

            metrics[f'{tags[i]}_acc'] = acc
            metrics[f'{tags[i]}_f1'] = f1
        return metrics

    # Train
    trainer = MultilabelTrainer(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        train_dataset=train_dataset,     # training dataset
        eval_dataset=valid_dataset,      # evaluation dataset
        args=training_args,              # training arguments, defined above 
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    pred_results = trainer.predict(test_dataset)
    
    # Dump Result
    json.dump(pred_results, open('./results/{}_{}_{}_{}_{}_lr-{}_bs-{}_ep-{}_ws-{}_wd-{}'.format(
        model_args.model, data_args.dataset_type, data_args.max_text_len, data_args.max_duration,
        training_args.learning_rate, training_args.train_batch_size, training_args.num_train_epochs,
        training_args.warmup_steps, training_args.weight_decay, training_args.per_device_train_batch_size
    ), 'w'))
    
    # Print Result
    print('== Test Result ==')
    print(pred_results) 
    
if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EHRTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)