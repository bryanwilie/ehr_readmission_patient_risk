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

from src.data_utils.biodataset import BioDataset
from src.modules.trainer import MultilabelTrainer
from src.utils.load_dataset import load_tokenized_dataset
from src.utils.load_model import load_model_and_tokenizer
from src.utils.args_helper import DataTrainingArguments, ModelArguments, EHRTrainingArguments


def main(model_args, data_args, training_args):
    
    # Model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # Dataset
#     encodings, labels = load_tokenized_dataset(data_args, tokenizer)

#     train_dataset = BioDataset(encodings, labels)
#     val_dataset = BioDataset(encodings, labels)
#     test_dataset = BioDataset(encodings, labels)
    
    # Dataset Preparation

    ## Load Metadata
    create_dtm = list(np.arange(0,100,1))
    diff_in_hour_create_dtm = list(np.arange(0,100,1))
    upper_note_type = ['D']*100
    consult_text = ['consult_text']*100
    pseudo_record_key = ['dasdSacaASDQF21241WDAS3']*100
    pseudo_episode_key = ['dasdSacaASDQF21241WDAS3']*100
    pseudo_patient_key = ['dasdSacaASDQF21241WDAS3']*100

    first_relative_timestamp = list(np.arange(1000,2000,10))
    features = [['0.0','0.0','0.0','0.0','0.0']]*100
    texts = ['Post-injury 5 days only<CR><LF> with suspected ..']*100
    diagnosis = [-100]*100
    mortality = [0]*100
    relative_readmission = list(np.arange(0,100,1))
    next_diagnosis = [-100]*100
    next_mortality = [1]*100
    next_relative_readmission = list(np.arange(100,200,1))

    last_pseudo_episode_key = pseudo_episode_key
    num_episode = 10
    num_labelled_episode = 15
    last_timestamp = [datetime.datetime.now().date()]*100
    no_train = [True]*100

    ### Load patient_id
    df = pd.DataFrame({'pseudo_patient_key':pseudo_patient_key,
                       'pseudo_episode_key':pseudo_episode_key,
                       'first_relative_timestamp':first_relative_timestamp,
                        'features':features,
                        'texts':texts,
                        'diagnosis':diagnosis,
                        'mortality':mortality,
                        'relative_readmission':relative_readmission,
                        'next_diagnosis':next_diagnosis,
                        'next_mortality':next_mortality,
                        'next_relative_readmission':next_relative_readmission})

    ### Load valid file
    valid_patient_df = pd.DataFrame({'pseudo_patient_key':pseudo_patient_key,
                                     'last_pseudo_episode_key':last_pseudo_episode_key,
                                     'relative_readmission':relative_readmission,
                                     'num_episode':num_episode,
                                     'num_labelled_episode':num_labelled_episode,
                                     'last_timestamp':last_timestamp,
                                     'no_train':no_train})

    ### Load test file
    test_patient_df = pd.DataFrame({'pseudo_patient_key':pseudo_patient_key,
                                    'last_pseudo_episode_key':last_pseudo_episode_key,
                                    'relative_readmission':relative_readmission,
                                    'num_episode':num_episode,
                                    'num_labelled_episode':num_labelled_episode,
                                    'last_timestamp':last_timestamp,
                                    'no_train':no_train})

    ### Load clinical notes file
    clinical_notes_df = pd.DataFrame({'create_dtm':create_dtm,
                                      'diff_in_hour_create_dtm':diff_in_hour_create_dtm,
                                      'upper_(note_type)':upper_note_type,
                                      'pseudo_record_key':pseudo_record_key,
                                      'pseudo_episode_key':pseudo_episode_key,
                                      'pseudo_patient_key':pseudo_patient_key})


    ## Instantiate dataset
    ### Filter patient_id with no_train = True

    ### Load instantiated EHR dataset train (patient_id_list, valid_test_id_list, clinical_notes_df)
    patient_ids = df.pseudo_patient_key.tolist()
    valid_ids = valid_patient_df.pseudo_patient_key.tolist()
    test_ids = test_patient_df.pseudo_patient_key.tolist()
    valid_test_ids = valid_ids + test_ids

    train_dataset = EHRDataset(patient_ids, valid_test_ids, clinical_notes_df)

    ### Load instantiated EHR dataset valid (valid_id_list, None, clinical_notes_df) and test (test_id_list, None, clinical_notes_df)
    valid_dataset = EHRDataset(valid_ids, None, clinical_notes_df)
    test_dataset = EHRDataset(test_ids, None, clinical_notes_df)
    
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