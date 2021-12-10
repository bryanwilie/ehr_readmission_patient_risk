import numpy as np
import pandas as pd
from itertools import chain
import torch
from torch.utils.data import Dataset

###
# EHR Dataset
###
class EHREpisodeDataset(Dataset):
    @staticmethod
    def group_readmission(relative_readmission):
        if relative_readmission >= 180:
            return 6
        else:
            return relative_readmission // 30

    @staticmethod
    def construct_patient_text(patient_df, tokenizer):
        def format_patient_row(row):
            return [
                tokenizer.cls_token,  tokenizer.sep_token,  row['timestamp'],  ':',  
                ('. '.join(row['texts']) + '.').replace('<CR><LF>','.\n')
            ]
        texts = patient_df.apply(format_patient_row, axis=1).values.tolist()
        
        return ' '.join(list(chain(*texts)))
        
    # Arguments:
    # base_path - base folder path to the patient data
    # patient_id_list - id list of the patient
    # tokenizer - tokenizer for tokenizing patient clinical notes
    # max_len - maximum length of the clinical notes data
    # max_duration - maximum duration of the previous health record data
    # use_tabular - add tabular features to the returned data
    def __init__(self, base_path, patient_id_list, tokenizer, max_length, max_duration, use_tabular=False):
        self.base_path = base_path
        self.patient_id_list = patient_id_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_duration = max_duration
        self.use_tabular = use_tabular
        
    def __getitem__(self, index):
        # Load file
        patient_id = self.patient_id_list[index]['pseudo_patient_key']
        episode_id = self.patient_id_list[index]['pseudo_episode_key']
        patient_df = pd.read_pickle(f'{self.base_path}/{patient_id}.pkl.gz')
                
        # Filter patient data
        max_pred_timestamp = patient_df.loc[patient_df['pseudo_episode_key'] == episode_id, 'first_relative_timestamp'].values[0]
        patient_df = patient_df[
            (patient_df['first_relative_timestamp'] >= max_pred_timestamp - self.max_duration) & 
            (patient_df['first_relative_timestamp'] <= max_pred_timestamp)
        ]
        
        # Extract texts, features, & labels
        next_diags = patient_df['next_diagnosis'].values[-1]
        next_mortals = patient_df['next_mortality'].values[-1]
        next_rel_readmis = EHREpisodeDataset.group_readmission(patient_df['next_relative_readmission'].values[-1])

        features = np.stack(patient_df['features'].values)[:,:-1]
        texts = EHREpisodeDataset.construct_patient_text(patient_df, self.tokenizer)
        labels = [next_diags, next_rel_readmis, next_mortals]

        # Process texts
        patient_data = self.tokenizer(texts, max_length=self.max_length, return_attention_mask=True, truncation=True)
        patient_data['labels'] = labels
        
        if self.use_tabular:
            patient_data['features'] = features
        
        return patient_data
        
    def __len__(self):
        return len(self.patient_id_list)