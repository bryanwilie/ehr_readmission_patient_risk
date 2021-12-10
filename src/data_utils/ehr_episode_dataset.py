import numpy as np
import pandas as pd
from itertools import chain

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
        texts = patient_df.apply(
            lambda x: [tokenizer.cls_token,  tokenizer.sep_token,  x['timestamp'],  ':',  ('. '.join(x['texts']) + '.').replace('<CR><LF>','.\n')]
        , axis=1).values.tolist()
        return ' '.join(list(chain(*texts)))
        
    # Arguments:
    # base_path - base folder path to the patient data
    # patient_id_list - id list of the patient
    # tokenizer - tokenizer for tokenizing patient clinical notes
    # clinical_notes_df - dataframe of clinical notes for all patients
    # eval_patient_id_list - validation & test patient id list. This is used for 
    #               filtering out last episode label from the training data
    def __init__(self, base_path, patient_id_list, tokenizer, max_length):
        self.base_path = base_path
        self.patient_id_list = patient_id_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, index):
        # Load file
        patient_id = self.patient_id_list[index]['pseudo_patient_key']
        episode_id = self.patient_id_list[index]['pseudo_episode_key']
        patient_df = pd.read_pickle(f'{self.base_path}/{patient_id}.pkl.gz')
        
        # Extract texts, features, & labels
        next_diags = patient_df['next_diagnosis'].values
        next_mortals = patient_df['next_mortality'].values
        next_rel_readmis = EHREpisodeDataset.group_readmission(patient_df['next_relative_readmission'].values)

        last_readmission = patient_df['readmission'].values[-1]
        features = np.stack(patient_df['features'].values)[:,:-1]
        texts = EHREpisodeDataset.construct_patient_text(patient_df['features'], self.tokenizer)
        labels = [next_diags, next_rel_readmis, next_mortals]

        # Process texts
        patient_data = self.tokenizer(texts, max_length=self.max_length, return_attention_mask=True)
        patient_data['labels'] = labels
        patient_data['features'] = features
        patient_data['input_ids'] = list(chain.from_iterable(patient_data['input_ids']))
        patient_data['attention_mask'] = list(chain.from_iterable(patient_data['attention_mask']))            
        if 'token_type_ids' in notes_data:
            patient_data['token_type_ids'] = [list(chain.from_iterable(patient_data['token_type_ids']))]

        return patient_data
        
    def __len__(self):
        return len(self.patient_id_list)