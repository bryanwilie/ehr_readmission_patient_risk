## UNTESTED - USE WITH CAUTION

###
# EHR Dataset
###
class EHRDataset(Dataset):
    @staticmethod
    def group_readmission(relative_readmission):
        relative_readmission[relative_readmission <= 30] = 0
        relative_readmission[relative_readmission <= 60] = 1
        relative_readmission[relative_readmission <= 90] = 2
        relative_readmission[relative_readmission <= 120] = 3
        relative_readmission[relative_readmission <= 150] = 4
        relative_readmission[relative_readmission <= 180] = 5
        relative_readmission[relative_readmission > 180] = 6
        return relative_readmission
        
    def __init__(self, patient_to_path_dict, tokenizer, eval_patient_to_path_dict=None):
        self.patient_to_path_dict = patient_to_path_dict
        self.patient_id_list = list(self.patient_to_path_dict.keys())
        self.tokenizer = tokenizer
        
        self.eval_patient_to_path_dict = None
        self.eval_patient_id_list = []
        if eval_patient_to_path_dict is not None:
            self.eval_patient_to_path_dict = eval_patient_to_path_dict
            self.eval_patient_id_list = list(self.eval_patient_to_path_dict.keys())
            
        
    def __getitem__(self, index):
        if index < len(self.patient_id_list):
            # Load file
            patient_id = self.patient_id_list[index]
            data_path, note_path = self.patient_to_path_dict[patient_id]
            patient_df = pd.read_pickle(data_path).dropna()
            notes_df = pd.read_pickle(note_path).dropna()
            
            # Extract texts, features, & labels            
            next_diags = patient_df['next_diagnosis'].values
            next_mortals = patient_df['next_mortality'].values
            next_rel_readmis = self.group_readmission(patient_df['next_relative_readmission'].values)
            labels = [next_diags, next_rel_readmis, next_mortals]

            last_readmission = patient_df['readmission'].values[-1]
            features = np.stack(patient_df['features'].values)[:,:-1]
            texts = notes_df.loc[notes_df['timestamp'] < patient_df.iloc[-1]['timestamp'],'texts'].values
        else:
            # Load file
            patient_id = self.eval_patient_id_list[index - len(self.patient_id_list)]
            data_path = self.eval_patient_to_path_dict[patient_id]
            patient_df = pd.read_pickle(data_path).dropna()
               
            # Extract texts, features & labels                  
            next_diags = patient_df['next_diagnosis'].values[:-1],
            next_mortals = patient_df['next_mortality'].values[:-1]
            next_rel_readmis = self.group_readmission(patient_df['next_relative_readmission'].values[:-1])
            labels = [next_diags, next_rel_readmis, next_mortals]
            
            last_readmission = patient_df['readmission'].values[-2]
            features = np.stack(patient_df['features'].values)[:-1,:-1]
            texts = patient_df.loc[patient_df['readmission'] < last_readmission, 'texts'].values            
            
    # Process texts
    patient_data = self.tokenizer(texts)
    patient_data['labels'] = labels
    patient_data['features'] = features
    patient_data['input_ids'] = list(chain.from_iterable(patient_data['input_ids']))
    patient_data['attention_mask'] = list(chain.from_iterable(patient_data['attention_mask']))            
    if 'token_type_ids' in notes_data:
        patient_data['token_type_ids'] = [list(chain.from_iterable(patient_data['token_type_ids']))]

    return patient_data
        
    def __len__(self):
        return len(self.patient_id_list) + len(self.eval_patient_id_list)