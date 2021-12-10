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
        
    # Arguments:
    # base_path - base folder path to the patient data
    # patient_id_list - id list of the patient
    # tokenizer - tokenizer for tokenizing patient clinical notes
    # clinical_notes_df - dataframe of clinical notes for all patients
    # eval_patient_id_list - validation & test patient id list. This is used for 
    #               filtering out last episode label from the training data
    def __init__(self, base_path, patient_episode_data, tokenizer, 
                    clinical_notes_df=None, eval_patient_episode_data=None):
        self.base_path = base_path
        self.patient_episode_data = patient_episode_data
        self.tokenizer = tokenizer
        self.clinical_notes_df = clinical_notes_df 
        self.eval_patient_episode_data = eval_patient_episode_data
        
    def __getitem__(self, index):
        # Load file
        patient_id = self.patient_id_list[index]
        patient_df = pd.read_pickle(f'{self.base_path}/{patient_id}.pkl.gz')
        if patient_id in self.eval_patient_id_list:
            # Extract texts, features, & labels
            next_diags = patient_df['next_diagnosis'].values
            next_mortals = patient_df['next_mortality'].values
            next_rel_readmis = self.group_readmission(patient_df['next_relative_readmission'].values)

            last_readmission = patient_df['readmission'].values[-1]
            features = np.stack(patient_df['features'].values)[:,:-1]
            texts = 
        else:
            # Extract texts, features & labels                  
            next_diags = patient_df['next_diagnosis'].values[:-1],
            next_mortals = patient_df['next_mortality'].values[:-1]
            next_rel_readmis = self.group_readmission(patient_df['next_relative_readmission'].values[:-1])
            
            last_readmission = patient_df['readmission'].values[-2]
            features = np.stack(patient_df['features'].values)[:-1,:-1]
            texts = 
        labels = [next_diags, next_rel_readmis, next_mortals]
            
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