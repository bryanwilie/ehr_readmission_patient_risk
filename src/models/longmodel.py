from transformers import ElectraConfig, RobertaConfig, ElectraTokenizerFast, RobertaTokenizer
from transformers import LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer


def convert_roberta_like_to_longformer(state_dict, model_name):
    orig_keys = [key for key in state_dict]
    for key in orig_keys:
        if model_name in key:
            new_key = key.replace(model_name,'longformer')
            state_dict[new_key] = state_dict[key]
            if 'query.' in new_key:
                state_dict[new_key.replace('.query.','.query_global.')] = state_dict[key]
            if 'key.' in new_key:
                state_dict[new_key.replace('.key.','.key_global.')] = state_dict[key]
            if 'value.' in new_key:
                state_dict[new_key.replace('.value.','.value_global.')] = state_dict[key]

            if '.position_embeddings' in new_key:
                state_dict[new_key] = state_dict[new_key].repeat([8,1])

            if '.position_ids' in new_key:
                    state_dict[new_key] = torch.arange(state_dict[key].shape[1] * 8).view(1, -1)
            del state_dict[key]
    return state_dict


class LongBioLMModel():
    def __init__(self, model_path, num_labels, num_multi_labels, attention_window):

        self.config = RobertaConfig.from_pretrained(model_path)
        self.config.num_labels = num_labels
        self.config.num_multi_labels = num_multi_labels
        self.config.attention_window = attention_window
        self.config.max_position_embeddings = self.config.max_position_embeddings * 8
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = LongformerForSequenceClassification(config=self.config)
        
        state_dict = convert_roberta_like_to_longformer(self.model.state_dict(), 'roberta')
        self.model.load_state_dict(state_dict, strict = True)
        
    def forward(self, inputs, return_tensors='pt'):
    
        tokenized_inputs = self.tokenizer(inputs, return_tensors=return_tensors)
        out = self.model(**tokenized_inputs)
        
        return out
    
    
class LongBioELECTRAModel():
    def __init__(self, model_path, num_labels, num_multi_labels, attention_window):

        self.config = ElectraConfig.from_pretrained(model_path)
        self.config.num_labels = num_labels
        self.config.num_multi_labels = num_multi_labels
        self.config.attention_window = attention_window
        self.config.max_position_embeddings = self.config.max_position_embeddings * 8
        
        self.tokenizer = ElectraTokenizerFast.from_pretrained(model_path)
        self.model = LongformerForSequenceClassification(config=self.config)
        
        state_dict = convert_roberta_like_to_longformer(self.model.state_dict(), 'electra')
        self.model.load_state_dict(state_dict, strict = True)
        
    def forward(self, inputs, return_tensors='pt'):
    
        tokenized_inputs = self.tokenizer(inputs, return_tensors=return_tensors)
        out = self.model(**tokenized_inputs)
        
        return out