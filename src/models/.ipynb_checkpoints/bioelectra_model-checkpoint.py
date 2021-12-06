import torch
from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizerFast


class BioELECTRAModel():
    def __init__(self, model_path, num_labels, num_multi_labels):
        
        self.config = ElectraConfig.from_pretrained(model_path)
        self.config.num_labels = num_labels
        self.config.num_multi_labels = num_multi_labels
        
        self.tokenizer = ElectraTokenizerFast.from_pretrained(model_path)
        self.model = ElectraForSequenceClassification.from_pretrained(
                        model_path,
                        config=self.config)
        
    def forward(self, inputs, return_tensors='pt'):
    
        tokenized_inputs = self.tokenizer(inputs, return_tensors=return_tensors)
        out = self.model(**tokenized_inputs)
        
        return out