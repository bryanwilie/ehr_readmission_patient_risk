import torch
from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizerFast
import torch.nn as nn

class BioLMModel(nn.Module):
    def __init__(self, model_path, num_labels, num_multi_labels):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(model_path)
        self.config.num_labels = num_labels
        self.config.num_multi_labels = num_multi_labels
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(
                        model_path,
                        config=self.config
                     )
        
    def forward(self, inputs):
        out = self.model(**inputs)
        return out