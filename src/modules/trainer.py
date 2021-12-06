import torch
from transformers import Trainer, TrainingArguments


class MultilabelTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()

        rem_logits = logits
        loss = 0
                
        if 'num_multi_labels' in self.model.config.__dict__:
            print('a')
            for i, num_label in enumerate(self.model.config.num_multi_labels):
                class_logits = rem_logits[:,:num_label]
                rem_logits = rem_logits[:,num_label:]
                
                class_loss = loss_fct(class_logits, labels[:,i])
                loss += class_loss
        else:
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss