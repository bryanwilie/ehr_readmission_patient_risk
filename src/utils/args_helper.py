from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

from datetime import datetime


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model: str = field(
        default = 'scratch',
        metadata = {
            'help': '',
            'choices':['scratch', 'biolm', 'bioelectra', 'longbiolm', 'longbioelectra']}
    )
    finetune_whole_lm: Optional[bool] = field(
        default = False,
        metadata = {
            'help': 'to finetune the whole lm or not'}
    ) 
    pretrained_config_name: Optional[str] = field(
        default = None, 
        metadata = {"help": "Pretrained config name or path if not the same as model_name"}
    )
    early_stop: Optional[bool] = field(
        default = False,
        metadata = {"help": "Whether to do early stopping in the training process."}
    )
    early_stopping_patience: Optional[int] = field(
        default = 1,
        metadata = {"help": "`metric_for_best_model` to stop training when"
                            "the specified metric worsens for `early_stopping_patience`"
                            "evaluation calls."}
    )
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_type: str = field(
        default = 'text',
        metadata = {
            'help':'To select whether the training will include text only data,'
                   'or text and tabular data',
            'choices':['text', 'text_tabular', 'short_dataset_1', 'long_mixed_dataset_1']}
    )
    patient_base_path: str = field(
        default = './dataset/preprocess_patient_data',
        metadata = {
            'help':'Patient folder base path'
        }
    )
    train_path: str = field(
        default = './dataset/train_patient_episode_metadata.json',
        metadata = {
            'help':'Train data path'
        }
    )
    valid_path: str = field(
        default = './dataset/valid_patient_episode_metadata.json',
        metadata = {
            'help':'Valid data path'
        }
    )
    test_path: str = field(
        default = './dataset/test_patient_episode_metadata.json',
        metadata = {
            'help':'Test data path'
        }
    )
    max_text_len: int = field(
        default = 4096,
        metadata = {
            'help':'max text length'}
    )
    max_duration: int = field(
        default = 24 * 365 * 10,
        metadata = {
            'help':'max duration from last diagnosis in hour'}
    )
    diagnosis_aggregation: str = field(
        default = 'sum',
        metadata = {
            'help': 'sum , etc'}
    )    
    clipping_criteria: str = field(
        default = '',
        metadata = {
            'help': 'string to choose which [list of index] should be used'
            'to clip some of the low frequencies categories'}
    )
    train_file: Optional[str] = field(
        default = None, 
        metadata = {"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default = None,
        metadata = {"help": "An optional input evaluation data file to evaluate"
                            "the perplexity on (a text file)."}
    )
    overwrite_cache: bool = field(
        default = False, 
        metadata = {"help": "Overwrite the cached training and evaluation sets"}
    )
    cache_dir: Optional[str] = field(
        default = None,
        metadata = {"help": "Path to directory to store the pretrained models "
                            "downloaded from huggingface.co"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default = 8,
        metadata = {"help": "The number of processes to use for the preprocessing."}
    )
    pad_to_max_length: bool = field(
        default = True,
        metadata = {
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when "
                    "batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."}
    )
    

@dataclass
class EHRTrainingArguments(TrainingArguments):
    exp: Optional[str] = field(
        default = 'ehr_hadcl_' + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-7]+'_',
        metadata = {"help": "experiment path to store models and results"},
    )
#     max_save_num: Optional[int] = field(
#         default=3, 
#         metadata={"help": "the maximum number of models to save"}
#     )
#     patience: Optional[int] = field(
#         default=3, 
#         metadata={"help": "patient epochs before early stopping"}
#     )
#     epochs: Optional[int] = field(
#         default=1, 
#         metadata={"help": "number of training epochs"}
#     )
#     batch_size: Optional[int] = field(
#         default=2, 
#         metadata={"help": "batch size"},
#     )
#     lr: Optional[float] = field(
#         default=1e-5, 
#         metadata={"help": "ppo learning rate"},
#     )
#     use_lr_scheduler: Optional[bool] = field(
#         default=False, 
#         metadata={"help": "use scheduler or not"},
#     )
#     fp16: Optional[bool] = field(
#         default=False
#     )
#     optimizer: Optional[str] = field(
#         default="Adam", 
#         metadata={"help": "Choose the optimizer to use. Default Adam.",
#                   "choices": ["Adam", "RecAdam"]}
#     )