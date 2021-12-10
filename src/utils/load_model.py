from src.models.biolm_model import BioLMModel
from src.models.bioelectra_model import BioELECTRAModel
from src.models.longmodel import LongBioLMModel, LongBioELECTRAModel


def load_model_and_tokenizer(model_args):
    if model_args.model == 'scratch':
        pass
    elif model_args.model == 'biolm':
        model_class = BioLMModel(model_path = "models/RoBERTa-base-PM-M3-Voc-distill-align-hf",
                           num_labels = 1699 + 7 + 2,
                           num_multi_labels = [1699, 7, 2])
        model = model_class.model
        tokenizer = model_class.tokenizer
    elif model_args.model == 'bioelectra':
        model_class = BioELECTRAModel(model_path = "kamalkraj/bioelectra-base-discriminator-pubmed",
                           num_labels = 1699 + 7 + 2,
                           num_multi_labels = [1699, 7, 2])
        model = model_class.model
        tokenizer = model_class.tokenizer
    elif model_args.model == 'longbiolm':
        model_class = LongBioLMModel(model_path = "models/RoBERTa-base-PM-M3-Voc-distill-align-hf",
                                num_labels = 1699 + 7 + 2,
                                num_multi_labels = [1699, 7, 2],
                                attention_window = [512] * 12)
        model = model_class.model
        tokenizer = model_class.tokenizer
    elif model_args.model == 'longbioelectra':
        model_class = LongBioELECTRAModel(model_path = "kamalkraj/bioelectra-base-discriminator-pubmed",
                                num_labels = 1699 + 7 + 2,
                                num_multi_labels = [1699, 7, 2],
                                attention_window = [512] * 12)
        model = model_class.model
        tokenizer = model_class.tokenizer
    else:
        raise ValueError(f'Invalid `mode_type`: {model_args.model}')
    return model, tokenizer