from src.datasets.trial_datasets import short_dataset_1, long_mixed_dataset_1


def load_tokenized_dataset(data_args, tokenizer):
    if data_args.dataset_type == 'short_dataset_1':
        return short_dataset_1(tokenizer)
    elif data_args.dataset_type == 'long_mixed_dataset_1':
        return long_mixed_dataset_1(tokenizer)