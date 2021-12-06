from itertools import chain


def short_dataset_1(tokenizer):
    sentence = "The quick brown fox jumps over the lazy dog"
    labels = [[0, 1, 2],[0, 2, 1]]
    dataset = [sentence, sentence]
    tokenized_dataset = tokenizer(dataset, return_tensors="pt")
    
    return tokenized_dataset, labels


def long_mixed_dataset_1(tokenizer):
    labels = [[0, 1, 2]]
    features = [0,1,1,1,0,0]
    patient_notes = [
        "The quick brown fox jumps over the lazy dog",
        "round the rugged rocks the ragged rascal ran",
        "peter piper pickled pepper picker",
        "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod",
        "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,",
        "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo",
        "consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse",
        "cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non",
        "proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    ] * 10

    notes_data = tokenizer(patient_notes)
    # notes_data['labels'] = [labels]
    # notes_data['features'] = features
    notes_data['input_ids'] = [list(chain.from_iterable(notes_data['input_ids']))]
    notes_data['attention_mask'] = [list(chain.from_iterable(notes_data['attention_mask']))]
    if 'token_type_ids' in notes_data:
        notes_data['token_type_ids'] = [list(chain.from_iterable(notes_data['token_type_ids']))]

    notes_data = tokenizer.pad(notes_data, pad_to_multiple_of=512, return_tensors='pt')
    
    return notes_data, labels