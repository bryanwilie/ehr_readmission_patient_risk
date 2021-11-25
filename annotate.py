import csv
import pandas as pd
from ordered_set import OrderedSet #4.0.2
from negex.python.negex import negTagger, sortRules


def find_marked_terms(sentence, mark='[NEGATED]'):
    marked_term_count = sentence.count(mark)//2
    
    marked_terms = OrderedSet()
    if marked_term_count > 0:
        unpack_marked_words = sentence.split(mark)
        for i in range(marked_term_count):
            # store negated terms
            marked_term = unpack_marked_words[2*(i+1)-1]
            marked_terms.append(marked_term)

    return marked_terms

def get_patients_annotated_terms(unique_terms, df_to_annotate,
                               negex_triggers_text_filepath = './negex/python/negex_triggers.txt'):
    
    rfile = open(negex_triggers_text_filepath)
    irules = sortRules(rfile.readlines())

    count = 0
    tagged_sentences = []
    for i, row in df_to_annotate.iterrows():
        #skip header
        if count == 0:
            count = count+1
            continue
        if count == 1:
            tagger = negTagger(sentence = row[2], phrases = unique_terms, rules = irules, negP=False)
            tagged_sentences.append(tagger.getNegTaggedSentence())

    patients_negated_terms, patients_recognized_terms = [], []
    for i, sentence in enumerate(tagged_sentences):
        negated_terms = find_marked_terms(sentence, '[NEGATED]')
        recognized_terms = find_marked_terms(sentence, '[PHRASE]')

        # store the patient's negated terms
        patients_negated_terms.append(negated_terms)
        patients_recognized_terms.append(recognized_terms)
        
    rfile.close()
    
    df_result = pd.DataFrame({'negated_terms': patients_negated_terms,
                              'recognized_terms': patients_recognized_terms,
                              'tagged_sentences': tagged_sentences})
        
    return df_result