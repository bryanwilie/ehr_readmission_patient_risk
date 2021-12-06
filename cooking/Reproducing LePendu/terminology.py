import re
import pandas as pd
from owlready2 import get_ontology
from owlready2.pymedtermino2.umls import import_umls

from helper import choose_label

path = 'data_pool/'


def obtain_lexicons(terminologies = ["ICD9", "SNOMEDCT_US", "RXNORM", "NDF"], 
                         hdo = True, 
                         additional_from_negex=True):
    """
    read more from: https://owlready2.readthedocs.io/en/latest/pymedtermino2.html
    """
    
    import_umls("umls-2021AB-full.zip", terminologies = terminologies)

    PYM = get_ontology("http://PYM/").load()
    PYM_classes_list = list(PYM.classes())

    terms = []
    for i, PYM_class in enumerate(PYM_classes_list):
        labels = PYM_class.label
        if len(labels) > 0:
            for label in labels:
                terms.append(label)
                
    unique_terms = list(set(terms))
    while '' in unique_terms:
        unique_terms.pop(unique_terms.index(''))

    if hdo:
        ### Human Disease Ontology
        print('Obtaining Human Disease Ontology')
        df_hdo = pd.read_csv(path + 'Human Disease Ontology/DOID.csv.gz', compression='gzip')

        df_hdo['processed_label'] = df_hdo.apply(lambda x: choose_label(x), axis=1)
        df_hdo_selected = df_hdo[df_hdo['Obsolete']==False][['processed_label', 'definition', 'Synonyms']]

        hdo_unique_terms = list(set(df_hdo_selected[~df_hdo_selected['processed_label'].isna()].processed_label.tolist()))

        ### Combine them all together
        unique_terms = list(set(unique_terms + hdo_unique_terms))
        
    if additional_from_negex:
        ### Samples from Negex github
        print('Obtaining Negex samples\' Concept')
        patients_clinical_reports_text_filepath = './negex/python/Annotations-1-120.txt'
        df = pd.read_csv(patients_clinical_reports_text_filepath, delimiter='\t')
        
        additional_terms = [re.sub(' +', ' ', concept) for concept in df.Concept.tolist()]
        unique_terms = list(set(unique_terms + additional_terms))
    
    print(len(unique_terms), 'unique terms obtained')
    
    return unique_terms